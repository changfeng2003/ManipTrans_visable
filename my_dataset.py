import torch
from torch.utils.data import Dataset
import numpy as np
import os
from .transform import aa_to_rotmat, rotmat_to_aa

class MyDatasetMTDexHand(Dataset):
    def __init__(self, data_root, split='all', hand_order: str = 'rh_first'):
        self.data_root = data_root
        pth_files = [f for f in os.listdir(self.data_root) if f.endswith('.pth')]
        pth_files.sort()
        self.data_files = [os.path.join(self.data_root, f) for f in pth_files]
        self.split = split
        # 新增：控制上游左右手顺序（'rh_first' | 'lh_first'）
        self.hand_order = hand_order

    def __len__(self):
        # 方案B：把一个目录内的所有 .pth 聚成一个时间序列样本
        return 1 if len(self.data_files) > 0 else 0

    def __getitem__(self, idx):
        # 将目录下所有 .pth 合并为单个样本的时序数据
        assert len(self.data_files) > 0, "Empty data directory"

        def _orient_to_aa(x):
            # 支持 (1,3,3)/(3,3) 旋转矩阵 或 (3,) 轴角
            t = x if torch.is_tensor(x) else torch.tensor(x)
            if t.ndim == 3 and t.shape[-2:] == (3, 3):
                R = t[0] if t.shape[0] == 1 else t
                return rotmat_to_aa(R)
            elif t.ndim == 2 and t.shape[-2:] == (3, 3):
                return rotmat_to_aa(t)
            elif t.shape[-1] == 3 and t.ndim == 1:
                return t
            else:
                return t.reshape(-1)[-3:]

        def _idx_or_zero(i: int, size: int) -> int:
            i = int(i)
            return i if size > i else 0

        rh_idx, lh_idx = (0, 1) if self.hand_order == 'rh_first' else (1, 0)

        wrist_pos_rh_list, wrist_pos_lh_list = [], []
        wrist_rot_rh_list, wrist_rot_lh_list = [], []
        obj_T_list = []
        cam_t_rh_list, cam_t_lh_list = [], []  

        # 逐文件累积为时间序列
        for f in self.data_files:
            data = torch.load(f)
            obj_pose = data['obj_pose']
            obj_pose = obj_pose if torch.is_tensor(obj_pose) else torch.tensor(obj_pose)
            obj_T_list.append(obj_pose)

            pred_mano_params = data['pred_mano_params']
            pred_vertices = data['pred_vertices']
            kp3d = data.get('pred_keypoints_3d', None)

            # global_orient 支持形状 [2,1,3,3] / [2,3,3]
            go = pred_mano_params['global_orient']
            # 右手/左手旋转（根据 hand_order 取索引），对单手数据做安全索引
            if torch.is_tensor(go) and go.ndim >= 3:
                if go.ndim == 4:  # [N,1,3,3]
                    idx_rh = _idx_or_zero(rh_idx, go.shape[0])
                    idx_lh = _idx_or_zero(lh_idx, go.shape[0])
                    rh_R = go[idx_rh, 0]
                    lh_R = go[idx_lh, 0]
                else:  # [N,3,3]
                    idx_rh = _idx_or_zero(rh_idx, go.shape[0])
                    idx_lh = _idx_or_zero(lh_idx, go.shape[0])
                    rh_R = go[idx_rh]
                    lh_R = go[idx_lh]
            else:
                # 退化情况：按单手处理
                rh_R = go
                lh_R = go
            wrist_rot_rh = _orient_to_aa(rh_R)
            wrist_rot_lh = _orient_to_aa(lh_R)

            # wrist 位置（优先用 MediaPipe 21 点做 hack）
            if kp3d is not None and torch.is_tensor(kp3d) and kp3d.ndim == 3 and kp3d.shape[1] >= 10:
                idx_rh = _idx_or_zero(rh_idx, kp3d.shape[0])
                idx_lh = _idx_or_zero(lh_idx, kp3d.shape[0])
                rh_wrist = kp3d[idx_rh, 0]
                rh_middle_mcp = kp3d[idx_rh, 9]
                lh_wrist = kp3d[idx_lh, 0]
                lh_middle_mcp = kp3d[idx_lh, 9]
                wrist_pos_rh = rh_wrist - (rh_middle_mcp - rh_wrist) * 0.25
                wrist_pos_lh = lh_wrist - (lh_middle_mcp - lh_wrist) * 0.25
            else:
                # 回退用 pred_vertices 的 0 号顶点（或相应定义的手腕点）
                pv = pred_vertices if torch.is_tensor(pred_vertices) else torch.tensor(pred_vertices)
                idx_rh = _idx_or_zero(rh_idx, pv.shape[0])
                idx_lh = _idx_or_zero(lh_idx, pv.shape[0])
                wrist_pos_rh = pv[idx_rh, 0]
                wrist_pos_lh = pv[idx_lh, 0]

            wrist_pos_rh_list.append(wrist_pos_rh)
            wrist_pos_lh_list.append(wrist_pos_lh)
            wrist_rot_rh_list.append(wrist_rot_rh)
            wrist_rot_lh_list.append(wrist_rot_lh)

            # 新增：读取相机位移 pred_cam_t_full（或 pred_cam_t）
            cam_t = data.get('pred_cam_t_full', data.get('pred_cam_t', None))
            if cam_t is not None:
                t = cam_t if torch.is_tensor(cam_t) else torch.tensor(cam_t)
                if t.ndim == 2 and t.shape[-1] == 3:
                    # 形如 [N,3]（双手或单手）
                    idx_rh = _idx_or_zero(rh_idx, t.shape[0])
                    idx_lh = _idx_or_zero(lh_idx, t.shape[0])
                    t_rh = t[idx_rh]
                    t_lh = t[idx_lh]
                elif t.ndim == 1 and t.numel() == 3:
                    t_rh = t
                    t_lh = t
                else:
                    # 其他情况：尽力取前3个数
                    flat = t.reshape(-1)
                    t_rh = flat[:3]
                    t_lh = flat[:3]
                cam_t_rh_list.append(t_rh)
                cam_t_lh_list.append(t_lh)

        # 拼接为时序张量
        wrist_pos_rh = torch.stack([p if torch.is_tensor(p) else torch.tensor(p) for p in wrist_pos_rh_list], dim=0)
        wrist_pos_lh = torch.stack([p if torch.is_tensor(p) else torch.tensor(p) for p in wrist_pos_lh_list], dim=0)
        wrist_rot_rh = torch.stack([a if torch.is_tensor(a) else torch.tensor(a) for a in wrist_rot_rh_list], dim=0)
        wrist_rot_lh = torch.stack([a if torch.is_tensor(a) else torch.tensor(a) for a in wrist_rot_lh_list], dim=0)
        state_manip_obj = torch.stack([T if torch.is_tensor(T) else torch.tensor(T) for T in obj_T_list], dim=0)

        # 兼容单手可视化：默认用右手序列（受 hand_order 影响且经过安全索引）
        wrist_pos_single = wrist_pos_rh
        wrist_rot_single = wrist_rot_rh

        out = {
            "wrist_pos": wrist_pos_single,
            "wrist_rot": wrist_rot_single,
            "wrist_pos_rh": wrist_pos_rh,
            "wrist_rot_rh": wrist_rot_rh,
            "wrist_pos_lh": wrist_pos_lh,
            "wrist_rot_lh": wrist_rot_lh,
            "state_manip_obj_rh": state_manip_obj,
            "state_manip_obj_lh": state_manip_obj,
            "dexhand": "inspire",
            "oid_rh": "0",
            "oid_lh": "0",
            "obj_rh_path": "/home/azh/桌面/code/ManipTrans/data/HoldPaperCup2MT/coacd_object_preview/obj_urdf_example.urdf",
            "obj_lh_path": "/home/azh/桌面/code/ManipTrans/data/HoldPaperCup2MT/coacd_object_preview/obj_urdf_example.urdf",
            "scene_objs": [],
        }

        if len(cam_t_rh_list) > 0:
            pred_cam_t_full_rh = torch.stack(cam_t_rh_list, dim=0)
            pred_cam_t_full_lh = torch.stack(cam_t_lh_list, dim=0)
            out["pred_cam_t_full_rh"] = pred_cam_t_full_rh
            out["pred_cam_t_full_lh"] = pred_cam_t_full_lh
            # 单手默认取 RH
            out["pred_cam_t_full"] = pred_cam_t_full_rh

        return out