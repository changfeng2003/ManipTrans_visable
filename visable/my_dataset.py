import torch
from torch.utils.data import Dataset
import numpy as np
import os
import math
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
        # 新增：左右手本地旋转偏置（度），支持环境变量覆盖
        def _parse_deg_env(name: str, default_tuple):
            v = os.getenv(name, None)
            if v is None:
                return default_tuple
            try:
                parts = [float(x.strip()) for x in v.split(',')]
                if len(parts) < 3:
                    parts += [0.0] * (3 - len(parts))
                return (parts[0], parts[1], parts[2])
            except Exception:
                return default_tuple
        # 默认：左手绕 X 轴 180°，其余 0°；右手默认不偏置
        self.lh_rot_offset_deg = _parse_deg_env('LH_EXTRA_ROT_DEG', (150.0, 150.0, 0.0))
        self.rh_rot_offset_deg = _parse_deg_env('RH_EXTRA_ROT_DEG', (0.0, 0.0, 0.0))
        print(f"[MyDatasetMTDexHand] LH_EXTRA_ROT_DEG={self.lh_rot_offset_deg}, RH_EXTRA_ROT_DEG={self.rh_rot_offset_deg}")

    def __len__(self):
        # 方案B：把一个目录内的所有 .pth 聚成一个时间序列样本
        return 1 if len(self.data_files) > 0 else 0

    # 新增：将度制 XYZ 欧拉角转换为旋转矩阵（按本地后乘意图，顺序为 Rx @ Ry @ Rz）
    def _euler_xyz_to_rotmat(self, rx_deg: float, ry_deg: float, rz_deg: float, device=None, dtype=None):
        rx = math.radians(rx_deg)
        ry = math.radians(ry_deg)
        rz = math.radians(rz_deg)
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        Rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], device=device, dtype=dtype)
        Ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], device=device, dtype=dtype)
        Rz = torch.tensor([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        return Rx @ Ry @ Rz

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
        # 新增：逐帧记录 right_bool（右手是否出镜）
        right_bool_list = []

        # 逐文件累积为时间序列
        for f in self.data_files:
            data = torch.load(f)
            obj_pose = data['obj_pose']
            obj_pose = obj_pose if torch.is_tensor(obj_pose) else torch.tensor(obj_pose)
            obj_T_list.append(obj_pose)

            pred_mano_params = data['pred_mano_params']
            pred_vertices = data['pred_vertices']
            kp3d = data.get('pred_keypoints_3d', None)
            # 读取 right_bool 字段（若存在）
            rb = data.get('right_bool', None)
            if rb is not None:
                rb_t = rb if torch.is_tensor(rb) else torch.tensor(rb)
                # 转为 Python bool（支持 [1] 张量或标量；若为更长向量则用 any() 判定）
                if rb_t.numel() > 1:
                    rb_flag = bool((rb_t != 0).any().item())
                else:
                    rb_flag = bool(rb_t.view(-1)[0].item() != 0)
                right_bool_list.append(rb_flag)

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
        

            # 新增：应用左右手本地后乘旋转偏置（若 rh_R/lh_R 为 3x3 旋转矩阵）
            try:
                # 统一转为 torch.Tensor 以便设备/类型一致
                if isinstance(rh_R, np.ndarray):
                    rh_R = torch.from_numpy(rh_R).float()
                if isinstance(lh_R, np.ndarray):
                    lh_R = torch.from_numpy(lh_R).float()
                # 仅当形状为 [3,3] 时应用
                if torch.is_tensor(rh_R) and rh_R.ndim == 2 and rh_R.shape == (3, 3):
                    R_off_rh = self._euler_xyz_to_rotmat(*self.rh_rot_offset_deg, device=rh_R.device, dtype=rh_R.dtype)
                    rh_R = rh_R @ R_off_rh
                if torch.is_tensor(lh_R) and lh_R.ndim == 2 and lh_R.shape == (3, 3):
                    R_off_lh = self._euler_xyz_to_rotmat(*self.lh_rot_offset_deg, device=lh_R.device, dtype=lh_R.dtype)
                    lh_R = lh_R @ R_off_lh
            except Exception:
                pass

            # 调整左手手腕旋转：将3x3旋转矩阵的第2、3列乘以-1（等价于右乘diag(1,-1,-1)）
            

            wrist_rot_rh = _orient_to_aa(rh_R)
            wrist_rot_lh = _orient_to_aa(lh_R)

            # wrist 位置（当前回退使用 pred_vertices 的 0 号顶点）
            pv = pred_vertices if torch.is_tensor(pred_vertices) else torch.tensor(pred_vertices)
            idx_rh = _idx_or_zero(rh_idx, pv.shape[0])
            idx_lh = _idx_or_zero(lh_idx, pv.shape[0])
            wrist_pos_rh = pv[idx_rh, 0]
            wrist_pos_lh = pv[idx_lh, 0]
            # 对左手手腕位置的 x 轴做镜像：乘以 -1
            if torch.is_tensor(wrist_pos_lh):
                wrist_pos_lh = wrist_pos_lh.clone()
                wrist_pos_lh[..., 0] = -wrist_pos_lh[..., 0]
            else:
                wrist_pos_lh = wrist_pos_lh.copy() if hasattr(wrist_pos_lh, 'copy') else wrist_pos_lh
                wrist_pos_lh[0] = -wrist_pos_lh[0]

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

        # 兼容单手可视化：若提供 right_bool，则逐帧选择 RH/LH；否则默认用 RH
        if len(right_bool_list) == len(self.data_files):
            _device = wrist_pos_rh.device
            right_bool_tensor = torch.tensor(right_bool_list, dtype=torch.bool, device=_device)
            rb_exp = right_bool_tensor[:, None].expand_as(wrist_pos_rh)
            wrist_pos_single = torch.where(rb_exp, wrist_pos_rh, wrist_pos_lh)
            rb_exp_rot = right_bool_tensor[:, None].expand_as(wrist_rot_rh)
            wrist_rot_single = torch.where(rb_exp_rot, wrist_rot_rh, wrist_rot_lh)
        else:
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
            "obj_rh_path": "/home/azh/桌面/code/ManipTrans/data/HoldBowl/coacd_object_preview/obj_urdf_example.urdf",
            "obj_lh_path": "/home/azh/桌面/code/ManipTrans/data/HoldBowl/coacd_object_preview/obj_urdf_example.urdf",
            "scene_objs": [],
        }

        # 若存在 right_bool，加入输出，方便下游按需使用（仅当长度与帧数一致）
        if len(right_bool_list) == len(self.data_files):
            out["right_bool"] = torch.tensor(right_bool_list, dtype=torch.bool, device=wrist_pos_rh.device)

        if len(cam_t_rh_list) > 0:
            pred_cam_t_full_rh = torch.stack(cam_t_rh_list, dim=0)
            pred_cam_t_full_lh = torch.stack(cam_t_lh_list, dim=0)
            out["pred_cam_t_full_rh"] = pred_cam_t_full_rh
            out["pred_cam_t_full_lh"] = pred_cam_t_full_lh
            # 单手默认取 RH（保留原有字段行为，避免破坏下游）
            out["pred_cam_t_full"] = pred_cam_t_full_rh

        return out