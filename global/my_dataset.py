import torch
from torch.utils.data import Dataset
import numpy as np
import os
from .transform import aa_to_rotmat, rotmat_to_aa

# 新增：简单的 XYZ 欧拉角（单位：度）到旋转矩阵
def _euler_xyz_to_rotmat(rx_deg, ry_deg, rz_deg, dtype=torch.float32, device=None):
    rad = torch.tensor([rx_deg, ry_deg, rz_deg], dtype=dtype, device=device) * torch.pi / 180.0
    rx, ry, rz = rad[0], rad[1], rad[2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    Rx = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0,  cx, -sx],
                       [0.0,  sx,  cx]], dtype=dtype, device=device)
    Ry = torch.tensor([[ cy, 0.0,  sy],
                       [0.0, 1.0, 0.0],
                       [-sy, 0.0,  cy]], dtype=dtype, device=device)
    Rz = torch.tensor([[ cz, -sz, 0.0],
                       [ sz,  cz, 0.0],
                       [0.0, 0.0, 1.0]], dtype=dtype, device=device)
    # 以手局部坐标为先后顺序：X->Y->Z
    return Rx @ Ry @ Rz


class MyDatasetMTDexHand(Dataset):
    def __init__(self, data_root, split='all', hand_order: str = 'rh_first',
                 lh_rot_offset_deg=(180.0, 0.0, 90.0), rh_rot_offset_deg=(0.0, 0.0, 0.0)):
        self.data_root = data_root
        pth_files = [f for f in os.listdir(self.data_root) if f.endswith('.pth')]
        pth_files.sort()
        self.data_files = [os.path.join(self.data_root, f) for f in pth_files]
        self.split = split
        self.hand_order = hand_order
        # 新增：左右手的额外旋转（度）可调
        self.lh_rot_offset_deg = lh_rot_offset_deg
        self.rh_rot_offset_deg = rh_rot_offset_deg

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        def _orient_to_aa(R):
            if torch.is_tensor(R):
                return rotmat_to_aa(R)
            else:
                return rotmat_to_aa(torch.tensor(R))

        wrist_pos_rh_list, wrist_pos_lh_list = [], []
        wrist_rot_rh_list, wrist_rot_lh_list = [], []
        obj_T_list = []
        cam_t_rh_list, cam_t_lh_list = [], []  
        right_bool_list = []

        # 逐文件累积为时间序列
        for f in self.data_files:
            data = torch.load(f)
            obj_pose = data['obj_pose']
            obj_pose = obj_pose if torch.is_tensor(obj_pose) else torch.tensor(obj_pose)
            obj_T_list.append(obj_pose)

            pred_mano_params = data['pred_mano_params']
            pred_vertices = data['pred_vertices']
            
            # 读取 right_bool 字段
            rb = data.get('right_bool', None)
            if rb is not None:
                rb_t = rb if torch.is_tensor(rb) else torch.tensor(rb)
                if rb_t.numel() > 1:
                    rb_flag = bool((rb_t != 0).any().item())
                else:
                    rb_flag = bool(rb_t.view(-1)[0].item() != 0)
                right_bool_list.append(rb_flag)
            # 确定该帧是否为右手（若缺省则默认右手）
            is_right_hand = rb_flag if rb is not None else True

            # 处理 global_orient - 适配新的单手数据格式
            go = pred_mano_params['global_orient']
            go = go if torch.is_tensor(go) else torch.tensor(go)
            
            # 根据数据形状提取旋转矩阵
            if go.ndim == 4:  # [1,1,3,3]
                R_single = go[0, 0]
            elif go.ndim == 3:  # [1,3,3] 
                R_single = go[0]
            elif go.ndim == 2:  # [3,3]
                R_single = go
            else:
                R_single = aa_to_rotmat(go.view(-1)[:3])
            
            # 在源数据阶段施加“额外旋转”以校正左右手坐标差（默认 LH: X 轴 180 度）
            if is_right_hand:
                R_off = _euler_xyz_to_rotmat(*self.rh_rot_offset_deg, dtype=R_single.dtype, device=R_single.device)
            else:
                R_off = _euler_xyz_to_rotmat(*self.lh_rot_offset_deg, dtype=R_single.dtype, device=R_single.device)
            # 以手的局部坐标为基，后乘偏置
            R_single_corr = R_single @ R_off
            
            # 数据已在世界坐标系，直接使用（带校正）
            wrist_rot_single = _orient_to_aa(R_single_corr)
            
            # 根据 right_bool 决定手的类型
            if is_right_hand:
                wrist_rot_rh = wrist_rot_single
                wrist_rot_lh = torch.zeros_like(wrist_rot_single)
            else:
                wrist_rot_lh = wrist_rot_single
                wrist_rot_rh = torch.zeros_like(wrist_rot_single)

            # wrist 位置 - 适配新格式 [1, 778, 3]
            pv = pred_vertices if torch.is_tensor(pred_vertices) else torch.tensor(pred_vertices)
            # wrist 位置 - 使用 pred_keypoints_3d 的手腕关节 [1, 21, 3]
            kp3d = data['pred_keypoints_3d']
            kp3d = kp3d if torch.is_tensor(kp3d) else torch.tensor(kp3d)
            
            # MANO关节：0=wrist, 4=middle_proximal
            wrist_pos_single = kp3d[0, 0]  # 手腕关节
            middle_pos = kp3d[0, 4]  # 中指近端关节
            # 手腕位置调整（与其他数据集保持一致）
            wrist_pos_single = wrist_pos_single - (middle_pos - wrist_pos_single) * 0.25
            
            if is_right_hand:
                wrist_pos_rh = wrist_pos_single
                wrist_pos_lh = torch.zeros_like(wrist_pos_single)
            else:
                wrist_pos_lh = wrist_pos_single
                wrist_pos_rh = torch.zeros_like(wrist_pos_single)

            wrist_pos_rh_list.append(wrist_pos_rh)
            wrist_pos_lh_list.append(wrist_pos_lh)
            wrist_rot_rh_list.append(wrist_rot_rh)
            wrist_rot_lh_list.append(wrist_rot_lh)

            # 读取相机位移
            cam_t = data.get('pred_cam_t_full', None)
            if cam_t is not None:
                t = cam_t if torch.is_tensor(cam_t) else torch.tensor(cam_t)
                if t.ndim == 2 and t.shape[0] == 1:  # [1, 3]
                    t_single = t[0]
                elif t.ndim == 1 and t.numel() == 3:  # [3]
                    t_single = t
                else:
                    flat = t.reshape(-1)
                    t_single = flat[:3]
                
                if is_right_hand:
                    t_rh = t_single
                    t_lh = torch.zeros_like(t_single)
                else:
                    t_lh = t_single
                    t_rh = torch.zeros_like(t_single)
                    
                cam_t_rh_list.append(t_rh)
                cam_t_lh_list.append(t_lh)

        # 拼接为时序张量
        wrist_pos_rh = torch.stack(wrist_pos_rh_list, dim=0)
        wrist_pos_lh = torch.stack(wrist_pos_lh_list, dim=0)
        wrist_rot_rh = torch.stack(wrist_rot_rh_list, dim=0)
        wrist_rot_lh = torch.stack(wrist_rot_lh_list, dim=0)
        state_manip_obj = torch.stack(obj_T_list, dim=0)

        # 根据 right_bool 选择主要输出
        if len(right_bool_list) == len(self.data_files):
            rb_tensor = torch.tensor(right_bool_list, dtype=torch.bool)
            rb_exp = rb_tensor[:, None].expand_as(wrist_pos_rh)
            wrist_pos_single = torch.where(rb_exp, wrist_pos_rh, wrist_pos_lh)
            rb_exp_rot = rb_tensor[:, None].expand_as(wrist_rot_rh)
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

        if len(cam_t_rh_list) > 0:
            pred_cam_t_full_rh = torch.stack(cam_t_rh_list, dim=0)
            pred_cam_t_full_lh = torch.stack(cam_t_lh_list, dim=0)
            out["pred_cam_t_full_rh"] = pred_cam_t_full_rh
            out["pred_cam_t_full_lh"] = pred_cam_t_full_lh
            if len(right_bool_list) == len(self.data_files):
                rb_exp_cam = rb_tensor[:, None].expand_as(pred_cam_t_full_rh)
                out["pred_cam_t_full"] = torch.where(rb_exp_cam, pred_cam_t_full_rh, pred_cam_t_full_lh)
            else:
                out["pred_cam_t_full"] = pred_cam_t_full_rh

        return out