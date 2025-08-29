from isaacgym import gymtorch, gymutil,gymapi
import torch
import math
import os
from pathlib import Path
import imageio
from main.dataset.transform import aa_to_quat, rotmat_to_quat, aa_to_rotmat, rotmat_to_aa
from main.dataset.my_dataset import MyDatasetMTDexHand

from DexManipNet.dexmanip_sh import DexManipSH_RH
from DexManipNet.dexmanip_sh import DexManipSH_LH
from DexManipNet.dexmanip_bih import DexManipBiH


# ===== 重构的手部数据处理模块 =====

class HandDataProcessor:
    """手部数据处理的基类，定义通用接口和共享功能"""
    
    def __init__(self, vis_env, src2gym_T: torch.Tensor, z_offset: float = 0.0, cam_cfg: dict = None):
        self.vis_env = vis_env
        self.src2gym_T = src2gym_T
        self.z_offset = z_offset
        self.cam_cfg = cam_cfg
        self.device = vis_env.sim_device
    
    def update_state(self, rollout_seq, iter_idx):
        """更新手部和物体状态的主接口方法，由子类实现"""
        raise NotImplementedError("子类必须实现 update_state 方法")
    
    def _apply_relative_transform(self, pos, aa, dexhand_obj):
        """应用 MANO->DexHand 的相对变换（在源坐标系下）"""
        try:
            rel_R = getattr(dexhand_obj, "relative_rotation", None)
            rel_t = getattr(dexhand_obj, "relative_translation", None)
            rel_R = torch.tensor(rel_R, device=self.device, dtype=torch.float32) if rel_R is not None else torch.eye(3, device=self.device, dtype=torch.float32)
            rel_t = torch.tensor(rel_t, device=self.device, dtype=torch.float32) if rel_t is not None else torch.zeros(3, device=self.device, dtype=torch.float32)
            return _apply_relative(pos, aa, rel_R, rel_t)
        except Exception:
            return pos, aa
    
    def _apply_src_to_gym_transform(self, pos, aa):
        """从源坐标系转换到 Gym 坐标系"""
        if self.src2gym_T is not None:
            return _apply_T_to_pos_aa(self.src2gym_T.to(self.device), pos.to(self.device), aa.to(self.device))
        return pos, aa
    
    def _apply_camera_compensation(self, pos, side_key, rollout_seq, iter_idx):
        """应用基于 pred_cam_t_full 的相机深度位移补偿"""
        if not (self.cam_cfg and self.cam_cfg.get("mode", "off") != "off" and self.cam_cfg.get("apply", "hand") in ("hand", "both")):
            return pos
        
        dt = _fetch_cam_dt(rollout_seq, side_key, iter_idx, self.cam_cfg.get("key_base", "pred_cam_t_full"))
        if dt is None:
            return pos
            
        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, device=self.device, dtype=torch.float32)
        dt = dt.to(self.device)
        
        if self.cam_cfg.get("mode") == "z":
            dt = torch.tensor([0.0, 0.0, float(dt[2].item())], device=self.device)
        
        scale = float(self.cam_cfg.get("scale", 1.0))
        dt = dt * scale
        
        R = self.src2gym_T[:3, :3].to(self.device) if self.src2gym_T is not None else torch.eye(3, device=self.device)
        return pos + R @ dt
    
    def _apply_object_camera_compensation(self, mat, side_key, rollout_seq, iter_idx):
        """应用物体的相机深度位移补偿"""
        if not (self.cam_cfg and self.cam_cfg.get("mode", "off") != "off" and self.cam_cfg.get("apply", "hand") in ("obj", "both")):
            return mat
        
        dt = _fetch_cam_dt(rollout_seq, side_key, iter_idx, self.cam_cfg.get("key_base", "pred_cam_t_full"))
        if dt is None:
            return mat
            
        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, device=self.device, dtype=torch.float32)
        dt = dt.to(self.device)
        
        if self.cam_cfg.get("mode") == "z":
            dt = torch.tensor([0.0, 0.0, float(dt[2].item())], device=self.device)
        
        scale = float(self.cam_cfg.get("scale", 1.0))
        dt = dt * scale
        
        R = self.src2gym_T[:3, :3].to(self.device) if self.src2gym_T is not None else torch.eye(3, device=self.device)
        t = mat[:3, 3] + R @ dt
        mat = mat.clone()
        mat[:3, 3] = t
        return mat
    
    def _debug_log_position(self, side_key, iter_idx, tag, raw_pos, final_pos):
        """记录位姿变换的调试日志"""
        if self.cam_cfg:
            _debug_log(self.cam_cfg, side_key, self.cam_cfg.get("sample_idx", -1), iter_idx, tag, raw_pos, final_pos)


class RightHandProcessor(HandDataProcessor):
    """右手数据处理器"""
    
    def update_state(self, rollout_seq, iter_idx):
        """更新右手环境的状态"""
        vis_env = self.vis_env
        device = self.device
        side_key = "rh"
        env0 = vis_env.envs[0]
        dexhand_handle = vis_env.gym.find_actor_handle(env0, "dexhand")

        # 1) 手的根状态
        state_key = f"state_{side_key}"
        state_val = rollout_seq.get(state_key)
        if state_val is not None and torch.is_tensor(state_val):
            if state_val.ndim == 2 and state_val.size(-1) == 13:
                vis_env._root_state[:, dexhand_handle] = state_val[[iter_idx]].to(device)
            elif state_val.ndim == 1 and state_val.numel() == 13:
                vis_env._root_state[:, dexhand_handle] = state_val[None].to(device)
        else:
            pos = None; aa = None
            if (f"wrist_pos_{side_key}" in rollout_seq) and (f"wrist_rot_{side_key}" in rollout_seq):
                pos = rollout_seq[f"wrist_pos_{side_key}"]
                aa = rollout_seq[f"wrist_rot_{side_key}"]
            elif ("wrist_pos" in rollout_seq) and ("wrist_rot" in rollout_seq):
                pos = rollout_seq["wrist_pos"]; aa = rollout_seq["wrist_rot"]
            if pos is not None and aa is not None:
                if torch.is_tensor(pos) and pos.ndim == 2:
                    pos = pos[iter_idx]
                if torch.is_tensor(aa) and aa.ndim == 2:
                    aa = aa[iter_idx]
                debug_pos_raw = torch.as_tensor(pos, device=device, dtype=torch.float32).clone()
                pos = pos.to(device) if torch.is_tensor(pos) else torch.tensor(pos, device=device, dtype=torch.float32)
                aa = aa.to(device) if torch.is_tensor(aa) else torch.tensor(aa, device=device, dtype=torch.float32)
                pos, aa = self._apply_relative_transform(pos, aa, getattr(vis_env, "dexhand", None))
                pos, aa = self._apply_src_to_gym_transform(pos, aa)
                pos = self._apply_camera_compensation(pos, side_key, rollout_seq, iter_idx)
                self._debug_log_position(side_key, iter_idx, "wrist", debug_pos_raw, pos)
                vis_env._root_state[:, dexhand_handle] = _aa_to_root13(pos, aa, device, self.z_offset)

        # 2) 物体根状态
        obj_key = f"state_manip_obj_{side_key}"
        if obj_key in rollout_seq:
            obj_val = rollout_seq[obj_key]
            if torch.is_tensor(obj_val) and obj_val.ndim == 3 and obj_val.shape[1:] == (4, 4):
                mat = obj_val[iter_idx].to(device)
                debug_t_raw = mat[:3, 3].clone()
                if self.src2gym_T is not None:
                    mat = _apply_T_to_mat4(self.src2gym_T.to(device), mat)
                mat = self._apply_object_camera_compensation(mat, side_key, rollout_seq, iter_idx)
                self._debug_log_position(side_key, iter_idx, "obj", debug_t_raw, mat[:3, 3])
                vis_env._manip_obj_root_state[:] = _mat4_to_root13(mat, device, self.z_offset)
            elif torch.is_tensor(obj_val) and obj_val.ndim in (1, 2) and obj_val.size(-1) == 13:
                if obj_val.ndim == 2:
                    vis_env._manip_obj_root_state[:] = obj_val[[iter_idx]].to(device)
                else:
                    vis_env._manip_obj_root_state[:] = obj_val[None].to(device)
            else:
                try:
                    mat = torch.as_tensor(obj_val, device=device)
                    if mat.ndim == 3:
                        mat = mat[iter_idx]
                    if mat.shape[-2:] == (4, 4):
                        debug_t_raw = mat[:3, 3].clone()
                        if self.src2gym_T is not None:
                            mat = _apply_T_to_mat4(self.src2gym_T.to(device), mat)
                        mat = self._apply_object_camera_compensation(mat, side_key, rollout_seq, iter_idx)
                        self._debug_log_position(side_key, iter_idx, "obj", debug_t_raw, mat[:3, 3])
                        vis_env._manip_obj_root_state[:] = _mat4_to_root13(mat, device, self.z_offset)
                except Exception:
                    pass

        vis_env.gym.set_actor_root_state_tensor(vis_env.sim, gymtorch.unwrap_tensor(vis_env._root_state))


class LeftHandProcessor(HandDataProcessor):
    """左手数据处理器"""
    
    def update_state(self, rollout_seq, iter_idx):
        """更新左手环境的状态"""
        vis_env = self.vis_env
        device = self.device
        side_key = "lh"
        env0 = vis_env.envs[0]
        dexhand_handle = vis_env.gym.find_actor_handle(env0, "dexhand")
        
        # 1) 手的根状态
        state_key = f"state_{side_key}"
        state_val = rollout_seq.get(state_key)
        if state_val is not None and torch.is_tensor(state_val):
            if state_val.ndim == 2 and state_val.size(-1) == 13:
                vis_env._root_state[:, dexhand_handle] = state_val[[iter_idx]].to(device)
            elif state_val.ndim == 1 and state_val.numel() == 13:
                vis_env._root_state[:, dexhand_handle] = state_val[None].to(device)
        else:
            pos = None; aa = None
            if (f"wrist_pos_{side_key}" in rollout_seq) and (f"wrist_rot_{side_key}" in rollout_seq):
                pos = rollout_seq[f"wrist_pos_{side_key}"]
                aa = rollout_seq[f"wrist_rot_{side_key}"]
            elif ("wrist_pos" in rollout_seq) and ("wrist_rot" in rollout_seq):
                pos = rollout_seq["wrist_pos"]; aa = rollout_seq["wrist_rot"]
            if pos is not None and aa is not None:
                if torch.is_tensor(pos) and pos.ndim == 2:
                    pos = pos[iter_idx]
                if torch.is_tensor(aa) and aa.ndim == 2:
                    aa = aa[iter_idx]
                debug_pos_raw = torch.as_tensor(pos, device=device, dtype=torch.float32).clone()
                pos = pos.to(device) if torch.is_tensor(pos) else torch.tensor(pos, device=device, dtype=torch.float32)
                aa = aa.to(device) if torch.is_tensor(aa) else torch.tensor(aa, device=device, dtype=torch.float32)
                pos, aa = self._apply_relative_transform(pos, aa, getattr(vis_env, "dexhand", None))
                pos, aa = self._apply_src_to_gym_transform(pos, aa)
                pos = self._apply_camera_compensation(pos, side_key, rollout_seq, iter_idx)
                self._debug_log_position(side_key, iter_idx, "wrist", debug_pos_raw, pos)
                vis_env._root_state[:, dexhand_handle] = _aa_to_root13(pos, aa, device, self.z_offset)
        
        # 2) 物体根状态
        obj_key = f"state_manip_obj_{side_key}"
        if obj_key in rollout_seq:
            obj_val = rollout_seq[obj_key]
            if torch.is_tensor(obj_val) and obj_val.ndim == 3 and obj_val.shape[1:] == (4, 4):
                mat = obj_val[iter_idx].to(device)
                debug_t_raw = mat[:3, 3].clone()
                if self.src2gym_T is not None:
                    mat = _apply_T_to_mat4(self.src2gym_T.to(device), mat)
                mat = self._apply_object_camera_compensation(mat, side_key, rollout_seq, iter_idx)
                self._debug_log_position(side_key, iter_idx, "obj", debug_t_raw, mat[:3, 3])
                vis_env._manip_obj_root_state[:] = _mat4_to_root13(mat, device, self.z_offset)
            elif torch.is_tensor(obj_val) and obj_val.ndim in (1, 2) and obj_val.size(-1) == 13:
                if obj_val.ndim == 2:
                    vis_env._manip_obj_root_state[:] = obj_val[[iter_idx]].to(device)
                else:
                    vis_env._manip_obj_root_state[:] = obj_val[None].to(device)
            else:
                try:
                    mat = torch.as_tensor(obj_val, device=device)
                    if mat.ndim == 3:
                        mat = mat[iter_idx]
                    if mat.shape[-2:] == (4, 4):
                        debug_t_raw = mat[:3, 3].clone()
                        if self.src2gym_T is not None:
                            mat = _apply_T_to_mat4(self.src2gym_T.to(device), mat)
                        mat = self._apply_object_camera_compensation(mat, side_key, rollout_seq, iter_idx)
                        self._debug_log_position(side_key, iter_idx, "obj", debug_t_raw, mat[:3, 3])
                        vis_env._manip_obj_root_state[:] = _mat4_to_root13(mat, device, self.z_offset)
                except Exception:
                    pass
        
        vis_env.gym.set_actor_root_state_tensor(vis_env.sim, gymtorch.unwrap_tensor(vis_env._root_state))


class BothHandsProcessor(HandDataProcessor):
    """双手数据处理器"""
    
    def update_state(self, rollout_seq, iter_idx):
        """更新双手环境的状态"""
        vis_env = self.vis_env
        device = self.device
        
        # RH
        rh_ok = False
        state_rh = rollout_seq.get("state_rh")
        if state_rh is not None and torch.is_tensor(state_rh):
            if state_rh.ndim == 2 and state_rh.size(-1) == 13:
                vis_env.dexhand_rh_root_state[:] = state_rh[[iter_idx]].to(device)
                rh_ok = True
            elif state_rh.ndim == 1 and state_rh.numel() == 13:
                vis_env.dexhand_rh_root_state[:] = state_rh[None].to(device)
                rh_ok = True
        if not rh_ok:
            pos_rh = rollout_seq.get("wrist_pos_rh"); aa_rh = rollout_seq.get("wrist_rot_rh")
            if pos_rh is not None and aa_rh is not None:
                if torch.is_tensor(pos_rh) and pos_rh.ndim == 2:
                    pos_rh = pos_rh[iter_idx]
                if torch.is_tensor(aa_rh) and aa_rh.ndim == 2:
                    aa_rh = aa_rh[iter_idx]
                debug_pos_rh_raw = torch.as_tensor(pos_rh, device=device, dtype=torch.float32).clone()
                pos_rh = pos_rh.to(device) if torch.is_tensor(pos_rh) else torch.tensor(pos_rh, device=device, dtype=torch.float32)
                aa_rh = aa_rh.to(device) if torch.is_tensor(aa_rh) else torch.tensor(aa_rh, device=device, dtype=torch.float32)
                pos_rh, aa_rh = self._apply_relative_transform(pos_rh, aa_rh, getattr(vis_env, "dexhand_rh", None))
                pos_rh, aa_rh = self._apply_src_to_gym_transform(pos_rh, aa_rh)
                pos_rh = self._apply_camera_compensation(pos_rh, "rh", rollout_seq, iter_idx)
                if self.cam_cfg and self.cam_cfg.get("debug", False):
                    raw = [round(x, 4) for x in debug_pos_rh_raw.detach().cpu().tolist()]
                    fin = [round(x, 4) for x in pos_rh.detach().cpu().tolist()]
                    print(f"[DEBUG] hand=rh iter={iter_idx} wrist_raw={raw} wrist_final={fin}")
                vis_env.dexhand_rh_root_state[:] = _aa_to_root13(pos_rh, aa_rh, device, self.z_offset)
                rh_ok = True
        
        # LH
        lh_ok = False
        state_lh = rollout_seq.get("state_lh")
        if state_lh is not None and torch.is_tensor(state_lh):
            if state_lh.ndim == 2 and state_lh.size(-1) == 13:
                vis_env.dexhand_lh_root_state[:] = state_lh[[iter_idx]].to(device)
                lh_ok = True
            elif state_lh.ndim == 1 and state_lh.numel() == 13:
                vis_env.dexhand_lh_root_state[:] = state_lh[None].to(device)
                lh_ok = True
        if not lh_ok:
            pos_lh = rollout_seq.get("wrist_pos_lh"); aa_lh = rollout_seq.get("wrist_rot_lh")
            if pos_lh is not None and aa_lh is not None:
                if torch.is_tensor(pos_lh) and pos_lh.ndim == 2:
                    pos_lh = pos_lh[iter_idx]
                if torch.is_tensor(aa_lh) and aa_lh.ndim == 2:
                    aa_lh = aa_lh[iter_idx]
                debug_pos_lh_raw = torch.as_tensor(pos_lh, device=device, dtype=torch.float32).clone()
                pos_lh = pos_lh.to(device) if torch.is_tensor(pos_lh) else torch.tensor(pos_lh, device=device, dtype=torch.float32)
                aa_lh = aa_lh.to(device) if torch.is_tensor(aa_lh) else torch.tensor(aa_lh, device=device, dtype=torch.float32)
                pos_lh, aa_lh = self._apply_relative_transform(pos_lh, aa_lh, getattr(vis_env, "dexhand_lh", None))
                pos_lh, aa_lh = self._apply_src_to_gym_transform(pos_lh, aa_lh)
                pos_lh = self._apply_camera_compensation(pos_lh, "lh", rollout_seq, iter_idx)
                if self.cam_cfg and self.cam_cfg.get("debug", False):
                    raw = [round(x, 4) for x in debug_pos_lh_raw.detach().cpu().tolist()]
                    fin = [round(x, 4) for x in pos_lh.detach().cpu().tolist()]
                    print(f"[DEBUG] hand=lh iter={iter_idx} wrist_raw={raw} wrist_final={fin}")
                vis_env.dexhand_lh_root_state[:] = _aa_to_root13(pos_lh, aa_lh, device, self.z_offset)
                lh_ok = True
        
        # Objects RH
        if "state_manip_obj_rh" in rollout_seq:
            mat_rh = rollout_seq["state_manip_obj_rh"]
            if torch.is_tensor(mat_rh) and mat_rh.ndim == 3:
                mat_rh = mat_rh[iter_idx]
            debug_t_rh_raw = torch.as_tensor(mat_rh, device=device)[:3, 3].clone()
            if self.src2gym_T is not None:
                mat_rh = _apply_T_to_mat4(self.src2gym_T.to(device), torch.as_tensor(mat_rh, device=device))
            mat_rh = self._apply_object_camera_compensation(mat_rh, "rh", rollout_seq, iter_idx)
            if self.cam_cfg and self.cam_cfg.get("debug", False):
                fin = mat_rh[:3, 3]
                raw = [round(x, 4) for x in debug_t_rh_raw.detach().cpu().tolist()]
                fin = [round(x, 4) for x in fin.detach().cpu().tolist()]
                print(f"[DEBUG] obj_rh iter={iter_idx} t_raw={raw} t_final={fin}")
            vis_env._manip_obj_rh_root_state[:] = _mat4_to_root13(mat_rh, device, self.z_offset)
        
        # Objects LH
        if "state_manip_obj_lh" in rollout_seq:
            mat_lh = rollout_seq["state_manip_obj_lh"]
            if torch.is_tensor(mat_lh) and mat_lh.ndim == 3:
                mat_lh = mat_lh[iter_idx]
            debug_t_lh_raw = torch.as_tensor(mat_lh, device=device)[:3, 3].clone()
            if self.src2gym_T is not None:
                mat_lh = _apply_T_to_mat4(self.src2gym_T.to(device), torch.as_tensor(mat_lh, device=device))
            mat_lh = self._apply_object_camera_compensation(mat_lh, "lh", rollout_seq, iter_idx)
            if self.cam_cfg and self.cam_cfg.get("debug", False):
                fin = mat_lh[:3, 3]
                raw = [round(x, 4) for x in debug_t_lh_raw.detach().cpu().tolist()]
                fin = [round(x, 4) for x in fin.detach().cpu().tolist()]
                print(f"[DEBUG] obj_lh iter={iter_idx} t_raw={raw} t_final={fin}")
            vis_env._manip_obj_lh_root_state[:] = _mat4_to_root13(mat_lh, device, self.z_offset)
        
        # push root states
        vis_env.gym.set_actor_root_state_tensor(vis_env.sim, gymtorch.unwrap_tensor(vis_env._root_state))


def _update_state_single_hand(vis_env, rollout_seq, iter_idx, side_key, src2gym_T: torch.Tensor, z_offset: float = 0.0, cam_cfg: dict = None):
    """Legacy proxy function for single-hand update. Delegates to the appropriate processor for backward compatibility."""
    processor = HandProcessorFactory.create_processor(side_key, vis_env, src2gym_T, z_offset, cam_cfg)
    processor.update_state(rollout_seq, iter_idx)


def _apply_T_to_mat4(T: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    return T @ mat


def _apply_T_to_pos_aa(T: torch.Tensor, pos: torch.Tensor, aa: torch.Tensor):
    R = aa_to_rotmat(aa)
    R_T = T[:3, :3]
    t = T[:3, 3]
    pos_new = R_T @ pos + t
    R_new = R_T @ R
    aa_new = rotmat_to_aa(R_new)
    return pos_new, aa_new


def _apply_relative(pos: torch.Tensor, aa: torch.Tensor, rel_R: torch.Tensor, rel_t: torch.Tensor):
    R = aa_to_rotmat(aa)
    pos_new = rel_R @ pos + rel_t
    R_new = rel_R @ R
    aa_new = rotmat_to_aa(R_new)
    return pos_new, aa_new


def _fetch_cam_dt(rollout_seq: dict, side_key: str, iter_idx: int, key_base: str = "pred_cam_t_full"):
    key_side = f"{key_base}_{side_key}"
    if key_side in rollout_seq:
        val = rollout_seq[key_side]
    elif key_base in rollout_seq:
        val = rollout_seq[key_base]
    else:
        return None
    try:
        dt = torch.as_tensor(val)
        if dt.ndim == 2 and dt.shape[-1] == 3:
            dt = dt[iter_idx]
        elif dt.ndim == 1 and dt.shape[0] == 3:
            pass
        else:
            return None
        return dt
    except Exception:
        return None


def _debug_log(cam_cfg: dict, side: str, sample_idx: int, iter_idx: int, tag: str, raw_pos, final_pos):
    try:
        raw = torch.as_tensor(raw_pos, dtype=torch.float32).detach().cpu().tolist()
        fin = torch.as_tensor(final_pos, dtype=torch.float32).detach().cpu().tolist()
        if cam_cfg.get("log_fp", None) is not None:
            fp = cam_cfg["log_fp"]
            fp.write(f"{side},{sample_idx},{iter_idx},{tag},{raw[0]:.6f},{raw[1]:.6f},{raw[2]:.6f},{fin[0]:.6f},{fin[1]:.6f},{fin[2]:.6f}\n")
            fp.flush()
        elif cam_cfg.get("debug", False):
            r3 = [round(x, 4) for x in raw]
            f3 = [round(x, 4) for x in fin]
            print(f"[DEBUG] side={side} iter={iter_idx} {tag}_raw={r3} {tag}_final={f3}")
    except Exception:
        pass


def _mat4_to_root13(mat: torch.Tensor, device, z_offset: float = 0.0):
    pos = mat[:3, 3]
    rot = mat[:3, :3]
    quat = rotmat_to_quat(rot)[[1, 2, 3, 0]]
    root = torch.zeros(13, device=device, dtype=torch.float32)
    root[0:3] = pos
    root[2] += z_offset
    root[3:7] = quat
    return root[None, :]


def _aa_to_root13(pos: torch.Tensor, aa: torch.Tensor, device, z_offset: float = 0.0):
    rot = aa_to_rotmat(aa)
    quat = rotmat_to_quat(rot)[[1, 2, 3, 0]]
    root = torch.zeros(13, device=device, dtype=torch.float32)
    root[0:3] = pos
    root[2] += z_offset
    root[3:7] = quat
    return root[None, :]


def _get_total_len(sample: dict, side_key: str) -> int:
    keys = []
    if side_key in ("rh", "lh"):
        keys.extend([f"state_{side_key}", f"wrist_pos_{side_key}", f"wrist_rot_{side_key}", f"state_manip_obj_{side_key}"])
        keys.extend(["wrist_pos", "wrist_rot"])  # fallback
    else:
        keys.extend(["state_rh", "state_lh", "wrist_pos_rh", "wrist_rot_rh", "wrist_pos_lh", "wrist_rot_lh", "state_manip_obj_rh", "state_manip_obj_lh"])
    max_len = 1
    for k in keys:
        if k in sample:
            v = sample[k]
            try:
                t = torch.as_tensor(v)
                L = t.shape[0] if t.ndim >= 2 else 1
                if L > max_len:
                    max_len = int(L)
            except Exception:
                pass
    return max_len


def _build_src2gym(src_frame: str, device):
    T = torch.eye(4, device=device, dtype=torch.float32)
    return T


def _parse_idx_list(s: str, total: int):
    """解析逗号分隔的索引列表与区间如 '0,2,5-8'；支持 'all' 或 '*' 表示全选。"""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if isinstance(s, str) and s.lower() in ("all", "*"):
        return list(range(total))
    out = []
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                a, b = int(a), int(b)
            except Exception:
                continue
            if a <= b:
                out.extend(list(range(a, b + 1)))
            else:
                out.extend(list(range(a, b - 1, -1)))
        else:
            try:
                out.append(int(part))
            except Exception:
                continue
    # 去重且保持顺序，并裁剪到合法范围
    seen = set()
    cleaned = []
    for i in out:
        if 0 <= i < total and i not in seen:
            cleaned.append(i)
            seen.add(i)
    return cleaned or None


class HandProcessorFactory:
    @staticmethod
    def create_processor(side_key: str, vis_env, src2gym_T: torch.Tensor, z_offset: float, cam_cfg: dict):
        if side_key == "rh":
            return RightHandProcessor(vis_env, src2gym_T, z_offset, cam_cfg)
        elif side_key == "lh":
            return LeftHandProcessor(vis_env, src2gym_T, z_offset, cam_cfg)
        elif side_key == "bih":
            return BothHandsProcessor(vis_env, src2gym_T, z_offset, cam_cfg)
        else:
            raise ValueError(f"Unknown side_key: {side_key}")


def _update_state_bih(vis_env, rollout_seq, iter_idx, src2gym_T: torch.Tensor, z_offset: float = 0.0, cam_cfg: dict = None):
    """Legacy proxy function for bi-hand update. Delegates to BothHandsProcessor for backward compatibility."""
    processor = HandProcessorFactory.create_processor("bih", vis_env, src2gym_T, z_offset, cam_cfg)
    processor.update_state(rollout_seq, iter_idx)


if __name__ == "__main__":
    args = gymutil.parse_arguments(
        description="Visualize my_dataset with a single window",
        headless=False,
        custom_parameters=[
            {"name": "--idx", "type": int, "default": 0, "help": "Sample index"},
            {"name": "--side", "type": str, "default": "rh", "help": "[rh | lh | bih]"},
            {"name": "--headless", "action": "store_true", "help": "Run without viewer (overrides default)"},
            {
                "name": "--data_root",
                "type": str,
                "default": "/home/azh/桌面/code/ManipTrans/data/HoldPaperCup2MT/HANDS_with_OBJ",
                "help": "Directory containing .pth files",
            },
            {
                "name": "--hand_order",
                "type": str,
                "default": "rh_first",
                "help": "Upstream hand order in tensors: [rh_first | lh_first]",
            },
            {
                "name": "--max_steps",
                "type": int,
                "default": -1,
                "help": ">0 to run finite steps; <=0 to run until window closed",
            },
            {
                "name": "--src_frame",
                "type": str,
                "default": "gym",
                "help": "Source coordinate frame of your data: [gym | mujoco | opencv | object_world]",
            },
            {
                "name": "--z_offset",
                "type": float,
                "default": 0.4,
                "help": "Add a constant height offset (meters) in Isaac Gym frame (applied to hand and object)",
            },
            {
                "name": "--idx_list",
                "type": str,
                "default": "*",
                "help": "Comma-separated indices or ranges (e.g., 0,2,5-10). Use 'all' or '*' for all samples",
            },
            # 新增：基于 pred_cam_t_full 的深度位移补偿
            {
                "name": "--use_pred_cam_t",
                "type": str,
                "default": "off",
                "help": "Use pred_cam_t_full translation compensation: [off | z | xyz]",
            },
            {
                "name": "--cam_t_apply",
                "type": str,
                "default": "hand",
                "help": "Apply compensation to: [hand | obj | both]",
            },
            {
                "name": "--cam_t_key",
                "type": str,
                "default": "pred_cam_t_full",
                "help": "Key name for camera translation in sample (.pth). Supports suffix _rh/_lh",
            },
            {
                "name": "--cam_t_scale",
                "type": float,
                "default": 1.0,
                "help": "Scale factor for pred_cam_t_full values (unit conversion)",
            },
            # 新增：单帧导出选项
            {
                "name": "--export_frames",
                "action": "store_true",
                "help": "Export individual frames as images for video comparison",
            },
            {
                "name": "--frame_dir",
                "type": str,
                "default": "exported_frames",
                "help": "Directory to save exported frame images",
            },
            {
                "name": "--frame_width",
                "type": int,
                "default": 1280,
                "help": "Width of exported frame images",
            },
            {
                "name": "--frame_height",
                "type": int,
                "default": 720,
                "help": "Height of exported frame images",
            },
            # 新增：相机 FOV 与位姿可配置
            {
                "name": "--frame_fov",
                "type": float,
                "default": 69.4,
                "help": "Horizontal field-of-view (degrees) for export camera",
            },
            {
                "name": "--cam_pos",
                "type": str,
                "default": "",
                "help": "Export camera position as 'x,y,z' in meters (comma or space separated). Empty = default",
            },
            {
                "name": "--cam_target",
                "type": str,
                "default": "",
                "help": "Export camera look-at target as 'x,y,z' in meters (comma or space separated). Empty = default",
            },
            # 新增：调试打印开关
            {
                "name": "--debug_pose",
                "action": "store_true",
                "help": "Print wrist/object positions before and after transforms each frame",
            },
            {
                "name": "--pose_log",
                "type": str,
                "default": "",
                "help": "CSV file path to record per-frame wrist/object positions (suppresses console debug)",
            },
        ],
    )

    # Auto-fallback to headless if no display server is available
    if not getattr(args, "headless", False):
        no_x = (os.name != "nt") and (not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"))
        if no_x:
            print("[INFO] No DISPLAY/WAYLAND_DISPLAY detected. Forcing headless mode.")
            args.headless = True
    # dataset
    dataset = MyDatasetMTDexHand(data_root=args.data_root, hand_order=getattr(args, "hand_order", "rh_first"))
    print(f"[INFO] Using hand_order={getattr(args, 'hand_order', 'rh_first')} (0->RH, 1->LH if rh_first; swapped if lh_first)")

    # indices to play
    indices = _parse_idx_list(getattr(args, "idx_list", "*"), len(dataset)) or [args.idx]

    # Removed: inspect mode and sample summary printing for simplicity
    # if getattr(args, "inspect", False):
    #     sample = dataset[indices[0]]
    #     _print_sample_summary(sample)
    #     raise SystemExit(0)

    def _build_env(side, sample):
        if side == "rh":
            return DexManipSH_RH(args, sample), "rh"
        elif side == "lh":
            return DexManipSH_LH(args, sample), "lh"
        elif side == "bih":
            return DexManipBiH(args, sample), "bih"
        else:
            raise ValueError("Invalid side. Choose from [rh | lh | bih].")

    # build first env once and reuse viewer; assumes same object URDF
    cur_sample_idx = 0
    sample = dataset[indices[cur_sample_idx]]
    vis_env, side_key = _build_env(args.side, sample)

    # build source->gym transform (rotation only)
    src2gym_T = _build_src2gym(args.src_frame, vis_env.sim_device)

    # configure wrist driving source
    cam_cfg = {
        "mode": getattr(args, "use_pred_cam_t", "off"),  # off | z | xyz
        "apply": getattr(args, "cam_t_apply", "hand"),   # hand | obj | both
        "key_base": getattr(args, "cam_t_key", "pred_cam_t_full"),
        "scale": float(getattr(args, "cam_t_scale", 1.0)),
        "debug": bool(getattr(args, "debug_pose", False)),
    }
    # 如果指定了日志文件，则打开并写入表头
    pose_log = getattr(args, "pose_log", "")
    if isinstance(pose_log, str) and pose_log.strip():
        pose_path = Path(pose_log).expanduser()
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        cam_cfg["log_fp"] = open(pose_path, "w", encoding="utf-8")
        cam_cfg["log_fp"].write("side,sample_idx,iter,tag,raw_x,raw_y,raw_z,final_x,final_y,final_z\n")
        cam_cfg["log_fp"].flush()
    # 初始化当前样本索引到 cam_cfg，便于日志标注
    cam_cfg["sample_idx"] = indices[cur_sample_idx]

    # New: camera for single-frame export
    export_frames = bool(getattr(args, "export_frames", False))
    cam_handle = None
    cam_props = None
    cur_sample_dir = None
    frame_idx = 0
    if export_frames:
        base_dir = Path(getattr(args, "frame_dir", "exported_frames"))
        base_dir.mkdir(parents=True, exist_ok=True)
        # per-sample subfolder
        cur_sample_dir = base_dir / f"{args.side}_{indices[cur_sample_idx]:06d}"
        cur_sample_dir.mkdir(parents=True, exist_ok=True)
        # create a camera sensor
        cam_props = gymapi.CameraProperties()
        cam_props.enable_tensors = True
        cam_props.width = int(getattr(args, "frame_width", 1280))
        cam_props.height = int(getattr(args, "frame_height", 720))
        cam_props.horizontal_fov = float(getattr(args, "frame_fov", 69.4))
        cam_handle = vis_env.gym.create_camera_sensor(vis_env.envs[0], cam_props)
        # default camera pose similar to dexmanip_sh
        cam_pos = gymapi.Vec3(0.8, 0.0, 0.7)
        cam_target = gymapi.Vec3(-0.5, 0.0, 0.3)
        # 如果提供了覆盖的相机位姿参数，则解析并替换
        try:
            pos_str = getattr(args, "cam_pos", "")
            if isinstance(pos_str, str) and pos_str.strip():
                parts = [float(p) for p in pos_str.replace(",", " ").split()]
                if len(parts) == 3:
                    cam_pos = gymapi.Vec3(parts[0], parts[1], parts[2])
        except Exception:
            pass
        try:
            tgt_str = getattr(args, "cam_target", "")
            if isinstance(tgt_str, str) and tgt_str.strip():
                parts = [float(p) for p in tgt_str.replace(",", " ").split()]
                if len(parts) == 3:
                    cam_target = gymapi.Vec3(parts[0], parts[1], parts[2])
        except Exception:
            pass
        vis_env.gym.set_camera_location(cam_handle, vis_env.envs[0], cam_pos, cam_target)

    processor = HandProcessorFactory.create_processor(side_key, vis_env, src2gym_T, args.z_offset, cam_cfg)

    iter_idx = 0
    total_len = _get_total_len(sample, side_key)
    step_count = 0

    while True:
        processor.update_state(sample, iter_idx)
        vis_env.gym.simulate(vis_env.sim)
        if not args.headless:
            vis_env.gym.fetch_results(vis_env.sim, True)
            vis_env.gym.step_graphics(vis_env.sim)
            vis_env.gym.draw_viewer(vis_env.viewer, vis_env.sim, True)
            vis_env.gym.sync_frame_time(vis_env.sim)
            if vis_env.gym.query_viewer_has_closed(vis_env.viewer):
                break
        else:
            vis_env.gym.fetch_results(vis_env.sim, True)
            vis_env.gym.sync_frame_time(vis_env.sim)

        # export current frame as image if requested
        if export_frames and cam_handle is not None:
            vis_env.gym.render_all_camera_sensors(vis_env.sim)
            vis_env.gym.start_access_image_tensors(vis_env.sim)
            rgb = vis_env.gym.get_camera_image(vis_env.sim, vis_env.envs[0], cam_handle, gymapi.IMAGE_COLOR)
            vis_env.gym.end_access_image_tensors(vis_env.sim)
            rgb = rgb.reshape(cam_props.height, cam_props.width, 4)[..., :3]
            out_path = cur_sample_dir / f"{frame_idx:06d}.png"
            imageio.imwrite(str(out_path), rgb)
            frame_idx += 1

        iter_idx += 1
        step_count += 1
        if getattr(args, "max_steps", -1) > 0 and step_count >= args.max_steps:
            break
        if iter_idx >= total_len:
            iter_idx = 0
            # advance to next sample
            cur_sample_idx = (cur_sample_idx + 1) % len(indices)
            sample = dataset[indices[cur_sample_idx]]
            total_len = _get_total_len(sample, side_key)
            cam_cfg["sample_idx"] = indices[cur_sample_idx]
            # update export folder for new sample
            if export_frames:
                base_dir = Path(getattr(args, "frame_dir", "exported_frames"))
                cur_sample_dir = base_dir / f"{args.side}_{indices[cur_sample_idx]:06d}"
                cur_sample_dir.mkdir(parents=True, exist_ok=True)
                frame_idx = 0
        # src2gym_T unchanged; assume same coord frame
        # Note: assuming same object URDF; if different, need to rebuild env
        # Only refreshing root state is enough when the object mesh is unchanged
 
    if not args.headless:
        vis_env.gym.destroy_viewer(vis_env.viewer)
    vis_env.gym.destroy_sim(vis_env.sim)
    # 关闭日志文件
    if cam_cfg.get("log_fp", None) is not None:
        try:
            cam_cfg["log_fp"].close()
        except Exception:
            pass
