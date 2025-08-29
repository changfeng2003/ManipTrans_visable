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


def _mat4_to_root13(mat, device, z_offset: float = 0.0):
    """Convert a 4x4 transform to Isaac Gym 13-D root state (pos, quat, linvel, angvel)."""
    if not torch.is_tensor(mat):
        mat = torch.tensor(mat, device=device, dtype=torch.float32)
    pos = mat[:3, 3]
    rot = mat[:3, :3]
    # Isaac Gym expects quaternion in [x,y,z,w]; our utils return [w,x,y,z], so reorder
    quat = rotmat_to_quat(rot)[[1, 2, 3, 0]]
    root = torch.zeros(13, device=device, dtype=torch.float32)
    root[0:3] = pos
    root[2] += z_offset
    root[3:7] = quat
    return root[None, :]


def _aa_to_root13(pos, aa, device, z_offset: float = 0.0):
    """Convert position and axis-angle (3,) to Isaac Gym 13-D root state."""
    if not torch.is_tensor(pos):
        pos = torch.tensor(pos, device=device, dtype=torch.float32)
    if not torch.is_tensor(aa):
        aa = torch.tensor(aa, device=device, dtype=torch.float32)
    # Isaac Gym expects quaternion in [x,y,z,w]; our utils return [w,x,y,z], so reorder
    quat = aa_to_quat(aa)[[1, 2, 3, 0]]
    root = torch.zeros(13, device=device, dtype=torch.float32)
    root[0:3] = pos
    root[2] += z_offset
    root[3:7] = quat
    return root[None, :]


def _get_total_len(sample, side):#用于单帧数据，可以删掉
    """Best-effort compute sequence length for a sample.
    Returns 1 when sequence fields are missing (e.g., my_dataset single-frame)."""
    try:
        if side == "bih":
            # prefer RH sequence if present
            if "dq_rh" in sample:
                return int(len(sample["dq_rh"]))
            if "state_manip_obj_rh" in sample and torch.is_tensor(sample["state_manip_obj_rh"]) and sample["state_manip_obj_rh"].ndim == 3:
                return int(sample["state_manip_obj_rh"].shape[0])
            if "state_manip_obj_lh" in sample and torch.is_tensor(sample["state_manip_obj_lh"]) and sample["state_manip_obj_lh"].ndim == 3:
                return int(sample["state_manip_obj_lh"].shape[0])
            return 1
        key = f"dq_{side}"
        if key in sample:
            return int(len(sample[key]))
        key_obj = f"state_manip_obj_{side}"
        if key_obj in sample:
            smo = sample[key_obj]
            if torch.is_tensor(smo) and smo.ndim == 3:
                return int(smo.shape[0])
        return 1
    except Exception:
        return 1


def _parse_idx_list(s: str, total: int):
    """Parse comma-separated indices and ranges like '0,2,5-8'. Support 'all' or '*' to include all."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if s.lower() in ("all", "*"):
        return list(range(total))
    out = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a, b = int(a), int(b)
            if a <= b:
                out.extend(list(range(a, b + 1)))
            else:
                out.extend(list(range(a, b - 1, -1)))
        else:
            out.append(int(part))
    # clamp to valid range and unique preserve order
    seen = set()
    cleaned = []
    for i in out:
        if 0 <= i < total and i not in seen:
            cleaned.append(i)
            seen.add(i)
    return cleaned or None


def _build_src2gym(frame: str, device: torch.device):
    """Return a 4x4 transform from given source frame to Isaac Gym world frame.
    - gym: identity (already in Isaac Gym frame)
    - mujoco: rotate Rz(-90deg) then Rx(+90deg) (commonly used in this codebase)
    - opencv: camera coords (x right, y down, z forward) -> gym (x forward, y left, z up)
    """
    frame = (frame or "gym").lower()
    T = torch.eye(4, device=device, dtype=torch.float32)
    if frame == "gym":
        return T
    elif frame == "mujoco":
        Rz = aa_to_rotmat(torch.tensor([0.0, 0.0, -math.pi / 2], device=device))
        Rx = aa_to_rotmat(torch.tensor([math.pi / 2, 0.0, 0.0], device=device))
        T[:3, :3] = Rz @ Rx
        return T
    elif frame == "opencv":
        # Xg = Zc, Yg = -Xc, Zg = -Yc
        R = torch.tensor([[0.0, 0.0, 1.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0]], device=device, dtype=torch.float32)
        T[:3, :3] = R
        return T
    else:
        # Fallback: identity
        return T


def _apply_T_to_mat4(T: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Apply 4x4 transform T to 4x4 pose M: return T @ M (rotation+translation only)."""
    return T @ M


def _apply_T_to_pos_aa(T: torch.Tensor, pos: torch.Tensor, aa: torch.Tensor):
    """Apply 4x4 transform T to position+axis-angle."""
    R = T[:3, :3]
    pos_t = R @ pos
    rot_t = R @ aa_to_rotmat(aa)
    aa_t = rotmat_to_aa(rot_t)
    return pos_t, aa_t


# 基于 .pth 中的 pred_cam_t_full 深度位移做补偿（输入为 opencv/camera 坐标系下的位移向量）
# 返回形状为 (3,) 的张量或 None
def _fetch_cam_dt(rollout_seq, side_key: str, iter_idx: int, key_base: str):
    # 优先侧别键，其次全局键
    if side_key in ("rh", "lh"):
        side_key_name = f"{key_base}_{side_key}"
        if side_key_name in rollout_seq:
            v = rollout_seq[side_key_name]
        else:
            v = rollout_seq.get(key_base, None)
    else:
        v = rollout_seq.get(key_base, None)
    if v is None:
        return None
    try:
        if torch.is_tensor(v):
            if v.ndim == 2 and v.shape[-1] == 3:
                return v[iter_idx]
            if v.ndim == 1 and v.numel() == 3:
                return v
        # 尝试转 tensor
        tv = torch.as_tensor(v, dtype=torch.float32)
        if tv.ndim == 2 and tv.shape[-1] == 3:
            return tv[iter_idx]
        if tv.ndim == 1 and tv.numel() == 3:
            return tv
    except Exception:
        return None


def _apply_relative(wrist_pos, wrist_aa, rel_R, rel_t):
    R_curr = aa_to_rotmat(wrist_aa)
    R_rel = rel_R if torch.is_tensor(rel_R) else torch.tensor(rel_R, dtype=torch.float32, device=wrist_pos.device)
    t_rel = rel_t if torch.is_tensor(rel_t) else torch.tensor(rel_t, dtype=torch.float32, device=wrist_pos.device)
    pos = wrist_pos + R_curr @ t_rel
    
    aa = rotmat_to_aa(R_curr @ R_rel)
    return pos, aa


# 新增：将调试信息写入文件或打印到控制台（当未指定文件时）
def _debug_log(cam_cfg: dict, side: str, sample_idx: int, iter_idx: int, tag: str, raw_vec, fin_vec):
    if cam_cfg is None:
        return
    fh = cam_cfg.get("log_fp", None)
    to_console = cam_cfg.get("debug", False) and fh is None

    def _tolist3(v):
        if v is None:
            return [float("nan"), float("nan"), float("nan")]
        if torch.is_tensor(v):
            v = v.detach().cpu().flatten()
            if v.numel() < 3:
                pad = torch.full((3 - v.numel(),), float("nan"))
                v = torch.cat([v, pad])
            return [float(v[0].item()), float(v[1].item()), float(v[2].item())]
        lst = list(v)
        if len(lst) < 3:
            lst = lst + [float("nan")] * (3 - len(lst))
        return [float(lst[0]), float(lst[1]), float(lst[2])]

    raw = _tolist3(raw_vec)
    fin = _tolist3(fin_vec)

    if fh is not None:
        fh.write(
            f"{side},{sample_idx},{iter_idx},{tag},{raw[0]:.6f},{raw[1]:.6f},{raw[2]:.6f},{fin[0]:.6f},{fin[1]:.6f},{fin[2]:.6f}\n"
        )
        fh.flush()
    elif to_console:
        raw_r = [round(x, 4) for x in raw]
        fin_r = [round(x, 4) for x in fin]
        print(f"[DEBUG] {tag} side={side} iter={iter_idx} raw={raw_r} final={fin_r}")


# 新增：Inspire 手的一个通用“抓握”默认姿态（12 DOF）
# DOF 顺序参考 inspire.py 中的 dof_names：
# [index_proximal, index_intermediate, middle_proximal, middle_intermediate,
#  pinky_proximal, pinky_intermediate, ring_proximal, ring_intermediate,
#  thumb_proximal_yaw, thumb_proximal_pitch, thumb_intermediate, thumb_distal]



def _update_state_single_hand(vis_env, rollout_seq, iter_idx, side_key, src2gym_T: torch.Tensor, z_offset: float = 0.0, cam_cfg: dict = None):
    """Update sim tensors for a single-hand environment (RH or LH) for a given frame index."""
    env0 = vis_env.envs[0]
    dexhand_handle = vis_env.gym.find_actor_handle(env0, "dexhand")
    device = vis_env.sim_device

    # hand root state: prefer explicit 13-D, fallback to wrist pos/aa
    state_key = f"state_{side_key}"
    if state_key in rollout_seq:
        state_val = rollout_seq[state_key]
        if torch.is_tensor(state_val) and state_val.ndim == 2 and state_val.size(-1) == 13:
            vis_env._root_state[:, dexhand_handle] = state_val[[iter_idx]].to(device)
        elif torch.is_tensor(state_val) and state_val.ndim == 1 and state_val.numel() == 13:
            vis_env._root_state[:, dexhand_handle] = state_val[None].to(device)
    else:
        # 优先使用侧别键，防止单手 LH 模式误用 RH 的通用键
        pos = None
        aa = None
        if (f"wrist_pos_{side_key}" in rollout_seq) and (f"wrist_rot_{side_key}" in rollout_seq):
            pos = rollout_seq[f"wrist_pos_{side_key}"]
            aa = rollout_seq[f"wrist_rot_{side_key}"]
        elif ("wrist_pos" in rollout_seq) and ("wrist_rot" in rollout_seq):
            pos = rollout_seq["wrist_pos"]
            aa = rollout_seq["wrist_rot"]
        if pos is not None and aa is not None:
            # 支持 (T,3) 或 (3,) 形式
            if torch.is_tensor(pos) and pos.ndim == 2:
                pos = pos[iter_idx]
            if torch.is_tensor(aa) and aa.ndim == 2:
                aa = aa[iter_idx]
            # 原始手腕位置（用于调试/日志）
            debug_pos_raw = torch.as_tensor(pos, device=device, dtype=torch.float32).clone()
            if not torch.is_tensor(pos):
                pos = torch.tensor(pos, device=device, dtype=torch.float32)
            if not torch.is_tensor(aa):
                aa = torch.tensor(aa, device=device, dtype=torch.float32)
            # 先应用 MANO->DexHand 的相对变换（在源坐标系下）
            try:
                rel_R = getattr(vis_env.dexhand, "relative_rotation", None)
                rel_t = getattr(vis_env.dexhand, "relative_translation", None)
                rel_R = torch.tensor(rel_R, device=device, dtype=torch.float32) if rel_R is not None else torch.eye(3, device=device, dtype=torch.float32)
                rel_t = torch.tensor(rel_t, device=device, dtype=torch.float32) if rel_t is not None else torch.zeros(3, device=device, dtype=torch.float32)
                pos, aa = _apply_relative(pos, aa, rel_R, rel_t)
            except Exception:
                pass
            # 再从源坐标系转换到 Gym 坐标系
            if src2gym_T is not None:
                pos, aa = _apply_T_to_pos_aa(src2gym_T.to(device), pos.to(device), aa.to(device))
            # 深度位移补偿（hand）
            if cam_cfg and cam_cfg.get("mode", "off") != "off" and cam_cfg.get("apply", "hand") in ("hand", "both"):
                dt = _fetch_cam_dt(rollout_seq, side_key, iter_idx, cam_cfg.get("key_base", "pred_cam_t_full"))
                if dt is not None:
                    if not torch.is_tensor(dt):
                        dt = torch.tensor(dt, device=device, dtype=torch.float32)
                    dt = dt.to(device)
                    if cam_cfg.get("mode") == "z":
                        dt = torch.tensor([0.0, 0.0, float(dt[2].item())], device=device)
                    scale = float(cam_cfg.get("scale", 1.0))
                    dt = dt * scale
                    R = src2gym_T[:3, :3].to(device) if src2gym_T is not None else torch.eye(3, device=device)
                    pos = pos + R @ dt
            if cam_cfg:
                _debug_log(cam_cfg, side_key, cam_cfg.get("sample_idx", -1), iter_idx, "wrist", debug_pos_raw, pos)
            vis_env._root_state[:, dexhand_handle] = _aa_to_root13(pos, aa, device, z_offset)

    # object root state: 13-D or 4x4 matrix (single or sequence)
    obj_key = f"state_manip_obj_{side_key}"
    if obj_key in rollout_seq:
        obj_val = rollout_seq[obj_key]
        if torch.is_tensor(obj_val) and obj_val.ndim == 3 and obj_val.shape[1:] == (4, 4):
            mat = obj_val[iter_idx]
            debug_t_raw = torch.as_tensor(mat, device=device)[:3, 3].clone()
            if src2gym_T is not None:
                mat = _apply_T_to_mat4(src2gym_T.to(device), mat.to(device))
            # 深度位移补偿（object）
            if cam_cfg and cam_cfg.get("mode", "off") != "off" and cam_cfg.get("apply", "hand") in ("obj", "both"):
                dt = _fetch_cam_dt(rollout_seq, side_key, iter_idx, cam_cfg.get("key_base", "pred_cam_t_full"))
                if dt is not None:
                    if not torch.is_tensor(dt):
                        dt = torch.tensor(dt, device=device, dtype=torch.float32)
                    dt = dt.to(device)
                    if cam_cfg.get("mode") == "z":
                        dt = torch.tensor([0.0, 0.0, float(dt[2].item())], device=device)
                    scale = float(cam_cfg.get("scale", 1.0))
                    dt = dt * scale
                    R = src2gym_T[:3, :3].to(device) if src2gym_T is not None else torch.eye(3, device=device)
                    t = mat[:3, 3] + R @ dt
                    mat = mat.clone()
                    mat[:3, 3] = t
            # 统一调试/日志输出
            if cam_cfg:
                _debug_log(cam_cfg, side_key, cam_cfg.get("sample_idx", -1), iter_idx, "obj", debug_t_raw, mat[:3, 3])
            root13 = _mat4_to_root13(mat, device, z_offset)
            vis_env._manip_obj_root_state[:] = root13
        elif torch.is_tensor(obj_val) and obj_val.ndim == 2 and obj_val.size(-1) == 13:
            vis_env._manip_obj_root_state[:] = obj_val[[iter_idx]].to(device)
        elif torch.is_tensor(obj_val) and obj_val.ndim == 1 and obj_val.numel() == 13:
            vis_env._manip_obj_root_state[:] = obj_val[None].to(device)
        else:
            try:
                mat = torch.tensor(obj_val, device=device)
                if mat.ndim == 3:
                    mat = mat[iter_idx]
                if mat.shape[-2:] == (4, 4):
                    debug_t_raw = mat[:3, 3].clone()
                    if src2gym_T is not None:
                        mat = _apply_T_to_mat4(src2gym_T.to(device), mat.to(device))
                    # 深度位移补偿（object）
                    if cam_cfg and cam_cfg.get("mode", "off") != "off" and cam_cfg.get("apply", "hand") in ("obj", "both"):
                        dt = _fetch_cam_dt(rollout_seq, side_key, iter_idx, cam_cfg.get("key_base", "pred_cam_t_full"))
                        if dt is not None:
                            if not torch.is_tensor(dt):
                                dt = torch.tensor(dt, device=device, dtype=torch.float32)
                            dt = dt.to(device)
                            if cam_cfg.get("mode") == "z":
                                dt = torch.tensor([0.0, 0.0, float(dt[2].item())], device=device)
                            scale = float(cam_cfg.get("scale", 1.0))
                            dt = dt * scale
                            R = src2gym_T[:3, :3].to(device) if src2gym_T is not None else torch.eye(3, device=device)
                            t = mat[:3, 3] + R @ dt
                            mat = mat.clone()
                            mat[:3, 3] = t
                    # 统一调试/日志输出
                    if cam_cfg:
                        _debug_log(cam_cfg, side_key, cam_cfg.get("sample_idx", -1), iter_idx, "obj", debug_t_raw, mat[:3, 3])
                    vis_env._manip_obj_root_state[:] = _mat4_to_root13(mat, device, z_offset)
            except Exception:
                pass

    # Removed q/dq injection for minimal pipeline; keep current finger posture
    pass

    # push root states
    vis_env.gym.set_actor_root_state_tensor(vis_env.sim, gymtorch.unwrap_tensor(vis_env._root_state))


def _update_state_bih(vis_env, rollout_seq, iter_idx, src2gym_T: torch.Tensor, z_offset: float = 0.0, cam_cfg: dict = None):
    device = vis_env.sim_device

    # RH
    rh_ok = False
    # Prefer 13-D root state if provided
    state_rh = rollout_seq.get("state_rh")
    if state_rh is not None:
        if torch.is_tensor(state_rh) and state_rh.ndim == 2 and state_rh.size(-1) == 13:
            vis_env.dexhand_rh_root_state[:] = state_rh[[iter_idx]].to(device)
            rh_ok = True
        elif torch.is_tensor(state_rh) and state_rh.ndim == 1 and state_rh.numel() == 13:
            vis_env.dexhand_rh_root_state[:] = state_rh[None].to(device)
            rh_ok = True
    if not rh_ok:
        pos_rh = rollout_seq.get("wrist_pos_rh")
        aa_rh = rollout_seq.get("wrist_rot_rh")
        if pos_rh is not None and aa_rh is not None:
            if torch.is_tensor(pos_rh) and pos_rh.ndim == 2:
                pos_rh = pos_rh[iter_idx]
            if torch.is_tensor(aa_rh) and aa_rh.ndim == 2:
                aa_rh = aa_rh[iter_idx]
            pos_rh = pos_rh.to(device) if torch.is_tensor(pos_rh) else torch.tensor(pos_rh, device=device, dtype=torch.float32)
            aa_rh = aa_rh.to(device) if torch.is_tensor(aa_rh) else torch.tensor(aa_rh, device=device, dtype=torch.float32)
            debug_pos_rh_raw = pos_rh.clone()
            # 先应用 MANO->DexHand 的相对变换（在源坐标系下）
            try:
                rel_R = torch.tensor(getattr(vis_env.dexhand_rh, "relative_rotation", torch.eye(3)), device=device, dtype=torch.float32)
                rel_t = torch.tensor(getattr(vis_env.dexhand_rh, "relative_translation", torch.zeros(3)), device=device, dtype=torch.float32)
                pos_rh, aa_rh = _apply_relative(pos_rh, aa_rh, rel_R, rel_t)
            except Exception:
                pass
            # 再从源坐标系转换到 Gym 坐标系
            if src2gym_T is not None:
                pos_rh, aa_rh = _apply_T_to_pos_aa(src2gym_T.to(device), pos_rh, aa_rh)
            # 深度位移补偿（hand RH）
            if cam_cfg and cam_cfg.get("mode", "off") != "off" and cam_cfg.get("apply", "hand") in ("hand", "both"):
                dt = _fetch_cam_dt(rollout_seq, "rh", iter_idx, cam_cfg.get("key_base", "pred_cam_t_full"))
                if dt is not None:
                    if not torch.is_tensor(dt):
                        dt = torch.tensor(dt, device=device, dtype=torch.float32)
                    dt = dt.to(device)
                    if cam_cfg.get("mode") == "z":
                        dt = torch.tensor([0.0, 0.0, float(dt[2].item())], device=device)
                    scale = float(cam_cfg.get("scale", 1.0))
                    dt = dt * scale
                    R = src2gym_T[:3, :3].to(device) if src2gym_T is not None else torch.eye(3, device=device)
                    pos_rh = pos_rh + R @ dt
            if cam_cfg and cam_cfg.get("debug", False):
                raw = [round(x, 4) for x in debug_pos_rh_raw.detach().cpu().tolist()]
                fin = [round(x, 4) for x in pos_rh.detach().cpu().tolist()]
                print(f"[DEBUG] hand=rh iter={iter_idx} wrist_raw={raw} wrist_final={fin}")
            vis_env.dexhand_rh_root_state[:] = _aa_to_root13(pos_rh, aa_rh, device, z_offset)
            rh_ok = True

    # LH
    lh_ok = False
    state_lh = rollout_seq.get("state_lh")
    if state_lh is not None:
        if torch.is_tensor(state_lh) and state_lh.ndim == 2 and state_lh.size(-1) == 13:
            vis_env.dexhand_lh_root_state[:] = state_lh[[iter_idx]].to(device)
            lh_ok = True
        elif torch.is_tensor(state_lh) and state_lh.ndim == 1 and state_lh.numel() == 13:
            vis_env.dexhand_lh_root_state[:] = state_lh[None].to(device)
            lh_ok = True
    if not lh_ok:
        pos_lh = rollout_seq.get("wrist_pos_lh")
        aa_lh = rollout_seq.get("wrist_rot_lh")
        if pos_lh is not None and aa_lh is not None:
            if torch.is_tensor(pos_lh) and pos_lh.ndim == 2:
                pos_lh = pos_lh[iter_idx]
            if torch.is_tensor(aa_lh) and aa_lh.ndim == 2:
                aa_lh = aa_lh[iter_idx]
            pos_lh = pos_lh.to(device) if torch.is_tensor(pos_lh) else torch.tensor(pos_lh, device=device, dtype=torch.float32)
            aa_lh = aa_lh.to(device) if torch.is_tensor(aa_lh) else torch.tensor(aa_lh, device=device, dtype=torch.float32)
            debug_pos_lh_raw = pos_lh.clone()
            # 先应用 MANO->DexHand 的相对变换（在源坐标系下）
            try:
                rel_R = torch.tensor(getattr(vis_env.dexhand_lh, "relative_rotation", torch.eye(3)), device=device, dtype=torch.float32)
                rel_t = torch.tensor(getattr(vis_env.dexhand_lh, "relative_translation", torch.zeros(3)), device=device, dtype=torch.float32)
                pos_lh, aa_lh = _apply_relative(pos_lh, aa_lh, rel_R, rel_t)
            except Exception:
                pass
            # 再从源坐标系转换到 Gym 坐标系
            if src2gym_T is not None:
                pos_lh, aa_lh = _apply_T_to_pos_aa(src2gym_T.to(device), pos_lh, aa_lh)
            # 深度位移补偿（hand LH）
            if cam_cfg and cam_cfg.get("mode", "off") != "off" and cam_cfg.get("apply", "hand") in ("hand", "both"):
                dt = _fetch_cam_dt(rollout_seq, "lh", iter_idx, cam_cfg.get("key_base", "pred_cam_t_full"))
                if dt is not None:
                    if not torch.is_tensor(dt):
                        dt = torch.tensor(dt, device=device, dtype=torch.float32)
                    dt = dt.to(device)
                    if cam_cfg.get("mode") == "z":
                        dt = torch.tensor([0.0, 0.0, float(dt[2].item())], device=device)
                    scale = float(cam_cfg.get("scale", 1.0))
                    dt = dt * scale
                    R = src2gym_T[:3, :3].to(device) if src2gym_T is not None else torch.eye(3, device=device)
                    pos_lh = pos_lh + R @ dt
            if cam_cfg and cam_cfg.get("debug", False):
                raw = [round(x, 4) for x in debug_pos_lh_raw.detach().cpu().tolist()]
                fin = [round(x, 4) for x in pos_lh.detach().cpu().tolist()]
                print(f"[DEBUG] hand=lh iter={iter_idx} wrist_raw={raw} wrist_final={fin}")
            vis_env.dexhand_lh_root_state[:] = _aa_to_root13(pos_lh, aa_lh, device, z_offset)
            lh_ok = True

    # Objects
    if "state_manip_obj_rh" in rollout_seq:
        mat_rh = rollout_seq["state_manip_obj_rh"]
        if torch.is_tensor(mat_rh) and mat_rh.ndim == 3:
            mat_rh = mat_rh[iter_idx]
        if src2gym_T is not None:
            debug_t_rh_raw = torch.as_tensor(mat_rh, device=device)[:3, 3].clone()
            mat_rh = _apply_T_to_mat4(src2gym_T.to(device), torch.as_tensor(mat_rh, device=device))
        else:
            debug_t_rh_raw = torch.as_tensor(mat_rh, device=device)[:3, 3].clone()
        if cam_cfg and cam_cfg.get("mode", "off") != "off" and cam_cfg.get("apply", "hand") in ("obj", "both"):
            dt = _fetch_cam_dt(rollout_seq, "rh", iter_idx, cam_cfg.get("key_base", "pred_cam_t_full"))
            if dt is not None:
                if not torch.is_tensor(dt):
                    dt = torch.tensor(dt, device=device, dtype=torch.float32)
                dt = dt.to(device)
                if cam_cfg.get("mode") == "z":
                    dt = torch.tensor([0.0, 0.0, float(dt[2].item())], device=device)
                scale = float(cam_cfg.get("scale", 1.0))
                dt = dt * scale
                R = src2gym_T[:3, :3].to(device) if src2gym_T is not None else torch.eye(3, device=device)
                t = mat_rh[:3, 3] + R @ dt
                mat_rh = mat_rh.clone(); mat_rh[:3, 3] = t
        if cam_cfg and cam_cfg.get("debug", False):
            fin = mat_rh[:3, 3]
            raw = [round(x, 4) for x in debug_t_rh_raw.detach().cpu().tolist()]
            fin = [round(x, 4) for x in fin.detach().cpu().tolist()]
            print(f"[DEBUG] obj_rh iter={iter_idx} t_raw={raw} t_final={fin}")
        vis_env._manip_obj_rh_root_state[:] = _mat4_to_root13(mat_rh, device, z_offset)

    if "state_manip_obj_lh" in rollout_seq:
        mat_lh = rollout_seq["state_manip_obj_lh"]
        if torch.is_tensor(mat_lh) and mat_lh.ndim == 3:
            mat_lh = mat_lh[iter_idx]
        if src2gym_T is not None:
            debug_t_lh_raw = torch.as_tensor(mat_lh, device=device)[:3, 3].clone()
            mat_lh = _apply_T_to_mat4(src2gym_T.to(device), torch.as_tensor(mat_lh, device=device))
        else:
            debug_t_lh_raw = torch.as_tensor(mat_lh, device=device)[:3, 3].clone()
        if cam_cfg and cam_cfg.get("mode", "off") != "off" and cam_cfg.get("apply", "hand") in ("obj", "both"):
            dt = _fetch_cam_dt(rollout_seq, "lh", iter_idx, cam_cfg.get("key_base", "pred_cam_t_full"))
            if dt is not None:
                if not torch.is_tensor(dt):
                    dt = torch.tensor(dt, device=device, dtype=torch.float32)
                dt = dt.to(device)
                if cam_cfg.get("mode") == "z":
                    dt = torch.tensor([0.0, 0.0, float(dt[2].item())], device=device)
                scale = float(cam_cfg.get("scale", 1.0))
                dt = dt * scale
                R = src2gym_T[:3, :3].to(device) if src2gym_T is not None else torch.eye(3, device=device)
                t = mat_lh[:3, 3] + R @ dt
                mat_lh = mat_lh.clone(); mat_lh[:3, 3] = t
        if cam_cfg and cam_cfg.get("debug", False):
            fin = mat_lh[:3, 3]
            raw = [round(x, 4) for x in debug_t_lh_raw.detach().cpu().tolist()]
            fin = [round(x, 4) for x in fin.detach().cpu().tolist()]
            print(f"[DEBUG] obj_lh iter={iter_idx} t_raw={raw} t_final={fin}")
        vis_env._manip_obj_lh_root_state[:] = _mat4_to_root13(mat_lh, device, z_offset)

    # 不注入默认 DOF；若数据未提供 q/dq 则保持当前手指 posture

    # Push
    vis_env.gym.set_actor_root_state_tensor(vis_env.sim, gymtorch.unwrap_tensor(vis_env._root_state))


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
                "help": "Source coordinate frame of your data: [gym | mujoco | opencv]",
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

    iter_idx = 0
    total_len = _get_total_len(sample, side_key)
    step_count = 0
 
    while True:
        if side_key == "bih":
            _update_state_bih(vis_env, sample, iter_idx, src2gym_T, args.z_offset, cam_cfg)
        else:
            _update_state_single_hand(vis_env, sample, iter_idx, side_key, src2gym_T, args.z_offset, cam_cfg)
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
