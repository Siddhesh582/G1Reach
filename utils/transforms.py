"""
utils/transforms.py - Coordinate frame math and rotation helpers used across the project.
All functions operate on torch tensors and are batch-aware (first dim = N).
"""
from __future__ import annotations
import torch


def quat_to_rot_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.
    Args:
        quat : (N, 4) - [w, x, y, z] convention
    Returns:
        R    : (N, 3, 3)
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    N = quat.shape[0]
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(N, 3, 3)
    return R


def rot_matrix_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """
    Rotation matrix to XYZ Euler angles (radians).
    Args:
        R   : (N, 3, 3)
    Returns:
        eul : (N, 3) - [roll, pitch, yaw]
    """
    pitch = torch.asin((-R[:, 2, 0]).clamp(-1.0, 1.0))
    roll  = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    yaw   = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    return torch.stack([roll, pitch, yaw], dim=-1)


def transform_points(
    points: torch.Tensor,
    T:      torch.Tensor,
) -> torch.Tensor:
    """
    Apply a 4x4 homogeneous transform to a batch of 3D points.
    Args:
        points : (N, 3)
        T      : (4, 4) or (N, 4, 4)
    Returns:
        (N, 3)
    """
    ones  = torch.ones((*points.shape[:-1], 1), device=points.device, dtype=points.dtype)
    pts_h = torch.cat([points, ones], dim=-1)  # (N, 4)
    if T.dim() == 2:
        return (T[:3, :3] @ points.unsqueeze(-1)).squeeze(-1) + T[:3, 3]
    return (T[:, :3, :3] @ points.unsqueeze(-1)).squeeze(-1) + T[:, :3, 3]


def make_homogeneous(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Build a 4x4 homogeneous matrix from rotation + translation.
    Args:
        R : (3, 3)
        t : (3,)
    Returns:
        T : (4, 4)
    """
    T = torch.eye(4, device=R.device, dtype=R.dtype)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def compute_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Euclidean distance between two batches of 3D points.
    Args:
        a, b : (N, 3)
    Returns:
        dist : (N,)
    """
    return torch.norm(a - b, dim=-1)


def world_to_base_frame(
    pos_world:       torch.Tensor,
    root_pos_world:  torch.Tensor,
    root_quat_world: torch.Tensor,
) -> torch.Tensor:
    """
    Transform positions from world frame to robot base frame.
    Args:
        pos_world       : (N, 3) - positions in world frame
        root_pos_world  : (N, 3) - robot root position in world frame
        root_quat_world : (N, 4) - robot root orientation [w, x, y, z]
    Returns:
        pos_base        : (N, 3)
    """
    R_wb = quat_to_rot_matrix(root_quat_world)  # world to base rotation: (N, 3, 3)
    R_bw = R_wb.transpose(1, 2)                 # transpose gives base to world
    delta = pos_world - root_pos_world           # (N, 3)
    pos_base = (R_bw @ delta.unsqueeze(-1)).squeeze(-1)
    return pos_base