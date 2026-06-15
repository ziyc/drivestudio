"""Camera pose manipulation and trajectory generation."""

from typing import Callable, Dict

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


TrajectoryFn = Callable[..., torch.Tensor]
_TRAJECTORIES: dict[str, TrajectoryFn] = {}


def register_trajectory(name: str):
    def decorator(fn: TrajectoryFn) -> TrajectoryFn:
        _TRAJECTORIES[name] = fn
        return fn

    return decorator


def list_registered_trajectories() -> list[str]:
    return sorted(_TRAJECTORIES.keys())


def interpolate_poses(key_poses: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Interpolate between key poses to generate a smooth trajectory."""
    device = key_poses.device
    key_poses_np = key_poses.detach().cpu().numpy()

    translations = key_poses_np[:, :3, 3]
    rotations = key_poses_np[:, :3, :3]

    times = np.linspace(0.0, 1.0, len(key_poses_np))
    target_times = np.linspace(0.0, 1.0, target_frames)

    interp_translations = np.stack(
        [np.interp(target_times, times, translations[:, i]) for i in range(3)],
        axis=-1,
    )

    key_rots = R.from_matrix(rotations)
    slerp = Slerp(times, key_rots)
    interp_rotations = slerp(target_times).as_matrix()

    interp_poses = np.repeat(np.eye(4, dtype=np.float32)[None], target_frames, axis=0)
    interp_poses[:, :3, :3] = interp_rotations
    interp_poses[:, :3, 3] = interp_translations

    return torch.tensor(interp_poses, dtype=torch.float32, device=device)


def sample_raw_poses(poses: torch.Tensor, target_frames: int) -> torch.Tensor:
    if poses.shape[0] == target_frames:
        return poses.clone()
    indices = torch.linspace(0, poses.shape[0] - 1, target_frames, device=poses.device)
    return poses.index_select(0, indices.round().long())


def front_center_key_frame_indices(original_frames: int) -> list[int]:
    stride = max(original_frames // 4, 1)
    frames = list(range(0, original_frames, stride))
    if not frames:
        frames = [0]
    return frames


def front_center_key_poses(
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    base_camera_id: int = 0,
) -> torch.Tensor:
    if base_camera_id not in per_cam_poses:
        raise ValueError(f"Camera {base_camera_id} is not available")
    frame_indices = front_center_key_frame_indices(original_frames)
    key_poses = per_cam_poses[base_camera_id][frame_indices]
    if key_poses.shape[0] < 2:
        key_poses = per_cam_poses[base_camera_id]
    return key_poses



def get_interp_novel_trajectories(
    dataset_type: str,
    scene_idx: str,
    per_cam_poses: Dict[int, torch.Tensor],
    traj_type: str = "front_center_interp",
    target_frames: int | None = None,
    traj_kwargs: dict | None = None,
) -> torch.Tensor:
    del dataset_type, scene_idx

    if traj_type not in _TRAJECTORIES:
        raise ValueError(
            f"Unknown trajectory type: {traj_type}. Available: {', '.join(list_registered_trajectories())}"
        )

    original_frames = per_cam_poses[list(per_cam_poses.keys())[0]].shape[0]
    kwargs = {} if traj_kwargs is None else dict(traj_kwargs)
    return _TRAJECTORIES[traj_type](per_cam_poses, original_frames, target_frames, **kwargs)


@register_trajectory("front_center_interp")
def front_center_interp(
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    key_poses = front_center_key_poses(per_cam_poses, original_frames, base_camera_id=0)
    if target_frames is None:
        target_frames = key_poses.shape[0]
    return interpolate_poses(key_poses, target_frames)


@register_trajectory("s_curve")
def s_curve(
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    assert all(cam in per_cam_poses for cam in [0, 1, 2]), "Front cameras 0, 1, 2 are required for s_curve"
    key_poses = torch.cat(
        [
            per_cam_poses[0][0:1],
            per_cam_poses[1][original_frames // 4 : original_frames // 4 + 1],
            per_cam_poses[0][original_frames // 2 : original_frames // 2 + 1],
            per_cam_poses[2][3 * original_frames // 4 : 3 * original_frames // 4 + 1],
            per_cam_poses[0][-1:],
        ],
        dim=0,
    )
    return interpolate_poses(key_poses, target_frames)


@register_trajectory("three_key_poses")
def three_key_poses_trajectory(
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    assert 0 in per_cam_poses, "Front center camera (ID 0) is required"
    assert 1 in per_cam_poses or 2 in per_cam_poses, "Either camera 1 or camera 2 is required"

    start_pose = per_cam_poses[0][0]
    middle_frame = int(original_frames // 2)
    chosen_cam = 1 if 1 in per_cam_poses else 2
    middle_pose = per_cam_poses[chosen_cam][middle_frame]

    start_rotation = R.from_matrix(start_pose[:3, :3].detach().cpu().numpy())
    middle_rotation = R.from_matrix(middle_pose[:3, :3].detach().cpu().numpy())
    slerp = Slerp([0, 1], R.from_quat([start_rotation.as_quat(), middle_rotation.as_quat()]))
    interpolated_rotation = slerp(0.5).as_matrix()

    middle_key_pose = torch.eye(4, device=start_pose.device, dtype=start_pose.dtype)
    middle_key_pose[:3, :3] = torch.tensor(interpolated_rotation, device=start_pose.device, dtype=start_pose.dtype)
    middle_key_pose[:3, 3] = middle_pose[:3, 3]

    key_poses = torch.stack([start_pose, middle_key_pose, per_cam_poses[0][-1]])
    return interpolate_poses(key_poses, target_frames)


def _normalize(vectors: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-8)
    return vectors / norms


def _trajectory_length(poses: torch.Tensor) -> torch.Tensor:
    if poses.shape[0] < 2:
        return poses.new_tensor(0.0)
    deltas = poses[1:, :3, 3] - poses[:-1, :3, 3]
    return torch.linalg.norm(deltas, dim=-1).sum()


def _ego_frame_lane_offset(
    per_cam_poses: Dict[int, torch.Tensor],
    base_camera_id: int,
    lane_offset_meters: float,
    lane_offset_ratio: float | None,
    ego_poses: torch.Tensor | None,
    offset_direction: torch.Tensor | None,
    target_frames: int,
    direction_sign: float,
) -> torch.Tensor:
    if base_camera_id not in per_cam_poses:
        raise ValueError(f"Camera {base_camera_id} is not available")
    if ego_poses is None:
        raise ValueError("lane offset trajectories require ego_poses")

    poses = per_cam_poses[base_camera_id]
    ego_poses = torch.as_tensor(ego_poses, dtype=poses.dtype, device=poses.device)
    if ego_poses.shape[0] != poses.shape[0]:
        raise ValueError("ego_poses must have the same number of frames as camera poses")

    ego_to_camera = torch.linalg.inv(ego_poses[0]) @ poses[0]
    ego_to_virtual_camera = ego_to_camera.clone()

    if offset_direction is not None:
        lateral = torch.as_tensor(offset_direction, dtype=poses.dtype, device=poses.device)
        if lateral.numel() != 3:
            raise ValueError("offset_direction must contain 3 values")
        lateral = _normalize(lateral.reshape(1, 3)).reshape(3)
    else:
        if 1 not in per_cam_poses or 2 not in per_cam_poses:
            raise ValueError(
                "lane offset trajectories require front-left (1) and front-right (2) camera poses"
            )
        ego_to_front_left = torch.linalg.inv(ego_poses[0]) @ per_cam_poses[1][0]
        ego_to_front_right = torch.linalg.inv(ego_poses[0]) @ per_cam_poses[2][0]
        lateral = ego_to_front_left[:3, 3] - ego_to_front_right[:3, 3]
        lateral = _normalize(lateral.reshape(1, 3)).reshape(3)
        lateral = lateral * direction_sign

    effective_lane_offset = float(lane_offset_meters)
    if lane_offset_ratio is not None:
        effective_lane_offset = float(_trajectory_length(poses).item() * lane_offset_ratio)

    offset_vector = effective_lane_offset * lateral
    print(
        "[lane_offset] "
        f"base_camera={base_camera_id} "
        f"meters={effective_lane_offset:.4f} "
        f"ratio={lane_offset_ratio if lane_offset_ratio is not None else 'None'} "
        f"lateral=[{lateral[0].item():.6f}, {lateral[1].item():.6f}, {lateral[2].item():.6f}] "
        f"offset=[{offset_vector[0].item():.6f}, {offset_vector[1].item():.6f}, {offset_vector[2].item():.6f}] "
        f"norm={torch.linalg.norm(offset_vector).item():.6f}"
    )

    ego_to_virtual_camera[:3, 3] = ego_to_virtual_camera[:3, 3] + offset_vector

    offset_poses = ego_poses @ ego_to_virtual_camera.unsqueeze(0)
    if target_frames is not None and target_frames != offset_poses.shape[0]:
        offset_poses = interpolate_poses(offset_poses, target_frames)
    return offset_poses

@register_trajectory("ego_raw")
def ego_raw(
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    base_camera_id: int = 0,
) -> torch.Tensor:
    if base_camera_id not in per_cam_poses:
        raise ValueError(f"Camera {base_camera_id} is not available")
    poses = per_cam_poses[base_camera_id]
    if target_frames is None or target_frames == poses.shape[0]:
        return poses.clone()
    return sample_raw_poses(poses, target_frames)


@register_trajectory("lane_offset_left")
def lane_offset_left(
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    base_camera_id: int = 0,
    lane_offset_meters: float = 3.5,
    lane_offset_ratio: float | None = None,
    ego_poses: torch.Tensor | None = None,
    offset_direction: torch.Tensor | None = None,
) -> torch.Tensor:
    return _ego_frame_lane_offset(
        per_cam_poses=per_cam_poses,
        base_camera_id=base_camera_id,
        lane_offset_meters=lane_offset_meters,
        lane_offset_ratio=lane_offset_ratio,
        ego_poses=ego_poses,
        offset_direction=offset_direction,
        target_frames=target_frames,
        direction_sign=1.0,
    )


@register_trajectory("lane_offset_right")
def lane_offset_right(
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
    base_camera_id: int = 0,
    lane_offset_meters: float = 3.5,
    lane_offset_ratio: float | None = None,
    ego_poses: torch.Tensor | None = None,
    offset_direction: torch.Tensor | None = None,
) -> torch.Tensor:
    return _ego_frame_lane_offset(
        per_cam_poses=per_cam_poses,
        base_camera_id=base_camera_id,
        lane_offset_meters=lane_offset_meters,
        lane_offset_ratio=lane_offset_ratio,
        ego_poses=ego_poses,
        offset_direction=offset_direction,
        target_frames=target_frames,
        direction_sign=-1.0,
    )
