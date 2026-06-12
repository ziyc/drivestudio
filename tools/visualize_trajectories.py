"""Visualize ego, camera, and registered trajectories directly from poses."""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from utils.camera import front_center_key_poses, get_interp_novel_trajectories
from utils.misc import import_str


CAMERA_DATA_CLASS = {
    "waymo": "datasets.waymo.waymo_sourceloader.WaymoCameraData",
    "nuscenes": "datasets.nuscenes.nuscenes_sourceloader.NuScenesCameraData",
    "argoverse": "datasets.argoverse.argoverse_sourceloader.ArgoVerseCameraData",
    "pandaset": "datasets.pandaset.pandaset_sourceloader.PandaCameraData",
    "nuplan": "datasets.nuplan.nuplan_sourceloader.NuPlanCameraData",
    "kitti": "datasets.kitti.kitti_sourceloader.KITTICameraData",
}


def parse_args():
    parser = argparse.ArgumentParser("Visualize registered trajectories from processed poses")
    parser.add_argument("--config_file", type=str, required=True, help="config file")
    parser.add_argument(
        "--traj_types",
        type=str,
        nargs="+",
        default=["lane_offset_left", "lane_offset_right"],
        help="registered trajectory names",
    )
    parser.add_argument("--frames", type=int, default=None, help="optional output frame count; defaults to all frames")
    parser.add_argument("--lane_offset", type=float, default=3.5, help="lateral offset in meters")
    parser.add_argument("--base_camera_id", type=int, default=0, help="base camera id for offset trajectories")
    parser.add_argument(
        "--offset_direction",
        type=str,
        default=None,
        help="optional forced lateral direction: +x,-x,+y,-y or comma-separated x,y,z",
    )
    parser.add_argument("--output_root", type=str, default="./outputs", help="root output directory")
    args, opts = parser.parse_known_args()
    args.opts = opts
    return args


def load_cfg(args):
    cfg = OmegaConf.load(args.config_file)
    args_from_cli = OmegaConf.from_cli(args.opts)

    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")

    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(os.path.join("configs", "datasets", f"{dataset_type}.yaml"))
        cfg = OmegaConf.merge(cfg, dataset_cfg)

    return OmegaConf.merge(cfg, args_from_cli)


def resolve_scene_dir(data_cfg) -> str:
    try:
        scene_name = f"{int(data_cfg.scene_idx):03d}"
    except Exception:
        scene_name = str(data_cfg.scene_idx)
    return os.path.join(data_cfg.data_root, scene_name)


def resolve_pose_dir(scene_dir: str) -> str:
    for dirname in ["ego_pose", "lidar_pose"]:
        pose_dir = os.path.join(scene_dir, dirname)
        if os.path.isdir(pose_dir):
            return pose_dir
    raise FileNotFoundError(f"Missing ego_pose/ or lidar_pose/ under {scene_dir}")


def get_total_frames(scene_dir: str) -> int:
    pose_dir = resolve_pose_dir(scene_dir)
    return len([name for name in os.listdir(pose_dir) if name.endswith(".txt")])


def get_reference_alignment(cfg, scene_dir: str, pose_dir: str, start_timestep: int) -> np.ndarray:
    pose_name = os.path.basename(pose_dir)
    if pose_name == "ego_pose":
        start_pose = np.loadtxt(os.path.join(pose_dir, f"{start_timestep:03d}.txt"))
        return np.linalg.inv(start_pose)

    if pose_name == "lidar_pose" and cfg.data.dataset == "nuscenes":
        front_camera_start = np.loadtxt(
            os.path.join(scene_dir, "extrinsics", f"{start_timestep:03d}_0.txt")
        )
        return np.linalg.inv(front_camera_start)

    start_pose = np.loadtxt(os.path.join(pose_dir, f"{start_timestep:03d}.txt"))
    return np.linalg.inv(start_pose)


def load_reference_poses(cfg, scene_dir: str, start_timestep: int, end_timestep: int) -> np.ndarray:
    pose_dir = resolve_pose_dir(scene_dir)
    world_to_reference = get_reference_alignment(cfg, scene_dir, pose_dir, start_timestep)
    poses = []
    for timestep in range(start_timestep, end_timestep):
        pose = np.loadtxt(os.path.join(pose_dir, f"{timestep:03d}.txt"))
        poses.append(world_to_reference @ pose)
    return np.stack(poses, axis=0)


def load_camera_poses(cfg, scene_dir: str, start_timestep: int, end_timestep: int) -> dict[int, torch.Tensor]:
    dataset_name = cfg.data.dataset
    if dataset_name not in CAMERA_DATA_CLASS:
        raise ValueError(f"Unsupported dataset for pose visualization: {dataset_name}")

    camera_data_cls = import_str(CAMERA_DATA_CLASS[dataset_name])
    camera_poses = {}
    for cam_id in cfg.data.pixel_source.cameras:
        camera_poses[int(cam_id)] = camera_data_cls.get_camera2worlds(
            scene_dir,
            int(cam_id),
            start_timestep,
            end_timestep,
        )
    return camera_poses


def parse_offset_direction(raw: str | None) -> torch.Tensor | None:
    if raw is None:
        return None

    mapping = {
        "+x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "+y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "+z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
    }
    if raw in mapping:
        return torch.tensor(mapping[raw], dtype=torch.float32)

    values = [float(x.strip()) for x in raw.split(",")]
    if len(values) != 3:
        raise ValueError("offset_direction must contain 3 values")
    return torch.tensor(values, dtype=torch.float32)


def build_traj_kwargs(args, ego_poses: np.ndarray) -> dict[str, dict]:
    kwargs = {}
    offset_direction = parse_offset_direction(args.offset_direction)
    for traj_type in args.traj_types:
        if traj_type in {"lane_offset_left", "lane_offset_right"}:
            traj_kwargs = {
                "base_camera_id": args.base_camera_id,
                "lane_offset_meters": args.lane_offset,
                "ego_poses": torch.from_numpy(ego_poses).float(),
            }
            if offset_direction is not None:
                traj_kwargs["offset_direction"] = offset_direction
            kwargs[traj_type] = traj_kwargs
    return kwargs


def key_pose_entries(per_cam_poses: dict[int, torch.Tensor], entries: list[tuple[int, int]]) -> list[dict]:
    return [
        {
            "cam_id": cam_id,
            "frame_idx": frame_idx,
            "pose": per_cam_poses[cam_id][frame_idx],
            "label": f"k{i}: cam{cam_id} f{frame_idx}",
        }
        for i, (cam_id, frame_idx) in enumerate(entries)
    ]


def trajectory_key_poses(
    traj_type: str,
    per_cam_poses: dict[int, torch.Tensor],
    original_frames: int,
) -> list[dict]:
    if traj_type == "front_center_interp":
        key_poses = front_center_key_poses(per_cam_poses, original_frames, base_camera_id=0)
        stride = max(original_frames // 4, 1)
        frames = list(range(0, original_frames, stride))
        if key_poses.shape[0] != len(frames):
            frames = list(range(key_poses.shape[0]))
        return key_pose_entries(per_cam_poses, [(0, frame_idx) for frame_idx in frames])

    if traj_type == "s_curve":
        return key_pose_entries(
            per_cam_poses,
            [
                (0, 0),
                (1, original_frames // 4),
                (0, original_frames // 2),
                (2, 3 * original_frames // 4),
                (0, original_frames - 1),
            ],
        )

    if traj_type == "three_key_poses":
        chosen_cam = 1 if 1 in per_cam_poses else 2
        middle_frame = original_frames // 2
        middle_pose = per_cam_poses[chosen_cam][middle_frame]
        start_pose = per_cam_poses[0][0]

        start_rotation = R.from_matrix(start_pose[:3, :3].detach().cpu().numpy())
        middle_rotation = R.from_matrix(middle_pose[:3, :3].detach().cpu().numpy())
        slerp = Slerp([0, 1], R.from_quat([start_rotation.as_quat(), middle_rotation.as_quat()]))
        interpolated_rotation = slerp(0.5).as_matrix()
        middle_key_pose = torch.eye(4, device=start_pose.device, dtype=start_pose.dtype)
        middle_key_pose[:3, :3] = torch.tensor(interpolated_rotation, device=start_pose.device, dtype=start_pose.dtype)
        middle_key_pose[:3, 3] = middle_pose[:3, 3]

        entries = key_pose_entries(
            per_cam_poses,
            [(0, 0), (chosen_cam, middle_frame), (0, original_frames - 1)],
        )
        entries[1]["pose"] = middle_key_pose
        return entries

    return []


def set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 1.0)

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def camera_forward_vectors(poses: np.ndarray, length: float = 6.0) -> np.ndarray:
    forward = poses[:, :3, 2].copy()
    norms = np.linalg.norm(forward, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return forward / norms * length


def pose_forward_vector(pose: np.ndarray, length: float = 4.5) -> np.ndarray:
    forward = pose[:3, 2].copy()
    norm = max(np.linalg.norm(forward), 1e-8)
    return forward / norm * length


def plot_trajectory_3d(
    ego_poses: np.ndarray,
    camera_poses: dict[int, np.ndarray],
    base_camera_id: int,
    trajectory: np.ndarray,
    key_pose_entries: list[dict],
    title: str,
    output_path: str,
) -> None:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    base_poses = camera_poses[base_camera_id]
    all_points = [base_poses[:, :3, 3], trajectory[:, :3, 3]]

    ax.plot(
        base_poses[:, 0, 3],
        base_poses[:, 1, 3],
        base_poses[:, 2, 3],
        color="#0057ff",
        linewidth=2.4,
        label=f"camera_{base_camera_id}_trajectory",
    )

    stride = max(len(base_poses) // 10, 1)
    sampled_positions = base_poses[::stride, :3, 3]
    sampled_forward = camera_forward_vectors(base_poses[::stride])
    ax.scatter(
        sampled_positions[:, 0],
        sampled_positions[:, 1],
        sampled_positions[:, 2],
        color="#111111",
        s=36,
        alpha=1.0,
    )
    ax.quiver(
        sampled_positions[:, 0],
        sampled_positions[:, 1],
        sampled_positions[:, 2],
        sampled_forward[:, 0],
        sampled_forward[:, 1],
        sampled_forward[:, 2],
        color="#111111",
        length=1.0,
        normalize=False,
        linewidth=2.8,
        arrow_length_ratio=0.35,
        alpha=1.0,
        label=f"camera_{base_camera_id}_heading",
    )

    ax.plot(
        trajectory[:, 0, 3],
        trajectory[:, 1, 3],
        trajectory[:, 2, 3],
        color="#ff2d55",
        linewidth=2.8,
        label="trajectory",
    )
    ax.scatter(trajectory[0, 0, 3], trajectory[0, 1, 3], trajectory[0, 2, 3], color="#ff2d55", s=36)
    ax.scatter(trajectory[-1, 0, 3], trajectory[-1, 1, 3], trajectory[-1, 2, 3], color="#ff2d55", s=36, marker="x")

    if key_pose_entries:
        key_colors = plt.cm.Set1(np.linspace(0, 1, max(len(key_pose_entries), 1)))
        for idx, entry in enumerate(key_pose_entries):
            pose = entry["pose"].detach().cpu().numpy() if isinstance(entry["pose"], torch.Tensor) else entry["pose"]
            pos = pose[:3, 3]
            forward = pose_forward_vector(pose)
            ax.scatter(pos[0], pos[1], pos[2], color=key_colors[idx], s=70, edgecolors="black", linewidths=0.8)
            ax.quiver(
                pos[0],
                pos[1],
                pos[2],
                forward[0],
                forward[1],
                forward[2],
                color=key_colors[idx],
                length=1.0,
                normalize=False,
                linewidth=2.2,
                arrow_length_ratio=0.3,
                alpha=1.0,
            )
            ax.text(pos[0], pos[1], pos[2], f" {entry['label']}", color="black", fontsize=9)

    points = np.concatenate(all_points, axis=0)
    set_equal_axes(ax, points)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = load_cfg(args)

    scene_dir = resolve_scene_dir(cfg.data)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")

    total_frames = get_total_frames(scene_dir)
    start_timestep = int(cfg.data.start_timestep)
    end_timestep = total_frames if int(cfg.data.end_timestep) == -1 else int(cfg.data.end_timestep) + 1

    ego_poses = load_reference_poses(cfg, scene_dir, start_timestep, end_timestep)
    per_cam_poses = load_camera_poses(cfg, scene_dir, start_timestep, end_timestep)

    target_frames = args.frames if args.frames is not None else ego_poses.shape[0]
    traj_kwargs = build_traj_kwargs(args, ego_poses)

    for traj_type in args.traj_types:
        key_pose_entries = trajectory_key_poses(
            traj_type,
            per_cam_poses,
            per_cam_poses[list(per_cam_poses.keys())[0]].shape[0],
        )
        trajectory = get_interp_novel_trajectories(
            cfg.data.dataset,
            str(cfg.data.scene_idx),
            per_cam_poses,
            traj_type=traj_type,
            target_frames=target_frames,
            traj_kwargs=traj_kwargs.get(traj_type),
        )
        trajectory_np = trajectory.detach().cpu().numpy()
        camera_poses_np = {
            cam_id: poses.detach().cpu().numpy() for cam_id, poses in per_cam_poses.items()
        }

        traj_tag = traj_type
        if traj_type in {"lane_offset_left", "lane_offset_right"}:
            traj_tag = f"{traj_type}@{args.lane_offset:g}"

        output_dir = os.path.join(
            args.output_root,
            "traj",
            f"{cfg.data.dataset}-s{cfg.data.scene_idx}-{traj_tag}",
        )
        os.makedirs(output_dir, exist_ok=True)

        plot_trajectory_3d(
            ego_poses=ego_poses,
            camera_poses=camera_poses_np,
            base_camera_id=args.base_camera_id,
            trajectory=trajectory_np,
            key_pose_entries=key_pose_entries,
            title=f"{cfg.data.dataset} scene {cfg.data.scene_idx} {traj_type}",
            output_path=os.path.join(output_dir, "trajectory_3d.png"),
        )

        print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
