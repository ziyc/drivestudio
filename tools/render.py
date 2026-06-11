"""Render registered novel trajectories and export trajectory visualizations."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.driving_dataset import DrivingDataset
from utils.camera import list_registered_trajectories
from utils.misc import import_str
from utils.output_paths import build_auto_output_dir, count_cameras


CAM_COLORS = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#9b59b6",
    "#f39c12",
    "#1abc9c",
    "#16a085",
    "#34495e",
]
NOVEL_COLOR = "#ff2d55"
REFERENCE_COLOR = "#111111"


def short_camera_tag(cam_name: str) -> str:
    name = cam_name.lower()
    prefixes = ["cam_", "camera_"]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("_", "-")


def parse_args():
    parser = argparse.ArgumentParser("Render registered novel trajectories")
    parser.add_argument("--resume_from", type=str, required=False, help="path to checkpoint")
    parser.add_argument("--traj_types", type=str, nargs="+", default=None, help="registered trajectory names")
    parser.add_argument("--frames", type=int, default=None, help="number of frames to render")
    parser.add_argument("--fps", type=int, default=None, help="output video fps")
    parser.add_argument("--output_root", type=str, default="./outputs", help="root directory for command outputs")
    parser.add_argument("--output_dir", type=str, default=None, help="directory for videos and trajectory files")
    parser.add_argument("--traj_arrow_stride", type=int, default=10, help="arrow stride for trajectory plots")
    parser.add_argument("--traj_arrow_length", type=float, default=1.5, help="arrow length for trajectory plots")
    parser.add_argument("--lane_offset", type=float, default=None, help="override lateral offset in meters for raw_lane_offset_left/right")
    parser.add_argument("--offset_direction", type=str, default=None, help="forced offset direction: +x,-x,+y,-y or comma-separated x,y,z e.g. 0,2,0")
    parser.add_argument("--base_camera_id", type=int, default=None, help="base camera id for raw offset trajectories and GT export; defaults to the repo reference camera")
    parser.add_argument("--skip_html", action="store_true", help="skip interactive 3D html output")
    parser.add_argument("--list_traj_types", action="store_true", help="print registered trajectory names and exit")
    parser.add_argument("opts", help="OmegaConf overrides", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def maybe_pad_frame(rgb: np.ndarray, macro_block_size: int = 16) -> np.ndarray:
    h, w = rgb.shape[:2]
    pad_h = (macro_block_size - h % macro_block_size) % macro_block_size
    pad_w = (macro_block_size - w % macro_block_size) % macro_block_size
    if pad_h == 0 and pad_w == 0:
        return rgb
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    rgb_t = F.pad(rgb_t, (0, pad_w, 0, pad_h), mode="replicate")
    return rgb_t.squeeze(0).permute(1, 2, 0).numpy()


def render_novel_video(trainer, render_data: list, save_path: str, fps: int = 30) -> None:
    trainer.set_eval()
    writer = None
    with torch.no_grad():
        for frame_data in render_data:
            for key, value in frame_data["cam_infos"].items():
                frame_data["cam_infos"][key] = value.cuda(non_blocking=True)
            for key, value in frame_data["image_infos"].items():
                frame_data["image_infos"][key] = value.cuda(non_blocking=True)

            outputs = trainer(
                image_infos=frame_data["image_infos"],
                camera_infos=frame_data["cam_infos"],
                novel_view=True,
            )
            rgb = outputs["rgb"].detach().cpu().numpy().clip(min=1.0e-6, max=1 - 1.0e-6)
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            rgb_uint8 = maybe_pad_frame(rgb_uint8)
            if writer is None:
                writer = imageio.get_writer(save_path, mode="I", fps=fps, macro_block_size=None)
            writer.append_data(rgb_uint8)
    if writer is not None:
        writer.close()
    print(f"Video saved to {save_path}")


def get_reference_cam_id(dataset) -> int:
    return 1 if dataset.type == "argoverse" else 0


def resolve_base_camera_id(dataset, args) -> int:
    if args.base_camera_id is not None:
        return args.base_camera_id
    return get_reference_cam_id(dataset)


def save_gt_video(dataset, render_data: list, save_path: str, fps: int = 30, cam_id: int = 0) -> None:
    camera = dataset.pixel_source.camera_data[cam_id]
    writer = None
    for frame_data in render_data:
        frame_idx = int(frame_data["image_infos"]["frame_idx"][0, 0].item())
        gt = camera.images[frame_idx].detach().cpu().numpy().clip(0.0, 1.0)
        gt_uint8 = (gt * 255).astype(np.uint8)
        gt_uint8 = maybe_pad_frame(gt_uint8)
        if writer is None:
            writer = imageio.get_writer(save_path, mode="I", fps=fps, macro_block_size=None)
        writer.append_data(gt_uint8)
    if writer is not None:
        writer.close()
    print(f"GT video saved to {save_path}")


def build_traj_kwargs(traj_types: list[str], base_camera_id: int, lane_offset: float | None, offset_direction: str | None, reference_traj: np.ndarray | None) -> dict[str, dict]:
    dir_vec = _parse_offset_direction(offset_direction) if offset_direction else None
    traj_kwargs = {}
    for traj_type in traj_types:
        if traj_type in {"raw_lane_offset_right", "raw_lane_offset_left"}:
            kwargs = {"base_camera_id": base_camera_id}
            if lane_offset is not None:
                kwargs["lane_offset_meters"] = lane_offset
            if dir_vec is not None:
                kwargs["offset_direction"] = dir_vec
            elif reference_traj is not None:
                kwargs["ego_positions"] = torch.from_numpy(reference_traj).float()
            traj_kwargs[traj_type] = kwargs
    return traj_kwargs


def _parse_offset_direction(s: str) -> torch.Tensor:
    m = {"+x": (1., 0., 0.), "-x": (-1., 0., 0.), "x": (1., 0., 0.),
         "+y": (0., 1., 0.), "-y": (0., -1., 0.), "y": (0., 1., 0.),
         "+z": (0., 0., 1.), "-z": (0., 0., -1.), "z": (0., 0., 1.)}
    if s.lower() in m:
        return torch.tensor(m[s.lower()], dtype=torch.float32)
    parts = [float(x.strip()) for x in s.split(",")]
    return torch.tensor(parts, dtype=torch.float32)


def load_camera_poses(scene_dir: str, cameras: list[int]) -> dict[int, list[tuple[int, np.ndarray]]]:
    extrinsics_dir = os.path.join(scene_dir, "extrinsics")
    camera_front_start = np.loadtxt(os.path.join(extrinsics_dir, "000_0.txt"))
    world_to_plot = np.linalg.inv(camera_front_start)

    poses = {cam_id: [] for cam_id in cameras}
    frame_idx = 0
    while True:
        any_found = False
        for cam_id in cameras:
            pose_path = os.path.join(extrinsics_dir, f"{frame_idx:03d}_{cam_id}.txt")
            if not os.path.exists(pose_path):
                continue
            any_found = True
            cam_to_world = np.loadtxt(pose_path)
            poses[cam_id].append((frame_idx, world_to_plot @ cam_to_world))
        if not any_found:
            break
        frame_idx += 1
    return poses


def load_reference_trajectory(scene_dir: str) -> np.ndarray | None:
    pose_dir = os.path.join(scene_dir, "ego_pose")
    if not os.path.isdir(pose_dir):
        pose_dir = os.path.join(scene_dir, "lidar_pose")
    if not os.path.isdir(pose_dir):
        return None

    camera_front_start = np.loadtxt(os.path.join(scene_dir, "extrinsics", "000_0.txt"))
    world_to_plot = np.linalg.inv(camera_front_start)

    traj = []
    frame_idx = 0
    while True:
        pose_path = os.path.join(pose_dir, f"{frame_idx:03d}.txt")
        if not os.path.exists(pose_path):
            break
        pose_to_world = np.loadtxt(pose_path)
        pose_in_plot = world_to_plot @ pose_to_world
        traj.append(pose_in_plot[:3, 3])
        frame_idx += 1
    if not traj:
        return None
    return np.stack(traj, axis=0)


def get_arrow_xy(c2w: np.ndarray, length: float) -> np.ndarray:
    forward = c2w[:3, 2].copy()
    forward[2] = 0.0
    norm = np.linalg.norm(forward[:2])
    if norm < 1e-6:
        return np.zeros(2, dtype=float)
    return forward[:2] / norm * length


def get_bounds(source_poses, reference_traj: np.ndarray | None, novel_traj: np.ndarray, margin: float = 5.0):
    points = []
    for cam_poses in source_poses.values():
        for _, c2w in cam_poses:
            points.append(c2w[:2, 3])
    if reference_traj is not None:
        points.extend(reference_traj[:, :2])
    points.extend(novel_traj[:, :2, 3])
    pts = np.asarray(points)
    return (
        pts[:, 0].min() - margin,
        pts[:, 0].max() + margin,
        pts[:, 1].min() - margin,
        pts[:, 1].max() + margin,
    )


def save_topdown_trajectory_plot(
    source_poses,
    reference_traj: np.ndarray | None,
    novel_traj: np.ndarray,
    cam_names: dict[int, str],
    base_camera_id: int,
    title: str,
    output_path: str,
    arrow_stride: int,
    arrow_length: float,
):
    xmin, xmax, ymin, ymax = get_bounds(source_poses, reference_traj, novel_traj)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    if reference_traj is not None:
        ax.plot(
            reference_traj[:, 0],
            reference_traj[:, 1],
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            label="Reference Pose",
        )

    for idx, (cam_id, cam_poses) in enumerate(source_poses.items()):
        if not cam_poses:
            continue
        pts = np.array([pose[:3, 3] for _, pose in cam_poses])
        color = CAM_COLORS[idx % len(CAM_COLORS)]
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.2, alpha=0.8, label=cam_names.get(cam_id, f"Cam{cam_id}"))
        for i in range(0, len(cam_poses), arrow_stride):
            _, c2w = cam_poses[i]
            pos = c2w[:3, 3]
            arrow_xy = get_arrow_xy(c2w, arrow_length)
            ax.arrow(pos[0], pos[1], arrow_xy[0], arrow_xy[1], color=color, head_width=0.2, width=0.03, alpha=0.55, length_includes_head=True)

    if base_camera_id in source_poses and source_poses[base_camera_id]:
        base_pts = np.array([pose[:3, 3] for _, pose in source_poses[base_camera_id]])
        ax.plot(
            base_pts[:, 0],
            base_pts[:, 1],
            color="#0057ff",
            linewidth=2.5,
            alpha=0.95,
            label=f"Base Camera ({cam_names.get(base_camera_id, base_camera_id)})",
        )

    ax.plot(novel_traj[:, 0, 3], novel_traj[:, 1, 3], color=NOVEL_COLOR, linewidth=2.5, alpha=0.95, label="Novel Trajectory")
    for i in range(0, len(novel_traj), arrow_stride):
        pos = novel_traj[i, :3, 3]
        arrow_xy = get_arrow_xy(novel_traj[i], arrow_length)
        ax.arrow(pos[0], pos[1], arrow_xy[0], arrow_xy[1], color=NOVEL_COLOR, head_width=0.25, width=0.04, alpha=0.8, length_includes_head=True)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"saved: {output_path}")


def add_direction_segments(fig: go.Figure, traj: np.ndarray, color: str, stride: int, length: float, name: str):
    for i in range(0, len(traj), stride):
        pose = traj[i]
        origin = pose[:3, 3]
        forward = pose[:3, 2]
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            continue
        tip = origin + forward / norm * length
        fig.add_trace(
            go.Scatter3d(
                x=[origin[0], tip[0]],
                y=[origin[1], tip[1]],
                z=[origin[2], tip[2]],
                mode="lines",
                line=dict(color=color, width=5),
                name=name,
                showlegend=False,
                hoverinfo="skip",
            )
        )


def save_trajectory_html(
    source_poses,
    reference_traj: np.ndarray | None,
    novel_traj: np.ndarray,
    cam_names: dict[int, str],
    base_camera_id: int,
    title: str,
    output_path: str,
    arrow_stride: int,
    arrow_length: float,
):
    fig = go.Figure()

    if reference_traj is not None:
        fig.add_trace(
            go.Scatter3d(
                x=reference_traj[:, 0],
                y=reference_traj[:, 1],
                z=reference_traj[:, 2],
                mode="lines",
                line=dict(color=REFERENCE_COLOR, width=6),
                name="Reference Pose",
            )
        )

    for idx, (cam_id, cam_poses) in enumerate(source_poses.items()):
        if not cam_poses:
            continue
        pts = np.array([pose[:3, 3] for _, pose in cam_poses])
        color = CAM_COLORS[idx % len(CAM_COLORS)]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=2.5, color=color),
                name=cam_names.get(cam_id, f"Cam{cam_id}"),
            )
        )

    if base_camera_id in source_poses and source_poses[base_camera_id]:
        base_pts = np.array([pose[:3, 3] for _, pose in source_poses[base_camera_id]])
        fig.add_trace(
            go.Scatter3d(
                x=base_pts[:, 0],
                y=base_pts[:, 1],
                z=base_pts[:, 2],
                mode="lines",
                line=dict(color="#0057ff", width=7),
                name=f"Base Camera ({cam_names.get(base_camera_id, base_camera_id)})",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=novel_traj[:, 0, 3],
            y=novel_traj[:, 1, 3],
            z=novel_traj[:, 2, 3],
            mode="lines+markers",
            line=dict(color=NOVEL_COLOR, width=7),
            marker=dict(size=2.5, color=NOVEL_COLOR),
            name="Novel Trajectory",
        )
    )
    add_direction_segments(fig, novel_traj, NOVEL_COLOR, arrow_stride, arrow_length, "Novel Forward")

    fig.update_layout(
        title=title,
        template="plotly_white",
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"saved: {output_path}")


def load_cfg_and_dataset(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    dataset = DrivingDataset(data_cfg=cfg.data)
    return cfg, dataset, log_dir


def build_trainer(cfg, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    return trainer


def main():
    args = parse_args()

    if args.list_traj_types:
        for name in list_registered_trajectories():
            print(name)
        return

    if args.resume_from is None:
        raise ValueError("--resume_from is required unless --list_traj_types is used")

    cfg, dataset, log_dir = load_cfg_and_dataset(args)
    trainer = build_trainer(cfg, dataset)
    trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)

    render_novel_cfg = cfg.render.get("render_novel", OmegaConf.create())
    traj_types = args.traj_types if args.traj_types is not None else render_novel_cfg.get("traj_types", list_registered_trajectories())
    frames = args.frames if args.frames is not None else render_novel_cfg.get("frames", dataset.frame_num)
    fps = args.fps if args.fps is not None else render_novel_cfg.get("fps", cfg.render.fps)
    base_camera_id = resolve_base_camera_id(dataset, args)
    base_camera_name = dataset.pixel_source.camera_data[base_camera_id].cam_name
    base_camera_tag = short_camera_tag(base_camera_name)
    traj_name_tag = "+".join(traj_types)
    if args.lane_offset is not None:
        traj_name_tag = f"{traj_name_tag}@{args.lane_offset:g}"

    scene_dir = dataset.data_path
    source_cameras = list(dataset.pixel_source.camera_list)
    source_poses = load_camera_poses(scene_dir, source_cameras)
    reference_traj = load_reference_trajectory(scene_dir)
    cam_names = {cam_id: dataset.pixel_source.camera_data[cam_id].cam_name for cam_id in source_cameras}

    traj_kwargs = build_traj_kwargs(traj_types, base_camera_id, args.lane_offset, args.offset_direction, reference_traj)

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        render_tags = [
            cfg.data.dataset,
            f"s{cfg.data.scene_idx}",
            f"cam{count_cameras(cfg.data.pixel_source.cameras)}",
            f"step{trainer.step}",
            base_camera_tag,
            traj_name_tag,
        ]
        output_dir = build_auto_output_dir(args.output_root, "render", *render_tags)
    os.makedirs(output_dir, exist_ok=True)

    render_traj = dataset.get_novel_render_traj(traj_types=traj_types, target_frames=frames, traj_kwargs=traj_kwargs)
    for traj_type, traj in render_traj.items():
        render_data = dataset.prepare_novel_view_render_data(traj, cam_id=base_camera_id)
        video_path = os.path.join(output_dir, f"{traj_type}.mp4")
        render_novel_video(trainer, render_data, video_path, fps=fps)
        gt_video_path = os.path.join(output_dir, f"{traj_type}_gt.mp4")
        save_gt_video(dataset, render_data, gt_video_path, fps=fps, cam_id=base_camera_id)

        traj_np = traj.detach().cpu().numpy()
        npy_path = os.path.join(output_dir, f"{traj_type}.npy")
        np.save(npy_path, traj_np)
        print(f"saved: {npy_path}")

        title = f"{cfg.data.dataset} scene {dataset.scene_idx} - {traj_type}"
        png_path = os.path.join(output_dir, f"{traj_type}_traj.png")
        save_topdown_trajectory_plot(
            source_poses=source_poses,
            reference_traj=reference_traj,
            novel_traj=traj_np,
            cam_names=cam_names,
            base_camera_id=base_camera_id,
            title=title,
            output_path=png_path,
            arrow_stride=args.traj_arrow_stride,
            arrow_length=args.traj_arrow_length,
        )

        if not args.skip_html:
            html_path = os.path.join(output_dir, f"{traj_type}_traj.html")
            save_trajectory_html(
                source_poses=source_poses,
                reference_traj=reference_traj,
                novel_traj=traj_np,
                cam_names=cam_names,
                base_camera_id=base_camera_id,
                title=title,
                output_path=html_path,
                arrow_stride=args.traj_arrow_stride,
                arrow_length=args.traj_arrow_length,
            )


if __name__ == "__main__":
    main()
