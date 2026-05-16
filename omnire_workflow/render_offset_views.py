"""Render OmniRe novel views from an offset camera trajectory.

This script is intentionally kept outside the core DriveStudio modules. It loads an
existing checkpoint, builds the standard novel trajectory, applies user-controlled
camera-space offsets/rotations, and renders a video from the reconstructed scene.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.driving_dataset import DrivingDataset  # noqa: E402
from models.video_utils import render_novel_views  # noqa: E402
from utils.misc import import_str  # noqa: E402


def _axis_angle_matrix(axis: Sequence[float], angle_degrees: float, device, dtype) -> torch.Tensor:
    angle = torch.tensor(np.deg2rad(angle_degrees), device=device, dtype=dtype)
    axis_t = torch.tensor(axis, device=device, dtype=dtype)
    axis_t = axis_t / torch.clamp(torch.linalg.norm(axis_t), min=1e-8)
    x, y, z = axis_t
    zero = torch.zeros((), device=device, dtype=dtype)
    k = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    eye = torch.eye(3, device=device, dtype=dtype)
    return eye + torch.sin(angle) * k + (1.0 - torch.cos(angle)) * (k @ k)


def _local_delta_rotation(yaw: float, pitch: float, roll: float, device, dtype) -> torch.Tensor:
    delta = torch.eye(3, device=device, dtype=dtype)
    if yaw:
        # Positive yaw turns toward camera-local left/right around physical up.
        delta = delta @ _axis_angle_matrix([0.0, -1.0, 0.0], yaw, device, dtype)
    if pitch:
        delta = delta @ _axis_angle_matrix([1.0, 0.0, 0.0], pitch, device, dtype)
    if roll:
        delta = delta @ _axis_angle_matrix([0.0, 0.0, 1.0], roll, device, dtype)
    return delta


def apply_camera_offset(
    poses: torch.Tensor,
    right_m: float,
    up_m: float,
    forward_m: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> torch.Tensor:
    """Apply camera-local translation and rotation offsets to c2w poses."""
    poses = poses.clone()
    rotations = poses[:, :3, :3]
    delta_rot = _local_delta_rotation(yaw_deg, pitch_deg, roll_deg, poses.device, poses.dtype)
    rotations = rotations @ delta_rot
    poses[:, :3, :3] = rotations

    # Camera convention in DriveStudio rays: +x is image right, +y is image down,
    # +z is forward. For user controls, "up" is physical up, i.e. camera-local -y.
    world_offset = (
        right_m * rotations[:, :, 0]
        - up_m * rotations[:, :, 1]
        + forward_m * rotations[:, :, 2]
    )
    poses[:, :3, 3] = poses[:, :3, 3] + world_offset
    return poses


def load_checkpoint_scene(resume_from: Path, cli_opts: Sequence[str]):
    log_dir = resume_from.parent
    cfg_path = log_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config beside checkpoint: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(list(cli_opts)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DrivingDataset(data_cfg=cfg.data)
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
    trainer.resume_from_checkpoint(ckpt_path=str(resume_from), load_only_model=True)
    trainer.set_eval()
    return cfg, dataset, trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a reconstructed DriveStudio/OmniRe scene from an offset trajectory."
    )
    parser.add_argument("--resume_from", required=True, type=Path, help="Checkpoint path, e.g. checkpoint_final.pth")
    parser.add_argument("--base_traj", default="front_center_interp", help="Existing trajectory type to offset")
    parser.add_argument("--frames", type=int, default=None, help="Number of rendered frames; default uses dataset length")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--right_m", type=float, default=0.0, help="Camera-local right offset in meters")
    parser.add_argument("--up_m", type=float, default=0.0, help="Physical upward offset in meters")
    parser.add_argument("--forward_m", type=float, default=0.0, help="Camera-local forward offset in meters")
    parser.add_argument("--yaw_deg", type=float, default=0.0, help="Yaw offset in degrees")
    parser.add_argument("--pitch_deg", type=float, default=0.0, help="Pitch offset in degrees")
    parser.add_argument("--roll_deg", type=float, default=0.0, help="Roll offset in degrees")
    parser.add_argument("--output", type=Path, default=None, help="Output mp4 path")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Optional OmegaConf overrides, e.g. data.preload_device=cuda",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume_from = args.resume_from.resolve()
    cfg, dataset, trainer = load_checkpoint_scene(resume_from, args.opts)

    target_frames = args.frames or dataset.frame_num
    base = dataset.get_novel_render_traj(
        traj_types=[args.base_traj],
        target_frames=target_frames,
    )[args.base_traj]
    offset = apply_camera_offset(
        base,
        right_m=args.right_m,
        up_m=args.up_m,
        forward_m=args.forward_m,
        yaw_deg=args.yaw_deg,
        pitch_deg=args.pitch_deg,
        roll_deg=args.roll_deg,
    )

    if args.output is None:
        video_dir = resume_from.parent / "videos_offset"
        video_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"{args.base_traj}_r{args.right_m:g}_u{args.up_m:g}_f{args.forward_m:g}"
            f"_yaw{args.yaw_deg:g}_pitch{args.pitch_deg:g}_roll{args.roll_deg:g}"
        )
        output = video_dir / f"{stem}.mp4"
    else:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

    traj_path = output.with_suffix(".npy")
    np.save(traj_path, offset.detach().cpu().numpy())

    render_data = dataset.prepare_novel_view_render_data(offset)
    render_novel_views(trainer, render_data, str(output), fps=args.fps)
    print(f"Saved offset trajectory to {traj_path}")
    print(f"Saved offset-view video to {output}")


if __name__ == "__main__":
    main()
