"""Render registered novel trajectories and save videos plus trajectory plots."""

import argparse
import os
import sys
from io import BytesIO

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageFilter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.dataset_meta import DATASETS_CONFIG
from datasets.driving_dataset import DrivingDataset
from tools.visualize_trajectories import (
    get_total_frames,
    load_camera_poses,
    load_reference_poses,
    plot_trajectory_3d,
    resolve_scene_dir,
    trajectory_key_poses,
)
from utils.camera import get_interp_novel_trajectories, list_registered_trajectories
from utils.misc import import_str
from utils.output_paths import build_auto_output_dir, count_cameras


def parse_args():
    parser = argparse.ArgumentParser("Render registered novel trajectories")
    parser.add_argument("--resume_from", type=str, required=False, help="path to checkpoint")
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="optional config overrides to merge on top of the run config",
    )
    parser.add_argument(
        "--traj_types",
        type=str,
        nargs="+",
        default=None,
        help="registered trajectory names",
    )
    parser.add_argument("--frames", type=int, default=None, help="number of frames to render")
    parser.add_argument("--fps", type=int, default=None, help="output video fps")
    parser.add_argument(
        "--output_root",
        type=str,
        default="./outputs",
        help="root directory for command outputs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory for videos and trajectory files",
    )
    parser.add_argument(
        "--lane_offset",
        type=float,
        default=None,
        help="override lateral offset in meters for lane_offset_left/right",
    )
    parser.add_argument(
        "--lane_offset_ratio",
        type=float,
        default=None,
        help="override lateral offset as a fraction of base-camera trajectory length",
    )
    parser.add_argument(
        "--base_camera_id",
        type=int,
        default=None,
        help="base camera id for novel rendering; defaults to the repo reference camera",
    )
    parser.add_argument(
        "--list_traj_types",
        action="store_true",
        help="print registered trajectory names and exit",
    )
    parser.add_argument("opts", help="OmegaConf overrides", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def short_camera_tag(cam_name: str) -> str:
    name = cam_name.lower()
    for prefix in ["cam_", "camera_"]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.replace("_", "-")


def maybe_pad_frame(rgb: np.ndarray, macro_block_size: int = 16) -> np.ndarray:
    h, w = rgb.shape[:2]
    pad_h = (macro_block_size - h % macro_block_size) % macro_block_size
    pad_w = (macro_block_size - w % macro_block_size) % macro_block_size
    if pad_h == 0 and pad_w == 0:
        return rgb
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    rgb_t = F.pad(rgb_t, (0, pad_w, 0, pad_h), mode="replicate")
    return rgb_t.squeeze(0).permute(1, 2, 0).numpy()


def _to_uint8(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(rgb, 0.0, 1.0)
        return (rgb * 255).astype(np.uint8)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def maybe_resize_frame(rgb: np.ndarray, output_size, output_size_mode: str = "stretch") -> np.ndarray:
    if output_size is None:
        return rgb
    width, height = int(output_size[0]), int(output_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid render.output_size: {output_size}")
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb

    image = Image.fromarray(rgb)
    if output_size_mode == "stretch":
        return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))
    if output_size_mode != "cover_crop":
        raise ValueError(
            f"Unsupported render.output_size_mode: {output_size_mode}. "
            "Expected one of ['stretch', 'cover_crop']"
        )

    scale = max(width / image.width, height / image.height)
    resized_w = max(width, int(round(image.width * scale)))
    resized_h = max(height, int(round(image.height * scale)))
    image = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    left = max(0, (resized_w - width) // 2)
    top = max(0, (resized_h - height) // 2)
    image = image.crop((left, top, left + width, top + height))
    return np.asarray(image)


def apply_render_degradation(rgb: np.ndarray, degradation_cfg) -> np.ndarray:
    rgb_uint8 = _to_uint8(rgb)
    if degradation_cfg is None:
        return rgb_uint8

    resize_scale = float(degradation_cfg.get("resize_scale", 1.0))
    blur_kernel_size = int(degradation_cfg.get("blur_kernel_size", 0) or 0)
    gaussian_noise_std = float(degradation_cfg.get("gaussian_noise_std", 0.0) or 0.0)
    jpeg_quality = degradation_cfg.get("jpeg_quality", None)

    image = Image.fromarray(rgb_uint8)
    if resize_scale != 1.0:
        new_width = max(1, int(round(image.width * resize_scale)))
        new_height = max(1, int(round(image.height * resize_scale)))
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    if blur_kernel_size > 1:
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_kernel_size / 2.0))

    rgb_uint8 = np.asarray(image)
    if gaussian_noise_std > 0:
        noise = np.random.normal(0.0, gaussian_noise_std * 255.0, rgb_uint8.shape)
        rgb_uint8 = np.clip(rgb_uint8.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    if jpeg_quality is not None:
        buffer = BytesIO()
        Image.fromarray(rgb_uint8).save(buffer, format="JPEG", quality=int(jpeg_quality))
        buffer.seek(0)
        rgb_uint8 = np.asarray(Image.open(buffer).convert("RGB"))

    return rgb_uint8


def render_novel_video(
    trainer,
    render_data: list,
    save_path: str,
    fps: int = 30,
    degradation_cfg=None,
    output_size=None,
    output_size_mode: str = "stretch",
    clean_save_path: str | None = None,
) -> None:
    trainer.set_eval()
    writer = None
    clean_writer = None
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
            clean_rgb_uint8 = _to_uint8(rgb)
            clean_rgb_uint8 = maybe_resize_frame(clean_rgb_uint8, output_size, output_size_mode)
            clean_rgb_uint8 = maybe_pad_frame(clean_rgb_uint8)
            if clean_save_path is not None:
                if clean_writer is None:
                    clean_writer = imageio.get_writer(clean_save_path, mode="I", fps=fps, macro_block_size=None)
                clean_writer.append_data(clean_rgb_uint8)

            rgb_uint8 = apply_render_degradation(rgb, degradation_cfg)
            rgb_uint8 = maybe_resize_frame(rgb_uint8, output_size, output_size_mode)
            rgb_uint8 = maybe_pad_frame(rgb_uint8)
            if writer is None:
                writer = imageio.get_writer(save_path, mode="I", fps=fps, macro_block_size=None)
            writer.append_data(rgb_uint8)
    if writer is not None:
        writer.close()
    if clean_writer is not None:
        clean_writer.close()
        print(f"Clean render video saved to {clean_save_path}")
    print(f"Video saved to {save_path}")


def save_gt_video(
    dataset,
    render_data: list,
    save_path: str,
    fps: int = 30,
    cam_id: int = 0,
    output_size=None,
    output_size_mode: str = "stretch",
) -> None:
    camera = dataset.pixel_source.camera_data[cam_id]
    writer = None
    for frame_data in render_data:
        source_frame_idx = int(frame_data["image_infos"]["frame_idx"][0, 0].item())
        gt = camera.images[source_frame_idx].detach().cpu().numpy().clip(0.0, 1.0)
        gt_uint8 = (gt * 255).astype(np.uint8)
        gt_uint8 = maybe_resize_frame(gt_uint8, output_size, output_size_mode)
        gt_uint8 = maybe_pad_frame(gt_uint8)
        if writer is None:
            writer = imageio.get_writer(save_path, mode="I", fps=fps, macro_block_size=None)
        writer.append_data(gt_uint8)
    if writer is not None:
        writer.close()
    print(f"GT video saved to {save_path}")


def get_reference_cam_id(dataset) -> int:
    return 1 if dataset.type == "argoverse" else 0


def resolve_base_camera_id(dataset, args) -> int:
    if args.base_camera_id is not None:
        return args.base_camera_id
    return get_reference_cam_id(dataset)


def build_traj_kwargs(
    traj_types: list[str],
    base_camera_id: int,
    lane_offset: float | None,
    lane_offset_ratio: float | None,
    reference_poses: np.ndarray | None,
) -> dict[str, dict]:
    traj_kwargs = {}
    for traj_type in traj_types:
        if traj_type in {"lane_offset_right", "lane_offset_left"}:
            kwargs = {"base_camera_id": base_camera_id}
            if lane_offset is not None:
                kwargs["lane_offset_meters"] = lane_offset
            if lane_offset_ratio is not None:
                kwargs["lane_offset_ratio"] = lane_offset_ratio
            if reference_poses is not None:
                kwargs["ego_poses"] = torch.from_numpy(reference_poses).float()
            traj_kwargs[traj_type] = kwargs
    return traj_kwargs


def get_reference_poses(dataset, base_camera_id: int) -> np.ndarray | None:
    pose_dir = os.path.join(dataset.data_path, "ego_pose")
    if not os.path.isdir(pose_dir):
        pose_dir = os.path.join(dataset.data_path, "lidar_pose")
    if not os.path.isdir(pose_dir):
        return None

    pose_files = sorted([name for name in os.listdir(pose_dir) if name.endswith(".txt")])
    if not pose_files:
        return None

    pose_name = os.path.basename(pose_dir)
    if pose_name == "lidar_pose" and dataset.type == "nuscenes":
        frame_stem = os.path.splitext(pose_files[0])[0]
        camera_start = np.loadtxt(
            os.path.join(dataset.data_path, "extrinsics", f"{frame_stem}_{base_camera_id}.txt")
        )
        world_to_start = np.linalg.inv(camera_start)
    else:
        start_pose = np.loadtxt(os.path.join(pose_dir, pose_files[0]))
        world_to_start = np.linalg.inv(start_pose)

    poses = []
    for pose_file in pose_files:
        pose = np.loadtxt(os.path.join(pose_dir, pose_file))
        poses.append(world_to_start @ pose)
    return np.stack(poses, axis=0)


def load_cfg(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    if args.config_file:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_file))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    return cfg


def get_checkpoint_step(ckpt_path: str) -> int | None:
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = state_dict.get("step") if isinstance(state_dict, dict) else None
    return int(step) if step is not None else None


def build_trainer(cfg, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )


def main():
    args = parse_args()

    if args.list_traj_types:
        for name in list_registered_trajectories():
            print(name)
        return

    if args.resume_from is None:
        raise ValueError("--resume_from is required unless --list_traj_types is used")

    if not os.path.isfile(args.resume_from):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")

    cfg = load_cfg(args)

    render_novel_cfg = cfg.render.get("render_novel", OmegaConf.create())
    traj_types = args.traj_types if args.traj_types is not None else render_novel_cfg.get(
        "traj_types", list_registered_trajectories()
    )
    registered = set(list_registered_trajectories())
    unknown = [t for t in traj_types if t not in registered]
    if unknown:
        raise ValueError(
            f"Unknown trajectory type(s): {unknown}. "
            f"Registered trajectories: {sorted(registered)}"
        )

    degradation_cfg = cfg.render.get("degradation", None)
    output_size = cfg.render.get("output_size", None)
    output_size_mode = cfg.render.get("output_size_mode", "stretch")
    scene_dir = resolve_scene_dir(cfg.data)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")
    total_frames = get_total_frames(scene_dir)
    start_timestep = int(cfg.data.start_timestep)
    end_timestep = total_frames if int(cfg.data.end_timestep) == -1 else int(cfg.data.end_timestep) + 1
    per_cam_poses = load_camera_poses(cfg, scene_dir, start_timestep, end_timestep)
    reference_poses = load_reference_poses(cfg, scene_dir, start_timestep, end_timestep)

    original_frames = per_cam_poses[int(cfg.data.pixel_source.cameras[0])].shape[0]
    frames = args.frames if args.frames is not None else render_novel_cfg.get("frames", original_frames)
    fps = args.fps if args.fps is not None else render_novel_cfg.get("fps", cfg.render.fps)
    effective_lane_offset = args.lane_offset if args.lane_offset is not None else render_novel_cfg.get(
        "lane_offset", None
    )
    effective_lane_offset_ratio = (
        args.lane_offset_ratio
        if args.lane_offset_ratio is not None
        else render_novel_cfg.get("lane_offset_ratio", None)
    )

    base_camera_id = args.base_camera_id if args.base_camera_id is not None else 0
    if cfg.data.dataset == "argoverse" and args.base_camera_id is None:
        base_camera_id = 1
    base_camera_name = DATASETS_CONFIG[cfg.data.dataset][base_camera_id]["camera_name"]
    base_camera_tag = short_camera_tag(base_camera_name)
    traj_name_tag = "+".join(traj_types)
    if effective_lane_offset_ratio is not None:
        traj_name_tag = f"{traj_name_tag}@ratio{effective_lane_offset_ratio:g}"
    elif effective_lane_offset is not None:
        traj_name_tag = f"{traj_name_tag}@{effective_lane_offset:g}"

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        checkpoint_step = get_checkpoint_step(args.resume_from)
        step_tag = f"step{checkpoint_step}" if checkpoint_step is not None else os.path.splitext(os.path.basename(args.resume_from))[0]
        output_dir = build_auto_output_dir(
            args.output_root,
            "render",
            cfg.data.dataset,
            f"s{cfg.data.scene_idx}",
            f"cam{count_cameras(cfg.data.pixel_source.cameras)}",
            step_tag,
            base_camera_tag,
            traj_name_tag,
        )
    os.makedirs(output_dir, exist_ok=True)

    traj_kwargs = build_traj_kwargs(
        traj_types,
        base_camera_id,
        effective_lane_offset,
        effective_lane_offset_ratio,
        reference_poses,
    )

    render_traj = {
        traj_type: get_interp_novel_trajectories(
            cfg.data.dataset,
            str(cfg.data.scene_idx),
            per_cam_poses,
            traj_type=traj_type,
            target_frames=frames,
            traj_kwargs=traj_kwargs.get(traj_type),
        )
        for traj_type in traj_types
    }
    camera_poses_np = {
        cam_id: poses.detach().cpu().numpy() for cam_id, poses in per_cam_poses.items()
    }
    plot_reference_poses = reference_poses
    if plot_reference_poses is None:
        plot_reference_poses = camera_poses_np[base_camera_id]

    for traj_type, traj in render_traj.items():
        traj_np = traj.detach().cpu().numpy()
        key_pose_entries = trajectory_key_poses(traj_type, per_cam_poses, original_frames)

        plot_trajectory_3d(
            ego_poses=plot_reference_poses,
            camera_poses=camera_poses_np,
            base_camera_id=base_camera_id,
            trajectory=traj_np,
            key_pose_entries=key_pose_entries,
            title=f"{cfg.data.dataset} scene {cfg.data.scene_idx} - {traj_type}",
            output_path=os.path.join(output_dir, f"{traj_type}_traj.png"),
        )

    dataset = DrivingDataset(data_cfg=cfg.data)

    trainer = build_trainer(cfg, dataset)
    trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)

    for traj_type, traj in render_traj.items():
        render_data = dataset.prepare_novel_view_render_data(traj, cam_id=base_camera_id)

        render_novel_video(
            trainer,
            render_data,
            os.path.join(output_dir, f"{traj_type}.mp4"),
            fps=fps,
            degradation_cfg=degradation_cfg,
            output_size=output_size,
            output_size_mode=output_size_mode,
            clean_save_path=os.path.join(output_dir, f"{traj_type}_render.mp4"),
        )
        save_gt_video(
            dataset,
            render_data,
            os.path.join(output_dir, f"{traj_type}_gt.mp4"),
            fps=fps,
            cam_id=base_camera_id,
            output_size=output_size,
            output_size_mode=output_size_mode,
        )
if __name__ == "__main__":
    main()
