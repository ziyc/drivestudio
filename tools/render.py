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

from datasets.driving_dataset import DrivingDataset
from tools.visualize_trajectories import plot_trajectory_3d, trajectory_key_poses
from utils.camera import list_registered_trajectories
from utils.config import merge_optional_config
from utils.misc import import_str
from utils.output_paths import build_auto_output_dir, count_cameras


def short_camera_tag(cam_name: str) -> str:
    name = cam_name.lower()
    for prefix in ["cam_", "camera_"]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.replace("_", "-")


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
) -> None:
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
            rgb_uint8 = maybe_pad_frame(apply_render_degradation(rgb, degradation_cfg))
            if writer is None:
                writer = imageio.get_writer(save_path, mode="I", fps=fps, macro_block_size=None)
            writer.append_data(rgb_uint8)
    if writer is not None:
        writer.close()
    print(f"Video saved to {save_path}")


def save_gt_video(
    dataset,
    render_data: list,
    save_path: str,
    fps: int = 30,
    cam_id: int = 0,
) -> None:
    camera = dataset.pixel_source.camera_data[cam_id]
    writer = None
    for frame_data in render_data:
        source_frame_idx = int(frame_data["image_infos"]["frame_idx"][0, 0].item())
        gt = camera.images[source_frame_idx].detach().cpu().numpy().clip(0.0, 1.0)
        gt_uint8 = maybe_pad_frame((gt * 255).astype(np.uint8))
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
    reference_poses: np.ndarray | None,
) -> dict[str, dict]:
    traj_kwargs = {}
    for traj_type in traj_types:
        if traj_type in {"lane_offset_right", "lane_offset_left"}:
            kwargs = {"base_camera_id": base_camera_id}
            if lane_offset is not None:
                kwargs["lane_offset_meters"] = lane_offset
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


def load_cfg_and_dataset(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = merge_optional_config(cfg, args.config_file)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    dataset = DrivingDataset(data_cfg=cfg.data)
    return cfg, dataset


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

    cfg, dataset = load_cfg_and_dataset(args)
    trainer = build_trainer(cfg, dataset)
    trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)

    render_novel_cfg = cfg.render.get("render_novel", OmegaConf.create())
    degradation_cfg = cfg.render.get("degradation", None)
    traj_types = args.traj_types if args.traj_types is not None else render_novel_cfg.get(
        "traj_types", list_registered_trajectories()
    )
    frames = args.frames if args.frames is not None else render_novel_cfg.get("frames", dataset.frame_num)
    fps = args.fps if args.fps is not None else render_novel_cfg.get("fps", cfg.render.fps)
    effective_lane_offset = args.lane_offset if args.lane_offset is not None else render_novel_cfg.get(
        "lane_offset", None
    )

    base_camera_id = resolve_base_camera_id(dataset, args)
    base_camera_name = dataset.pixel_source.camera_data[base_camera_id].cam_name
    base_camera_tag = short_camera_tag(base_camera_name)
    traj_name_tag = "+".join(traj_types)
    if effective_lane_offset is not None:
        traj_name_tag = f"{traj_name_tag}@{effective_lane_offset:g}"

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = build_auto_output_dir(
            args.output_root,
            "render",
            cfg.data.dataset,
            f"s{cfg.data.scene_idx}",
            f"cam{count_cameras(cfg.data.pixel_source.cameras)}",
            f"step{trainer.step}",
            base_camera_tag,
            traj_name_tag,
        )
    os.makedirs(output_dir, exist_ok=True)

    reference_poses = get_reference_poses(dataset, base_camera_id)
    traj_kwargs = build_traj_kwargs(
        traj_types,
        base_camera_id,
        effective_lane_offset,
        reference_poses,
    )

    render_traj = dataset.get_novel_render_traj(
        traj_types=traj_types,
        target_frames=frames,
        traj_kwargs=traj_kwargs,
    )
    per_cam_poses = {
        cam_id: camera.cam_to_worlds for cam_id, camera in dataset.pixel_source.camera_data.items()
    }
    camera_poses_np = {
        cam_id: poses.detach().cpu().numpy() for cam_id, poses in per_cam_poses.items()
    }
    plot_reference_poses = reference_poses
    if plot_reference_poses is None:
        plot_reference_poses = camera_poses_np[base_camera_id]

    for traj_type, traj in render_traj.items():
        render_data = dataset.prepare_novel_view_render_data(traj, cam_id=base_camera_id)

        render_novel_video(
            trainer,
            render_data,
            os.path.join(output_dir, f"{traj_type}.mp4"),
            fps=fps,
            degradation_cfg=degradation_cfg,
        )
        save_gt_video(
            dataset,
            render_data,
            os.path.join(output_dir, f"{traj_type}_gt.mp4"),
            fps=fps,
            cam_id=base_camera_id,
        )

        traj_np = traj.detach().cpu().numpy()
        key_pose_entries = trajectory_key_poses(traj_type, per_cam_poses, dataset.frame_num)
        plot_trajectory_3d(
            ego_poses=plot_reference_poses,
            camera_poses=camera_poses_np,
            base_camera_id=base_camera_id,
            trajectory=traj_np,
            key_pose_entries=key_pose_entries,
            title=f"{cfg.data.dataset} scene {dataset.scene_idx} - {traj_type}",
            output_path=os.path.join(output_dir, f"{traj_type}_traj.png"),
        )


if __name__ == "__main__":
    main()
