"""Build the canonical OmniRe Waymo dataset layout for Cosmos post-training.

Output layout::

    {output_root}/
        videos/           GT videos (shared across modes)
        render/           clean 3DGS renders
        light/            Gaussian drop 0.10
        medium/           Gaussian drop 0.25
        heavy/            Gaussian drop 0.40
"""

import argparse
import os
import sys
from types import MethodType

import torch
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets.dataset_meta import DATASETS_CONFIG
from datasets.driving_dataset import DrivingDataset
from models.gaussians.basics import dataclass_gs
from tools.render import (
    _to_uint8,
    apply_render_degradation,
    build_trainer,
    build_traj_kwargs,
    maybe_pad_frame,
    maybe_resize_frame,
    short_camera_tag,
)
from tools.visualize_trajectories import get_total_frames, load_camera_poses, load_reference_poses, resolve_scene_dir
from utils.camera import get_interp_novel_trajectories, list_registered_trajectories

# ---------------------------------------------------------------------------
# helpers reused / adapted from the existing data‑builder
# ---------------------------------------------------------------------------

MODE_DEFAULTS = {
    "render": {"gaussian_drop_ratio": 0.0, "checkpoint": "checkpoint_final.pth"},
    "light": {"gaussian_drop_ratio": 0.10, "checkpoint": "checkpoint_20000.pth"},
    "medium": {"gaussian_drop_ratio": 0.25, "checkpoint": "checkpoint_10000.pth"},
    "heavy": {"gaussian_drop_ratio": 0.40, "checkpoint": "checkpoint_5000.pth"},
}


def gaussian_keep_mask(num_gaussians: int, device, drop_ratio: float, seed: int) -> torch.Tensor:
    if drop_ratio <= 0.0:
        return torch.ones(num_gaussians, dtype=torch.bool, device=device)
    if drop_ratio >= 1.0:
        raise ValueError(f"gaussian_drop_ratio must be < 1.0, got {drop_ratio}")
    if num_gaussians <= 1:
        return torch.ones(num_gaussians, dtype=torch.bool, device=device)
    keep_count = max(1, int(round(num_gaussians * (1.0 - drop_ratio))))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    scores = torch.rand(num_gaussians, device=device, generator=generator)
    keep_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
    keep_mask[torch.topk(scores, k=keep_count, largest=True, sorted=False).indices] = True
    return keep_mask


def _subset_extras(extras, keep_mask: torch.Tensor):
    if extras is None:
        return None
    subset = {}
    for key, value in extras.items():
        if torch.is_tensor(value) and value.shape[:1] == keep_mask.shape:
            subset[key] = value[keep_mask]
        else:
            subset[key] = value
    return subset


def drop_gaussians(gs: dataclass_gs, keep_mask: torch.Tensor) -> dataclass_gs:
    if keep_mask.all():
        return gs
    return dataclass_gs(
        _means=gs._means[keep_mask],
        _scales=gs._scales[keep_mask],
        _quats=gs._quats[keep_mask],
        _rgbs=gs._rgbs[keep_mask],
        _opacities=gs._opacities[keep_mask],
        detach_keys=list(gs.detach_keys),
        extras=_subset_extras(gs.extras, keep_mask),
    )


def patch_trainer_collect_gaussians(trainer, drop_ratio: float, seed: int):
    if drop_ratio <= 0.0:
        return
    original_collect = trainer.collect_gaussians

    def collect_gaussians_with_drop(self, cam, image_ids):
        gs = original_collect(cam, image_ids)
        keep_mask = gaussian_keep_mask(gs._means.shape[0], gs._means.device, drop_ratio=drop_ratio, seed=seed)
        self.pts_labels = self.pts_labels[keep_mask]
        if getattr(self, "dynamic_pts_mask", None) is not None:
            self.dynamic_pts_mask = self.dynamic_pts_mask[keep_mask]
        return drop_gaussians(gs, keep_mask=keep_mask)

    trainer.collect_gaussians = MethodType(collect_gaussians_with_drop, trainer)


def remap_render_data_to_full_scene_indices(render_data: list, dataset, cam_id: int, frame_indices: list[int]):
    camera = dataset.pixel_source.camera_data[cam_id]
    normalized_time = dataset.pixel_source.normalized_time
    for frame_data, source_frame_idx in zip(render_data, frame_indices):
        image_infos = frame_data["image_infos"]
        H, W = image_infos["img_idx"].shape
        device = image_infos["img_idx"].device
        img_idx = int(camera.unique_img_idx[source_frame_idx].item())
        image_infos["img_idx"] = torch.full((H, W), img_idx, dtype=torch.long, device=device)
        image_infos["frame_idx"] = torch.full((H, W), source_frame_idx, dtype=torch.long, device=device)
        image_infos["normed_time"] = torch.full(
            (H, W), float(normalized_time[source_frame_idx].item()), dtype=torch.float32, device=device,
        )


def frame_to_output(rgb_uint8, output_size, output_size_mode: str):
    rgb_uint8 = maybe_resize_frame(rgb_uint8, output_size, output_size_mode)
    return maybe_pad_frame(rgb_uint8)


# ---------------------------------------------------------------------------
# Per‑sample chunk rendering
# ---------------------------------------------------------------------------

def render_sample(
    trainer,
    dataset,
    base_camera_id: int,
    per_cam_poses,
    full_frame_indices,
    traj_kwargs: dict,
    traj_type: str,
    full_frames: int,
    fps: int,
    chunk_start: int,
    chunk_end: int,
    output_size,
    output_size_mode: str,
    degradation_cfg,
):
    """Render one chunk [chunk_start, chunk_end) and return vis + gt as lists of uint8 frames."""
    trainer.set_eval()
    camera = dataset.pixel_source.camera_data[base_camera_id]
    traj_device = dataset.pixel_source.device

    traj = get_interp_novel_trajectories(
        dataset.type, str(dataset.scene_idx),
        per_cam_poses,
        traj_type=traj_type,
        target_frames=full_frames,
        traj_kwargs=traj_kwargs.get(traj_type),
    )
    render_data = dataset.prepare_novel_view_render_data(traj.to(traj_device), cam_id=base_camera_id)
    remap_render_data_to_full_scene_indices(render_data, dataset, base_camera_id, full_frame_indices)

    vis_frames = []
    gt_frames = []
    with torch.no_grad():
        for frame_idx in range(chunk_start, chunk_end):
            frame_data = render_data[frame_idx]
            for info_key, value in frame_data["cam_infos"].items():
                frame_data["cam_infos"][info_key] = value.cuda(non_blocking=True)
            for info_key, value in frame_data["image_infos"].items():
                frame_data["image_infos"][info_key] = value.cuda(non_blocking=True)

            outputs = trainer(
                image_infos=frame_data["image_infos"],
                camera_infos=frame_data["cam_infos"],
                novel_view=True,
            )
            rgb = outputs["rgb"].detach().cpu().numpy().clip(min=1.0e-6, max=1 - 1.0e-6)

            vis_uint8 = frame_to_output(apply_render_degradation(rgb, degradation_cfg), output_size, output_size_mode)

            source_frame_idx = int(frame_data["image_infos"]["frame_idx"][0, 0].item())
            gt = camera.images[source_frame_idx].detach().cpu().numpy().clip(0.0, 1.0)
            gt_uint8 = frame_to_output(_to_uint8(gt), output_size, output_size_mode)

            vis_frames.append(vis_uint8)
            gt_frames.append(gt_uint8)
    return vis_frames, gt_frames


def write_mp4(frames: list, path: str, fps: int):
    import imageio.v2 as imageio
    writer = imageio.get_writer(path, mode="I", fps=fps, macro_block_size=None)
    for f in frames:
        writer.append_data(f)
    writer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("Build OmniRe canonical Waymo dataset for Cosmos")
    parser.add_argument(
        "--scene_ids", type=str, nargs="+", required=True,
        help="scene IDs to process, e.g. 001 002 003 ...",
    )
    parser.add_argument(
        "--checkpoint_root", type=str, required=True,
        help="template path with {scene} placeholder, "
             "e.g. outputs/train/waymo/scene{scene}/5cams_step30000_nosmpl",
    )
    parser.add_argument(
        "--output_root", type=str, default="data/omnire/waymo",
        help="canonical dataset root",
    )
    parser.add_argument(
        "--config_file", type=str, default="configs/render/ego_raw.yaml",
        help="render config overlays",
    )
    parser.add_argument(
        "--base_camera_ids", type=int, nargs="+", default=None,
        help="camera ids; defaults to all cameras from cfg",
    )
    parser.add_argument("--modes", type=str, nargs="+", default=["render", "light", "medium", "heavy"])
    parser.add_argument("--traj_types", type=str, nargs="+", default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--chunk_length", type=int, default=93)
    parser.add_argument("--chunk_stride", type=int, default=None)
    parser.add_argument("--gaussian_drop_seed", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print plan without loading models or rendering")
    parser.add_argument("opts", help="OmegaConf overrides", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    registered = set(list_registered_trajectories())

    output_root = args.output_root
    videos_root = os.path.join(output_root, "videos")
    for d in [output_root, videos_root]:
        os.makedirs(d, exist_ok=True)
    for mode in args.modes:
        os.makedirs(os.path.join(output_root, mode), exist_ok=True)

    # --- resolve scene info up‑front ---
    base_cfg = OmegaConf.load(args.config_file) if args.config_file else OmegaConf.create()
    if args.traj_types:
        base_cfg.render.render_novel.traj_types = args.traj_types
    traj_types = list(base_cfg.render.render_novel.get("traj_types", ["ego_raw"]))
    unknown = [t for t in traj_types if t not in registered]
    if unknown:
        raise ValueError(f"Unknown trajectory type(s): {unknown}")

    output_size = base_cfg.render.get("output_size", None)
    output_size_mode = base_cfg.render.get("output_size_mode", "stretch")

    # --- process each scene ---
    for scene_id in args.scene_ids:
        ckpt_root = args.checkpoint_root.format(scene=scene_id)
        print(f"\n=== scene {scene_id} ===")

        # First, load any checkpoint to get scene config / frame info
        first_ckpt = os.path.join(ckpt_root, MODE_DEFAULTS[args.modes[0]]["checkpoint"])
        if not os.path.isfile(first_ckpt):
            print(f"  skipping scene {scene_id}: no checkpoint at {first_ckpt}")
            continue
        cfg = OmegaConf.load(os.path.join(os.path.dirname(first_ckpt), "config.yaml"))
        cfg = OmegaConf.merge(cfg, base_cfg)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))

        scene_dir = resolve_scene_dir(cfg.data)
        total_frames = get_total_frames(scene_dir)
        start_timestep = int(cfg.data.start_timestep)
        end_timestep = total_frames if int(cfg.data.end_timestep) == -1 else int(cfg.data.end_timestep) + 1
        base_frames = end_timestep - start_timestep

        chunk_length = args.chunk_length
        chunk_stride = args.chunk_stride if args.chunk_stride is not None else chunk_length
        chunk_starts = list(range(0, base_frames - chunk_length + 1, chunk_stride))
        dropped_tail = base_frames - ((chunk_starts[-1] + chunk_length) if chunk_starts else 0)
        if dropped_tail > 0:
            print(f"  dropping tail frames: {dropped_tail}")

        camera_ids = args.base_camera_ids
        if camera_ids is None:
            camera_ids = [int(cam_id) for cam_id in cfg.data.pixel_source.cameras]

        fps = args.fps if args.fps is not None else base_cfg.render.render_novel.get("fps", 10)
        lane_offset = base_cfg.render.render_novel.get("lane_offset", None)
        lane_offset_ratio = base_cfg.render.render_novel.get("lane_offset_ratio", None)

        per_cam_poses = load_camera_poses(cfg, scene_dir, start_timestep, end_timestep)
        reference_poses = load_reference_poses(cfg, scene_dir, start_timestep, end_timestep)
        full_frame_indices = list(range(start_timestep, end_timestep))

        sample_name_prefix = f"{cfg.data.dataset}_scene{int(cfg.data.scene_idx):03d}"

        # --- build sample list ---
        sn_by_chunk = {}  # (cam_id, chunk_idx) -> sample_name
        for base_camera_id in camera_ids:
            base_camera_name = DATASETS_CONFIG[cfg.data.dataset][base_camera_id]["camera_name"]
            camera_tag = short_camera_tag(base_camera_name)
            for chunk_idx, chunk_start in enumerate(chunk_starts):
                sample_name = f"{sample_name_prefix}_{camera_tag}_cam{base_camera_id}_{traj_types[0]}_chunk{chunk_idx:03d}"
                sn_by_chunk[(base_camera_id, chunk_idx)] = sample_name

        if args.dry_run:
            print(f"  total_frames={total_frames} start={start_timestep} end={end_timestep} base_frames={base_frames}")
            print(f"  chunk_length={chunk_length} chunk_stride={chunk_stride} chunks={len(chunk_starts)}")
            print(f"  fps={fps} cameras={camera_ids}")
            print(f"  traj_types={traj_types}")
            print(f"  modes={args.modes}")
            for base_camera_id in camera_ids:
                for chunk_idx, chunk_start in enumerate(chunk_starts):
                    chunk_end = chunk_start + chunk_length
                    sample_name = sn_by_chunk[(base_camera_id, chunk_idx)]
                    gt_path = os.path.join(videos_root, f"{sample_name}.mp4")
                    print(f"  [{sample_name}]")
                    print(f"    source frames {start_timestep + chunk_start}-{start_timestep + chunk_end - 1} ({chunk_length} frames)")
                    print(f"    GT      -> {gt_path}")
                    for mode in args.modes:
                        out_path = os.path.join(output_root, mode, f"{sample_name}.mp4")
                        exists = os.path.isfile(out_path)
                        tag = " (EXISTS)" if exists else ""
                        print(f"    {mode:8s} -> {out_path}{tag}")
            continue

        # GT is shared, only write once. We write GT in the *first* mode processed.
        gt_written = False

        for mode_idx, mode in enumerate(args.modes):
            drop_ratio = MODE_DEFAULTS[mode]["gaussian_drop_ratio"]
            ckpt_path = os.path.join(ckpt_root, MODE_DEFAULTS[mode]["checkpoint"])
            if not os.path.isfile(ckpt_path):
                print(f"  mode {mode}: missing checkpoint {ckpt_path}, skipping")
                continue
            print(f"  mode {mode}: gaussian_drop={drop_ratio} checkpoint={ckpt_path}")

            # load with full-scene config
            dataset = DrivingDataset(data_cfg=cfg.data)
            trainer = build_trainer(cfg, dataset)
            trainer.resume_from_checkpoint(ckpt_path=ckpt_path, load_only_model=True)
            patch_trainer_collect_gaussians(trainer, drop_ratio, args.gaussian_drop_seed)

            mode_dir = os.path.join(output_root, mode)

            for base_camera_id in camera_ids:
                traj_kwargs = build_traj_kwargs(traj_types, base_camera_id, lane_offset, lane_offset_ratio, reference_poses)
                for traj_type in traj_types:
                    for chunk_idx, chunk_start in enumerate(chunk_starts):
                        chunk_end = chunk_start + chunk_length
                        sample_name = sn_by_chunk[(base_camera_id, chunk_idx)]
                        out_path = os.path.join(mode_dir, f"{sample_name}.mp4")
                        gt_path = os.path.join(videos_root, f"{sample_name}.mp4")

                        if args.skip_existing and os.path.isfile(out_path) and (gt_written or os.path.isfile(gt_path)):
                            print(f"    skip {sample_name}")
                            continue

                        vis_frames, gt_frames = render_sample(
                            trainer, dataset, base_camera_id,
                            per_cam_poses, full_frame_indices, traj_kwargs, traj_type,
                            base_frames, fps,
                            chunk_start, chunk_end,
                            output_size, output_size_mode,
                            None,  # degradation_cfg -- structural only
                        )
                        write_mp4(vis_frames, out_path, fps)
                        print(f"    wrote {out_path}")

                        if not gt_written:
                            write_mp4(gt_frames, gt_path, fps)
                            print(f"    wrote GT {gt_path}")

            del dataset
            del trainer
            torch.cuda.empty_cache()
            gt_written = True  # GT was saved during the first mode


if __name__ == "__main__":
    main()
