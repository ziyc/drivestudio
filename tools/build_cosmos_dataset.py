import argparse
import json
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


def parse_args():
    parser = argparse.ArgumentParser("Build a Cosmos vis dataset directly from checkpoints")
    parser.add_argument("--resume_from", type=str, nargs="+", required=True, help="one or more checkpoint paths")
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="optional render config overrides merged on top of each run config",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="target Cosmos dataset root; defaults to outputs/datasets/<dataset>_<mode>",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["render", "light", "medium", "heavy"],
        required=True,
        help="render uses baseline 3DGS render; light/medium/heavy apply structural degradation on top of the chosen checkpoint",
    )
    parser.add_argument(
        "--traj_types",
        type=str,
        nargs="+",
        default=None,
        help="registered trajectory names; defaults to render.render_novel.traj_types",
    )
    parser.add_argument("--frames", type=int, default=None, help="override number of frames to render")
    parser.add_argument("--fps", type=int, default=None, help="override output video fps")
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=None,
        help="number of source frames per exported chunk; defaults to the effective render frame count",
    )
    parser.add_argument(
        "--chunk_stride",
        type=int,
        default=None,
        help="stride between chunk starts; defaults to chunk_length",
    )
    parser.add_argument(
        "--base_camera_ids",
        type=int,
        nargs="+",
        default=None,
        help="camera ids to export; defaults to all cameras from cfg.data.pixel_source.cameras",
    )
    parser.add_argument(
        "--sample_name_template",
        type=str,
        default="{dataset}_scene{scene:03d}_{camera_tag}_cam{camera_id}_{traj_type}_{mode}_chunk{chunk_idx:03d}",
        help="python format template for sample names",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=(
            "A front-view driving video recorded from a moving vehicle in daytime. "
            "The scene contains roads, lane markings, vehicles, roadside objects, and urban surroundings."
        ),
        help="caption text written to each sample json",
    )
    parser.add_argument(
        "--gaussian_drop_ratio",
        type=float,
        default=None,
        help="override mode default fraction of gaussians to drop from the vis render",
    )
    parser.add_argument(
        "--gaussian_drop_seed",
        type=int,
        default=0,
        help="seed used for deterministic gaussian dropping in non-render modes",
    )
    parser.add_argument("--skip_existing", action="store_true", help="skip samples whose videos and caption already exist")
    parser.add_argument("opts", help="OmegaConf overrides", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def load_cfg(resume_from: str, config_file: str | None, opts: list[str]):
    log_dir = os.path.dirname(resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    if config_file:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_file))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(opts))
    return cfg


def ensure_dataset_dirs(dataset_dir: str):
    videos_dir = os.path.join(dataset_dir, "videos")
    vis_dir = os.path.join(dataset_dir, "vis")
    captions_dir = os.path.join(dataset_dir, "captions")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(captions_dir, exist_ok=True)
    return videos_dir, vis_dir, captions_dir


def resolve_dataset_dir(dataset_dir: str | None, cfg, mode: str) -> str:
    if dataset_dir is not None:
        return dataset_dir
    dataset_tag = str(cfg.data.dataset).replace("/", "_")
    return os.path.join("outputs", "datasets", f"{dataset_tag}_{mode}")


def write_caption(path: str, caption: str):
    with open(path, "w", encoding="ascii") as f:
        json.dump({"caption": caption}, f, ensure_ascii=True, indent=2)
        f.write("\n")


def resolve_gaussian_drop_ratio(mode: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    return {
        "render": 0.0,
        "light": 0.10,
        "medium": 0.25,
        "heavy": 0.40,
    }[mode]


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
    keep_indices = torch.topk(scores, k=keep_count, largest=True, sorted=False).indices
    keep_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
    keep_mask[keep_indices] = True
    return keep_mask


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
            (H, W),
            float(normalized_time[source_frame_idx].item()),
            dtype=torch.float32,
            device=device,
        )


def frame_to_output(rgb_uint8, output_size, output_size_mode: str):
    rgb_uint8 = maybe_resize_frame(rgb_uint8, output_size, output_size_mode)
    return maybe_pad_frame(rgb_uint8)


def write_chunked_dataset_videos(
    trainer,
    dataset,
    render_data: list,
    chunk_specs: list[tuple[int, int, str, str, str, str]],
    fps: int,
    cam_id: int,
    degradation_cfg,
    output_size,
    output_size_mode: str,
):
    trainer.set_eval()
    camera = dataset.pixel_source.camera_data[cam_id]
    vis_writers = {}
    gt_writers = {}

    def active_chunk(frame_idx: int):
        for chunk_start, chunk_end, sample_name, gt_path, vis_path, caption_path in chunk_specs:
            if chunk_start <= frame_idx < chunk_end:
                return chunk_start, chunk_end, sample_name, gt_path, vis_path, caption_path
        return None

    try:
        with torch.no_grad():
            for frame_idx, frame_data in enumerate(render_data):
                chunk = active_chunk(frame_idx)
                if chunk is None:
                    continue
                chunk_start, chunk_end, sample_name, gt_path, vis_path, caption_path = chunk
                key = (chunk_start, chunk_end, sample_name, gt_path, vis_path, caption_path)

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
                vis_uint8 = frame_to_output(
                    apply_render_degradation(rgb, degradation_cfg),
                    output_size,
                    output_size_mode,
                )

                source_frame_idx = int(frame_data["image_infos"]["frame_idx"][0, 0].item())
                gt = camera.images[source_frame_idx].detach().cpu().numpy().clip(0.0, 1.0)
                gt_uint8 = frame_to_output(_to_uint8(gt), output_size, output_size_mode)

                if key not in vis_writers:
                    import imageio.v2 as imageio

                    vis_writers[key] = imageio.get_writer(vis_path, mode="I", fps=fps, macro_block_size=None)
                    gt_writers[key] = imageio.get_writer(gt_path, mode="I", fps=fps, macro_block_size=None)

                vis_writers[key].append_data(vis_uint8)
                gt_writers[key].append_data(gt_uint8)
    finally:
        for writer in vis_writers.values():
            writer.close()
        for writer in gt_writers.values():
            writer.close()


def main():
    args = parse_args()
    registered = set(list_registered_trajectories())
    videos_dir = vis_dir = captions_dir = None

    for resume_from in args.resume_from:
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

        cfg = load_cfg(resume_from, args.config_file, args.opts)
        dataset_dir = resolve_dataset_dir(args.dataset_dir, cfg, args.mode)
        if videos_dir is None:
            videos_dir, vis_dir, captions_dir = ensure_dataset_dirs(dataset_dir)
        render_novel_cfg = cfg.render.get("render_novel", OmegaConf.create())
        traj_types = args.traj_types if args.traj_types is not None else render_novel_cfg.get("traj_types", ["ego_raw"])
        unknown = [t for t in traj_types if t not in registered]
        if unknown:
            raise ValueError(f"Unknown trajectory type(s): {unknown}. Registered trajectories: {sorted(registered)}")

        degradation_cfg = None
        gaussian_drop_ratio = resolve_gaussian_drop_ratio(args.mode, args.gaussian_drop_ratio)

        output_size = cfg.render.get("output_size", None)
        output_size_mode = cfg.render.get("output_size_mode", "stretch")
        scene_dir = resolve_scene_dir(cfg.data)
        if not os.path.isdir(scene_dir):
            raise FileNotFoundError(f"Scene directory does not exist: {scene_dir}")

        total_frames = get_total_frames(scene_dir)
        start_timestep = int(cfg.data.start_timestep)
        end_timestep = total_frames if int(cfg.data.end_timestep) == -1 else int(cfg.data.end_timestep) + 1
        base_frames = end_timestep - start_timestep
        full_frames = base_frames
        chunk_length = args.chunk_length if args.chunk_length is not None else int(
            args.frames if args.frames is not None else render_novel_cfg.get("frames", base_frames)
        )
        chunk_stride = args.chunk_stride if args.chunk_stride is not None else chunk_length
        if chunk_length <= 0 or chunk_stride <= 0:
            raise ValueError("chunk_length and chunk_stride must be positive")
        if chunk_length > base_frames:
            raise ValueError(
                f"Chunk length {chunk_length} exceeds available frame count {base_frames} for scene {cfg.data.scene_idx}"
            )

        fps = args.fps if args.fps is not None else render_novel_cfg.get("fps", cfg.render.fps)
        lane_offset = render_novel_cfg.get("lane_offset", None)
        lane_offset_ratio = render_novel_cfg.get("lane_offset_ratio", None)
        camera_ids = args.base_camera_ids
        if camera_ids is None:
            camera_ids = [int(cam_id) for cam_id in cfg.data.pixel_source.cameras]

        dataset = DrivingDataset(data_cfg=cfg.data)
        trainer = build_trainer(cfg, dataset)
        trainer.resume_from_checkpoint(ckpt_path=resume_from, load_only_model=True)
        patch_trainer_collect_gaussians(trainer, gaussian_drop_ratio, args.gaussian_drop_seed)
        traj_device = dataset.pixel_source.device

        print(
            f"Using mode={args.mode} checkpoint={resume_from} gaussian_drop_ratio={gaussian_drop_ratio:g}"
        )

        chunk_starts = list(range(0, base_frames - chunk_length + 1, chunk_stride))
        dropped_tail = base_frames - ((chunk_starts[-1] + chunk_length) if chunk_starts else 0)
        if dropped_tail > 0:
            print(f"Dropping tail frames for scene {cfg.data.scene_idx}: {dropped_tail}")

        per_cam_poses = load_camera_poses(cfg, scene_dir, start_timestep, end_timestep)
        reference_poses = load_reference_poses(cfg, scene_dir, start_timestep, end_timestep)
        full_frame_indices = list(range(start_timestep, end_timestep))

        for base_camera_id in camera_ids:
            base_camera_name = DATASETS_CONFIG[cfg.data.dataset][base_camera_id]["camera_name"]
            base_camera_tag = short_camera_tag(base_camera_name)
            traj_kwargs = build_traj_kwargs(
                traj_types,
                base_camera_id,
                lane_offset,
                lane_offset_ratio,
                reference_poses,
            )

            for traj_type in traj_types:
                chunk_specs = []
                for chunk_idx, chunk_start in enumerate(chunk_starts):
                    chunk_end = chunk_start + chunk_length
                    sample_name = args.sample_name_template.format(
                        dataset=cfg.data.dataset,
                        scene=int(cfg.data.scene_idx),
                        camera_tag=base_camera_tag,
                        camera_id=base_camera_id,
                        traj_type=traj_type,
                        mode=args.mode,
                        chunk_idx=chunk_idx,
                        chunk_start=start_timestep + chunk_start,
                        chunk_end=start_timestep + chunk_end - 1,
                    )
                    gt_path = os.path.join(videos_dir, f"{sample_name}.mp4")
                    vis_path = os.path.join(vis_dir, f"{sample_name}.mp4")
                    caption_path = os.path.join(captions_dir, f"{sample_name}.json")
                    if args.skip_existing and os.path.isfile(gt_path) and os.path.isfile(vis_path) and os.path.isfile(caption_path):
                        print(f"Skipping existing sample {sample_name}")
                        continue
                    chunk_specs.append((chunk_start, chunk_end, sample_name, gt_path, vis_path, caption_path))

                if not chunk_specs:
                    continue

                traj = get_interp_novel_trajectories(
                    cfg.data.dataset,
                    str(cfg.data.scene_idx),
                    per_cam_poses,
                    traj_type=traj_type,
                    target_frames=full_frames,
                    traj_kwargs=traj_kwargs.get(traj_type),
                )
                render_data = dataset.prepare_novel_view_render_data(traj.to(traj_device), cam_id=base_camera_id)
                remap_render_data_to_full_scene_indices(render_data, dataset, base_camera_id, full_frame_indices)
                write_chunked_dataset_videos(
                    trainer,
                    dataset,
                    render_data,
                    chunk_specs,
                    fps,
                    base_camera_id,
                    degradation_cfg,
                    output_size,
                    output_size_mode,
                )

                for chunk_start, chunk_end, sample_name, gt_path, vis_path, caption_path in chunk_specs:
                    write_caption(caption_path, args.caption)
                    print(
                        f"Built dataset sample {sample_name} "
                        f"for frames [{start_timestep + chunk_start}, {start_timestep + chunk_end - 1}]"
                    )


if __name__ == "__main__":
    main()
