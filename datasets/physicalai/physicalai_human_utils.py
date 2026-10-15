"""Native-camera pedestrian boxes for the existing human-pose pipeline."""

import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from utils.geometry import camera_midpoint_pose, project_world_camera

CAMERA_LIST = list(range(7))


def project_human_boxes(
    scene_dir: str,
    camera_list: List[int],
    save_temp=True,
    verbose=False,
    narrow_width_ratio=0.2,
    fps=10
):
    """Project human boxes to 2D image space and save the results to pkl files.

    Args:
        scene_dir: str, path to the scene directory
        camera_list: List[int], a list of camera ids to be processed
        save_temp: bool, whether to save the intermediate results
        verbose: bool, whether to visualize the projected boxes
        narrow_width_ratio: narrow the projected boxes horizontally by this ratio
        fps: int, FPS for the visualization video

    Returns:
        results: dict, projected boxes and track ids for each camera and frame
    """
    if verbose and fps <= 0:
        raise ValueError("Visualization FPS must be positive")
    root = Path(scene_dir)
    save_dir = root / 'humanpose/temp/Pedes_GTTracks'
    instances = json.loads((root / 'instances/instances_info.json').read_text())
    frames = json.loads((root / 'instances/frame_instances.json').read_text())
    signs = torch.tensor([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
    results = {}
    for cam_id in camera_list:
        p = json.loads((root / 'intrinsics' / f'{cam_id}.json').read_text())
        width, height = p['resolution']
        result = {}
        video_writer = None
        if verbose:
            video_dir = save_dir / 'vis'
            per_cam_vis_dir = video_dir / 'images' / str(cam_id)
            per_cam_vis_dir.mkdir(parents=True, exist_ok=True)
            output_path = video_dir / f'cam_{cam_id}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if not video_writer.isOpened():
                video_writer.release()
                raise RuntimeError(f"Could not open debug video: {output_path}")
        try:
            for frame, track_ids in frames.items():
                entry = {'gt_bbox': [], 'extra_data': {'gt_track_id': [], 'gt_class': []}}
                if verbose:
                    image_path = root / 'images' / f'{int(frame):03d}_{cam_id}.png'
                    ori_image = cv2.imread(str(image_path))
                    if ori_image is None:
                        raise FileNotFoundError(f"Could not read camera image: {image_path}")
                    if ori_image.shape[:2] != (height, width):
                        raise ValueError(
                            f"Camera image dimensions do not match calibration: {image_path}"
                        )
                    image_plotted = ori_image.copy()
                start = np.loadtxt(
                    root / 'camera_pose_start' / f'{int(frame):03d}_{cam_id}.txt', dtype=np.float32
                )
                end = np.loadtxt(
                    root / 'camera_pose' / f'{int(frame):03d}_{cam_id}.txt', dtype=np.float32
                )
                pose_mid = torch.from_numpy(camera_midpoint_pose(start, end)).float()
                for track_id in track_ids:
                    track = instances[str(track_id)]
                    if track['class_name'] != 'Pedestrian':
                        continue
                    ann = track['frame_annotations']
                    index = ann['frame_idx'].index(int(frame))
                    pose, size = torch.tensor(ann['obj_to_world'][index]), torch.tensor(
                        ann['box_size'][index]
                    )
                    points = (signs * size / 2) @ pose[:3, :3].T + pose[:3, 3]
                    uv, _, valid = project_world_camera(
                        points, pose_mid, torch.eye(3), width, height, 'ftheta', ftheta_parameters=p
                    )
                    if valid.sum() < 4:
                        continue
                    lo, hi = uv[valid].amin(0), uv[valid].amax(0)
                    inset = (hi[0] - lo[0]) * narrow_width_ratio
                    lo[0] += inset
                    hi[0] -= inset
                    if (hi - lo).min() < 4:
                        continue
                    entry['gt_bbox'].append(
                        [float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])]
                    )
                    entry['extra_data']['gt_track_id'].append(track_id)
                    entry['extra_data']['gt_class'].append([0])
                    if verbose:
                        top_left = (int(lo[0]), int(lo[1]))
                        bottom_right = (int(hi[0]), int(hi[1]))
                        raw_image = cv2.rectangle(
                            ori_image.copy(), top_left, bottom_right, (0, 255, 0), 2
                        )
                        image_path = per_cam_vis_dir / f'{int(frame)}_{track_id}.jpg'
                        if not cv2.imwrite(str(image_path), raw_image):
                            raise OSError(f"Could not write debug image: {image_path}")
                        cv2.rectangle(image_plotted, top_left, bottom_right, (0, 255, 0), 2)
                result[int(frame)] = entry
                if verbose:
                    video_writer.write(image_plotted)
        finally:
            if video_writer is not None:
                video_writer.release()
        if save_temp:
            path = save_dir / f'{cam_id}.pkl'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
        results[cam_id] = result
    return results
