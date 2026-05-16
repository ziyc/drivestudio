"""Open the modified 3D Detection & Tracking Viewer on converted Waymo-KITTI data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWER_ROOT = REPO_ROOT / "external_tools" / "3D-Detection-Tracking-Viewer"
if str(VIEWER_ROOT) not in sys.path:
    sys.path.insert(0, str(VIEWER_ROOT))

from dataset.kitti_dataset import KittiDetectionDataset  # noqa: E402
from viewer.viewer import Viewer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize converted Waymo data in KITTI-like format.")
    parser.add_argument("--kitti_root", required=True, type=Path, help="Directory containing calib/image_x/label_x/velodyne")
    parser.add_argument("--camera_id", default="0", help="Camera folder suffix, e.g. 0 for image_0 and label_0")
    parser.add_argument("--classes", default="Car", help="Comma-separated class filter, or ALL")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1, help="-1 means last frame")
    parser.add_argument("--points_radius", type=float, default=2.0)
    parser.add_argument("--skip_2d", action="store_true")
    parser.add_argument("--skip_3d", action="store_true")
    parser.add_argument("--show_cars", action="store_true", help="Render boxes as car meshes in addition to wire boxes")
    return parser.parse_args()


def run_viewer(
    kitti_root: str | Path,
    camera_id: str = "0",
    classes: str = "Car",
    start_frame: int = 0,
    end_frame: int = -1,
    points_radius: float = 2.0,
    skip_2d: bool = False,
    skip_3d: bool = False,
    show_cars: bool = False,
) -> None:
    kitti_root = Path(kitti_root).resolve()
    image_dir = kitti_root / f"image_{camera_id}"
    label_dir = kitti_root / f"label_{camera_id}"
    velo_dir = kitti_root / "velodyne"
    calib_dir = kitti_root / "calib"

    missing = [p for p in [image_dir, label_dir, velo_dir, calib_dir] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing converted KITTI directories: " + ", ".join(str(p) for p in missing))

    dataset = KittiDetectionDataset(str(kitti_root), label_path=str(label_dir))
    dataset.image_path = str(image_dir)
    dataset.all_ids = sorted(dataset.all_ids)

    class_filter = None
    if classes.strip().upper() != "ALL":
        class_filter = {item.strip() for item in classes.split(",") if item.strip()}

    start = max(start_frame, 0)
    end = len(dataset) if end_frame < 0 else min(end_frame + 1, len(dataset))
    if start >= end:
        raise ValueError(f"Empty frame range: start={start}, end={end}, dataset_len={len(dataset)}")

    viewer = Viewer(box_type="Kitti")
    viewer.set_ob_color_map("gnuplot")

    print(f"Viewing {kitti_root}")
    print(f"Camera image/label folders: image_{camera_id}, label_{camera_id}")
    print(f"Frames: {start}..{end - 1}; classes: {classes}")
    print("Focus the visualization window and press Q, Enter, or Esc to step frames.")

    for frame_idx in range(start, end):
        p2, v2c, points, image, labels, label_names = dataset[frame_idx]

        if labels.size == 0:
            labels = labels.reshape(0, 7)
            label_names = np.array([])

        if class_filter is not None and label_names.size > 0:
            mask = np.array([name in class_filter for name in label_names])
            labels = labels[mask]
            label_names = label_names[mask]

        viewer.add_points(
            points[:, :3],
            radius=points_radius,
            scatter_filed=points[:, 2] if len(points) > 0 else None,
            color_map_name="viridis",
        )
        if len(labels) > 0:
            viewer.add_3D_boxes(labels, box_info=label_names)
            if show_cars:
                viewer.add_3D_cars(labels, box_info=label_names)

        viewer.add_image(image)
        viewer.set_extrinsic_mat(v2c)
        viewer.set_intrinsic_mat(p2)

        if not skip_2d:
            viewer.show_2D()
        if not skip_3d:
            viewer.show_3D()


def main() -> None:
    args = parse_args()
    run_viewer(
        kitti_root=args.kitti_root,
        camera_id=args.camera_id,
        classes=args.classes,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        points_radius=args.points_radius,
        skip_2d=args.skip_2d,
        skip_3d=args.skip_3d,
        show_cars=args.show_cars,
    )


if __name__ == "__main__":
    main()
