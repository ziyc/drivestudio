"""CLI for Waymo-to-KITTI dataset visualization workflows.

The batch files in this folder are intentionally thin Windows launchers. This
module owns the actual path conventions, ten-scene mapping, conversion, checking,
and viewer dispatch.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .check_kitti_io import validate_kitti_root


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBLEM_TFRECORD = (
    "train_segment-12505030131868863688_1740_000_1760_000_with_camera_labels.tfrecord"
)
DEFAULT_PROJECT_OUTPUT = Path("OutPut") / "waymo_training_10scenes"
DEFAULT_RAW_DIR = Path("data") / "waymo" / "raw"
DEFAULT_MAP_FILE = Path("omnire_workflow") / "dataset_visualization" / "waymo_10scene_map.txt"
DEFAULT_CONVERTER_DIR = Path("external_tools") / "waymo_kitti_converter"


@dataclass(frozen=True)
class SceneEntry:
    scene_idx: str
    tfrecord: str


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_scene_map(map_file: str | Path) -> list[SceneEntry]:
    entries: list[SceneEntry] = []
    with repo_path(map_file).open("r", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(f"Bad scene map row: {row}")
            entries.append(SceneEntry(scene_idx=row[0].strip(), tfrecord=row[1].strip()))
    if not entries:
        raise ValueError(f"No scenes found in map file: {map_file}")
    return entries


def ensure_converter(converter_dir: str | Path) -> Path:
    converter = repo_path(converter_dir) / "converter.py"
    if not converter.exists():
        raise FileNotFoundError(f"converter.py not found: {converter}")
    return converter


def prepare_single_tfrecord_dir(tfrecord_file: str | Path) -> Path:
    tfrecord = repo_path(tfrecord_file)
    if not tfrecord.exists():
        raise FileNotFoundError(f"TFRecord not found: {tfrecord}")
    staging = REPO_ROOT / "visualization_outputs" / "_staging" / tfrecord.stem
    staging.mkdir(parents=True, exist_ok=True)
    staged_file = staging / tfrecord.name
    if not staged_file.exists() or staged_file.stat().st_size != tfrecord.stat().st_size:
        if staged_file.exists() or staged_file.is_symlink():
            staged_file.unlink()
        try:
            os.link(tfrecord, staged_file)
        except OSError:
            try:
                os.symlink(tfrecord, staged_file)
            except OSError:
                shutil.copy2(tfrecord, staged_file)
    return staging


def run_converter(raw_dir: Path, kitti_out: Path, converter_dir: str | Path, prefix: str, num_proc: int) -> None:
    converter = ensure_converter(converter_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    kitti_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(converter),
        str(raw_dir),
        str(kitti_out),
        "--prefix",
        prefix,
        "--num_proc",
        str(num_proc),
    ]
    print("[RUN] " + " ".join(f'"{item}"' if " " in item else item for item in cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def convert_one(args: argparse.Namespace) -> None:
    if args.convert_all:
        raw_dir = repo_path(args.raw_dir)
    else:
        tfrecord = args.tfrecord_file or repo_path(DEFAULT_RAW_DIR) / DEFAULT_PROBLEM_TFRECORD
        raw_dir = prepare_single_tfrecord_dir(tfrecord)

    kitti_out = repo_path(args.kitti_out)
    print("Converting:")
    print(f"  raw_dir   = {raw_dir}")
    print(f"  kitti_out = {kitti_out}")
    print(f"  prefix    = {args.prefix}")
    print(f"  num_proc  = {args.num_proc}")
    run_converter(raw_dir, kitti_out, args.converter_dir, args.prefix, args.num_proc)
    if args.check:
        validate_kitti_root(kitti_out, args.camera_id)


def convert_ten(args: argparse.Namespace) -> None:
    raw_dir = repo_path(args.raw_dir)
    project_output = repo_path(args.project_output)
    entries = load_scene_map(args.map_file)

    for entry in entries:
        tfrecord_file = raw_dir / entry.tfrecord
        kitti_out = project_output / f"scene_{entry.scene_idx}" / "dataset_visualization" / "kitti"
        if not tfrecord_file.exists():
            raise FileNotFoundError(f"Missing TFRecord for scene_{entry.scene_idx}: {tfrecord_file}")
        if (kitti_out / "velodyne").exists() and not args.overwrite:
            print(f"[SKIP] scene_{entry.scene_idx} already converted: {kitti_out}")
            if args.check:
                validate_kitti_root(kitti_out, args.camera_id)
            continue

        print(f"\n===== Converting scene_{entry.scene_idx} =====")
        single_raw_dir = prepare_single_tfrecord_dir(tfrecord_file)
        run_converter(single_raw_dir, kitti_out, args.converter_dir, args.prefix, args.num_proc)
        if args.check:
            validate_kitti_root(kitti_out, args.camera_id)

    print("\nAll requested scenes are converted.")


def check_one(args: argparse.Namespace) -> None:
    validate_kitti_root(repo_path(args.kitti_root), args.camera_id)


def check_ten(args: argparse.Namespace) -> None:
    project_output = repo_path(args.project_output)
    for entry in load_scene_map(args.map_file):
        kitti_root = project_output / f"scene_{entry.scene_idx}" / "dataset_visualization" / "kitti"
        print(f"\n===== Checking scene_{entry.scene_idx} =====")
        validate_kitti_root(kitti_root, args.camera_id)
    print("\nAll converted scene folders passed I/O alignment checks.")


def view(args: argparse.Namespace) -> None:
    from .view_kitti_scene import run_viewer

    run_viewer(
        kitti_root=repo_path(args.kitti_root),
        camera_id=args.camera_id,
        classes=args.classes,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        points_radius=args.points_radius,
        skip_2d=args.skip_2d,
        skip_3d=args.skip_3d,
        show_cars=args.show_cars,
    )


def view_scene(args: argparse.Namespace) -> None:
    kitti_root = (
        repo_path(args.project_output)
        / f"scene_{args.scene_idx}"
        / "dataset_visualization"
        / "kitti"
    )
    args.kitti_root = kitti_root
    view(args)


def add_common_convert_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--converter_dir", default=str(DEFAULT_CONVERTER_DIR))
    parser.add_argument("--prefix", default="")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--camera_id", default="0")
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)


def add_view_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera_id", default="0")
    parser.add_argument("--classes", default="Car")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--points_radius", type=float, default=2.0)
    parser.add_argument("--skip_2d", action="store_true")
    parser.add_argument("--skip_3d", action="store_true")
    parser.add_argument("--show_cars", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OmniRe Waymo dataset visualization workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    one = subparsers.add_parser("convert-one", help="Convert one TFRecord or one raw directory")
    one.add_argument("--tfrecord_file", default=None)
    one.add_argument("--raw_dir", default=str(DEFAULT_RAW_DIR))
    one.add_argument(
        "--kitti_out",
        default=str(DEFAULT_PROJECT_OUTPUT / "scene_114" / "dataset_visualization" / "kitti"),
    )
    one.add_argument("--convert_all", action="store_true")
    add_common_convert_args(one)
    one.set_defaults(func=convert_one)

    ten = subparsers.add_parser("convert-ten", help="Convert the ten local Waymo scenes")
    ten.add_argument("--raw_dir", default=str(DEFAULT_RAW_DIR))
    ten.add_argument("--project_output", default=str(DEFAULT_PROJECT_OUTPUT))
    ten.add_argument("--map_file", default=str(DEFAULT_MAP_FILE))
    ten.add_argument("--overwrite", action="store_true")
    add_common_convert_args(ten)
    ten.set_defaults(func=convert_ten)

    check = subparsers.add_parser("check", help="Check one converted KITTI-like folder")
    check.add_argument(
        "--kitti_root",
        default=str(DEFAULT_PROJECT_OUTPUT / "scene_114" / "dataset_visualization" / "kitti"),
    )
    check.add_argument("--camera_id", default="0")
    check.set_defaults(func=check_one)

    check_all = subparsers.add_parser("check-ten", help="Check all ten converted folders")
    check_all.add_argument("--project_output", default=str(DEFAULT_PROJECT_OUTPUT))
    check_all.add_argument("--map_file", default=str(DEFAULT_MAP_FILE))
    check_all.add_argument("--camera_id", default="0")
    check_all.set_defaults(func=check_ten)

    view_parser = subparsers.add_parser("view", help="Open one KITTI-like folder in the viewer")
    view_parser.add_argument(
        "--kitti_root",
        default=str(DEFAULT_PROJECT_OUTPUT / "scene_114" / "dataset_visualization" / "kitti"),
    )
    add_view_args(view_parser)
    view_parser.set_defaults(func=view)

    view_scene_parser = subparsers.add_parser("view-scene", help="Open scene_<idx> beside reconstruction output")
    view_scene_parser.add_argument("--scene_idx", default="114")
    view_scene_parser.add_argument("--project_output", default=str(DEFAULT_PROJECT_OUTPUT))
    add_view_args(view_scene_parser)
    view_scene_parser.set_defaults(func=view_scene)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
