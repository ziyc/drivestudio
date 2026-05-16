"""Check converted Waymo-KITTI visualization inputs are aligned."""

from __future__ import annotations

import argparse
from pathlib import Path


def ids_in(folder: Path, suffix: str) -> set[str]:
    if not folder.exists():
        return set()
    return {path.stem for path in folder.glob(f"*{suffix}") if path.is_file()}


def sample(items: set[str], n: int = 10) -> str:
    values = sorted(items)
    return ", ".join(values[:n]) + (" ..." if len(values) > n else "")


def validate_kitti_root(kitti_root: str | Path, camera_id: str = "0") -> None:
    root = Path(kitti_root).resolve()
    folders = {
        "velodyne": root / "velodyne",
        "calib": root / "calib",
        "pose": root / "pose",
        f"image_{camera_id}": root / f"image_{camera_id}",
        f"label_{camera_id}": root / f"label_{camera_id}",
        "label_all": root / "label_all",
    }

    missing_dirs = [name for name, folder in folders.items() if not folder.exists()]
    if missing_dirs:
        raise SystemExit(f"[ERROR] Missing directories under {root}: {', '.join(missing_dirs)}")

    id_sets = {
        "velodyne": ids_in(folders["velodyne"], ".bin"),
        "calib": ids_in(folders["calib"], ".txt"),
        "pose": ids_in(folders["pose"], ".txt"),
        f"image_{camera_id}": ids_in(folders[f"image_{camera_id}"], ".png"),
        f"label_{camera_id}": ids_in(folders[f"label_{camera_id}"], ".txt"),
        "label_all": ids_in(folders["label_all"], ".txt"),
    }

    base = id_sets["velodyne"]
    if not base:
        raise SystemExit(f"[ERROR] No .bin files found in {folders['velodyne']}")

    print(f"[OK] Root: {root}")
    for name, values in id_sets.items():
        print(f"[INFO] {name}: {len(values)} files")

    failed = False
    for name, values in id_sets.items():
        missing = base - values
        extra = values - base
        if missing:
            failed = True
            print(f"[ERROR] {name} missing IDs present in velodyne: {sample(missing)}")
        if extra:
            failed = True
            print(f"[WARN] {name} has extra IDs not present in velodyne: {sample(extra)}")

    sorted_ids = sorted(base)
    print(f"[INFO] First frame ID: {sorted_ids[0]}")
    print(f"[INFO] Last frame ID:  {sorted_ids[-1]}")
    if len(sorted_ids) > 1:
        numeric = [frame_id for frame_id in sorted_ids if frame_id.isdigit()]
        if len(numeric) == len(sorted_ids):
            gaps = []
            previous = int(numeric[0])
            for frame_id in numeric[1:]:
                current = int(frame_id)
                if current != previous + 1:
                    gaps.append((previous, current))
                previous = current
            if gaps:
                print(
                    "[INFO] Numeric IDs are not continuous. This is expected when one output folder "
                    "contains multiple Waymo tfrecords; the viewer reads sorted file IDs directly."
                )

    if failed:
        raise SystemExit("[ERROR] Converted KITTI-like data is not fully aligned.")

    print("[OK] Converted KITTI-like data is aligned for visualization.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate KITTI-like conversion output for visualization.")
    parser.add_argument("--kitti_root", required=True, type=Path)
    parser.add_argument("--camera_id", default="0")
    args = parser.parse_args()
    validate_kitti_root(args.kitti_root, args.camera_id)


if __name__ == "__main__":
    main()
