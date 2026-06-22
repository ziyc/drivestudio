import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List


def download_file(filename, target_dir, source):
    result = subprocess.run(
        [
            "gsutil",
            "cp",
            "-n",
            f"{source}/{filename}.tfrecord",
            target_dir,
        ],
        capture_output=True,  # To capture stderr and stdout for detailed error information
        text=True,
    )

    # Check the return code of the gsutil command
    if result.returncode != 0:
        raise Exception(
            result.stderr
        )  # Raise an exception with the error message from the gsutil command


def download_files(
    file_names: List[str],
    target_dir: str,
    source: str = "gs://waymo_open_dataset_scene_flow/train",
) -> None:
    """
    Downloads a list of files from a given source to a target directory using multiple threads.

    Args:
        file_names (List[str]): A list of file names to download.
        target_dir (str): The target directory to save the downloaded files.
        source (str, optional): The source directory to download the files from. Defaults to "gs://waymo_open_dataset_scene_flow/train".
    """
    # Get the total number of file_names
    total_files = len(file_names)

    # Use ThreadPoolExecutor to manage concurrent downloads
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(download_file, filename, target_dir, source)
            for filename in file_names
        ]

        for counter, future in enumerate(futures, start=1):
            # Wait for the download to complete and handle any exceptions
            try:
                # inspects the result of the future and raises an exception if one occurred during execution
                future.result()
                print(f"[{counter}/{total_files}] Downloaded successfully!")
            except Exception as e:
                print(f"[{counter}/{total_files}] Failed to download. Error: {e}")


if __name__ == "__main__":
    print("note: `gcloud auth login` is required before running this script")
    print("Downloading Waymo dataset from Google Cloud Storage...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dir",
        type=str,
        default="data/waymo/raw",
        help="Path to the target directory",
    )
    parser.add_argument(
        "--scene_ids", type=int, nargs="+", help="scene ids to download"
    )
    parser.add_argument(
        "--scene_file", type=str, default=None,
        help="simple scene list: one scene ID per line, # comments ignored",
    )
    parser.add_argument(
        "--split_file", type=str, default=None,
        help="csv-style split file (scene_id,seg_name,...) with header",
    )
    parser.add_argument(
        "--processed_root", type=str, default="data/waymo/processed/training",
        help="check this dir for already-processed scenes to skip",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if processed dir exists")
    args = parser.parse_args()
    os.makedirs(args.target_dir, exist_ok=True)
    total_list = open("data/waymo_train_list.txt", "r").readlines()

    requested_ids = []
    if args.scene_ids is not None:
        requested_ids = list(args.scene_ids)
    elif args.scene_file is not None:
        for line in open(args.scene_file):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                requested_ids.append(int(line.split(",")[0]))
            else:
                requested_ids.append(int(line))
    elif args.split_file is not None:
        for line in open(args.split_file).readlines()[1:]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            requested_ids.append(int(line.split(",")[0]))
    else:
        print("ERROR: provide --scene_ids, --scene_file, or --split_file")
        exit(1)

    # filter already-processed scenes
    file_names = []
    for i in requested_ids:
        scene_dir = os.path.join(args.processed_root, f"{i:03d}")
        if not args.force and os.path.isdir(scene_dir):
            print(f"  skip scene {i}: already processed at {scene_dir}")
            continue
        file_names.append(total_list[i].strip())

    if not file_names:
        print("All requested scenes already processed. Nothing to download.")
        exit(0)

    download_files(file_names, args.target_dir)
