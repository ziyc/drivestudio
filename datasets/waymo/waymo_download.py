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
        "--split_file", type=str, default=None, help="split file in data/waymo_splits"
    )
    args = parser.parse_args()
    os.makedirs(args.target_dir, exist_ok=True)
    total_list = open("data/waymo_train_list.txt", "r").readlines()
    if args.split_file is None:
        file_names = [total_list[i].strip() for i in args.scene_ids]
    else:
        # parse the split file
        split_file = open(args.split_file, "r").readlines()[1:]
        scene_ids = [int(line.strip().split(",")[0]) for line in split_file]
        file_names = [total_list[i].strip() for i in scene_ids]
    download_files(file_names, args.target_dir)
