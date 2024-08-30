import argparse
import numpy as np

if __name__ == "__main__":
    """
    Unified Dataset preprocessing script
    ===========================

    This script facilitates the preprocessing of datasets:
    - Waymo
    - Argoverse
    - NuScenes
    - KITTI
    - NUPlan
    - PandaSet

    Usage:
    ------
    python datasets/preprocess.py \
        --data_root <path_to_dataset> \
        --dataset <dataset_name> \
        --split <split_name> \
        --target_dir <path_to_save_processed_data> \
        --workers <num_workers> \
        --scene_ids <scene_ids> \
        --split_file <split_file> \
        --start_idx <start_idx> \
        --num_scenes <num_scenes> \
        --process_keys <process_keys>
    
    Example:
    --------
    Waymo:
    python datasets/preprocess.py \
        --data_root data/waymo/raw/ \
        --target_dir data/waymo/processed \
        --split training \
        --process_keys images lidar calib pose dynamic_masks objects \
        --workers 8 \
        --scene_ids 23 114 327 621 703 172 552 788
    
    PandaSet:
    python datasets/preprocess.py \
        --data_root data/pandaset/raw \
        --dataset pandaset \
        --split_file data/pandaset_example_scenes.txt \
        --target_dir data/pandaset/processed \
        --workers 32 \
        --process_keys images lidar calib pose dynamic_masks objects
    
    Please refer to the documentation for more information on the available options.

    Arguments:
    ----------
    --data_root (str):
        The root directory where the Waymo dataset is stored. This is a required argument.

    --split (str):
        Specifies the name of the data split. Default is set to "training".

    --target_dir (str):
        Designates the directory where the processed data will be saved. This is a mandatory argument.

    --workers (int):
        The number of processing threads. Default is set to 4.

    --scene_ids (list[int]):
        List of specific scene IDs for processing. Should be integers separated by spaces.

    --split_file (str):
        If provided, indicates the path to a file located in `data/waymo_splits` that contains the desired scene IDs.

    --start_idx (int):
        Used in conjunction with `num_scenes` to generate a list of scene IDs when neither `scene_ids` nor `split_file` are provided.

    --num_scenes (int):
        The total number of scenes to be processed.

    --process_keys (list[str]):
        Denotes the types of data components to be processed. Options include but aren't limited to "images", "lidar", "calib", "pose", etc.

    Notes:
    ------
    The logic of the script ensures that if specific scene IDs (`scene_ids`) are provided, they are prioritized. 
    If a split file (`split_file`) is indicated, it is utilized next. 
    If neither is available, the script uses the `start_idx` and `num_scenes` parameters to determine the scene IDs.
    """
    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument(
        "--data_root", type=str, required=True, help="root path of dataset"
    )
    parser.add_argument("--dataset", type=str, default="waymo", help="dataset name")
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        help="split of the dataset, e.g. training, validation, testing, please specify the split name for different dataset",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="output directory of processed data",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of threads to be used"
    )
    # priority: scene_ids > split_file > start_idx + num_scenes
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed, a list of integers separated by space. Range: [0, 798] for training, [0, 202] for validation",
    )
    parser.add_argument(
        "--split_file", type=str, default=None, help="Split file in data/waymo_splits"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes to generate scene_ids_list",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument(
        "--max_frame_limit",
        type=int,
        default=300,
        help="maximum number of frames to be processed in a dataset, in nuplan dataset, \
            the scene duration super long, we can limit the number of frames to be processed, \
                this argument is used only for nuplan dataset",
    )
    parser.add_argument(
        "--start_frame_idx",
        type=int,
        default=1000,
        help="We skip the first start_frame_idx frames to avoid ego static frames",
    )
    parser.add_argument(
        "--interpolate_N",
        type=int,
        default=0,
        help="Interpolate to get frames at higher frequency, this is only used for nuscene dataset",
    )
    parser.add_argument(
        "--process_keys",
        nargs="+",
        default=[
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
            "objects"
        ],
    )
    args = parser.parse_args()
    if args.dataset != 'nuscenes' and args.interpolate_N > 0:
        parser.error("interpolate_N > 0 is only allowed when dataset is 'nuscenes'")
    
    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    elif args.split_file is not None:
        # parse the split file
        split_file = open(args.split_file, "r").readlines()[1:]
        scene_ids_list = [line.strip().split(",")[0] for line in split_file]
    else:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)

    if args.dataset == "waymo":
        from datasets.waymo.waymo_preprocess import WaymoProcessor
        
        scene_ids_list = [int(scene_id) for scene_id in scene_ids_list]
        dataset_processor = WaymoProcessor(
            load_dir=args.data_root,
            save_dir=args.target_dir,
            prefix=args.split,
            process_keys=args.process_keys,
            process_id_list=scene_ids_list,
            workers=args.workers,
        )
    elif args.dataset == "pandaset":
        from datasets.pandaset.pandaset_preprocess import PandaSetProcessor
        
        scene_ids_list = [str(scene_id).zfill(3) for scene_id in scene_ids_list]
        dataset_processor = PandaSetProcessor(
            load_dir=args.data_root,
            save_dir=args.target_dir,
            process_keys=args.process_keys,
            process_id_list=scene_ids_list,
            workers=args.workers,
        )
    elif args.dataset == "argoverse":
        from datasets.argoverse.argoverse_preprocess import ArgoVerseProcessor
        
        scene_ids_list = [int(scene_id) for scene_id in scene_ids_list]
        dataset_processor = ArgoVerseProcessor(
            load_dir=args.data_root,
            save_dir=args.target_dir,
            process_keys=args.process_keys,
            process_id_list=scene_ids_list,
            workers=args.workers,
        )
    elif args.dataset == "nuscenes":
        from datasets.nuscenes.nuscenes_preprocess import NuScenesProcessor
        
        scene_ids_list = [int(scene_id) for scene_id in scene_ids_list]
        dataset_processor = NuScenesProcessor(
            load_dir=args.data_root,
            save_dir=args.target_dir,
            split=args.split,
            interpolate_N=args.interpolate_N,
            process_keys=args.process_keys,
            process_id_list=scene_ids_list,
            workers=args.workers,
        )
    elif args.dataset == "kitti":
        from datasets.kitti.kitti_preprocess import KittiProcessor
        
        dataset_processor = KittiProcessor(
            load_dir=args.data_root,
            save_dir=args.target_dir,
            prefix=args.split,
            process_keys=args.process_keys,
            process_id_list=scene_ids_list,
            workers=args.workers,
        )
    elif args.dataset == "nuplan":
        from datasets.nuplan.nuplan_preprocess import NuPlanProcessor
        
        dataset_processor = NuPlanProcessor(
            load_dir=args.data_root,
            save_dir=args.target_dir,
            prefix=args.split,
            start_frame_idx=args.start_frame_idx,
            max_frame_limit=args.max_frame_limit,
            process_keys=args.process_keys,
            process_id_list=scene_ids_list,
            workers=args.workers,
        )
    else:
        raise ValueError(f"Unknown dataset {args.dataset}, please choose from waymo, pandaset, argoverse, nuscenes, kitti, nuplan")

    if args.scene_ids is not None and args.workers == 1:
        for scene_id in args.scene_ids:
            dataset_processor.convert_one(scene_id)
    else:
        dataset_processor.convert()
