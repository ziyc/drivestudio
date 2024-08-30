from typing import List, Callable
import os
import joblib
import logging
import argparse
import numpy as np

from datasets.tools.extract_smpl import run_4DHumans
from datasets.tools.postprocess import match_and_postprocess

logger = logging.getLogger()

def extract_humanpose(
    scene_dir,
    projection_fn: Callable,
    camera_list: List[str],
    save_temp: bool=True,
    verbose: bool=False,
    fps: int=12
):
    """Extract human pose from the waymo dataset
    
    Args:
        scene_dir: str, path to the scene directory
        save_temp: bool, whether to save the intermediate results
        verbose: bool, whether to visualize debug images
        fps: int, FPS for the visualization video
    """
    # project human boxes to 2D image space
    GTTracks_meta = projection_fn(
        scene_dir, camera_list=camera_list,
        save_temp=save_temp, verbose=verbose,
        narrow_width_ratio=0.2, fps=fps
    )
    
    # run 4DHuman to get predicted human tracks with SMPL parameters
    PredTracks_meta = run_4DHumans(
        scene_dir, camera_list=camera_list,
        save_temp=save_temp, verbose=verbose, fps=fps
    )
    
    # match the predicted tracks with the ground truth tracks
    smpl_meta = match_and_postprocess(
        scene_dir, camera_list=camera_list,
        GTTracksDict=GTTracks_meta, PredTracksDict=PredTracks_meta,
        save_temp=save_temp, verbose=verbose, fps=fps
    )
    
    joblib.dump(
        smpl_meta,
        os.path.join(scene_dir, "humanpose", "smpl.pkl")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument("--data_root", type=str, required=True, help="root path of waymo dataset")
    parser.add_argument("--dataset", type=str, default="waymo", help="dataset name")
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
        "--save_temp",
        action="store_true",
        help="Whether to save the intermediate results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to visualize the intermediate results",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="FPS for the visualization video if verbose is True",
    )
    args = parser.parse_args()
    
    if args.dataset == "waymo":
        from datasets.waymo.waymo_human_utils import project_human_boxes, CAMERA_LIST
    elif args.dataset == "pandaset":
        from datasets.pandaset.pandaset_human_utils import project_human_boxes, CAMERA_LIST
    elif args.dataset == "argoverse":
        from datasets.argoverse.argoverse_human_utils import project_human_boxes, CAMERA_LIST
    elif args.dataset == "nuscenes":
        from datasets.nuscenes.nuscenes_human_utils import project_human_boxes, CAMERA_LIST
    elif args.dataset == "kitti":
        from datasets.kitti.kitti_human_utils import project_human_boxes, CAMERA_LIST
    elif args.dataset == "nuplan":
        from datasets.nuplan.nuplan_human_utils import project_human_boxes, CAMERA_LIST
    else:
        raise ValueError(f"Unknown dataset {args.dataset}, please choose from waymo, pandaset, argoverse, nuscenes, kitti, nuplan")
    
    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    elif args.split_file is not None:
        # parse the split file
        split_file = open(args.split_file, "r").readlines()[1:]
        try:
            # Waymo, Pandaset, Argoverse, NuScenes
            scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
        except:
            # KITTI
            scene_ids_list = [line.strip().split(" ")[0] for line in split_file]
    else:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)
    
    for scene_id in scene_ids_list:
        try:
            scene_dir = f'{args.data_root}/{str(scene_id).zfill(3)}'
            extract_humanpose(
                scene_dir=scene_dir,
                projection_fn=project_human_boxes,
                camera_list=CAMERA_LIST,
                save_temp=args.save_temp,
                verbose=args.verbose,
                fps=args.fps
            )
            logger.info(f"Finished processing scene {scene_id}")
        except Exception as e:
            logger.error(f"Error processing scene {scene_id}: {e}")
            continue