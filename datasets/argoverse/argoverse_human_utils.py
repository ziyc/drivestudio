from typing import List
import os
import cv2
import json
import logging
import numpy as np
from tqdm import tqdm

from utils.geometry import (
    get_corners,
    project_camera_points_to_image
)
from .argoverse_sourceloader import (
    SMPLNODE_CLASSES,
    OPENCV2DATASET,
    AVAILABLE_CAM_LIST,
)

logger = logging.getLogger()

CAMERA_LIST = AVAILABLE_CAM_LIST

def project_human_boxes(
    scene_dir: str,
    camera_list: List[int],
    save_temp=True,
    verbose=False,
    narrow_width_ratio=0.2,
    fps=12
):
    """Project human boxes to 2D image space and save the results to pkl files
    
    Args:
        scene_dir: str, path to the scene directory
        camera_list: List[int], a list of camera ids to be processed
        save_temp: bool, whether to save the intermediate results
        verbose: bool, whether to visualize the projected boxes
        narrow_width_ratio: sometimes the projected boxes are too wide
            we can narrow them with this ratio to get more accurate results
        fps: int, FPS for the visualization video
    
    Returns:
        collector_all: dict, a dictionary containing the projected boxes for each camera
    """
    # check if the necessary directories exist
    images_dir = f'{scene_dir}/images'
    poses_dir = f'{scene_dir}/ego_pose'
    extrinsics_dir = f'{scene_dir}/extrinsics'
    intrinsics_dir = f'{scene_dir}/intrinsics'
    instances_dir = f'{scene_dir}/instances'
    valid_paths = [images_dir, poses_dir, extrinsics_dir, intrinsics_dir, instances_dir]
    for path in valid_paths:
        assert os.path.exists(path), \
            f"Path {path} does not exist, you need to run waymo preprocess to generate the necessary files"
    
    # create directories for saving the results
    save_dir = f'{scene_dir}/humanpose/temp/Pedes_GTTracks'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if verbose:
        # create directories for saving the intermediate visualization results
        video_dir = f'{scene_dir}/humanpose/temp/Pedes_GTTracks/vis'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        per_human_img_dir = f'{scene_dir}/humanpose/temp/Pedes_GTTracks/vis/images'
        if not os.path.exists(per_human_img_dir):
            os.makedirs(per_human_img_dir)
    
    # load instances and frame infos
    frame_infos = json.load(
        open(f'{instances_dir}/frame_instances.json')
    )
    instances_meta = json.load(
        open(f'{instances_dir}/instances_info.json')
    )
    
    collector_all = {}
    # iterate over each camera
    for cam_id in camera_list:
        # check if already processed
        pkl_path = os.path.join(save_dir, f"{cam_id}.pkl")
        if os.path.exists(pkl_path):
            collector_all[cam_id] = json.load(open(pkl_path))
            logger.info(f"Results for camera {cam_id} already exists at {pkl_path}")
            continue
                        
        if verbose:
            per_cam_vis_dir = os.path.join(per_human_img_dir, f"{cam_id}")
            if not os.path.exists(per_cam_vis_dir):
                os.makedirs(per_cam_vis_dir)
        
        collector = {}
        frames = []
        for frame_id, frame_ins_list in frame_infos.items():
            frame_id = int(frame_id)
            
            # define empty instance collector for each frame
            frame_collector = {
                "gt_bbox": [],
                "extra_data": {
                    "gt_track_id": [],
                    "gt_class": [],
                }
            }
            
            # load extrinsic
            cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{cam_id}.txt"))
            cam_to_ego = cam_to_ego @ OPENCV2DATASET
            ego_to_world = np.loadtxt(os.path.join(poses_dir, f"{str(frame_id).zfill(3)}.txt"))
            cam2world = ego_to_world @ cam_to_ego
            
            # load intrinsic
            Ks = np.loadtxt(os.path.join(intrinsics_dir, f"{cam_id}.txt"))
            fx, fy, cx, cy = Ks[0], Ks[1], Ks[2], Ks[3]
            # k1, k2, p1, p2, k3 = Ks[4], Ks[5], Ks[6], Ks[7], Ks[8]
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            
            # load image
            ori_image = cv2.imread(
                os.path.join(images_dir, f"{str(frame_id).zfill(3)}_{cam_id}.jpg")
            )
            image = ori_image.copy()
            H, W = image.shape[:2]
            
            # iterate over each instance in the frame
            if len(frame_ins_list) > 0:
                # if there are pedestrians in this frame, project the boxes to the image
                image_plotted = image.copy()
                for instance_id in frame_ins_list:
                    ins = instances_meta[str(instance_id)]
                    
                    if ins["class_name"] not in SMPLNODE_CLASSES:
                        continue
                    
                    ins_anno = ins["frame_annotations"]
                    index = ins_anno['frame_idx'].index(frame_id)
                    obj_to_world = np.array(ins_anno['obj_to_world'][index])
                    l, w, h = ins_anno['box_size'][index]
                    
                    # get box corners in object space
                    corners = get_corners(l, w, h)
                    # transform box corners to world space
                    corners_world = obj_to_world[:3, :3] @ corners + obj_to_world[:3, 3:4]
                    # transform box corners to image space
                    world2cam = np.linalg.inv(cam2world)
                    corners_cam = world2cam[:3, :3] @ corners_world + world2cam[:3, 3:4]
                    cam_points, depth = project_camera_points_to_image(corners_cam.T, intrinsic)
                    
                    x_min, y_min = np.min(cam_points, axis=0)
                    x_max, y_max = np.max(cam_points, axis=0)
                    # clip left and right with this ratio
                    if narrow_width_ratio > 0.:
                        length = x_max - x_min
                        x_min += length * narrow_width_ratio
                        x_max -= length * narrow_width_ratio

                    # clip the box to the image
                    original_area = (x_max - x_min) * (y_max - y_min)
                    x_min, x_max = np.clip(x_min, 0, W), np.clip(x_max, 0, W)
                    y_min, y_max = np.clip(y_min, 0, H), np.clip(y_max, 0, H)
                    new_area = (x_max - x_min) * (y_max - y_min)
                    
                    # filter out boxes that are too small or too large
                    behind = depth.max() < 0
                    too_small = new_area < W * H * (0.03)**2
                    too_large = new_area > W * H / 1.1
                    too_far = np.linalg.norm(obj_to_world[:3, 3] - cam2world[:3, 3]) > 40
                    clip_large = new_area / original_area < 1/3
                    if too_small or too_large or clip_large or behind or too_far:
                        continue
                    
                    # gt box on image
                    gt_box = [x_min, y_min, x_max - x_min, y_max - y_min]
                    
                    # save the projected box to the collector
                    frame_collector["gt_bbox"].append(gt_box)
                    frame_collector["extra_data"]["gt_track_id"].append(instance_id)
                    frame_collector["extra_data"]["gt_class"].append([0])
                    
                    if verbose:
                        # visualize the projected boxes of ONE instance
                        raw_image = cv2.rectangle(
                            ori_image.copy(), (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2
                        )
                        raw_image_path = os.path.join(per_cam_vis_dir, f"{frame_id}_{instance_id}.jpg")
                        cv2.imwrite(raw_image_path, raw_image)
                        
                        # add this instance to the image
                        image_plotted = cv2.rectangle(
                            image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2
                        )
                
                if verbose:
                    frames.append(image_plotted)
            else:
                # if no instance in this frame, just save the original image
                if verbose:
                    frames.append(ori_image)

            collector[frame_id] = frame_collector
        
        if verbose:
            height, width = frames[0].shape[:2]
            output_path = os.path.join(video_dir, f"cam_{cam_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for frame in tqdm(frames, desc=f"Writing video for camera {cam_id}"):
                out.write(frame)
            out.release()
        
        if save_temp:
            # save collector to pkl
            json.dump(collector, open(pkl_path, "w"))
        
        collector_all[cam_id] = collector
    
    return collector_all