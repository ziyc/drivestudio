import json
import os
from collections import Counter
from typing import List

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.nuscenes import LidarPointCloud, NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

from datasets.tools.multiprocess_utils import track_parallel_progress
from utils.geometry import get_corners
from utils.visualization import color_mapper, dump_3d_bbox_on_image

NUSCENES_LABELS = [
    'animal',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier',
    'movable_object.debris',
    'movable_object.pushable_pullable',
    'movable_object.trafficcone',
    'static_object.bicycle_rack',
    'vehicle.bicycle',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.truck'
]

NUSCENES_NONRIGID_DYNAMIC_CLASSES = [
    'animal',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'vehicle.bicycle',
    'vehicle.motorcycle',
]

NUSCENES_RIGID_DYNAMIC_CLASSES = [
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.trailer',
    'vehicle.truck'
]

NUSCENES_DYNAMIC_CLASSES = NUSCENES_NONRIGID_DYNAMIC_CLASSES + NUSCENES_RIGID_DYNAMIC_CLASSES

class NuScenesProcessor(object):
    """Process NuScenes dataset.

    NuScenes Data Frequencies:
    - CAMERA: 12 Hz
    - LIDAR: 20 Hz
    - Annotated Keyframes: 2 Hz

    This function processes the NuScenes dataset, offering two processing modes:

    1. Original Mode (2 Hz):
    - Uses only the annotated keyframes.
    - Sparse but contains original annotations.
    - Default mode when 'interpolate_N' is set to 0.

    2. Interpolated Mode:
    - Interpolate N frames between annotated keyframes, where N <= 4 is recommended.
    - Provides denser data by interpolating between annotated frames.
    - Activated by setting 'interpolate_N' to a positive integer.
    - Note: Using N > 4 may lead to insufficient image frames due to potential frame drops,
      despite the camera's 12 Hz capture rate.

    Interpolated mode may enhance training due to higher temporal resolution,
    but uses estimated annotations for non-keyframes. Choose the mode that 
    best fits your use case.

    Args:
        load_dir (str): Source directory of NuScenes data.
        save_dir (str): Target directory for processed data.
        split (str): Dataset split to process. Default: 'v1.0-mini'.
        interpolate_N (int): Number of frames to interpolate between keyframes. 
                             If 0, use original 2 Hz data.
        workers (int): Number of parallel processing workers.
        process_keys (list): Data types to process. 
                            Default: ["images", "lidar", "calib", "dynamic_masks", "objects"]
        process_id_list (list): Specific scene IDs to process.
    """

    def __init__(
        self,
        load_dir,
        save_dir,
        split='v1.0-mini',
        interpolate_N=0,
        process_keys=[
            "images",
            "lidar",
            "calib",
            "dynamic_masks",
            "objects"
        ],
        process_id_list=None,
        workers=64,
    ):
        self.process_id_list = process_id_list
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)
        self.interpolate_N = interpolate_N
        assert self.interpolate_N <= 4, \
            "Interpolation frames should be less than 4. \or there will be frame drop issue."
        if self.interpolate_N:
            print(f"We will interpolate {self.interpolate_N} frames between keyframes.")

        # NuScenes Provides 6 cameras
        self.cam_list = [          # {frame_idx}_{cam_id}.jpg
            "CAM_FRONT",        # "xxx_0.jpg"
            "CAM_FRONT_LEFT",   # "xxx_1.jpg"
            "CAM_FRONT_RIGHT",  # "xxx_2.jpg"
            "CAM_BACK_LEFT",    # "xxx_3.jpg"
            "CAM_BACK_RIGHT",   # "xxx_4.jpg"
            "CAM_BACK"          # "xxx_5.jpg"
        ]
        # For each keyframe, the total number of points from LIDAR_TOP is ~35000
        # the total number of points from ALL RADAR is ~300
        # Since Radar points are too sparse, we consider only LIDAR_TOP
        self.lidar_list = ['LIDAR_TOP']

        self.load_dir = load_dir
        post_fix = f"_{(interpolate_N+1)*2}Hz" if interpolate_N > 0 else ""
        save_dir = save_dir.replace("processed", "processed"+post_fix)
        self.save_dir = os.path.join(save_dir, split.split('-')[-1])
        self.workers = int(workers)
        self.nusc =  NuScenes(
            version=split, dataroot=load_dir, verbose=True
        )
        self.create_folder()

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        if self.interpolate_N > 0:
            convert_fn = self.convert_one_interpolated
        else:
            convert_fn = self.convert_one
        track_parallel_progress(convert_fn, id_list, self.workers)
        print("\nFinished ...")

    def convert_one(self, scene_idx):
        """Convert action for single file."""
        scene = self.nusc.scene[scene_idx]
        scene_data = self.nusc.get('scene', scene['token'])
        if "images" in self.process_keys:
            self.save_image(scene_data, scene_idx)
            print(f"Processed images for scene {str(scene_idx).zfill(3)}")
        if "calib" in self.process_keys:
            self.save_calib(scene_data, scene_idx)
            print(f"Processed calib for scene {str(scene_idx).zfill(3)}")
        if "lidar" in self.process_keys:
            self.save_lidar(scene_data, scene_idx)
            print(f"Processed lidar for scene {str(scene_idx).zfill(3)}")
        if "dynamic_masks" in self.process_keys:
            self.save_dynamic_mask(scene_data, scene_idx, class_valid='all')
            self.save_dynamic_mask(scene_data, scene_idx, class_valid='human')
            self.save_dynamic_mask(scene_data, scene_idx, class_valid='vehicle')
            print(f"Processed dynamic masks for scene {str(scene_idx).zfill(3)}")
        
        # process annotated objects
        if "objects" in self.process_keys:
            instances_info, frame_instances = self.save_objects(scene_data)
            print(f"Processed objects for scene {str(scene_idx).zfill(3)}")
            
            # Save instances info and frame instances
            instances_info_save_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/instances"
            with open(f"{instances_info_save_path}/instances_info.json", "w") as fp:
                json.dump(instances_info, fp, indent=4)
            with open(f"{instances_info_save_path}/frame_instances.json", "w") as fp:
                json.dump(frame_instances, fp, indent=4)
                
            if "objects_vis" in self.process_keys:                    
                self.visualize_dynamic_objects(scene_data, scene_idx, instances_info, frame_instances)
                print(f"Processed objects visualization for scene {str(scene_idx).zfill(3)}")

    def convert_one_interpolated(self, scene_idx):
        """Convert action for single file."""
        scene = self.nusc.scene[scene_idx]
        scene_data = self.nusc.get('scene', scene['token'])
        keyframe_timestamps = self.get_keyframe_timestamps(scene_data)
        interpolated_timestamps = self.get_interpolated_timestamps(keyframe_timestamps, self.interpolate_N)
        interpolated_timestamps = np.array(interpolated_timestamps, dtype=np.int64)
        if "images" in self.process_keys:
            self.save_image_interpolated(scene_data, scene_idx, interpolated_timestamps)
            print(f"Processed images for scene {str(scene_idx).zfill(3)}")
        if "calib" in self.process_keys:
            self.save_calib_interpolated(scene_data, scene_idx, interpolated_timestamps)
            print(f"Processed calib for scene {str(scene_idx).zfill(3)}")
        if "lidar" in self.process_keys:
            self.save_lidar_interpolated(scene_data, scene_idx, interpolated_timestamps)
            print(f"Processed lidar for scene {str(scene_idx).zfill(3)}")

        # process annotated objects
        if "objects" in self.process_keys:
            instances_info, _ = self.save_objects(scene_data)
            print(f"Processed objects for scene {str(scene_idx).zfill(3)}")
            
            # interpolate the instances info
            instances_info, frame_instances = self.interpolate_boxes(instances_info)
            print(f"Interpolated objects for scene {str(scene_idx).zfill(3)}")
            
            # Save instances info and frame instances
            instances_info_save_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/instances"
            with open(f"{instances_info_save_path}/instances_info.json", "w") as fp:
                json.dump(instances_info, fp, indent=4)
            with open(f"{instances_info_save_path}/frame_instances.json", "w") as fp:
                json.dump(frame_instances, fp, indent=4)

            if "objects_vis" in self.process_keys:                    
                self.visualize_dynamic_objects_interpolated(scene_data, scene_idx, interpolated_timestamps, instances_info, frame_instances)
                print(f"Processed objects visualization for scene {str(scene_idx).zfill(3)}")
            
            if "dynamic_masks" in self.process_keys:
                self.save_dynamic_mask_interpolated(scene_data, scene_idx, interpolated_timestamps, instances_info, class_valid='all')
                self.save_dynamic_mask_interpolated(scene_data, scene_idx, interpolated_timestamps, instances_info, class_valid='human')
                self.save_dynamic_mask_interpolated(scene_data, scene_idx, interpolated_timestamps, instances_info, class_valid='vehicle')
                print(f"Processed dynamic masks for scene {str(scene_idx).zfill(3)}")

    def __len__(self):
        """Length of the filename list."""
        return len(self.process_id_list)
    
    def get_percam_meta(self, scene_data):
        """Get the timestamp for each image."""
        first_sample_token, last_sample_token = scene_data['first_sample_token'], scene_data['last_sample_token']
        first_sample_record = self.nusc.get('sample', first_sample_token)
        cams_mata = {
            cam_name: {
            "timestamps": [],
            "is_key_frame": [],
            "token": []
            } for cam_name in self.cam_list
        }
        
        for cam_name in self.cam_list:
            cur_img_data = self.nusc.get('sample_data', first_sample_record['data'][cam_name])
            while True:
                cams_mata[cam_name]["timestamps"].append(cur_img_data['timestamp'])
                cams_mata[cam_name]["is_key_frame"].append(cur_img_data['is_key_frame'])
                cams_mata[cam_name]["token"].append(cur_img_data['token'])
                
                if cur_img_data['sample_token'] == last_sample_token:
                    break
                cur_img_data = self.nusc.get('sample_data', cur_img_data['next'])
        
        return cams_mata
    
    def get_keyframe_timestamps(self, scene_data):
        """Get timestamps for keyframes in the scene."""
        first_sample_token = scene_data['first_sample_token']
        last_sample_token = scene_data['last_sample_token']
        curr_sample_record = self.nusc.get('sample', first_sample_token)
        
        keyframe_timestamps = []
        
        while True:
            # Add the timestamp of the current keyframe
            keyframe_timestamps.append(curr_sample_record['timestamp'])
            
            if curr_sample_record['token'] == last_sample_token:
                break
            
            # Move to the next keyframe
            curr_sample_record = self.nusc.get('sample', curr_sample_record['next'])
        
        return keyframe_timestamps

    def get_interpolated_timestamps(self, keyframe_timestamps: List[int], N):
        """Interpolate timestamps between keyframes."""
        interpolated_timestamps = []
        
        for i in range(len(keyframe_timestamps) - 1):
            start_time = keyframe_timestamps[i]
            end_time = keyframe_timestamps[i + 1]
            
            # Calculate the time step for interpolation
            time_step = (end_time - start_time) / (N + 1)
            
            # Add the start timestamp
            interpolated_timestamps.append(start_time)
            
            # Add N interpolated timestamps
            for j in range(1, N + 1):
                interpolated_time = start_time + j * time_step
                interpolated_timestamps.append(int(interpolated_time))
        
        # Add the last keyframe timestamp
        interpolated_timestamps.append(keyframe_timestamps[-1])
        
        return interpolated_timestamps
    
    def find_cloest_lidar_tokens(self, scene_data, timestamps: List[int]):
        """Find the closest LiDAR tokens for given timestamps."""
        first_sample_token = scene_data['first_sample_token']
        first_sample_record = self.nusc.get('sample', first_sample_token)
        lidar_token = first_sample_record['data'][self.lidar_list[0]]
        lidar_data = self.nusc.get('sample_data', lidar_token)
        
        # Collect all LiDAR timestamps and tokens
        lidar_timestamps = []
        lidar_tokens = []
        current_lidar = lidar_data
        while True:
            lidar_timestamps.append(current_lidar['timestamp'])
            lidar_tokens.append(current_lidar['token'])
            if current_lidar['next'] == '':
                break
            current_lidar = self.nusc.get('sample_data', current_lidar['next'])
        
        lidar_timestamps = np.array(lidar_timestamps, dtype=np.int64)
        
        # Find closest LiDAR tokens for each timestamp
        closest_tokens = []
        for timestamp in timestamps:
            idx = np.argmin(np.abs(lidar_timestamps - timestamp))
            closest_tokens.append(lidar_tokens[idx])
            
        # DEBUG USAGE: find is there any duplicated tokens
        # if len(closest_tokens) != len(set(closest_tokens)):
        #     duplicates = [token for token, count in Counter(closest_tokens).items() if count > 1]
        #     print(f"\nWARNING: {len(duplicates)} duplicated tokens in lidar")
        
        return closest_tokens
    
    def find_closest_img_tokens(self, scene_data, timestamps: List[int], cam_name):
        """Find the closest image tokens for given timestamps for a specific camera."""
        first_sample_token = scene_data['first_sample_token']
        first_sample_record = self.nusc.get('sample', first_sample_token)
        img_token = first_sample_record['data'][cam_name]
        img_data = self.nusc.get('sample_data', img_token)
        
        # Collect all image timestamps and tokens for the specified camera
        img_timestamps = []
        img_tokens = []
        current_img = img_data
        while True:
            img_timestamps.append(current_img['timestamp'])
            img_tokens.append(current_img['token'])
            if current_img['next'] == '':
                break
            current_img = self.nusc.get('sample_data', current_img['next'])
        
        img_timestamps = np.array(img_timestamps, dtype=np.int64)
        
        # Find closest image tokens for each timestamp
        closest_tokens = []
        for timestamp in timestamps:
            idx = np.argmin(np.abs(img_timestamps - timestamp))
            closest_tokens.append(img_tokens[idx])
        
        # DEBUG USAGE: find is there any duplicated tokens
        # if len(closest_tokens) != len(set(closest_tokens)):
        #     duplicates = [token for token, count in Counter(closest_tokens).items() if count > 1]
        #     print(f"\nWARNING: {len(duplicates)} duplicated tokens in {cam_name}")
        
        return closest_tokens

    def save_image(self, scene_data, scene_idx):
        """Parse and save the images in jpg format."""
        first_sample_token, last_sample_token = scene_data['first_sample_token'], scene_data['last_sample_token']
        curr_sample_record = self.nusc.get('sample', first_sample_token)
        key_frame_idx = 0
        
        while True:
            for cam_idx, cam_name in enumerate(self.cam_list):
                cam_data = self.nusc.get('sample_data', curr_sample_record['data'][cam_name])
                source_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
                img_path = (
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                    + f"{str(key_frame_idx).zfill(3)}_{str(cam_idx)}.jpg"
                )
                # cp the image
                os.system(f"cp {source_path} {img_path}")
            
            if curr_sample_record['next'] == '' or curr_sample_record['token'] == last_sample_token:
                break
            key_frame_idx += 1
            curr_sample_record = self.nusc.get('sample', curr_sample_record['next'])

    def save_image_interpolated(self, scene_data, scene_idx, timestamps: np.array):
        """Parse and save the interpolated images in jpg format."""
        for cam_idx, cam_name in enumerate(self.cam_list):
            # Find the closest image tokens for each timestamp
            closest_tokens = self.find_closest_img_tokens(scene_data, timestamps, cam_name)
            
            for frame_idx, token in enumerate(closest_tokens):
                cam_data = self.nusc.get('sample_data', token)
                source_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
                img_path = (
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                    + f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.jpg"
                )
                # Copy the image
                os.system(f"cp {source_path} {img_path}")

    def save_calib(self, scene_data, scene_idx):
        """Parse and save the calibration data."""
        first_sample_token, last_sample_token = scene_data['first_sample_token'], scene_data['last_sample_token']
        curr_sample_record = self.nusc.get('sample', first_sample_token)
        key_frame_idx = 0
        
        while True:
            for cam_idx, cam_name in enumerate(self.cam_list):
                cam_data = self.nusc.get('sample_data', curr_sample_record['data'][cam_name])
                calib_data = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                
                # Extrinsics (camera to ego)
                extrinsics_cam_to_ego = np.eye(4)
                extrinsics_cam_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
                extrinsics_cam_to_ego[:3, 3] = np.array(calib_data['translation'])
                
                # Get ego pose (ego to world)
                ego_pose_data = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
                ego_to_world = np.eye(4)
                ego_to_world[:3, :3] = Quaternion(ego_pose_data['rotation']).rotation_matrix
                ego_to_world[:3, 3] = np.array(ego_pose_data['translation'])
                
                # Transform camera extrinsics to world coordinates
                extrinsics_cam_to_world = ego_to_world @ extrinsics_cam_to_ego
                
                np.savetxt(
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/extrinsics/"
                    f"{str(key_frame_idx).zfill(3)}_{str(cam_idx)}.txt",
                    extrinsics_cam_to_world
                )
                
                # Intrinsics
                intrinsics = np.array(calib_data['camera_intrinsic'])
                #      fx  fy  cx  cy 
                Ks = [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2], 0., 0., 0., 0., 0.]
                np.savetxt(
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/intrinsics/"
                    f"{str(cam_idx)}.txt",
                    Ks
                )
            
            if curr_sample_record['next'] == '' or curr_sample_record['token'] == last_sample_token:
                break
            key_frame_idx += 1
            curr_sample_record = self.nusc.get('sample', curr_sample_record['next'])
    
    def save_calib_interpolated(self, scene_data, scene_idx, timestamps: np.array):
        """Parse and save the interpolated calibration data."""
        for cam_idx, cam_name in enumerate(self.cam_list):
            # Find the closest image tokens for each timestamp
            closest_tokens = self.find_closest_img_tokens(scene_data, timestamps, cam_name)
            
            for frame_idx, token in enumerate(closest_tokens):
                cam_data = self.nusc.get('sample_data', token)
                calib_data = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                
                # Extrinsics (camera to ego)
                extrinsics_cam_to_ego = np.eye(4)
                extrinsics_cam_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
                extrinsics_cam_to_ego[:3, 3] = np.array(calib_data['translation'])
                
                # Get ego pose (ego to world)
                ego_pose_data = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
                ego_to_world = np.eye(4)
                ego_to_world[:3, :3] = Quaternion(ego_pose_data['rotation']).rotation_matrix
                ego_to_world[:3, 3] = np.array(ego_pose_data['translation'])
                
                # Transform camera extrinsics to world coordinates
                extrinsics_cam_to_world = ego_to_world @ extrinsics_cam_to_ego
                
                np.savetxt(
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/extrinsics/"
                    f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.txt",
                    extrinsics_cam_to_world
                )
                
                # Intrinsics
                intrinsics = np.array(calib_data['camera_intrinsic'])
                #      fx  fy  cx  cy 
                Ks = [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2], 0., 0., 0., 0., 0.]
                np.savetxt(
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/intrinsics/"
                    f"{str(cam_idx)}.txt",
                    Ks
                )

    def save_lidar(self, scene_data, scene_idx):
        """Parse and save the lidar data in bin format and lidar pose in world coordinates."""
        first_sample_token, last_sample_token = scene_data['first_sample_token'], scene_data['last_sample_token']
        curr_sample_record = self.nusc.get('sample', first_sample_token)
        key_frame_idx = 0
        
        while True:
            lidar_token = curr_sample_record['data'][self.lidar_list[0]]
            lidar_data = self.nusc.get('sample_data', lidar_token)
            lidar_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
            
            # Load point cloud
            pc = LidarPointCloud.from_file(lidar_path)
            
            # Get lidar extrinsics (lidar to ego)
            calib_data = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            lidar_to_ego = np.eye(4)
            lidar_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
            lidar_to_ego[:3, 3] = np.array(calib_data['translation'])
            
            # Save lidar points in ego frame
            lidar_save_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/lidar/{str(key_frame_idx).zfill(3)}.bin"
            pc.points.T.astype(np.float32).tofile(lidar_save_path)
            
            # Get ego pose (ego to world)
            ego_pose_data = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
            ego_to_world = np.eye(4)
            ego_to_world[:3, :3] = Quaternion(ego_pose_data['rotation']).rotation_matrix
            ego_to_world[:3, 3] = np.array(ego_pose_data['translation'])
            
            # Calculate lidar pose in world coordinates
            lidar_to_world = ego_to_world @ lidar_to_ego
            
            # Save lidar pose in world coordinates
            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/lidar_pose/"
                f"{str(key_frame_idx).zfill(3)}.txt",
                lidar_to_world
            )
            
            if curr_sample_record['next'] == '' or curr_sample_record['token'] == last_sample_token:
                break
            key_frame_idx += 1
            curr_sample_record = self.nusc.get('sample', curr_sample_record['next'])

    def save_lidar_interpolated(self, scene_data, scene_idx, timestamps: np.array):
        """Parse and save the interpolated lidar data in bin format and lidar pose in world coordinates."""
        # Find the closest LiDAR tokens for each timestamp
        closest_tokens = self.find_cloest_lidar_tokens(scene_data, timestamps)
        
        for frame_idx, token in enumerate(closest_tokens):
            lidar_data = self.nusc.get('sample_data', token)
            lidar_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
            
            # Load point cloud
            pc = LidarPointCloud.from_file(lidar_path)
            
            # Get lidar extrinsics (lidar to ego)
            calib_data = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            lidar_to_ego = np.eye(4)
            lidar_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
            lidar_to_ego[:3, 3] = np.array(calib_data['translation'])
            
            # Save lidar points in ego frame
            lidar_save_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/lidar/{str(frame_idx).zfill(3)}.bin"
            pc.points.T.astype(np.float32).tofile(lidar_save_path)
            
            # Get ego pose (ego to world)
            ego_pose_data = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
            ego_to_world = np.eye(4)
            ego_to_world[:3, :3] = Quaternion(ego_pose_data['rotation']).rotation_matrix
            ego_to_world[:3, 3] = np.array(ego_pose_data['translation'])
            
            # Calculate lidar pose in world coordinates
            lidar_to_world = ego_to_world @ lidar_to_ego
            
            # Save lidar pose in world coordinates
            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/lidar_pose/"
                f"{str(frame_idx).zfill(3)}.txt",
                lidar_to_world
            )

    def save_dynamic_mask(self, scene_data, scene_idx, class_valid='all'):
        """Parse and save the segmentation data."""
        assert class_valid in ['all', 'human', 'vehicle'], "Invalid class valid"
        if class_valid == 'all':
            VALID_CLASSES = NUSCENES_DYNAMIC_CLASSES
        elif class_valid == 'human':
            VALID_CLASSES = NUSCENES_NONRIGID_DYNAMIC_CLASSES
        elif class_valid == 'vehicle':
            VALID_CLASSES = NUSCENES_RIGID_DYNAMIC_CLASSES
        mask_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/dynamic_masks/{class_valid}"
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        
        first_sample_token, last_sample_token = scene_data['first_sample_token'], scene_data['last_sample_token']
        curr_sample_record = self.nusc.get('sample', first_sample_token)
        key_frame_idx = 0
        
        while True:
            for cam_idx, cam_name in enumerate(self.cam_list):
                cam_data = self.nusc.get('sample_data', curr_sample_record['data'][cam_name])
                img_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/{str(key_frame_idx).zfill(3)}_{str(cam_idx)}.jpg"

                img = cv2.imread(img_path)
                dynamic_mask = np.zeros(img.shape[:2], dtype=np.float32)
                
                anns = [self.nusc.get('sample_annotation', token) for token in curr_sample_record['anns']]
                valid_anns = [ann for ann in anns if ann['category_name'] in VALID_CLASSES]
                
                # Get camera calibration data
                cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                camera_intrinsic = np.array(cs_record['camera_intrinsic'])
                
                # Get ego pose
                pose_record = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
                
                # Project 3D boxes to 2D and create mask
                for ann in valid_anns:
                    # Create Box object
                    box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                    
                    # Move box to ego vehicle coordinate system
                    box.translate(-np.array(pose_record['translation']))
                    box.rotate(Quaternion(pose_record['rotation']).inverse)
                    
                    # Move box to sensor coordinate system
                    box.translate(-np.array(cs_record['translation']))
                    box.rotate(Quaternion(cs_record['rotation']).inverse)
                    
                    # Project 3D box to 2D
                    corners_3d = box.corners()
                    corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)

                    # Check if the object is in front of the camera and all corners are in the image
                    # NOTE: we use strict visibility check here, requiring all corners to be visible
                    in_front = np.all(corners_3d[2, :] > 0.1)
                    in_image = np.all(corners_2d[0, :] >= 0) & np.all(corners_2d[0, :] < img.shape[1]) & \
                            np.all(corners_2d[1, :] >= 0) & np.all(corners_2d[1, :] < img.shape[0])
                    if not (in_front and in_image):
                        continue

                    # If valid, extract x and y coordinates
                    corners_2d = corners_2d[:2, :]

                    # Fill the mask
                    u, v = corners_2d[0, :].astype(np.int32), corners_2d[1, :].astype(np.int32)
                    u = np.clip(u, 0, img.shape[1] - 1)
                    v = np.clip(v, 0, img.shape[0] - 1)

                    if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                        continue

                    xy = (u.min(), v.min())
                    width = u.max() - u.min()
                    height = v.max() - v.min()

                    dynamic_mask[
                        int(xy[1]): int(xy[1] + height),
                        int(xy[0]): int(xy[0] + width)
                    ] = np.maximum(
                        dynamic_mask[
                            int(xy[1]): int(xy[1] + height),
                            int(xy[0]): int(xy[0] + width)
                        ],
                        1
                    )
                
                # Save dynamic mask
                dynamic_mask = np.clip((dynamic_mask > 0.) * 255, 0, 255).astype(np.uint8)
                dynamic_mask = Image.fromarray(dynamic_mask, "L")
                dynamic_mask_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/dynamic_masks/{class_valid}/{str(key_frame_idx).zfill(3)}_{str(cam_idx)}.png"
                dynamic_mask.save(dynamic_mask_path)
                
            if curr_sample_record['next'] == '' or curr_sample_record['token'] == last_sample_token:
                break
            key_frame_idx += 1
            curr_sample_record = self.nusc.get('sample', curr_sample_record['next'])

    def save_dynamic_mask_interpolated(self, scene_data, scene_idx, timestamps, instances_info, class_valid='all'):
        """Parse and save the interpolated segmentation data."""
        assert class_valid in ['all', 'human', 'vehicle'], "Invalid class valid"
        if class_valid == 'all':
            VALID_CLASSES = NUSCENES_DYNAMIC_CLASSES
        elif class_valid == 'human':
            VALID_CLASSES = NUSCENES_NONRIGID_DYNAMIC_CLASSES
        elif class_valid == 'vehicle':
            VALID_CLASSES = NUSCENES_RIGID_DYNAMIC_CLASSES
        mask_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/dynamic_masks/{class_valid}"
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        
        for frame_idx, timestamp in enumerate(timestamps):
            for cam_idx, cam_name in enumerate(self.cam_list):
                # Find the closest image token for this timestamp
                closest_token = self.find_closest_img_tokens(scene_data, [timestamp], cam_name)[0]
                cam_data = self.nusc.get('sample_data', closest_token)
                img_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/{str(frame_idx).zfill(3)}_{str(cam_idx)}.jpg"

                img = cv2.imread(img_path)
                dynamic_mask = np.zeros(img.shape[:2], dtype=np.float32)
                
                objects = [obj_id for obj_id, obj_info in instances_info.items() 
                        if frame_idx in obj_info['frame_annotations']['frame_idx'] 
                        and obj_info['class_name'] in VALID_CLASSES]
                
                # Get camera calibration data
                cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                camera_intrinsic = np.array(cs_record['camera_intrinsic'])
                
                # Get ego pose
                pose_record = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
                
                # Project 3D boxes to 2D and create mask
                for obj_id in objects:
                    obj_info = instances_info[obj_id]
                    idx_in_obj = obj_info['frame_annotations']['frame_idx'].index(frame_idx)
                    o2w = np.array(obj_info['frame_annotations']['obj_to_world'][idx_in_obj])
                    length, width, height = obj_info['frame_annotations']['box_size'][idx_in_obj]
                    
                    # Create Box object
                    box = Box(o2w[:3, 3], [width, length, height], Quaternion(matrix=o2w[:3, :3]))
                    
                    # Move box to ego vehicle coordinate system
                    box.translate(-np.array(pose_record['translation']))
                    box.rotate(Quaternion(pose_record['rotation']).inverse)
                    
                    # Move box to sensor coordinate system
                    box.translate(-np.array(cs_record['translation']))
                    box.rotate(Quaternion(cs_record['rotation']).inverse)
                    
                    # Project 3D box to 2D
                    corners_3d = box.corners()
                    corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)

                    # Check if the object is in front of the camera and all corners are in the image
                    in_front = np.all(corners_3d[2, :] > 0.1)
                    in_image = np.all(corners_2d[0, :] >= 0) & np.all(corners_2d[0, :] < img.shape[1]) & \
                            np.all(corners_2d[1, :] >= 0) & np.all(corners_2d[1, :] < img.shape[0])
                    if not (in_front and in_image):
                        continue

                    # If valid, extract x and y coordinates
                    corners_2d = corners_2d[:2, :]

                    # Fill the mask
                    u, v = corners_2d[0, :].astype(np.int32), corners_2d[1, :].astype(np.int32)
                    u = np.clip(u, 0, img.shape[1] - 1)
                    v = np.clip(v, 0, img.shape[0] - 1)

                    if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                        continue

                    xy = (u.min(), v.min())
                    width = u.max() - u.min()
                    height = v.max() - v.min()

                    dynamic_mask[
                        int(xy[1]): int(xy[1] + height),
                        int(xy[0]): int(xy[0] + width)
                    ] = np.maximum(
                        dynamic_mask[
                            int(xy[1]): int(xy[1] + height),
                            int(xy[0]): int(xy[0] + width)
                        ],
                        1
                    )
                
                # Save dynamic mask
                dynamic_mask = np.clip((dynamic_mask > 0.) * 255, 0, 255).astype(np.uint8)
                dynamic_mask = Image.fromarray(dynamic_mask, "L")
                dynamic_mask_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/dynamic_masks/{class_valid}/{str(frame_idx).zfill(3)}_{str(cam_idx)}.png"
                dynamic_mask.save(dynamic_mask_path)
            
    def save_objects(self, scene_data):
        """Parse and save the objects annotation data."""
        first_sample_token, last_sample_token = scene_data['first_sample_token'], scene_data['last_sample_token']
        curr_sample_record = self.nusc.get('sample', first_sample_token)
        key_frame_idx = 0
        
        instances_info, frame_instances = {}, {}
        while True:
            anns = [self.nusc.get('sample_annotation', token) for token in curr_sample_record['anns']]
            
            for ann in anns:
                if ann['category_name'] not in NUSCENES_DYNAMIC_CLASSES:
                    continue
                
                instance_token = ann['instance_token']
                if instance_token not in instances_info:
                    instances_info[instance_token] = {
                        'id': instance_token,
                        'class_name': ann['category_name'],
                        'frame_annotations': {
                            'frame_idx': [],
                            'obj_to_world': [],
                            'box_size': [],
                        }
                    }
                
                # Object to world transformation
                o2w = np.eye(4)
                o2w[:3, :3] = Quaternion(ann['rotation']).rotation_matrix
                o2w[:3, 3] = np.array(ann['translation'])
                
                # Key frames are spaced (interpolate_N + 1) frames apart in the new sequence
                obj_frame_idx = key_frame_idx * (self.interpolate_N + 1)
                instances_info[instance_token]['frame_annotations']['frame_idx'].append(obj_frame_idx)
                instances_info[instance_token]['frame_annotations']['obj_to_world'].append(o2w.tolist())
                # convert wlh to lwh
                lwh = [ann['size'][1], ann['size'][0], ann['size'][2]]
                instances_info[instance_token]['frame_annotations']['box_size'].append(lwh)
            
            if key_frame_idx not in frame_instances:
                frame_instances[key_frame_idx] = []
            frame_instances[key_frame_idx].extend([ann['instance_token'] for ann in anns if ann['category_name'] in NUSCENES_DYNAMIC_CLASSES])
            
            if curr_sample_record['next'] == '' or curr_sample_record['token'] == last_sample_token:
                break
            key_frame_idx += 1
            curr_sample_record = self.nusc.get('sample', curr_sample_record['next'])
        
        # Correct ID mapping
        id_map = {}
        for i, (k, v) in enumerate(instances_info.items()):
            id_map[v["id"]] = i

        # Update keys in instances_info
        new_instances_info = {}
        for k, v in instances_info.items():
            new_instances_info[id_map[v["id"]]] = v

        # Update keys in frame_instances
        new_frame_instances = {}
        for k, v in frame_instances.items():
            new_frame_instances[k] = [id_map[i] for i in v]

        return new_instances_info, new_frame_instances
    
    def interpolate_boxes(self, instances_info):
        """Interpolate object positions and sizes between keyframes."""
        new_instances_info = {}
        new_frame_instances = {}

        for obj_id, obj_info in instances_info.items():
            frame_annotations = obj_info['frame_annotations']
            keyframe_indices = frame_annotations['frame_idx']
            obj_to_world_list = frame_annotations['obj_to_world']
            box_size_list = frame_annotations['box_size']

            new_frame_idx = []
            new_obj_to_world = []
            new_box_size = []

            for i in range(len(keyframe_indices) - 1):
                start_frame = keyframe_indices[i]
                start_transform = np.array(obj_to_world_list[i])
                end_transform = np.array(obj_to_world_list[i + 1])
                start_quat = Quaternion(matrix=start_transform[:3, :3])
                end_quat = Quaternion(matrix=end_transform[:3, :3])
                start_size = np.array(box_size_list[i])
                end_size = np.array(box_size_list[i + 1])

                for j in range(self.interpolate_N + 1):
                    t = j / (self.interpolate_N + 1)
                    current_frame = start_frame + j

                    # Interpolate translation
                    translation = (1 - t) * start_transform[:3, 3] + t * end_transform[:3, 3]

                    # Interpolate rotation using Quaternions
                    current_quat = Quaternion.slerp(start_quat, end_quat, t)

                    # Construct interpolated transformation matrix
                    current_transform = np.eye(4)
                    current_transform[:3, :3] = current_quat.rotation_matrix
                    current_transform[:3, 3] = translation

                    # Interpolate box size
                    current_size = (1 - t) * start_size + t * end_size

                    new_frame_idx.append(current_frame)
                    new_obj_to_world.append(current_transform.tolist())
                    new_box_size.append(current_size.tolist())

            # Add the last keyframe
            new_frame_idx.append(keyframe_indices[-1])
            new_obj_to_world.append(obj_to_world_list[-1])
            new_box_size.append(box_size_list[-1])

            # Update instance info
            new_instances_info[obj_id] = {
                'id': obj_info['id'],
                'class_name': obj_info['class_name'],
                'frame_annotations': {
                    'frame_idx': new_frame_idx,
                    'obj_to_world': new_obj_to_world,
                    'box_size': new_box_size,
                }
            }

            # Update frame instances
            for frame in new_frame_idx:
                if frame not in new_frame_instances:
                    new_frame_instances[frame] = []
                new_frame_instances[frame].append(obj_id)

        return new_instances_info, new_frame_instances

    def visualize_dynamic_objects(self, scene_data, scene_idx, instances_info, frame_instances):
        """DEBUG: Visualize the dynamic objects' boxes with different colors on the image."""
        output_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/instances/debug_vis"
        
        # Get all sample tokens for this scene
        first_sample_token, last_sample_token = scene_data['first_sample_token'], scene_data['last_sample_token']
        curr_sample_record = self.nusc.get('sample', first_sample_token)
        key_frame_idx = 0
        while True:
            for cam_idx, cam_name in enumerate(self.cam_list):
                cam_data = self.nusc.get('sample_data', curr_sample_record['data'][cam_name])
                img_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/{str(key_frame_idx).zfill(3)}_{str(cam_idx)}.jpg"
                canvas = np.array(Image.open(img_path))
                objects = frame_instances[key_frame_idx]
                
                if len(objects) > 0:
                    lstProj2d = []
                    color_list = []
                    for obj_id in objects:
                        idx_in_obj = instances_info[obj_id]['frame_annotations']['frame_idx'].index(key_frame_idx)
                        o2w = np.array(
                            instances_info[obj_id]['frame_annotations']['obj_to_world'][idx_in_obj]
                        )
                        length, width, height = instances_info[obj_id]['frame_annotations']['box_size'][idx_in_obj]
                        corners = get_corners(length, width, height)
                        
                        # Transform corners to world coordinates
                        corners_world = (o2w[:3, :3] @ corners + o2w[:3, 3:4]).T

                        # Get camera calibration data
                        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                        camera_intrinsic = np.array(cs_record['camera_intrinsic'])
                        
                        # Get ego pose
                        pose_record = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
                        
                        # Transform from world to ego vehicle coordinates
                        corners_ego = corners_world - np.array(pose_record['translation'])
                        corners_ego = Quaternion(pose_record['rotation']).inverse.rotation_matrix @ corners_ego.T
                        
                        # Transform from ego to camera coordinates
                        corners_cam = corners_ego - np.array(cs_record['translation']).reshape(3, 1)
                        corners_cam = Quaternion(cs_record['rotation']).inverse.rotation_matrix @ corners_cam
                        
                        # Project to 2D
                        corners_2d = view_points(corners_cam, camera_intrinsic, normalize=True)
                        
                        # Check if the object is in front of the camera and all corners are in the image
                        in_front = np.all(corners_cam[2, :] > 0.1)
                        in_image = np.all(corners_2d[0, :] >= 0) & np.all(corners_2d[0, :] < canvas.shape[1]) & \
                                np.all(corners_2d[1, :] >= 0) & np.all(corners_2d[1, :] < canvas.shape[0])
                        ok = in_front and in_image

                        if ok:
                            projected_points2d = corners_2d[:2, :].T
                            lstProj2d.append(projected_points2d)
                            color_list.append(color_mapper(str(obj_id)))
                            
                    lstProj2d = np.asarray(lstProj2d)
                    img_plotted = dump_3d_bbox_on_image(coords=lstProj2d, img=canvas, color=color_list)
                else:
                    img_plotted = canvas
                
                img_path = (
                    f"{output_path}/"
                    + f"{str(key_frame_idx).zfill(3)}_{str(cam_idx)}.jpg"
                )
                Image.fromarray(img_plotted).save(img_path)
            
            if curr_sample_record['next'] == '' or curr_sample_record['token'] == last_sample_token:
                break
            key_frame_idx += 1
            curr_sample_record = self.nusc.get('sample', curr_sample_record['next'])

    def visualize_dynamic_objects_interpolated(self, scene_data, scene_idx, timestamps, instances_info, frame_instances):
        """DEBUG: Visualize the interpolated dynamic objects' boxes with different colors on the image."""
        output_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/instances/debug_vis"
        
        for frame_idx, timestamp in enumerate(timestamps):
            for cam_idx, cam_name in enumerate(self.cam_list):
                # Find the closest image token for this timestamp
                closest_token = self.find_closest_img_tokens(scene_data, [timestamp], cam_name)[0]
                cam_data = self.nusc.get('sample_data', closest_token)
                
                img_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/{str(frame_idx).zfill(3)}_{str(cam_idx)}.jpg"
                canvas = np.array(Image.open(img_path))
                objects = frame_instances.get(frame_idx, [])
                
                if len(objects) > 0:
                    lstProj2d = []
                    color_list = []
                    for obj_id in objects:
                        obj_info = instances_info[obj_id]
                        idx_in_obj = obj_info['frame_annotations']['frame_idx'].index(frame_idx)
                        o2w = np.array(obj_info['frame_annotations']['obj_to_world'][idx_in_obj])
                        length, width, height = obj_info['frame_annotations']['box_size'][idx_in_obj]
                        corners = get_corners(length, width, height)
                        
                        # Transform corners to world coordinates
                        corners_world = (o2w[:3, :3] @ corners + o2w[:3, 3:4]).T

                        # Get camera calibration data
                        cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                        camera_intrinsic = np.array(cs_record['camera_intrinsic'])
                        
                        # Get ego pose
                        pose_record = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
                        
                        # Transform from world to ego vehicle coordinates
                        corners_ego = corners_world - np.array(pose_record['translation'])
                        corners_ego = Quaternion(pose_record['rotation']).inverse.rotation_matrix @ corners_ego.T
                        
                        # Transform from ego to camera coordinates
                        corners_cam = corners_ego - np.array(cs_record['translation']).reshape(3, 1)
                        corners_cam = Quaternion(cs_record['rotation']).inverse.rotation_matrix @ corners_cam
                        
                        # Project to 2D
                        corners_2d = view_points(corners_cam, camera_intrinsic, normalize=True)
                        
                        # Check if the object is in front of the camera and all corners are in the image
                        in_front = np.all(corners_cam[2, :] > 0.1)
                        in_image = np.all(corners_2d[0, :] >= 0) & np.all(corners_2d[0, :] < canvas.shape[1]) & \
                                np.all(corners_2d[1, :] >= 0) & np.all(corners_2d[1, :] < canvas.shape[0])
                        ok = in_front and in_image

                        if ok:
                            projected_points2d = corners_2d[:2, :].T
                            lstProj2d.append(projected_points2d)
                            color_list.append(color_mapper(str(obj_id)))
                            
                    lstProj2d = np.asarray(lstProj2d)
                    img_plotted = dump_3d_bbox_on_image(coords=lstProj2d, img=canvas, color=color_list)
                else:
                    img_plotted = canvas
                
                img_path = f"{output_path}/{str(frame_idx).zfill(3)}_{str(cam_idx)}.jpg"
                Image.fromarray(img_plotted).save(img_path)

    def create_folder(self):
        """Create folder for data preprocessing."""
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        for i in id_list:
            if "images" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/images", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/sky_masks", exist_ok=True)
            if "calib" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/extrinsics", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/intrinsics", exist_ok=True)
            if "lidar" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/lidar", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/lidar_pose", exist_ok=True)
            if "dynamic_masks" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/dynamic_masks/all", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/dynamic_masks/human", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/dynamic_masks/vehicle", exist_ok=True)
            if "objects" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/instances", exist_ok=True)
                if "objects_vis" in self.process_keys:
                    os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/instances/debug_vis", exist_ok=True)