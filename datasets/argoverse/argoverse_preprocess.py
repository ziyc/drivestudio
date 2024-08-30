import os
import json
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from av2.datasets.sensor.constants import RingCameras, StereoCameras
from av2.datasets.sensor.sensor_dataloader import SensorDataloader, SynchronizedSensorData
from datasets.tools.multiprocess_utils import track_parallel_progress
from utils.visualization import dump_3d_bbox_on_image, color_mapper

AV2_LABELS = [
    "ANIMAL", "ARTICULATED_BUS", "BICYCLE", "BICYCLIST",
    "BOLLARD", "BOX_TRUCK", "BUS", "CONSTRUCTION_BARREL",
    "CONSTRUCTION_CONE", "DOG", "LARGE_VEHICLE", "MESSAGE_BOARD_TRAILER",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN", "MOTORCYCLE", "MOTORCYCLIST",
    "OFFICIAL_SIGNALER", "PEDESTRIAN", "RAILED_VEHICLE", "REGULAR_VEHICLE",
    "SCHOOL_BUS", "SIGN", "STOP_SIGN", "STROLLER", "TRAFFIC_LIGHT_TRAILER",
    "TRUCK", "TRUCK_CAB", "VEHICULAR_TRAILER", "WHEELCHAIR",
    "WHEELED_DEVICE", "WHEELED_RIDER"
]

AV2_NONRIGID_DYNAMIC_CLASSES = [
    "BICYCLIST", "DOG", "MOTORCYCLIST", "PEDESTRIAN", "STROLLER",
    "WHEELCHAIR", "WHEELED_DEVICE", "WHEELED_RIDER"
]

AV2_RIGID_DYNAMIC_CLASSES = [
    "ARTICULATED_BUS", "BOX_TRUCK", "BUS", "LARGE_VEHICLE",
    "MOTORCYCLE", "RAILED_VEHICLE", "REGULAR_VEHICLE", "SCHOOL_BUS",
    "TRUCK", "TRUCK_CAB", "VEHICULAR_TRAILER"
]

AV2_DYNAMIC_CLASSES = AV2_NONRIGID_DYNAMIC_CLASSES + AV2_RIGID_DYNAMIC_CLASSES

valid_ring_cams = set([x.value for x in RingCameras])
valid_stereo_cams = set([x.value for x in StereoCameras])

class ArgoVerseProcessor(object):
    """Process ArgoVerse.
    
    LiDAR: 10Hz, Camera: 20Hz
    Since the LiDAR and Camera are not synchronized, we need to find the closest camera image for each LiDAR frame.
    Thus the actual frame rate of the processed data is 10Hz, which is aligned with the LiDAR.

    Args:
        load_dir (str): Directory to load data.
        save_dir (str): Directory to save data.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
            Defaults to False.
    """
    def __init__(
        self,
        load_dir,
        save_dir,
        process_keys=[
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
            "objects"
        ],
        process_id_list=None,
        workers=64,
    ):
        self.process_id_list = process_id_list
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)

        # ArgoVerse Provides 7 cameras, we process 5 of them
        self.cam_list = [          # {frame_idx}_{cam_id}.jpg
            "ring_front_center",   # "xxx_0.jpg"
            "ring_front_left",     # "xxx_1.jpg"
            "ring_front_right",    # "xxx_2.jpg"
            "ring_side_left",      # "xxx_3.jpg"
            "ring_side_right",     # "xxx_4.jpg"
            "ring_rear_left",      # "xxx_5.jpg"
            "ring_rear_right",     # "xxx_6.jpg"
        ]
        cam_enums: List[Union[RingCameras, StereoCameras]] = []
        for cam_name in self.cam_list:
            if cam_name in valid_ring_cams:
                cam_enums.append(RingCameras(cam_name))
            elif cam_name in valid_stereo_cams:
                cam_enums.append(StereoCameras(cam_name))
            else:
                raise ValueError("Must provide _valid_ camera names!")
        
        # Prepare dynamic objects' metadata
        self.load_dir = Path(load_dir)
        self.save_dir = f"{save_dir}"
        self.workers = int(workers)
        self.av2loader = SensorDataloader(
            dataset_dir=self.load_dir,
            with_annotations=True,
            with_cache=True,
            cam_names=tuple(cam_enums)
        )
        # a list of tfrecord pathnames
        self.training_files = open("data/argoverse_train_list.txt").read().splitlines()
        self.log_pathnames = [
            f"{self.load_dir}/{f}" for f in self.training_files
        ]
        self.create_folder()

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        track_parallel_progress(self.convert_one, id_list, self.workers)
        print("\nFinished ...")
        
    def get_lidar_indices(self, log_id: str):
        sensor_cache = self.av2loader.sensor_cache
        lidar_indices = sensor_cache.xs(key="lidar", level=2).index
        lidar_indices_mask = lidar_indices.get_level_values('log_id') == log_id
        # get the positions of the lidar indices, get nonzeros
        lidar_indices = np.nonzero(lidar_indices_mask)[0]
        return lidar_indices
    
    def filter_lidar_indices(self, lidar_indices):
        """
        Filter lidar indices whose corresponding synchronized camera images are not complete.
        These usually happen at the beginning and the end of the sequence.
        """
        valid_list = []
        invalid_list = []
        for idx in lidar_indices:
            datum = self.av2loader[idx]
            sweep = datum.sweep
            
            timestamp_city_SE3_ego_dict = datum.timestamp_city_SE3_ego_dict
            synchronized_imagery = datum.synchronized_imagery
            
            cnt = 0
            for _, cam in synchronized_imagery.items():
                if (
                    cam.timestamp_ns in timestamp_city_SE3_ego_dict
                    and sweep.timestamp_ns in timestamp_city_SE3_ego_dict
                ):
                    cnt += 1
                    
            if cnt != len(self.cam_list):
                invalid_list.append(idx)
                continue
            valid_list.append(idx)
        print(f"INFO: {len(invalid_list)} lidar indices filtered")
            
        return valid_list
        
    def convert_one(self, scene_idx):
        """Convert action for single file.

        Args:
            scene_idx (str): Scene index.
        """
        lidar_indices = self.get_lidar_indices(
            self.training_files[scene_idx]
        )
        lidar_indices = self.filter_lidar_indices(lidar_indices)
        
        # process each frame
        num_frames = len(lidar_indices)
        for frame_idx, lidar_idx in tqdm(
            enumerate(lidar_indices), desc=f"File {scene_idx}", total=num_frames, dynamic_ncols=True
        ):  
            datum = self.av2loader[lidar_idx]
            if "images" in self.process_keys:
                self.save_image(datum, scene_idx, frame_idx)
            if "calib" in self.process_keys:
                self.save_calib(datum, scene_idx, frame_idx)
            if "lidar" in self.process_keys:
                self.save_lidar(datum, scene_idx, frame_idx)
            if "pose" in self.process_keys:
                self.save_pose(datum, scene_idx, frame_idx)
            if "3dbox_vis" in self.process_keys:
                # visualize 3d box, debug usage
                self.visualize_3dbox(datum, scene_idx, frame_idx)
            if "dynamic_masks" in self.process_keys:
                self.save_dynamic_mask(datum, scene_idx, frame_idx, class_valid='all')
                self.save_dynamic_mask(datum, scene_idx, frame_idx, class_valid='human')
                self.save_dynamic_mask(datum, scene_idx, frame_idx, class_valid='vehicle')
                
        # sort and save objects info
        if "objects" in self.process_keys:
            instances_info, frame_instances = self.save_objects(lidar_indices)
            print(f"Processed instances info for {scene_idx}")
            
            # Save instances info and frame instances
            object_info_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/instances"
            with open(f"{object_info_dir}/instances_info.json", "w") as fp:
                json.dump(instances_info, fp, indent=4)
            with open(f"{object_info_dir}/frame_instances.json", "w") as fp:
                json.dump(frame_instances, fp, indent=4)
            
            # verbose: visualize the instances on the image (Debug Usage)
            if "objects_vis" in self.process_keys:
                self.visualize_dynamic_objects(
                    scene_idx, lidar_indices,
                    instances_info=instances_info,
                    frame_instances=frame_instances
                )
                print(f"Processed objects visualization for {scene_idx}")

    def __len__(self):
        """Length of the filename list."""
        return len(self.process_id_list)

    def save_image(self, datum: SynchronizedSensorData, scene_idx, frame_idx):
        """Parse and save the images in jpg format.

        Args:
            datum (:obj:`SynchronizedSensorData`): ArgoVerse synchronized sensor data.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        synchronized_imagery = datum.synchronized_imagery
        for cam_name, cam in synchronized_imagery.items():
            idx = self.cam_list.index(cam_name)
            img_path = (
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
            )
            image = Image.fromarray(cam.img[:, :, [2, 1, 0]]) # BGR to RGB
            image.save(img_path)

    def save_calib(self, datum: SynchronizedSensorData, scene_idx, frame_idx):
        """Parse and save the calibration data.

        Args:
            datum (:obj:`SynchronizedSensorData`): ArgoVerse synchronized sensor data.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        synchronized_imagery = datum.synchronized_imagery
        for idx, cam_name in enumerate(self.cam_list):
            cam = synchronized_imagery[cam_name]
            c2v = cam.camera_model.ego_SE3_cam.transform_matrix
            K = cam.camera_model.intrinsics
            intrinsics = [K.fx_px, K.fy_px, K.cx_px, K.cy_px, 0.0, 0.0, 0.0, 0.0, 0.0] 
    
            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/extrinsics/"
                + f"{str(idx)}.txt",
                c2v,
            )
            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/intrinsics/"
                + f"{str(idx)}.txt",
                intrinsics,
            )

    def save_lidar(self, datum: SynchronizedSensorData, scene_idx, frame_idx):
        """Parse and save the lidar data in psd format.

        Args:
            datum (:obj:`SynchronizedSensorData`): ArgoVerse synchronized sensor data.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        sweep = datum.sweep
        
        point_cloud = np.column_stack(
            (
                sweep.xyz, # lidar points in ego frame
                sweep.intensity, # intensity
            )
        )
        
        pc_path = (
            f"{self.save_dir}/"
            + f"{str(scene_idx).zfill(3)}/lidar/{str(frame_idx).zfill(3)}.bin"
        )
        point_cloud.astype(np.float32).tofile(pc_path)

    def save_pose(self, datum: SynchronizedSensorData, scene_idx, frame_idx):
        """Parse and save the pose data.

        Args:
            datum (:obj:`SynchronizedSensorData`): ArgoVerse synchronized sensor data.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        sweep = datum.sweep
        timestamp_city_SE3_ego_dict = datum.timestamp_city_SE3_ego_dict
        synchronized_imagery = datum.synchronized_imagery
        for cam_name, cam in synchronized_imagery.items():
            if cam_name == "ring_front_center":
                # we use the ego state at the time of the lidar as the ego pose
                ego_pose_t_lidar = timestamp_city_SE3_ego_dict[sweep.timestamp_ns]

                np.savetxt(
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/ego_pose/"
                    + f"{str(frame_idx).zfill(3)}.txt",
                    ego_pose_t_lidar.transform_matrix,
                )
        
    def visualize_3dbox(self, datum: SynchronizedSensorData, scene_idx, frame_idx):
        """DEBUG: Visualize the 3D bounding box on the image.
        Visualize the 3D bounding box all with the same COLOR.
        If you want to visualize the 3D bounding box with different colors, please use the `visualize_dynamic_objects` function.

        Args:
            datum (:obj:`SynchronizedSensorData`): ArgoVerse synchronized sensor data.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        annotations = datum.annotations
        synchronized_imagery = datum.synchronized_imagery
        for idx, cam_name in enumerate(self.cam_list):
            cam = synchronized_imagery[cam_name]
            if annotations is not None:
                img_plotted = annotations.project_to_cam(
                    img=cam.img,
                    cam_model=cam.camera_model,
                )
            else:
                img_plotted = cam.img.copy()
            
            # save
            img_path = (
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/3dbox_vis/"
                + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
            )
            Image.fromarray(
                img_plotted[:, :, [2, 1, 0]]
            ).save(img_path)
            
    def visualize_dynamic_objects(
        self, scene_idx, lidar_indices,
        instances_info, frame_instances
    ):
        """DEBUG: Visualize the dynamic objects'box with different colors on the image.

        Args:
            scene_idx (str): Current file index.
            lidar_indices (list): List of lidar indices.
            instances_info (dict): Instances information.
            frame_instances (dict): Frame instances.
        """
        output_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/instances/debug_vis"
        
        print("Visualizing dynamic objects ...")
        for frame_idx, lidar_idx in tqdm(
            enumerate(lidar_indices), desc=f"Visualizing dynamic objects of scene {scene_idx} ...", total=len(lidar_indices), dynamic_ncols=True
        ):
            datum = self.av2loader[lidar_idx]
            synchronized_imagery = datum.synchronized_imagery
            for cam_name, cam in synchronized_imagery.items():
                cam_idx = self.cam_list.index(cam_name)
                img_path = (
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                    + f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.jpg"
                )
                canvas = np.array(Image.open(img_path))
                
                if frame_idx in frame_instances:
                    objects = frame_instances[frame_idx]
                    
                    if len(objects) == 0:
                        img_plotted = canvas
                    else:
                        lstProj2d = []
                        color_list = []
                        for obj_id in objects:
                            idx_in_obj = instances_info[obj_id]['frame_annotations']['frame_idx'].index(frame_idx)
                            o2w = np.array(
                                instances_info[obj_id]['frame_annotations']['obj_to_world'][idx_in_obj]
                            )
                            length, width, height = instances_info[obj_id]['frame_annotations']['box_size'][idx_in_obj]
                            half_dim_x, half_dim_y, half_dim_z = length/2.0, width/2.0, height/2.0
                            corners = np.array(
                                [[half_dim_x, half_dim_y, -half_dim_z],
                                [half_dim_x, -half_dim_y, -half_dim_z],
                                [-half_dim_x, -half_dim_y, -half_dim_z],
                                [-half_dim_x, half_dim_y, -half_dim_z],
                                [half_dim_x, half_dim_y, half_dim_z],
                                [half_dim_x, -half_dim_y, half_dim_z],
                                [-half_dim_x, -half_dim_y, half_dim_z],
                                [-half_dim_x, half_dim_y, half_dim_z]]
                            )
                            corners = (o2w[:3, :3] @ corners.T + o2w[:3, [3]]).T
                            v2w = datum.timestamp_city_SE3_ego_dict[cam.timestamp_ns]
                            w2v = v2w.inverse()
                            corners_in_ego = w2v.transform_point_cloud(corners)
                            
                            projected_points2d, _, ok = cam.camera_model.project_ego_to_img(
                                corners_in_ego # cuboid corners in ego frame
                            )
                            projected_points2d = projected_points2d.tolist()
                            if all(ok):
                                lstProj2d.append(projected_points2d)
                                color_list.append(color_mapper(obj_id))
                                
                        lstProj2d = np.asarray(lstProj2d)
                        img_plotted = dump_3d_bbox_on_image(coords=lstProj2d, img=canvas, color=color_list)
                
                img_path = (
                    f"{output_path}/"
                    + f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.jpg"
                )
                Image.fromarray(img_plotted).save(img_path)

    def save_dynamic_mask(self, datum: SynchronizedSensorData, scene_idx, frame_idx, class_valid='all'):
        """Parse and save the segmentation data.

        Args:
            datum (:obj:`SynchronizedSensorData`): ArgoVerse synchronized sensor data.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
            class_valid (str): Class valid for dynamic mask.
        """
        assert class_valid in ['all', 'human', 'vehicle'], "Invalid class valid"
        if class_valid == 'all':
            VALID_CLASSES = AV2_DYNAMIC_CLASSES
        elif class_valid == 'human':
            VALID_CLASSES = AV2_NONRIGID_DYNAMIC_CLASSES
        elif class_valid == 'vehicle':
            VALID_CLASSES = AV2_RIGID_DYNAMIC_CLASSES
        mask_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/dynamic_masks/{class_valid}"
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
            
        annotations = datum.annotations
        synchronized_imagery = datum.synchronized_imagery
        
        for cam_idx, cam_name in enumerate(self.cam_list):
            cam = synchronized_imagery[cam_name]
            H, W = cam.img.shape[:2]
            dynamic_mask = np.zeros_like(cam.img, dtype=np.float32)[..., 0]

            for cuboid_idx in range(len(annotations)):
                cuboid = annotations[cuboid_idx]
                if cuboid.category not in VALID_CLASSES:
                    continue
                
                uv, _, ok = cam.camera_model.project_ego_to_img(
                    cuboid.vertices_m # cuboid corners in ego frame
                )

                # Skip object if any corner projection failed. Note that this is very
                # strict and can lead to exclusion of some partially visible objects.
                if not all(ok):
                    continue
                u, v= uv[:, 0], uv[:, 1]
                u = u.astype(np.int32)
                v = v.astype(np.int32)

                # Clip box to image bounds.
                u = np.clip(u, 0, W - 1)
                v = np.clip(v, 0, H - 1)

                if u.max() - u.min() == 0 or v.max() - v.min() == 0:
                    continue

                # Draw projected 2D box onto the image.
                xy = (u.min(), v.min())
                width = u.max() - u.min()
                height = v.max() - v.min()
                # max pooling
                dynamic_mask[
                    int(xy[1]) : int(xy[1] + height),
                    int(xy[0]) : int(xy[0] + width),
                ] = np.maximum(
                    dynamic_mask[
                        int(xy[1]) : int(xy[1] + height),
                        int(xy[0]) : int(xy[0] + width),
                    ],
                    1.,
                )
            dynamic_mask = np.clip((dynamic_mask > 0.) * 255, 0, 255).astype(np.uint8)
            dynamic_mask = Image.fromarray(dynamic_mask, "L")
            dynamic_mask_path = os.path.join(mask_dir, f"{str(frame_idx).zfill(3)}_{str(cam_idx)}.png")
            dynamic_mask.save(dynamic_mask_path)
            
    def save_objects(self, lidar_indices: List[int]):
        """Parse and save the objects annotation data.
        
        Args:
            lidar_indices (list): List of lidar indices.
        """
        instances_info, frame_instances = {}, {}
        for frame_idx, lidar_idx in enumerate(lidar_indices):
            datum = self.av2loader[lidar_idx]
            annotations = datum.annotations
            sweep = datum.sweep
            timestamp_city_SE3_ego_dict = datum.timestamp_city_SE3_ego_dict
            
            frame_instances[frame_idx] = []
            for cuboid_idx in range(len(annotations)):
                cuboid = annotations[cuboid_idx]
                track_id, label = cuboid.track_uuid, cuboid.category
                if label not in AV2_DYNAMIC_CLASSES:
                    continue
                
                if track_id not in instances_info:
                    instances_info[track_id] = dict(
                        id=track_id,
                        class_name=label,
                        frame_annotations={
                            "frame_idx": [],
                            "obj_to_world": [],
                            "box_size": [],
                        }
                    )
                
                o2v = cuboid.dst_SE3_object.transform_matrix
                v2w = timestamp_city_SE3_ego_dict[sweep.timestamp_ns].transform_matrix
                # [object to  world] transformation matrix
                o2w = v2w @ o2v
                
                # Dimensions of the box. length: dim x. width: dim y. height: dim z.
                # length: dim_x: along heading; dim_y: verticle to heading; dim_z: verticle up
                dimension = [cuboid.length_m, cuboid.width_m, cuboid.height_m]
                
                instances_info[track_id]['frame_annotations']['frame_idx'].append(frame_idx)
                instances_info[track_id]['frame_annotations']['obj_to_world'].append(o2w.tolist())
                instances_info[track_id]['frame_annotations']['box_size'].append(dimension)
                
                frame_instances[frame_idx].append(track_id)

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
            if "pose" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/ego_pose", exist_ok=True)
            if "lidar" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/lidar", exist_ok=True)
            if "3dbox_vis" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/3dbox_vis", exist_ok=True)
            if "dynamic_masks" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/dynamic_masks", exist_ok=True)
            if "objects" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/instances", exist_ok=True)
            if "objects_vis" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{str(i).zfill(3)}/instances/debug_vis", exist_ok=True)
