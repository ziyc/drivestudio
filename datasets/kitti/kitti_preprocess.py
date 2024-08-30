import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pykitti
from PIL import Image

from datasets.kitti.trackletparser import parseXML
from datasets.tools.multiprocess_utils import track_parallel_progress
from utils.geometry import get_corners, project_camera_points_to_image
from utils.visualization import color_mapper, dump_3d_bbox_on_image

KITTI_LABELS = [
    'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]

KITTI_NONRIGID_DYNAMIC_CLASSES = [
    'Pedestrian', 'Person_sitting', 'Cyclist'
]

KITTI_RIGID_DYNAMIC_CLASSES = [
    'Car', 'Van', 'Truck', 'Tram'
]

KITTI_DYNAMIC_CLASSES = KITTI_NONRIGID_DYNAMIC_CLASSES + KITTI_RIGID_DYNAMIC_CLASSES

class KittiProcessor(object):
    """Process KITTI dataset."""
    
    def __init__(
        self,
        load_dir: str,
        save_dir: str,
        process_keys: List[str] = [
            "images",
            "lidar",
            "calib",
            "dynamic_masks",
            "objects"
        ],
        prefix: str = "2011_09_26",
        process_id_list: List[str] = None,
        workers: int = 64,
    ):
        self.process_id_list = process_id_list
        self.process_keys = process_keys
        self.HW = (375, 1242)
        print("will process keys: ", self.process_keys)

        self.cam_list = [
            "CAM_LEFT",     # "xxx_0.jpg"
            "CAM_RIGHT"     # "xxx_1.jpg"
        ]

        self.split_dir = os.path.join(load_dir, prefix)
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.workers = int(workers)
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
        
    def get_kitti_data(self, basedir):
        date = basedir.split("/")[-2]
        drive = basedir.split("/")[-1].split('_')[-2]
        data = pykitti.raw(self.load_dir, date, drive)
        return data

    def convert_one(self, scene_name: str):
        """Convert action for single file."""
        basedir = os.path.join(self.split_dir, scene_name)
        kitti_data = self.get_kitti_data(basedir)
        if "images" in self.process_keys:
            self.save_image(kitti_data, scene_name)
        if "calib" in self.process_keys:
            self.save_calib(kitti_data, scene_name)
        if "pose" in self.process_keys:
            self.save_pose(kitti_data, scene_name)
        if "lidar" in self.process_keys:
            self.save_lidar(kitti_data, scene_name)
            
        tracklet_file = os.path.join(basedir, 'tracklet_labels.xml')
        tracklets = parseXML(tracklet_file)
        if "dynamic_masks" in self.process_keys:
            self.save_dynamic_mask(kitti_data, tracklets, scene_name, class_valid='all')
            self.save_dynamic_mask(kitti_data, tracklets, scene_name, class_valid='human')
            self.save_dynamic_mask(kitti_data, tracklets, scene_name, class_valid='vehicle')
        
        # process annotated objects
        if "objects" in self.process_keys:
            instances_info, frame_instances = self.save_objects(kitti_data, tracklets)
            
            # Save instances info and frame instances
            instances_info_save_path = f"{self.save_dir}/{str(scene_name).zfill(3)}/instances"
            os.makedirs(instances_info_save_path, exist_ok=True)
            with open(f"{instances_info_save_path}/instances_info.json", "w") as fp:
                json.dump(instances_info, fp, indent=4)
            with open(f"{instances_info_save_path}/frame_instances.json", "w") as fp:
                json.dump(frame_instances, fp, indent=4)
                
            if "objects_vis" in self.process_keys:
                objects_vis_path = f"{self.save_dir}/{str(scene_name).zfill(3)}/instances/debug_vis"
                if not os.path.exists(objects_vis_path):
                    os.makedirs(objects_vis_path)
                self.visualize_dynamic_objects(kitti_data, scene_name, objects_vis_path, instances_info, frame_instances)

    def __len__(self):
        """Length of the filename list."""
        return len(self.process_id_list) if self.process_id_list else 1

    def save_image(self, kitti_data, scene_name: str):
        for frame_idx in range(len(kitti_data)):
            images = kitti_data.get_rgb(frame_idx)
            # Save left and right images
            images[0].save(f"{self.save_dir}/{scene_name}/images/{str(frame_idx).zfill(3)}_0.jpg")
            images[1].save(f"{self.save_dir}/{scene_name}/images/{str(frame_idx).zfill(3)}_1.jpg")

    def save_calib(self, kitti_data, scene_name: str):
        # NOTE: assume ego vehicle is the same as velodyne frame        
        np.savetxt(
            f"{self.save_dir}/{scene_name}/extrinsics/0.txt",
            np.linalg.inv(kitti_data.calib.T_cam2_velo)
        )
        np.savetxt(
            f"{self.save_dir}/{scene_name}/extrinsics/1.txt",
            np.linalg.inv(kitti_data.calib.T_cam3_velo)
        )
        
        cam2_Ks = kitti_data.calib.K_cam2
        cam3_Ks = kitti_data.calib.K_cam3
        # fx, fy, cx, cy, p1, p2, k1, k2, k3
        Ks_left = np.array([cam2_Ks[0, 0], cam2_Ks[1, 1], cam2_Ks[0, 2], cam2_Ks[1, 2], 0, 0, 0, 0, 0])
        Ks_right = np.array([cam3_Ks[0, 0], cam3_Ks[1, 1], cam3_Ks[0, 2], cam3_Ks[1, 2], 0, 0, 0, 0, 0])
        np.savetxt(
            f"{self.save_dir}/{scene_name}/intrinsics/0.txt",
            Ks_left
        )
        np.savetxt(
            f"{self.save_dir}/{scene_name}/intrinsics/1.txt",
            Ks_right
        )

    def save_pose(self, kitti_data, scene_name: str):
        # NOTE: we assume the ego pose is the same as the velodyne pose
        for frame_idx in range(len(kitti_data)):
            imu2world = kitti_data.oxts[frame_idx].T_w_imu
            imu2velo = kitti_data.calib.T_velo_imu
            velo2world = imu2world @ imu2velo
            np.savetxt(
                f"{self.save_dir}/{scene_name}/ego_pose/{str(frame_idx).zfill(3)}.txt",
                velo2world
            )

    def save_lidar(self, kitti_data, scene_name: str):
        for frame_idx in range(len(kitti_data)):
            # Points are already in ego frame (velodyne frame), so we don't need to transform them
            points = kitti_data.get_velo(frame_idx)
            
            # Save lidar points
            lidar_save_path = f"{self.save_dir}/{scene_name}/lidar/{str(frame_idx).zfill(3)}.bin"
            points.astype(np.float32).tofile(lidar_save_path)
    
    def save_dynamic_mask(self, kitti_data, tracklets, scene_name: str, class_valid: str):
        # Define valid classes based on class_valid parameter
        if class_valid == 'all':
            valid_classes = KITTI_DYNAMIC_CLASSES
        elif class_valid == 'human':
            valid_classes = KITTI_NONRIGID_DYNAMIC_CLASSES
        elif class_valid == 'vehicle':
            valid_classes = KITTI_RIGID_DYNAMIC_CLASSES
        else:
            raise ValueError("Invalid class_valid parameter")
        
        for frame_idx in range(len(kitti_data)):
            for cam_idx, cam_name in enumerate(self.cam_list):
                img_path = f"{self.save_dir}/{scene_name}/images/{str(frame_idx).zfill(3)}_{str(cam_idx)}.jpg"

                img = cv2.imread(img_path)
                dynamic_mask = np.zeros(img.shape[:2], dtype=np.float32)
                
                for object in tracklets:
                    if frame_idx < object.firstFrame or frame_idx >= object.firstFrame + object.nFrames:
                        continue
                    
                    if object.objectType not in valid_classes:
                        continue

                    obj_step = frame_idx - object.firstFrame
                    t_obj = np.array(object.trans[obj_step])
                    roty = np.array(object.rots[obj_step][2])
                    
                    # Create 3D bounding box
                    h, w, l = object.size
                    corners_3d_local = get_corners(l=l, w=w, h=h)

                    # Rotate and translate 3D bounding box
                    c = np.cos(roty)
                    s = np.sin(roty)
                    tx, ty, tz = t_obj
                    tz += h / 2 # NOTE: objects are annotated at the bottom center in KITTI dataset
                    o2v = np.array([
                        [ c, -s,  0, tx],
                        [ s,  c,  0, ty],
                        [ 0,  0,  1, tz],
                        [ 0,  0,  0,  1]
                    ])
                    corners_3d_velo = o2v[:3, :3] @ corners_3d_local + o2v[:3, 3:4]
                    
                    # Project 3D bounding box to 2D
                    if cam_name == "CAM_LEFT":
                        velo2cam = kitti_data.calib.T_cam2_velo
                        intrinsics_3X3 = kitti_data.calib.K_cam2
                    elif cam_name == "CAM_RIGHT":
                        velo2cam = kitti_data.calib.T_cam3_velo
                        intrinsics_3X3 = kitti_data.calib.K_cam3
                    corners_3d_cam = velo2cam[:3, :3] @ corners_3d_velo + velo2cam[:3, 3:4]
                    corners_2d, _ = project_camera_points_to_image(corners_3d_cam.T, intrinsics_3X3)

                    # Check if the all object's corners are in the image
                    # NOTE: we use strict visibility check here, requiring all corners to be visible
                    in_image = np.all(corners_2d[0, :] >= 0) & np.all(corners_2d[0, :] < img.shape[1]) & \
                            np.all(corners_2d[1, :] >= 0) & np.all(corners_2d[1, :] < img.shape[0])
                    if not in_image:
                        continue

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

                # Save the mask
                dynamic_mask = np.clip((dynamic_mask > 0.) * 255, 0, 255).astype(np.uint8)
                dynamic_mask = Image.fromarray(dynamic_mask, "L")
                dynamic_mask_path = f"{self.save_dir}/{scene_name}/dynamic_masks/{class_valid}/{str(frame_idx).zfill(3)}_{cam_idx}.png"
                dynamic_mask.save(dynamic_mask_path)

    def save_objects(self, kitti_data, tracklets) -> Tuple[Dict, Dict]:
        instances_info = {}
        frame_instances = {i: [] for i in range(len(kitti_data))}
        for obj_id, object in enumerate(tracklets):
            if object.objectType not in KITTI_DYNAMIC_CLASSES:
                continue
                    
            instances_info[obj_id] = {
                "id": obj_id,
                "class_name": object.objectType,
                "frame_annotations": {
                    "frame_idx": [],
                    "obj_to_world": [],
                    "box_size": []
                }
            }

            for frame_idx in range(object.firstFrame, object.firstFrame + object.nFrames):
                if frame_idx >= len(kitti_data):
                    break

                obj_step = frame_idx - object.firstFrame
                t_obj = np.array(object.trans[obj_step])
                roty = np.array(object.rots[obj_step][2])

                # Create transformation matrix
                c = np.cos(roty)
                s = np.sin(roty)
                tx, ty, tz = t_obj
                h, _, _ = object.size
                tz += h / 2 # NOTE: objects are annotated at the bottom center in KITTI dataset
                o2v = np.array([
                    [ c, -s,  0, tx],
                    [ s,  c,  0, ty],
                    [ 0,  0,  1, tz],
                    [ 0,  0,  0,  1]])

                # Transform to world coordinates
                imu2world = kitti_data.oxts[frame_idx].T_w_imu
                imu2velo = kitti_data.calib.T_velo_imu
                velo2world = imu2world @ imu2velo
                obj_to_world = velo2world @ o2v
                
                # convert hwl to lwh
                lwh = [object.size[2], object.size[1], object.size[0]]

                instances_info[obj_id]["frame_annotations"]["frame_idx"].append(frame_idx)
                instances_info[obj_id]["frame_annotations"]["obj_to_world"].append(obj_to_world.tolist())
                instances_info[obj_id]["frame_annotations"]["box_size"].append(lwh)

                frame_instances[frame_idx].append(obj_id)
        
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
    
    def visualize_dynamic_objects(self, kitti_data, scene_name:str, objects_vis_path: str, instances_info: Dict, frame_instances: Dict):
        for frame_idx, obj_ids in frame_instances.items():
            for cam_idx, cam_name in enumerate(self.cam_list):
                # Load the image
                img_path = os.path.join(self.save_dir, scene_name, 'images', f'{frame_idx:03d}_{cam_idx}.jpg')
                canvas = np.array(Image.open(img_path))

                if len(obj_ids) == 0:
                    img_plotted = canvas
                else:
                    lstProj2d = []
                    color_list = []
                    for obj_id in obj_ids:
                        obj_info = instances_info[obj_id]
                        frame_ann_idx = obj_info['frame_annotations']['frame_idx'].index(frame_idx)
                        
                        # Get object pose and size
                        obj_to_world = np.array(obj_info['frame_annotations']['obj_to_world'][frame_ann_idx])
                        box_size = np.array(obj_info['frame_annotations']['box_size'][frame_ann_idx])

                        # Create 3D bounding box in object coordinates
                        l, w, h = box_size
                        corners = get_corners(l=l, w=w, h=h)

                        # Transform box to world coordinates
                        corners_3d_world = obj_to_world[:3, :3] @ corners + obj_to_world[:3, 3:4]
                        
                        # world to velo
                        imu2world = kitti_data.oxts[frame_idx].T_w_imu
                        imu2velo = kitti_data.calib.T_velo_imu
                        world2velo = np.linalg.inv(imu2world @ imu2velo)
                        corners_3d_velo = world2velo[:3, :3] @ corners_3d_world + world2velo[:3, 3:4]
                        
                        # velo to cam
                        if cam_name == "CAM_LEFT":
                            velo2cam = kitti_data.calib.T_cam2_velo
                            intrinsics_3X3 = kitti_data.calib.K_cam2
                        elif cam_name == "CAM_RIGHT":
                            velo2cam = kitti_data.calib.T_cam3_velo
                            intrinsics_3X3 = kitti_data.calib.K_cam3
                        corners_3d_cam = velo2cam[:3, :3] @ corners_3d_velo + velo2cam[:3, 3:4]

                        # Project to image plane
                        corners_2d, _ = project_camera_points_to_image(corners_3d_cam.T, intrinsics_3X3)
                            
                        # Check if the all object's corners are in the image
                        in_image = np.all(corners_2d[0, :] >= 0) & np.all(corners_2d[0, :] < canvas.shape[1]) & \
                                np.all(corners_2d[1, :] >= 0) & np.all(corners_2d[1, :] < canvas.shape[0])
                        if in_image:
                            projected_points2d = corners_2d[:2, :].T
                            lstProj2d.append(projected_points2d)
                            color_list.append(color_mapper(str(obj_id)))

                    # Draw all bounding boxes at once
                    img_plotted = dump_3d_bbox_on_image(coords=np.array(lstProj2d), img=canvas, color=color_list)

                # Save the visualized image
                out_path = os.path.join(objects_vis_path, f'{frame_idx:06d}_{cam_idx}.jpg')
                Image.fromarray(img_plotted).save(out_path)

    def create_folder(self):
        """Create folder for data preprocessing."""
        if self.process_id_list is None:
            id_list = range(len(self))
        else:
            id_list = self.process_id_list
        for scene_name in id_list:
            os.makedirs(f"{self.save_dir}/{scene_name}/images", exist_ok=True)
            os.makedirs(f"{self.save_dir}/{scene_name}/extrinsics", exist_ok=True)
            os.makedirs(f"{self.save_dir}/{scene_name}/intrinsics", exist_ok=True)
            os.makedirs(f"{self.save_dir}/{scene_name}/sky_masks", exist_ok=True)
            os.makedirs(f"{self.save_dir}/{scene_name}/ego_pose", exist_ok=True)
            if "lidar" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{scene_name}/lidar", exist_ok=True)
            if "dynamic_masks" in self.process_keys:
                os.makedirs(f"{self.save_dir}/{scene_name}/dynamic_masks/all", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{scene_name}/dynamic_masks/human", exist_ok=True)
                os.makedirs(f"{self.save_dir}/{scene_name}/dynamic_masks/vehicle", exist_ok=True)