import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from pandaset import DataSet as PandaSet, geometry
from pandaset.sequence import Sequence

from datasets.tools.multiprocess_utils import track_parallel_progress
from utils.visualization import color_mapper, dump_3d_bbox_on_image

PANDA_LABELS = [
    'Animals - Other', 'Bicycle', 'Bus', 'Car',
    'Cones', 'Construction Signs', 'Emergency Vehicle', 'Medium-sized Truck',
    'Motorcycle', 'Motorized Scooter', 'Other Vehicle - Construction Vehicle', 'Other Vehicle - Pedicab',
    'Other Vehicle - Uncommon', 'Pedestrian', 'Pedestrian with Object', 'Personal Mobility Device',
    'Pickup Truck', 'Pylons', 'Road Barriers', 'Rolling Containers',
    'Semi-truck', 'Signs', 'Temporary Construction Barriers', 'Towed Object',
    'Train', 'Tram / Subway'
]

PANDA_NONRIGID_DYNAMIC_CLASSES = [
    'Pedestrian', 'Pedestrian with Object', 'Bicycle', 'Animals - Other'
]

PANDA_RIGID_DYNAMIC_CLASSES = [
    'Bus', 'Car', 'Emergency Vehicle', 'Medium-sized Truck',
    'Motorcycle', 'Motorized Scooter', 'Other Vehicle - Construction Vehicle', 'Other Vehicle - Pedicab',
    'Other Vehicle - Uncommon', 'Personal Mobility Device', 'Pickup Truck',
    'Semi-truck', 'Train', 'Tram / Subway'
]

PANDA_DYNAMIC_CLASSES = PANDA_NONRIGID_DYNAMIC_CLASSES + PANDA_RIGID_DYNAMIC_CLASSES

class PandaSetProcessor(object):
    """Process PandaSet.

    Args:
        load_dir (str): Directory to load data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
            Defaults to False.
        save_cam_sync_labels (bool, optional): Whether to save cam sync labels.
            Defaults to True.
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

        # PandaSet Provides 6 cameras and 2 lidars
        self.cam_list = [          # {frame_idx}_{cam_id}.jpg
            "front_camera",        # "xxx_0.jpg"
            "front_left_camera",   # "xxx_1.jpg"
            "front_right_camera",  # "xxx_2.jpg"
            "left_camera",         # "xxx_3.jpg"
            "right_camera",        # "xxx_4.jpg"
            "back_camera"          # "xxx_5.jpg"
        ]
        # 0: mechanical 360° LiDAR, 1: front-facing LiDAR, -1: All LiDARs
        self.lidar_list = [-1]

        self.load_dir = load_dir
        self.save_dir = f"{save_dir}"
        self.workers = int(workers)
        self.pandaset = PandaSet(load_dir)
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

    def convert_one(self, scene_idx):
        """Convert action for single file.

        Args:
            scene_idx (str): Scene index.
        """
        scene_data = self.pandaset[scene_idx]
        scene_data.load()
        num_frames = sum(1 for _ in scene_data.timestamps)
        for frame_idx in tqdm(range(num_frames), desc=f"File {scene_idx}", total=num_frames, dynamic_ncols=True):
            if "images" in self.process_keys:
                self.save_image(scene_data, scene_idx, frame_idx)
            if "calib" in self.process_keys:
                self.save_calib(scene_data, scene_idx, frame_idx)
            if "lidar" in self.process_keys:
                self.save_lidar(scene_data, scene_idx, frame_idx)
            if "pose" in self.process_keys:
                self.save_pose(scene_data, scene_idx, frame_idx)
            if "3dbox_vis" in self.process_keys:
                # visualize 3d box, debug usage
                self.visualize_3dbox(scene_data, scene_idx, frame_idx)
            if "dynamic_masks" in self.process_keys:
                self.save_dynamic_mask(scene_data, scene_idx, frame_idx, class_valid='all')
                self.save_dynamic_mask(scene_data, scene_idx, frame_idx, class_valid='human')
                self.save_dynamic_mask(scene_data, scene_idx, frame_idx, class_valid='vehicle')
                
        # save instances info
        if "objects" in self.process_keys:
            instances_info = self.save_objects(scene_data, num_frames)
            
            # solve duplicated objects from different lidars
            duplicated_id_pairs = []
            for k, v in instances_info.items():
                if v["sibling_id"] != '-':
                    # find if the pair is already in the list
                    if (v["id"], v["sibling_id"]) in duplicated_id_pairs or (v["sibling_id"], v["id"]) in duplicated_id_pairs:
                        continue
                    else:
                        duplicated_id_pairs.append((v["id"], v["sibling_id"]))
            
            for pair in duplicated_id_pairs:
                # check if all in the pair are in the instances_info
                if pair[0] not in instances_info:
                    # print(f"WARN: {pair[0]} not in instances_info")
                    continue
                elif pair[1] not in instances_info:
                    # print(f"WARN: {pair[1]} not in instances_info")
                    continue
                else:
                    # keep the longer one in pairs
                    if len(instances_info[pair[0]]['frame_annotations']['frame_idx']) > \
                        len(instances_info[pair[1]]['frame_annotations']['frame_idx']):
                        instances_info.pop(pair[1])
                    else:
                        instances_info.pop(pair[0])
            
            # rough filter stationary objects
            # if all the annotations of an object are stationary, remove it
            static_ids = []
            for k, v in instances_info.items():
                if all(v['frame_annotations']['stationary']):
                    static_ids.append(v['id'])
            print(f"INFO: {len(static_ids)} static objects removed")
            for static_id in static_ids:
                instances_info.pop(static_id)
            print(f"INFO: Final number of objects: {len(instances_info)}")
            
            frame_instances = {}
            # update frame_instances
            for frame_idx in range(num_frames):
                # must ceate a object for each frame
                frame_instances[frame_idx] = []
                for k, v in instances_info.items():
                    if frame_idx in v['frame_annotations']['frame_idx']:
                        frame_instances[frame_idx].append(v["id"])
            
            # verbose: visualize the instances on the image (Debug Usage)
            if "objects_vis" in self.process_keys:
                self.visualize_dynamic_objects(
                    scene_data, scene_idx,
                    instances_info=instances_info,
                    frame_instances=frame_instances
                )
            
            # correct id
            id_map = {}
            for i, (k, v) in enumerate(instances_info.items()):
                id_map[v["id"]] = i
            # update keys in instances_info
            new_instances_info = {}
            for k, v in instances_info.items():
                new_instances_info[id_map[v["id"]]] = v
            # update keys in frame_instances
            new_frame_instances = {}
            for k, v in frame_instances.items():
                new_frame_instances[k] = [id_map[i] for i in v]
                
            # write as json
            instances_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/instances"
            with open(f"{instances_dir}/instances_info.json", "w") as fp:
                json.dump(new_instances_info, fp, indent=4)
            with open(f"{instances_dir}/frame_instances.json", "w") as fp:
                json.dump(new_frame_instances, fp, indent=4)

    def __len__(self):
        """Length of the filename list."""
        return len(self.process_id_list)

    def save_image(self, scene_data: Sequence, scene_idx, frame_idx):
        """Parse and save the images in jpg format.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        for idx, cam in enumerate(self.cam_list):
            img_path = (
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
            )
            # write PIL Image to jpg
            image = scene_data.camera[cam][frame_idx]
            image.save(img_path)

    def save_calib(self, scene_data: Sequence, scene_idx, frame_idx):
        """Parse and save the calibration data.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        for idx, cam in enumerate(self.cam_list):
            camera = scene_data.camera[cam]
            poses = camera.poses[frame_idx]
            c2w = geometry._heading_position_to_mat(poses['heading'], poses['position'])
            K = camera.intrinsics
            intrinsics = [K.fx, K.fy, K.cx, K.cy, 0.0, 0.0, 0.0, 0.0, 0.0] 
    
            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/extrinsics/"
                + f"{str(frame_idx).zfill(3)}_{str(idx)}.txt",
                c2w,
            )
            np.savetxt(
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/intrinsics/"
                + f"{str(idx)}.txt",
                intrinsics,
            )

    def save_lidar(self, scene_data: Sequence, scene_idx, frame_idx):
        """Parse and save the lidar data in psd format.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        pc_world = scene_data.lidar[frame_idx].to_numpy()
        # index        x           y         z        i         t       d                                                     
        # 0       -75.131138  -79.331690  3.511804   7.0  1.557540e+09  0
        # 1      -112.588306 -118.666002  1.423499  31.0  1.557540e+09  0
        # - `i`: `float`: Reflection intensity in a range `[0,255]`
        # - `t`: `float`: Recorded timestamp for specific point
        # - `d`: `int`: Sensor ID. `0` -> mechnical 360° LiDAR, `1` -> forward-facing LiDAR
        lidar_poses = scene_data.lidar.poses[frame_idx]

        # save lidar pts in ego coordinate system
        pcd_ego = geometry.lidar_points_to_ego(
            pc_world[:, :3], lidar_poses
        )
        intensity = pc_world[:, 3]
        laser_ids = pc_world[:, 5]
        
        point_cloud = np.column_stack(
            (
                pcd_ego,
                intensity,
                laser_ids
            )
        )
        
        pc_path = (
            f"{self.save_dir}/"
            + f"{str(scene_idx).zfill(3)}/lidar/{str(frame_idx).zfill(3)}.bin"
        )
        point_cloud.astype(np.float32).tofile(pc_path)

    def save_pose(self, scene_data: Sequence, scene_idx, frame_idx):
        """Parse and save the pose data.

        Since pandaset does not provide the ego pose, we use the lidar pose as the ego pose.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        lidar_poses = scene_data.lidar.poses[frame_idx]
        lidar_to_world = geometry._heading_position_to_mat(lidar_poses['heading'], lidar_poses['position'])

        np.savetxt(
            f"{self.save_dir}/{str(scene_idx).zfill(3)}/ego_pose/"
            + f"{str(frame_idx).zfill(3)}.txt",
            lidar_to_world,
        )
        
    def visualize_3dbox(self, scene_data: Sequence, scene_idx, frame_idx):
        """DEBUG: Visualize the 3D bounding box on the image.
        Visualize the 3D bounding box all with the same COLOR.
        If you want to visualize the 3D bounding box with different colors, please use the `visualize_dynamic_objects` function.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
        """
        for idx, cam in enumerate(self.cam_list):
            img_path = (
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
            )
            canvas = np.array(Image.open(img_path))
    
            camera = scene_data.camera[cam]
            cuboids = scene_data.cuboids[frame_idx]
            
            lstProj2d = []
            recorded_id = []
            for _, row in cuboids.iterrows():
                if row["label"] not in PANDA_DYNAMIC_CLASSES or row["stationary"]:
                    continue
                if not row["cuboids.sensor_id"] == -1:
                    if row["cuboids.sibling_id"] in recorded_id:
                        continue
                recorded_id.append(row["uuid"])
                box = [
                        row[  "position.x"], row[  "position.y"], row[  "position.z"],
                        row["dimensions.x"], row["dimensions.y"], row["dimensions.z"],
                        row["yaw"]
                    ]
                corners = geometry.center_box_to_corners(box)
                
                projected_points2d, _, _ = geometry.projection(
                    lidar_points=corners,                
                    camera_data=camera[frame_idx],
                    camera_pose=camera.poses[frame_idx],
                    camera_intrinsics=camera.intrinsics,
                    filter_outliers=True
                )
                projected_points2d = projected_points2d.tolist()
                if len(projected_points2d) == 8:
                    lstProj2d.append(projected_points2d)
            
            lstProj2d = np.asarray(lstProj2d)
            
            img_plotted = dump_3d_bbox_on_image(coords=lstProj2d, img=canvas, color=(255,0,0))
            
            # save
            img_path = (
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/3dbox_vis/"
                + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
            )
            Image.fromarray(img_plotted).save(img_path)
            
    def visualize_dynamic_objects(
        self, scene_data: Sequence, scene_idx,
        instances_info: dict, frame_instances: dict
    ):
        """DEBUG: Visualize the dynamic objects'box with different colors on the image.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            instances_info (dict): Instances information.
            frame_instances (dict): Frame instances.
        """
        output_path = f"{self.save_dir}/{str(scene_idx).zfill(3)}/instances/debug_vis"
        
        num_frames = sum(1 for _ in scene_data.timestamps)
        for frame_idx in range(num_frames):
            for idx, cam in enumerate(self.cam_list):
                img_path = (
                    f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                    + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
                )
                canvas = np.array(Image.open(img_path))
                
                camera = scene_data.camera[cam]
                if frame_idx in frame_instances:
                    objects = frame_instances[frame_idx]
                    
                    lstProj2d = []
                    color_list = []
                    for obj_id in objects:
                        idx_in_obj = instances_info[obj_id]['frame_annotations']['frame_idx'].index(frame_idx)
                        o2w = instances_info[obj_id]['frame_annotations']['obj_to_world'][idx_in_obj]
                        o2w = np.array(o2w)
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
                        
                        projected_points2d, _, _ = geometry.projection(
                            lidar_points=corners,                
                            camera_data=camera[frame_idx],
                            camera_pose=camera.poses[frame_idx],
                            camera_intrinsics=camera.intrinsics,
                            filter_outliers=True
                        )
                        projected_points2d = projected_points2d.tolist()
                        if len(projected_points2d) == 8:
                            lstProj2d.append(projected_points2d)
                            color_list.append(color_mapper(obj_id))
                            
                    lstProj2d = np.asarray(lstProj2d)
                    img_plotted = dump_3d_bbox_on_image(coords=lstProj2d, img=canvas, color=color_list)
                
                img_path = (
                    f"{output_path}/"
                    + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
                )
                Image.fromarray(img_plotted).save(img_path)

    def save_dynamic_mask(self, scene_data: Sequence, scene_idx, frame_idx, class_valid='all'):
        """Parse and save the segmentation data.

        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            scene_idx (str): Current file index.
            frame_idx (int): Current frame index.
            class_valid (str): Class valid for dynamic mask.
        """
        assert class_valid in ['all', 'human', 'vehicle'], "Invalid class valid"
        if class_valid == 'all':
            VALID_CLASSES = PANDA_DYNAMIC_CLASSES
        elif class_valid == 'human':
            VALID_CLASSES = PANDA_NONRIGID_DYNAMIC_CLASSES
        elif class_valid == 'vehicle':
            VALID_CLASSES = PANDA_RIGID_DYNAMIC_CLASSES
        mask_dir = f"{self.save_dir}/{str(scene_idx).zfill(3)}/dynamic_masks/{class_valid}"
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
            
        for idx, cam in enumerate(self.cam_list):
            # dynamic_mask
            img_path = (
                f"{self.save_dir}/{str(scene_idx).zfill(3)}/images/"
                + f"{str(frame_idx).zfill(3)}_{str(idx)}.jpg"
            )
            img_shape = np.array(Image.open(img_path))
            dynamic_mask = np.zeros_like(img_shape, dtype=np.float32)[..., 0]

            camera = scene_data.camera[cam]
            cuboids = scene_data.cuboids[frame_idx]
            
            recorded_id = []
            for _, row in cuboids.iterrows():
                if row["label"] not in VALID_CLASSES or row["stationary"]:
                    continue
                if not row["cuboids.sensor_id"] == -1:
                    if row["cuboids.sibling_id"] in recorded_id:
                        continue
                recorded_id.append(row["uuid"])

                box = [
                        row[  "position.x"], row[  "position.y"], row[  "position.z"],
                        row["dimensions.x"], row["dimensions.y"], row["dimensions.z"],
                        row["yaw"]
                    ]
                corners = geometry.center_box_to_corners(box)
                
                projected_points2d, _, _ = geometry.projection(
                    lidar_points=corners,                
                    camera_data=camera[frame_idx],
                    camera_pose=camera.poses[frame_idx],
                    camera_intrinsics=camera.intrinsics,
                    filter_outliers=True
                )
                # Skip object if any corner projection failed. Note that this is very
                # strict and can lead to exclusion of some partially visible objects.
                if not len(projected_points2d) == 8:
                    continue
                u, v= projected_points2d[:, 0], projected_points2d[:, 1]
                u = u.astype(np.int32)
                v = v.astype(np.int32)

                # Clip box to image bounds.
                u = np.clip(u, 0, camera[frame_idx].size[0])
                v = np.clip(v, 0, camera[frame_idx].size[1])

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
            dynamic_mask_path = os.path.join(mask_dir, f"{str(frame_idx).zfill(3)}_{str(idx)}.png")
            dynamic_mask.save(dynamic_mask_path)
            
    def save_objects(self, scene_data: Sequence, num_frames):
        """Parse and save the objects annotation data.
        
        Args:
            scene_data (:obj:`Sequence`): PandaSet sequence.
            num_frames (int): Number of frames.
        """
        instances_info = {}
        
        for frame_idx in range(num_frames):
            cuboids = scene_data.cuboids[frame_idx]
            for _, row in cuboids.iterrows():
                str_id = row["uuid"]
                label = row["label"]
                if label not in PANDA_DYNAMIC_CLASSES:
                    continue
                
                if str_id not in instances_info:
                    instances_info[str_id] = dict(
                        id=str_id,
                        class_name=row["label"],
                        sibling_id=row["cuboids.sibling_id"],
                        frame_annotations={
                            "frame_idx": [],
                            "obj_to_world": [],
                            "box_size": [],
                            "stationary": [],
                        }
                    )
                
                # Box coordinates in vehicle frame.
                tx, ty, tz = row["position.x"], row["position.y"], row["position.z"]
                
                # The heading of the bounding box (in radians).  The heading is the angle
                #   required to rotate +x to the surface normal of the box front face. It is
                #   normalized to [-pi, pi).
                c = np.math.cos(row["yaw"])
                s = np.math.sin(row["yaw"])
                
                # [object to  world] transformation matrix
                o2w = np.array([
                    [ c, -s,  0, tx],
                    [ s,  c,  0, ty],
                    [ 0,  0,  1, tz],
                    [ 0,  0,  0,  1]])
                
                # Dimensions of the box. length: dim x. width: dim y. height: dim z.
                # length: dim_x: along heading; dim_y: verticle to heading; dim_z: verticle up
                dimension = [row["dimensions.x"], row["dimensions.y"], row["dimensions.z"]]
                
                instances_info[str_id]['frame_annotations']['frame_idx'].append(frame_idx)
                instances_info[str_id]['frame_annotations']['obj_to_world'].append(o2w.tolist())
                instances_info[str_id]['frame_annotations']['box_size'].append(dimension)
                instances_info[str_id]['frame_annotations']['stationary'].append(row["stationary"])
        
        return instances_info

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
