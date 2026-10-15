"""PhysicalAI sources for the regular DrivingDataset training/evaluation path."""

import json
import logging
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from pytorch3d.transforms import matrix_to_quaternion

from datasets.dataset_meta import DATASETS_CONFIG
from datasets.base.pixel_source import CameraData, ScenePixelSource
from datasets.base.lidar_source import SceneLidarSource
from datasets.base.scene_dataset import ModelType
from utils.geometry import (
    camera_model_rays,
    camera_midpoint_pose,
    project_camera_model,
    project_world_camera,
    scale_ftheta_calibration,
)

logger = logging.getLogger(__name__)
OBJECT_CLASS_NODE_MAPPING = {
    'Vehicle': ModelType.RigidNodes,
    'Pedestrian': ModelType.SMPLNodes,
    'Cyclist': ModelType.DeformableNodes,
    'RigidObject': ModelType.RigidNodes,
}


class PhysicalAICameraData(CameraData):
    def __init__(self, rectified_fov=None, **kwargs):
        root = Path(kwargs['data_path'])
        meta = json.loads((root / 'metadata.json').read_text())
        cam_id = int(kwargs['cam_id'])
        self.camera_metadata = meta['cameras'][str(cam_id)]
        self.camera_metadata['camera_name'] = DATASETS_CONFIG['physicalai'][cam_id]['camera_name']
        self.rectified_fov = rectified_fov
        self._ray_cache = {}
        self.dynamic_masks = self.human_masks = self.vehicle_masks = self.sky_masks = None
        super().__init__(**kwargs)

    def create_all_filelist(self):
        super().create_all_filelist()
        self.img_filepaths = np.array(
            [str(Path(p).with_suffix('.png')) for p in self.img_filepaths]
        )

    @classmethod
    def get_camera2worlds(cls, data_path, cam_id, start_timestep, end_timestep):
        root = Path(data_path)
        anchor = np.linalg.inv(np.loadtxt(root / 'ego_pose' / f'{start_timestep:03d}.txt'))
        poses = []
        for t in range(start_timestep, end_timestep):
            start = np.loadtxt(root / 'camera_pose_start' / f'{t:03d}_{cam_id}.txt')
            end = np.loadtxt(root / 'camera_pose' / f'{t:03d}_{cam_id}.txt')
            poses.append(anchor @ camera_midpoint_pose(start, end))
        return torch.from_numpy(np.stack(poses)).float()

    def load_calibrations(self):
        root = Path(self.data_path)
        self.ftheta_parameters = scale_ftheta_calibration(
            json.loads((root / 'intrinsics' / f'{self.cam_id}.json').read_text()),
            self.WIDTH,
            self.HEIGHT,
        )
        args = self.data_path, self.cam_id, self.start_timestep, self.end_timestep
        self.cam_to_worlds = self.get_camera2worlds(*args)
        K = torch.eye(3)
        K[:2, 2] = torch.tensor(self.ftheta_parameters['principal_point']) + 0.5
        self.camera_model = 'ftheta'
        self.rectification = None
        if self.undistort:
            fov = self.rectified_fov
            if fov is None:
                fov = min(100.0, np.degrees(float(self.ftheta_parameters['max_angle'])) * 1.6)
            if not 0 < fov < 180:
                raise ValueError('rectified_fov must lie between 0 and 180 degrees')
            K[0, 0] = K[1, 1] = self.WIDTH / (2 * np.tan(np.radians(fov) / 2))
            K[:2, 2] = torch.tensor([self.WIDTH / 2, self.HEIGHT / 2])
            rays, _ = camera_model_rays(self.HEIGHT, self.WIDTH, K)
            pixels, valid = project_camera_model(
                rays, K, 'ftheta', ftheta_parameters=self.ftheta_parameters
            )
            valid &= (
                (pixels[..., 0] >= 0.5)
                & (pixels[..., 0] <= self.WIDTH - 0.5)
                & (pixels[..., 1] >= 0.5)
                & (pixels[..., 1] <= self.HEIGHT - 0.5)
            )
            self.rectification = (pixels.numpy().astype(np.float32) - 0.5, valid)
            self.camera_model = 'pinhole'
        self.intrinsics = K.repeat(len(self.cam_to_worlds), 1, 1)
        self.distortions = None

    def _read(self, path, mask=False):
        image = Image.open(path).convert('L' if mask else 'RGB')
        pixels = np.asarray(
            image.resize(
                (self.WIDTH, self.HEIGHT),
                Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR,
            )
        )
        if self.rectification is not None:
            grid, _ = self.rectification
            pixels = cv2.remap(
                pixels,
                grid[..., 0],
                grid[..., 1],
                cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        return torch.from_numpy(pixels.copy())

    def load_images(self):
        self.images = torch.stack([self._read(p) for p in self.img_filepaths]).float() / 255

    def load_egocar_mask(self):
        self.egocar_mask = None
        self.invalid_masks = torch.stack(
            [
                self._read(
                    Path(self.data_path) / 'invalid_masks' / f'{t:03d}_{self.cam_id}.png', True
                )
                > 0
                for t in range(self.start_timestep, self.end_timestep)
            ]
        )
        if self.rectification is not None:
            self.invalid_masks |= ~self.rectification[1]

    def load_dynamic_masks(self):
        for name in ('dynamic', 'human', 'vehicle'):
            paths = getattr(self, name + '_mask_filepaths')
            setattr(
                self, name + '_masks', torch.stack([self._read(p, True) > 0 for p in paths]).float()
            )

    def load_sky_masks(self):
        self.sky_masks = torch.stack(
            [self._read(p, True) > 0 for p in self.sky_mask_filepaths]
        ).float()

    def to(self, device):
        super().to(device)
        self.device = device
        self.invalid_masks = self.invalid_masks.to(device)
        self._ray_cache.clear()
        return self

    def get_image(self, frame_idx):
        image, camera = super().get_image(frame_idx)
        height, width = int(camera['height']), int(camera['width'])
        K = self.intrinsics[frame_idx].clone()
        K[:2] *= self.downscale_factor
        p = (
            scale_ftheta_calibration(
                self.ftheta_parameters, width, height, scale=self.downscale_factor
            )
            if self.camera_model == 'ftheta'
            else None
        )
        key = height, width
        if key not in self._ray_cache:
            self._ray_cache[key] = camera_model_rays(
                height, width, K, self.camera_model, ftheta_parameters=p
            )
        rays, valid = self._ray_cache[key]
        pose = self.cam_to_worlds[frame_idx]
        image['viewdirs'] = F.normalize(rays @ pose[:3, :3].T, dim=-1)
        image['origins'] = pose[:3, 3].expand(height, width, 3)
        image['direction_norm'] = torch.ones_like(rays[..., :1])
        invalid = (
            F.interpolate(
                self.invalid_masks[frame_idx][None, None].float(),
                scale_factor=self.downscale_factor, mode='nearest'
            )[0, 0].bool()
            | ~valid
        )
        image['egocar_masks'] = invalid.float()
        image['valid_masks'] = ~invalid
        camera.update(
            intrinsics=K, camera_model=self.camera_model, ftheta_parameters=p, camera_to_world=pose
        )
        return image, camera

    def project_points(self, world_points, frame_idx):
        pixels, depth, valid = project_world_camera(
            world_points,
            self.cam_to_worlds[frame_idx],
            self.intrinsics[frame_idx],
            self.WIDTH,
            self.HEIGHT,
            self.camera_model,
            ftheta_parameters=self.ftheta_parameters,
        )
        selected = valid.nonzero().flatten()
        uv = pixels[selected].long()
        valid[selected] &= ~self.invalid_masks[frame_idx, uv[:, 1], uv[:, 0]]
        return pixels, depth, valid


class PhysicalAIPixelSource(ScenePixelSource):
    def __init__(
        self,
        dataset_name,
        pixel_data_config,
        data_path,
        start_timestep,
        end_timestep,
        device=torch.device('cpu'),
    ):
        super().__init__(dataset_name, pixel_data_config, device=device)
        self.data_path, self.start_timestep, self.end_timestep = (
            data_path,
            start_timestep,
            end_timestep,
        )
        self.available_cameras = [
            int(i) for i in json.loads((Path(data_path) / 'metadata.json').read_text())['cameras']
        ]
        self.smpl_human_all = {}
        self.load_data()

    def load_cameras(self):
        self._timesteps = torch.arange(self.start_timestep, self.end_timestep)
        self.register_normalized_timestamps()
        if self.end_timestep - self.start_timestep == 1:
            self._normalized_time = torch.zeros_like(self._timesteps, dtype=torch.float32)
            self._unique_normalized_timestamps = torch.zeros(1)
        for index, cam_id in enumerate(self.camera_list):
            if cam_id not in self.available_cameras:
                raise ValueError(f'Camera {cam_id} is not present in this PhysicalAI scene')
            camera = PhysicalAICameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                load_dynamic_mask=self.data_cfg.load_dynamic_mask,
                load_sky_mask=self.data_cfg.load_sky_mask,
                downscale_when_loading=self.data_cfg.downscale_when_loading[index],
                undistort=self.data_cfg.undistort,
                rectified_fov=self.data_cfg.get('rectified_fov'),
                buffer_downscale=self.buffer_downscale,
                device=self.device,
            )
            camera.load_time(self.normalized_time)
            camera.set_unique_ids(index, torch.arange(len(camera)) * self.num_cams + index)
            self.camera_data[cam_id] = camera

    def load_objects(self):
        """
        get ground truth bounding boxes of the dynamic objects

        instances_info = {
            "0": # simplified instance id
                {
                    "id": str,
                    "class_name": str,
                    "frame_annotations": {
                        "frame_idx": List,
                        "obj_to_world": List,
                        "box_size": List,
                },
            ...
        }
        frame_instances = {
            "0": # frame idx
                List[int] # list of simplified instance ids
            ...
        }
        """
        instances_info_path = os.path.join(self.data_path, "instances", "instances_info.json")
        frame_instances_path = os.path.join(self.data_path, "instances", "frame_instances.json")
        with open(instances_info_path, "r") as f:
            instances_info = json.load(f)
        with open(frame_instances_path, "r") as f:
            frame_instances = json.load(f)
        # get pose of each instance at each frame
        # shape (num_frames, num_instances, 4, 4)
        num_instances = len(instances_info)
        num_full_frames = len(frame_instances)
        instances_pose = np.zeros((num_full_frames, num_instances, 4, 4))
        instances_size = np.zeros((num_full_frames, num_instances, 3))
        instances_true_id = np.arange(num_instances)
        instances_model_types = np.ones(num_instances) * -1

        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )
        for k, v in instances_info.items():
            instances_model_types[int(k)] = OBJECT_CLASS_NODE_MAPPING[v["class_name"]]
            for frame_idx, obj_to_world, box_size in zip(
                v["frame_annotations"]["frame_idx"],
                v["frame_annotations"]["obj_to_world"],
                v["frame_annotations"]["box_size"],
            ):
                # the first ego pose as the origin of the world coordinate system.
                obj_to_world = np.array(obj_to_world).reshape(4, 4)
                obj_to_world = np.linalg.inv(ego_to_world_start) @ obj_to_world
                instances_pose[frame_idx, int(k)] = np.array(obj_to_world)
                instances_size[frame_idx, int(k)] = np.array(box_size)

        # get frame valid instances
        # shape (num_frames, num_instances)
        per_frame_instance_mask = np.zeros((num_full_frames, num_instances))
        for frame_idx, valid_instances in frame_instances.items():
            per_frame_instance_mask[int(frame_idx), valid_instances] = 1

        # select the frames that are in the range of start_timestep and end_timestep
        instances_pose = torch.from_numpy(
            instances_pose[self.start_timestep : self.end_timestep]
        ).float()
        instances_size = torch.from_numpy(
            instances_size[self.start_timestep : self.end_timestep]
        ).float()
        instances_true_id = torch.from_numpy(instances_true_id).long()
        instances_model_types = torch.from_numpy(instances_model_types).long()
        per_frame_instance_mask = torch.from_numpy(
            per_frame_instance_mask[self.start_timestep : self.end_timestep]
        ).bool()

        # filter out the instances that are not visible in selected frames
        ins_frame_cnt = per_frame_instance_mask.sum(dim=0)
        instances_pose = instances_pose[:, ins_frame_cnt > 0]
        instances_size = instances_size[:, ins_frame_cnt > 0]
        instances_true_id = instances_true_id[ins_frame_cnt > 0]
        instances_model_types = instances_model_types[ins_frame_cnt > 0]
        per_frame_instance_mask = per_frame_instance_mask[:, ins_frame_cnt > 0]

        # assign to the class
        # (num_frames, num_instances, 4, 4)
        self.instances_pose = instances_pose
        # (num_instances, 3)
        self.instances_size = instances_size.sum(0) / per_frame_instance_mask.sum(0).unsqueeze(-1)
        # (num_frames, num_instances)
        self.per_frame_instance_mask = per_frame_instance_mask
        # (num_instances)
        self.instances_true_id = instances_true_id
        # (num_instances)
        self.instances_model_types = instances_model_types

        if self.data_cfg.load_smpl:
            # Collect camera-to-world matrices for all available cameras
            cam_to_worlds = {}
            for cam_id in self.available_cameras:
                cam_to_worlds[cam_id] = PhysicalAICameraData.get_camera2worlds(
                    self.data_path, str(cam_id), self.start_timestep, self.end_timestep
                )

            # load SMPL parameters
            smpl_dict = joblib.load(os.path.join(self.data_path, "humanpose", "smpl.pkl"))
            frame_num = self.end_timestep - self.start_timestep

            smpl_human_all = {}
            for fi in tqdm(range(self.start_timestep, self.end_timestep), desc="Loading SMPL"):
                for instance_id, ins_smpl in smpl_dict.items():
                    if instance_id not in smpl_human_all:
                        smpl_human_all[instance_id] = {
                            "smpl_quats": torch.zeros((frame_num, 24, 4), dtype=torch.float32),
                            "smpl_trans": torch.zeros((frame_num, 3), dtype=torch.float32),
                            "smpl_betas": torch.zeros((frame_num, 10), dtype=torch.float32),
                            "frame_valid": torch.zeros((frame_num), dtype=torch.bool),
                        }
                        smpl_human_all[instance_id]["smpl_quats"][:, :, 0] = 1.0
                    if ins_smpl["valid_mask"][fi]:
                        betas = ins_smpl["smpl"]["betas"][fi]
                        smpl_human_all[instance_id]["smpl_betas"][fi - self.start_timestep] = betas

                        body_pose = ins_smpl["smpl"]["body_pose"][fi]
                        smpl_orient = ins_smpl["smpl"]["global_orient"][fi]
                        cam_depend = ins_smpl["selected_cam_idx"][fi].item()

                        c2w = cam_to_worlds[cam_depend][fi - self.start_timestep]
                        world_orient = c2w[:3, :3].to(smpl_orient.device) @ smpl_orient.squeeze()
                        smpl_quats = matrix_to_quaternion(
                            torch.cat([world_orient[None, ...], body_pose], dim=0)
                        )

                        ii = instances_info[str(instance_id)]['frame_annotations'][
                            "frame_idx"
                        ].index(fi)
                        o2w = np.array(
                            instances_info[str(instance_id)]['frame_annotations']["obj_to_world"][
                                ii
                            ]
                        )
                        o2w = torch.from_numpy(np.linalg.inv(ego_to_world_start) @ o2w)
                        # box_size = instances_info[str(instance_id)]['frame_annotations']["box_size"][ii]

                        smpl_human_all[instance_id]["smpl_quats"][
                            fi - self.start_timestep
                        ] = smpl_quats
                        smpl_human_all[instance_id]["smpl_trans"][fi - self.start_timestep] = o2w[
                            :3, 3
                        ]
                        smpl_human_all[instance_id]["frame_valid"][fi - self.start_timestep] = True

            self.smpl_human_all = smpl_human_all


class PhysicalAILiDARSource(SceneLidarSource):
    def __init__(
        self, lidar_data_config, data_path, start_timestep, end_timestep, device=torch.device('cpu')
    ):
        super().__init__(lidar_data_config, device=device)
        self.data_path = Path(data_path)
        self.start_timestep, self.end_timestep = start_timestep, end_timestep
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        self.lidar_filepaths = [
            self.data_path / 'lidar' / f'{t:03d}.bin'
            for t in range(self.start_timestep, self.end_timestep)
        ]

    def load_calibrations(self):
        anchor = np.linalg.inv(
            np.loadtxt(self.data_path / 'ego_pose' / f'{self.start_timestep:03d}.txt')
        )
        self.lidar_to_worlds = torch.from_numpy(
            np.stack(
                [
                    anchor @ np.loadtxt(self.data_path / 'lidar_pose' / f'{t:03d}.txt')
                    for t in range(self.start_timestep, self.end_timestep)
                ]
            )
        ).float()

    def load_lidar(self):
        origins, directions, ranges, timesteps = [], [], [], []
        for index, path in enumerate(self.lidar_filepaths):
            xyz = torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].copy())
            distance = xyz.norm(dim=-1, keepdim=True)
            valid = torch.isfinite(xyz).all(-1) & (distance[:, 0] > 1e-6)
            xyz, distance = xyz[valid], distance[valid]
            pose = self.lidar_to_worlds[index]
            origins.append(pose[:3, 3].expand_as(xyz))
            directions.append((xyz / distance) @ pose[:3, :3].T)
            ranges.append(distance)
            timesteps.append(torch.full((len(xyz),), index, dtype=torch.long))
        if not origins or not sum(len(o) for o in origins):
            raise ValueError('PhysicalAI scene has no valid LiDAR points')
        self.origins = torch.cat(origins)
        self.directions = torch.cat(directions)
        self.ranges = torch.cat(ranges)
        self._timesteps = torch.cat(timesteps)
        self.register_normalized_timestamps()
        if self.end_timestep - self.start_timestep == 1:
            self._normalized_time = torch.zeros_like(self._timesteps, dtype=torch.float32)
            self._unique_normalized_timestamps = torch.zeros(1)
        self.visible_masks = torch.zeros(len(self.origins), dtype=torch.bool)
        self.colors = torch.ones_like(self.origins)
        self.flows = torch.zeros_like(self.origins)

    def to(self, device):
        super().to(device)
        self.flows = self.flows.to(device)
        return self
