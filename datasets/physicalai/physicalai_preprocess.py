"""Convert downloaded PhysicalAI NCore V4 clips to DriveStudio scene folders."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

from datasets.tools.multiprocess_utils import track_parallel_progress
from utils.geometry import camera_midpoint_pose, project_camera_model, scale_ftheta_calibration

CAMERAS = (
    'camera_front_wide_120fov',
    'camera_front_tele_30fov',
    'camera_cross_left_120fov',
    'camera_cross_right_120fov',
    'camera_rear_left_70fov',
    'camera_rear_right_70fov',
    'camera_rear_tele_30fov',
)


class PhysicalAIProcessor:
    def __init__(
        self,
        load_dir,
        save_dir,
        process_keys,
        process_id_list=None,
        workers=1,
    ):
        self.paths = sorted(Path(load_dir).glob('**/pai_*.json'))
        if not self.paths:
            raise FileNotFoundError('No pai_<clip_uuid>.json files found under ' + str(load_dir))
        self.save_dir = Path(save_dir)
        self.process_keys = set(process_keys)
        unknown = self.process_keys - {
            'images',
            'calib',
            'pose',
            'lidar',
            'objects',
            'dynamic_masks',
        }
        if unknown:
            raise ValueError('Unsupported PhysicalAI process_keys: ' + str(sorted(unknown)))
        self.process_id_list = (
            list(process_id_list) if process_id_list is not None else list(range(len(self.paths)))
        )
        self.workers = int(workers)
        print('will process keys: ', sorted(self.process_keys), flush=True)

    def convert(self):
        print('Start converting ...', flush=True)
        if self.workers == 1:
            for scene in self.process_id_list:
                self.convert_one(scene)
        else:
            track_parallel_progress(self.convert_one, self.process_id_list, self.workers)
        print('\nFinished ...', flush=True)

    def _source(self, scene_id):
        if str(scene_id).isdigit():
            index = int(scene_id)
            if not 0 <= index < len(self.paths):
                raise ValueError(
                    f'Scene {index} is outside the {len(self.paths)} downloaded clips; specify --scene_ids'
                )
            return self.paths[index], f'{index:03d}'
        matches = [p for p in self.paths if p.stem == 'pai_' + str(scene_id)]
        if len(matches) != 1:
            raise ValueError('Expected exactly one downloaded clip for ' + str(scene_id))
        return matches[0], str(scene_id)

    @staticmethod
    def _calibration(sensor):
        p = sensor.model_parameters
        if not hasattr(p, 'pixeldist_to_angle_poly'):
            raise ValueError(f'{sensor.sensor_id} is not an FTheta camera')
        # Additional windshield models require their own renderer support.
        if getattr(p, 'external_distortion_parameters', None) is not None:
            raise ValueError(
                'External/windshield distortion is not supported; refusing to discard calibration'
            )
        result = {
            key: np.asarray(getattr(p, key)).tolist()
            for key in (
                'resolution',
                'principal_point',
                'linear_cde',
                'pixeldist_to_angle_poly',
                'angle_to_pixeldist_poly',
            )
        }
        result.update(
            camera_model='ftheta',
            reference_poly=p.reference_poly.name,
            max_angle=float(p.max_angle),
            shutter_type=p.shutter_type.name,
        )
        return result

    @staticmethod
    def _tracks(loader, timestamps):
        tracks = {}
        mapping = {
            'automobile': 'Vehicle',
            'car': 'Vehicle',
            'vehicle': 'Vehicle',
            'other_vehicle': 'Vehicle',
            'heavy_truck': 'Vehicle',
            'truck': 'Vehicle',
            'bus': 'Vehicle',
            'trailer': 'Vehicle',
            'pedestrian': 'Pedestrian',
            'person': 'Pedestrian',
            'rider': 'Cyclist',
            'bicycle': 'Cyclist',
            'bicycle_with_rider': 'Cyclist',
            'cyclist': 'Cyclist',
            'motorcycle': 'Cyclist',
            'motorcycle_with_rider': 'Cyclist',
        }
        for observation in loader.get_cuboid_track_observations():
            category = mapping.get(str(observation.class_id).lower())
            if category is None or observation.source.name != 'AUTOLABEL':
                continue
            bbox = observation.bbox3
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler('xyz', bbox.rot).as_matrix()
            pose[:3, 3] = bbox.centroid
            reference = loader.pose_graph.evaluate_poses(
                observation.reference_frame_id,
                'world',
                np.asarray(observation.reference_frame_timestamp_us, dtype=np.uint64),
            )
            track_observation = {
                'timestamp': int(observation.timestamp_us),
                'pose': reference @ pose,
                'size': list(bbox.dim),
                'class_name': category,
            }
            tracks.setdefault(str(observation.track_id), []).append(track_observation)
        instances, frames = {}, {str(i): [] for i in range(len(timestamps))}
        for source_id, observations in sorted(tracks.items()):
            observations_by_time = {}
            for observation in observations:
                observations_by_time[observation['timestamp']] = observation
            times = np.asarray(sorted(observations_by_time), dtype=np.int64)
            observations = [observations_by_time[timestamp] for timestamp in times]
            annotations = {'frame_idx': [], 'obj_to_world': [], 'box_size': []}
            for frame, timestamp in enumerate(timestamps):
                if timestamp < times[0] or timestamp > times[-1]:
                    continue
                right = int(np.searchsorted(times, timestamp))
                if times[right] == timestamp:
                    pose = observations[right]['pose']
                    size = observations[right]['size']
                else:
                    left = right - 1
                    if times[right] - times[left] > 500000:
                        continue  # Do not bridge missing annotation tracks.
                    t = float(timestamp - times[left]) / float(times[right] - times[left])
                    start_pose = observations[left]['pose']
                    end_pose = observations[right]['pose']
                    pose = np.eye(4)
                    pose[:3, :3] = Slerp(
                        [0.0, 1.0],
                        Rotation.from_matrix(np.stack((start_pose[:3, :3], end_pose[:3, :3]))),
                    )(t).as_matrix()
                    pose[:3, 3] = (1 - t) * start_pose[:3, 3] + t * end_pose[:3, 3]
                    size = (
                        (1 - t) * np.asarray(observations[left]['size'])
                        + t * np.asarray(observations[right]['size'])
                    ).tolist()
                annotations['frame_idx'].append(frame)
                annotations['obj_to_world'].append(pose.tolist())
                annotations['box_size'].append(size)
            if annotations['frame_idx']:
                index = len(instances)
                instances[str(index)] = {
                    'id': source_id,
                    'class_name': observations[0]['class_name'],
                    'frame_annotations': annotations,
                }
                for frame in annotations['frame_idx']:
                    frames[str(frame)].append(index)
        return instances, frames

    def convert_one(self, scene_id):
        try:
            from ncore.data import FrameTimepoint, FThetaCameraModelParameters
            from ncore.data.v4 import SequenceComponentGroupsReader, SequenceLoaderV4
            from upath import UPath
        except ImportError as exc:
            raise ImportError(
                'PhysicalAI preprocessing needs nvidia-ncore and its dependencies; see docs/PhysicalAI.md'
            ) from exc
        if sys.version_info < (3, 10):
            # NCore's nested enum annotation fails to resolve when decoding FTheta on Python 3.9.
            FThetaCameraModelParameters.__annotations__['reference_poly'] = (
                FThetaCameraModelParameters.PolynomialType
            )
        source, name = self._source(scene_id)
        meta = json.loads(source.read_text())
        if not str(meta.get('version', '')).startswith('v4'):
            raise ValueError('Only NCore V4 clips are supported')
        loader = SequenceLoaderV4(
            SequenceComponentGroupsReader(
                [UPath(str(source.parent / s['path'])) for s in meta['component_stores']],
                open_consolidated=True,
            ),
            poses_component_group_name='default',
            intrinsics_component_group_name='default',
            masks_component_group_name='default',
            cuboids_component_group_name='default',
        )
        sensors = {
            i: loader.get_camera_sensor(name)
            for i, name in enumerate(CAMERAS)
            if name in loader.camera_ids
        }
        if 0 not in sensors:
            raise ValueError('The reference front-wide camera is required')
        # Sample the full 30 Hz clip at approximately 10 Hz, matching the dataset config.
        indices = np.arange(0, len(sensors[0].frames_timestamps_us), 3)
        if not len(indices):
            raise ValueError('Selected frame range is empty')
        timestamps = (
            np.asarray(sensors[0].frames_timestamps_us[indices]).mean(axis=-1).astype(np.int64)
        )
        out = self.save_dir / name
        out.mkdir(parents=True, exist_ok=True)
        manifest = out / 'metadata.json'
        cameras, selections = {}, {}
        for cam_id, sensor in sensors.items():
            p = self._calibration(sensor)
            size = [int(v) for v in p['resolution']]
            p = scale_ftheta_calibration(p, *size)
            selection = [
                sensor.get_closest_frame_index(int(t), relative_frame_time=0.5) for t in timestamps
            ]
            actual_times = np.asarray(sensor.frames_timestamps_us[selection], dtype=np.int64)
            if (
                len(set(selection)) != len(selection)
                or np.max(np.abs(actual_times.mean(-1) - timestamps)) > 50000
            ):
                raise ValueError(
                    f'Camera {cam_id}: duplicate frames or synchronization error >50 ms'
                )
            selections[cam_id] = selection
            cameras[str(cam_id)] = {
                'camera_name': CAMERAS[cam_id],
                'original_size': size[::-1],
                'source_frame_indices': [int(v) for v in selection],
                'timestamps_us': actual_times.tolist(),
                'calibration': p,
            }
        identity = {
            'schema': 'drivestudio-physicalai/1',
            'clip_id': meta['sequence_id'],
            'num_frames': len(timestamps),
            'timestamps_us': timestamps.tolist(),
            'cameras': cameras,
            'native': True,
            'image_downscale': 1,
            'lidar_format': 'float32 XYZI, motion compensated to sensor END frame',
            'pose_convention': 'OpenCV camera-to-world; all poses in NCore world',
            'annotations': 'NCore AUTOLABEL cuboids; sky/fine masks and SMPL are separate stages',
        }
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError(
                'Existing scene has different source/calibration/frame selection; use a new target directory'
            )
        if not manifest.exists() and any(out.iterdir()):
            raise ValueError('Nonempty scene without matching metadata; use a new target directory')
        manifest.write_text(json.dumps(identity, indent=2) + '\n')

        self.create_folder(out)
        np.savetxt(out / 'timestamps.txt', timestamps, fmt='%d')
        instances, frames = {}, {}
        if 'objects' in self.process_keys or 'dynamic_masks' in self.process_keys:
            instances, frames = self._tracks(loader, timestamps)
        if 'objects' in self.process_keys:
            self.save_objects(out, instances, frames)
        for cam_id, sensor in sensors.items():
            calibration = cameras[str(cam_id)]['calibration']
            width, height = calibration['resolution']
            if 'calib' in self.process_keys:
                self.save_calib(sensor, calibration, out, cam_id)
            ego = sensor.get_mask_images().get('ego')
            if ego is None:
                invalid = np.zeros((height, width), dtype=np.uint8)
            else:
                invalid = np.asarray(
                    ego.convert('L').resize((width, height), Image.Resampling.NEAREST)
                )
            for frame, source_index in enumerate(
                tqdm(
                    selections[cam_id], desc=f'File {name}, camera {cam_id}',
                    total=len(selections[cam_id]), dynamic_ncols=True
                )
            ):
                start = sensor.get_frames_T_sensor_target(
                    'world', source_index, FrameTimepoint.START
                )
                end = sensor.get_frames_T_sensor_target('world', source_index, FrameTimepoint.END)
                if 'images' in self.process_keys:
                    self.save_image(sensor, source_index, out, frame, cam_id, calibration, invalid)
                if 'pose' in self.process_keys:
                    self.save_pose(sensor, out, frame, cam_id, start, end)
                if 'dynamic_masks' in self.process_keys:
                    camera_pose = camera_midpoint_pose(start, end)
                    self.save_dynamic_mask(
                        out, frame, cam_id, calibration, camera_pose, instances, frames[str(frame)]
                    )
        if 'lidar' in self.process_keys:
            self.save_lidar(loader, timestamps, out)
        print(f'PhysicalAI {name}: {len(timestamps)} timesteps, {len(sensors)} cameras -> {out}')

    def create_folder(self, scene_dir):
        folders = []
        if 'images' in self.process_keys:
            folders.extend(['images', 'invalid_masks'])
        if 'calib' in self.process_keys:
            folders.extend(['intrinsics', 'extrinsics'])
        if 'pose' in self.process_keys:
            folders.extend(['camera_pose', 'camera_pose_start', 'ego_pose'])
        if 'objects' in self.process_keys:
            folders.append('instances')
        if 'dynamic_masks' in self.process_keys:
            folders.extend(['dynamic_masks/all', 'dynamic_masks/human', 'dynamic_masks/vehicle'])
        if 'lidar' in self.process_keys:
            folders.extend(['lidar', 'lidar_pose'])
        for folder in folders:
            (scene_dir / folder).mkdir(parents=True, exist_ok=True)

    def save_image(self, sensor, source_index, scene_dir, frame_idx, cam_id, calibration, invalid):
        width, height = calibration['resolution']
        image = sensor.get_frame_image(source_index).convert('RGB')
        if image.size != (width, height):
            raise ValueError(f'Camera {cam_id}: image dimensions do not match calibration')
        stem = f'{frame_idx:03d}_{cam_id}'
        image.save(scene_dir / 'images' / (stem + '.png'))
        Image.fromarray((invalid > 0).astype(np.uint8) * 255).save(
            scene_dir / 'invalid_masks' / (stem + '.png')
        )

    def save_calib(self, sensor, calibration, scene_dir, cam_id):
        (scene_dir / 'intrinsics' / f'{cam_id}.json').write_text(json.dumps(calibration, indent=2))
        np.savetxt(scene_dir / 'extrinsics' / f'{cam_id}.txt', sensor.T_sensor_rig)

    def save_pose(self, sensor, scene_dir, frame_idx, cam_id, start, end):
        stem = f'{frame_idx:03d}_{cam_id}'
        np.savetxt(scene_dir / 'camera_pose' / (stem + '.txt'), end)
        np.savetxt(scene_dir / 'camera_pose_start' / (stem + '.txt'), start)
        if cam_id == 0:
            np.savetxt(
                scene_dir / 'ego_pose' / f'{frame_idx:03d}.txt',
                end @ np.linalg.inv(sensor.T_sensor_rig),
            )

    def save_objects(self, scene_dir, instances, frames):
        (scene_dir / 'instances/instances_info.json').write_text(json.dumps(instances, indent=2))
        (scene_dir / 'instances/frame_instances.json').write_text(json.dumps(frames, indent=2))

    def save_dynamic_mask(
        self, scene_dir, frame_idx, cam_id, calibration, camera_pose, instances, track_ids
    ):
        width, height = calibration['resolution']
        masks = {
            kind: np.zeros((height, width), dtype=np.uint8) for kind in ('all', 'human', 'vehicle')
        }
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        world_to_camera = np.linalg.inv(camera_pose)
        for track_id in track_ids:
            track = instances[str(track_id)]
            annotations = track['frame_annotations']
            index = annotations['frame_idx'].index(frame_idx)
            pose = np.asarray(annotations['obj_to_world'][index])
            size = np.asarray(annotations['box_size'][index])
            world = (signs * size / 2) @ pose[:3, :3].T + pose[:3, 3]
            xyz = torch.from_numpy(
                world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
            ).float()
            pixels, valid = project_camera_model(
                xyz, torch.eye(3), 'ftheta', ftheta_parameters=calibration
            )
            if valid.sum() < 4:
                continue
            uv = pixels[valid].numpy()
            lo = np.maximum(np.floor(uv.min(0)), 0).astype(int)
            hi = np.minimum(np.ceil(uv.max(0)), [width, height]).astype(int)
            kind = 'human' if track['class_name'] == 'Pedestrian' else 'vehicle'
            if np.all(hi > lo):
                for key in ('all', kind):
                    masks[key][lo[1] : hi[1], lo[0] : hi[0]] = 255
        stem = f'{frame_idx:03d}_{cam_id}'
        for kind, mask in masks.items():
            Image.fromarray(mask).save(scene_dir / 'dynamic_masks' / kind / (stem + '.png'))

    def save_lidar(self, loader, timestamps, scene_dir):
        from ncore.data import FrameTimepoint

        lidar_ids = list(loader.lidar_ids)
        if not lidar_ids:
            raise ValueError('No LiDAR available in this clip')
        lidar_id = lidar_ids[0]
        for sensor_id in lidar_ids:
            if 'top' in sensor_id:
                lidar_id = sensor_id
                break
        lidar = loader.get_lidar_sensor(lidar_id)
        for frame, timestamp in enumerate(
            tqdm(timestamps, desc=f'File {scene_dir.name}, LiDAR', total=len(timestamps), dynamic_ncols=True)
        ):
            index = lidar.get_closest_frame_index(int(timestamp), relative_frame_time=0.5)
            if abs(float(np.mean(lidar.frames_timestamps_us[index])) - timestamp) > 100000:
                raise ValueError('LiDAR synchronization error >100 ms')
            cloud = lidar.get_frame_point_cloud(
                index, motion_compensation=True, with_start_points=False
            )
            xyz = np.asarray(cloud.xyz_m_end, dtype=np.float32)
            valid = np.asarray(lidar.get_frame_ray_bundle_return_valid_mask(index)).copy()
            distance = np.linalg.norm(xyz, axis=-1)
            valid &= np.isfinite(xyz).all(-1) & (distance > 0.5) & (distance < 120)
            intensity = np.asarray(
                lidar.get_frame_ray_bundle_return_intensity(index), dtype=np.float32
            )
            np.column_stack((xyz[valid], intensity[valid])).astype(np.float32).tofile(
                scene_dir / 'lidar' / f'{frame:03d}.bin'
            )
            np.savetxt(
                scene_dir / 'lidar_pose' / f'{frame:03d}.txt',
                lidar.get_frames_T_sensor_target('world', index, FrameTimepoint.END),
            )
