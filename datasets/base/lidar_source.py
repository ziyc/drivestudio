import abc
import logging
from typing import Dict

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor

logger = logging.getLogger()


class SceneLidarSource(abc.ABC):
    """
    The base class for the lidar source of a scene.
    """

    data_cfg: OmegaConf = None
    # the normalized time of all images (normalized to [0, 1]), shape: (num_frames,)
    _normalized_time: Tensor = None
    # timestep indices of frames, shape: (num_frames,)
    _timesteps: Tensor = None
    # origin of each lidar point, shape: (num_points, 3)
    origins: Tensor = None
    # unit direction of each lidar point, shape: (num_points, 3)
    directions: Tensor = None
    # range of each lidar point, shape: (num_points,)
    ranges: Tensor = None
    # the transformation matrices from lidar to world coordinate system,
    lidar_to_worlds: Tensor = None
    # indicate whether each lidar point is visible to the camera,
    visible_masks: Tensor = None
    # the color of each lidar point, shape: (num_points, 3)
    colors: Tensor = None
    # the indices of the lidar scans that are cached
    cached_indices: Tensor = None
    cached_origins: Tensor = None
    cached_directions: Tensor = None
    cached_ranges: Tensor = None
    cached_normalized_timestamps: Tensor = None

    def __init__(
        self,
        lidar_data_config: OmegaConf,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        # hold the config of the lidar data
        self.data_cfg = lidar_data_config
        self.device = device

    @abc.abstractmethod
    def create_all_filelist(self) -> None:
        """
        Create a list of all the files in the dataset.
        e.g., a list of all the lidar scans in the dataset.
        """
        raise NotImplementedError

    def load_data(self):
        self.load_calibrations()
        self.load_lidar()
        logger.info("[Lidar] All Lidar Data loaded.")

    def to(self, device: torch.device) -> "SceneLidarSource":
        """
        Move the dataset to the given device.
        Args:
            device: the device to move the dataset to.
        """
        self.device = device
        if self.origins is not None:
            self.origins = self.origins.to(device)
        if self.directions is not None:
            self.directions = self.directions.to(device)
        if self.ranges is not None:
            self.ranges = self.ranges.to(device)
        if self._timesteps is not None:
            self._timesteps = self._timesteps.to(device)
        if self._normalized_time is not None:
            self._normalized_time = self._normalized_time.to(device)
        if self.lidar_to_worlds is not None:
            self.lidar_to_worlds = self.lidar_to_worlds.to(device)
        if self.visible_masks is not None:
            self.visible_masks = self.visible_masks.to(device)
        if self.colors is not None:
            self.colors = self.colors.to(device)
        return self

    @abc.abstractmethod
    def load_calibrations(self) -> None:
        """
        Load the calibration files of the dataset.
        e.g., lidar to world transformation matrices.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_lidar(self) -> None:
        """
        Load the lidar data of the dataset from the filelist.
        """
        raise NotImplementedError

    def get_aabb(self) -> Tensor:
        """
        Returns:
            aabb_min, aabb_max: the min and max of the axis-aligned bounding box of the scene
        Note:
            we assume the lidar points are already in the world coordinate system
            we first downsample the lidar points, then compute the aabb by taking the
            given percentiles of the lidar coordinates in each dimension.
        """
        assert (
            self.origins is not None
            and self.directions is not None
            and self.ranges is not None
        ), "Lidar points not loaded, cannot compute aabb."
        logger.info("[Lidar] Computing auto AABB based on downsampled lidar points....")

        lidar_pts = self.origins + self.directions * self.ranges

        # downsample the lidar points by uniformly sampling a subset of them
        lidar_pts = lidar_pts[
            torch.randperm(len(lidar_pts))[
                : int(len(lidar_pts) / self.data_cfg.lidar_downsample_factor)
            ]
        ]
        # compute the aabb by taking the given percentiles of the lidar coordinates in each dimension
        aabb_min = torch.quantile(lidar_pts, self.data_cfg.lidar_percentile, dim=0)
        aabb_max = torch.quantile(lidar_pts, 1 - self.data_cfg.lidar_percentile, dim=0)
        del lidar_pts
        torch.cuda.empty_cache()

        # usually the lidar's height is very small, so we slightly increase the height of the aabb
        if aabb_max[-1] < 20:
            aabb_max[-1] = 20.0
        aabb = torch.tensor([*aabb_min, *aabb_max])
        logger.info(f"[Lidar] Auto AABB from LiDAR: {aabb}")
        return aabb
    
    @property
    def pts_xyz(self) -> Tensor:
        """
        Returns:
            the xyz coordinates of the lidar points.
            shape: (num_lidar_points, 3)
        """
        return self.origins + self.directions * self.ranges
    
    @property
    def num_points(self) -> int:
        """
        Returns:
            the number of lidar points in the dataset.
        """
        return self.origins.size(0)

    @property
    def num_timesteps(self) -> int:
        """
        Returns:
            the number of lidar timestamps in the dataset,
            usually the number of captured lidar scans.
        """
        return len(self.timesteps.unique())

    @property
    def timesteps(self) -> Tensor:
        """
        Returns:
            the integer timestep indices of each lidar timestamp,
            shape: (num_lidar_points,)
        Note:
            the difference between timestamps and timesteps is that
            timestamps are the actual timestamps (minus 1e9) of the lidar scans,
            while timesteps are the integer timestep indices of the lidar scans.
        """
        return self._timesteps

    @property
    def normalized_time(self) -> Tensor:
        """
        Returns:
            the normalized timestamps of the lidar scans
            (normalized to the range [0, 1]).
            shape: (num_lidar_points,)
        """
        return self._normalized_time

    @property
    def unique_normalized_timestamps(self) -> Tensor:
        """
        Returns:
            the unique normalized timestamps of the lidar scans
            (normalized to the range [0, 1]).
            shape: (num_timesteps,)
        """
        return self._unique_normalized_timestamps

    def register_normalized_timestamps(self) -> None:
        # normalized timestamps are between 0 and 1
        normalized_time = (self._timesteps - self._timesteps.min()) / (
            self._timesteps.max() - self._timesteps.min()
        )
        self._normalized_time = normalized_time.to(self.device)
        self._unique_normalized_timestamps = self._normalized_time.unique()

    def find_closest_timestep(self, normed_timestamp: float) -> int:
        """
        Find the closest timestep to the given timestamp.
        Args:
            normed_timestamp: the normalized timestamp to find the closest timestep for.
        Returns:
            the closest timestep to the given timestamp.
        """
        return torch.argmin(
            torch.abs(self.unique_normalized_timestamps - normed_timestamp)
        )

    def get_lidar_rays(self, time_idx: int) -> Dict[str, Tensor]:
        """
        Get the of rays for rendering at the given timestep.
        Args:
            time_idx: the index of the lidar scan to render.
        Returns:
            a dict of the sampled rays.
        """
        origins = self.origins[self.timesteps == time_idx]
        directions = self.directions[self.timesteps == time_idx]
        ranges = self.ranges[self.timesteps == time_idx]
        normalized_time = self.normalized_time[self.timesteps == time_idx]
        flows = self.flows[self.timesteps == time_idx]
        return {
            "lidar_origins": origins,
            "lidar_viewdirs": directions,
            "lidar_ranges": ranges,
            "lidar_normed_time": normalized_time,
            "lidar_mask": self.timesteps == time_idx,
            "lidar_flows": flows,
        }
    
    def delete_invisible_pts(self) -> None:
        """
        Clear the unvisible points.
        """
        if self.visible_masks is not None:
            num_bf = self.origins.shape[0]
            self.origins = self.origins[self.visible_masks]
            self.directions = self.directions[self.visible_masks]
            self.ranges = self.ranges[self.visible_masks]
            self.flows = self.flows[self.visible_masks]
            self._timesteps = self._timesteps[self.visible_masks]
            self._normalized_time = self._normalized_time[self.visible_masks]
            self.colors = self.colors[self.visible_masks]
            logger.info(
                f"[Lidar] {num_bf - self.visible_masks.sum()} out of {num_bf} points are cleared. {self.visible_masks.sum()} points left."
            )
            self.visible_masks = None
        else:
            logger.info("[Lidar] No unvisible points to clear.")
