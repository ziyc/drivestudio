import abc
import logging
from enum import IntEnum
from typing import List, Callable

import torch
from omegaconf import OmegaConf
from torch import Tensor

from .lidar_source import SceneLidarSource
from .pixel_source import ScenePixelSource
from .split_wrapper import SplitWrapper

logger = logging.getLogger()

class ModelType(IntEnum):
    RigidNodes = 0
    SMPLNodes = 1
    DeformableNodes = 2
    
class SceneDataset(abc.ABC):
    """
    Base class for scene dataset.
    """

    data_cfg: OmegaConf = None
    pixel_source: ScenePixelSource = None
    lidar_source: SceneLidarSource = None
    # training and testing indices are indices into the full dataset
    # train_indices are img indices, so the length is num_cams * num_timesteps
    train_indices: List[int] = None
    test_indices: List[int] = None
    # train_timesteps are timesteps, so the length is num_timesteps (len(unique_timesteps))
    train_timesteps: Tensor = None
    test_timesteps: Tensor = None
    
    # dataset layout for multi-camera visualization
    layout: Callable = None

    # dataset wrappers
    # full: includes all data
    full_image_set: SplitWrapper = None
    full_lidar_set: SplitWrapper = None
    # train: includes only training data
    train_image_set: SplitWrapper = None
    train_lidar_set: SplitWrapper = None
    # test: includes only testing data
    test_image_set: SplitWrapper = None
    test_lidar_set: SplitWrapper = None

    def __init__(
        self,
        data_config: OmegaConf,
    ):
        super().__init__()
        self.data_cfg = data_config

    @abc.abstractmethod
    def build_data_source(self):
        """
        Create the data source for the dataset.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def build_split_wrapper(self):
        """
        Makes each data source as a Pytorch Dataset.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def split_train_test(self):
        raise NotImplementedError

    def get_aabb(self) -> Tensor:
        if self.lidar_source is not None:
            aabb = self.lidar_source.get_aabb()
        else:
            aabb = self.pixel_source.get_aabb()
        return aabb
    
    def get_init_pcd(self) -> Tensor:
        raise NotImplementedError

    @property
    def num_cams(self) -> int:
        return self.pixel_source.num_cams

    @property
    def scene_idx(self) -> int:
        return self.data_cfg.scene_idx

    @property
    def num_img_timesteps(self) -> int:
        return self.pixel_source.num_timesteps

    @property
    def num_lidar_timesteps(self) -> int:
        if self.lidar_source is None:
            logger.warning("No lidar source, returning num_img_timesteps")
            return self.num_img_timesteps
        return self.lidar_source.num_timesteps

    @property
    def num_train_timesteps(self) -> int:
        return len(self.train_timesteps)

    @property
    def num_test_timesteps(self) -> int:
        return len(self.test_timesteps)

    @property
    def unique_normalized_training_timestamps(self) -> Tensor:
        return self.pixel_source.unique_normalized_timestamps[self.train_timesteps]

    @property
    def device(self):
        return self.data_cfg.preload_device
