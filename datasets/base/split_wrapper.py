from typing import List, Tuple
import torch

from .pixel_source import ScenePixelSource

class SplitWrapper(torch.utils.data.Dataset):

    # a sufficiently large number to make sure we don't run out of data
    _num_iters = 1000000

    def __init__(
        self,
        datasource: ScenePixelSource,
        split_indices: List[int] = None,
        split: str = "train",
    ):
        super().__init__()
        self.datasource = datasource
        self.split_indices = split_indices
        self.split = split

    def get_image(self, idx, camera_downscale) -> dict:
        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
        self.datasource.update_downscale_factor(downscale_factor)
        image_infos, cam_infos = self.datasource.get_image(self.split_indices[idx])
        self.datasource.reset_downscale_factor()
        return image_infos, cam_infos

    def next(self, camera_downscale) -> Tuple[dict, dict]:
        assert self.split == "train", "Only train split supports next()"
        
        img_idx = self.datasource.propose_training_image(
            candidate_indices=self.split_indices
        )
        
        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
        self.datasource.update_downscale_factor(downscale_factor)
        image_infos, cam_infos = self.datasource.get_image(img_idx)
        self.datasource.reset_downscale_factor()
        
        return image_infos, cam_infos
    
    def __getitem__(self, idx) -> dict:
        return self.get_image(idx, camera_downscale=1.0)

    def __len__(self) -> int:
        return len(self.split_indices)

    @property
    def num_iters(self) -> int:
        return self._num_iters

    def set_num_iters(self, num_iters) -> None:
        self._num_iters = num_iters
