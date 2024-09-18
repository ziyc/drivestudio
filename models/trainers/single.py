from typing import Dict

import torch
import logging

from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer, GSModelType
from models.gaussians import PeriodicVibrationGaussians
from utils.misc import import_str
from utils.geometry import uniform_sample_sphere

"""
Single Gaussian Model Trainer

Methods Available:
    Static Vanilla Gaussians,
    Defromable Gaussians for dynamic scenes,
    Periodic Vibration Gaussians for dynamic scenes,
"""

logger = logging.getLogger()

class SingleTrainer(BasicTrainer):
    def __init__(
        self, 
        num_timesteps: int,
        **kwargs
    ):
        self.num_timesteps = num_timesteps
        super().__init__(**kwargs)

    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)
        
    def _init_models(self):
        # gaussian model classes
        self.gaussian_classes["Background"] = GSModelType.Background
     
        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes:
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)
            
            if class_name in self.gaussian_classes.keys():
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device
                )
                
            if class_name in self.misc_classes_keys: 
                # ['Sky', 'Affine', 'CamPose', 'CamPosePerturb']
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)

            self.models[class_name] = model
        
        logger.info(f"Initialized models: {self.models.keys()}")

        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)

    def init_gaussians_from_dataset(
        self,
        dataset: DrivingDataset,
    ) -> None:
        for class_name in self.gaussian_classes:
            model_cfg = self.model_config[class_name]
            model = self.models[class_name]
            if class_name == 'Background':                
                # ------ initialize gaussians ------
                init_cfg = model_cfg.pop('init')
                # sample points from the lidar point clouds
                if init_cfg.get("from_lidar", None) is not None:
                    sampled_pts, sampled_color, sampled_time = dataset.get_lidar_samples(
                        **init_cfg.from_lidar, device=self.device
                    )
                else:
                    sampled_pts, sampled_color, sampled_time = \
                        torch.empty(0, 3).to(self.device), torch.empty(0, 3).to(self.device), None
                
                random_pts = []
                num_near_pts = init_cfg.get('near_randoms', 0)
                if num_near_pts > 0: # uniformly sample points inside the scene's sphere
                    num_near_pts *= 3 # since some invisible points will be filtered out
                    random_pts.append(uniform_sample_sphere(num_near_pts, self.device))
                num_far_pts = init_cfg.get('far_randoms', 0)
                if num_far_pts > 0: # inverse distances uniformly from (0, 1 / scene_radius)
                    num_far_pts *= 3
                    random_pts.append(uniform_sample_sphere(num_far_pts, self.device, inverse=True))
                
                if num_near_pts + num_far_pts > 0:
                    random_pts = torch.cat(random_pts, dim=0) 
                    random_pts = random_pts * self.scene_radius + self.scene_origin
                    visible_mask = dataset.check_pts_visibility(random_pts)
                    valid_pts = random_pts[visible_mask]
                    
                    sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)
                    sampled_color = torch.cat([sampled_color, torch.rand(valid_pts.shape, ).to(self.device)], dim=0)
                    if sampled_time is not None:
                        sampled_time = torch.cat([sampled_time, torch.zeros(valid_pts.shape[0], 1).to(self.device)], dim=0)
                
                if isinstance(model, PeriodicVibrationGaussians):
                    model.create_from_pcd(
                        init_means=sampled_pts, init_colors=sampled_color, init_times=sampled_time
                    )
                else:
                    model.create_from_pcd(
                        init_means=sampled_pts, init_colors=sampled_color
                    )
                
            logger.info(f"Initialized {class_name} gaussians")
        logger.info(f"Initialized gaussians from pcd")
        
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
                        novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """
        
        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )
        
        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set
                
        # assigne current frame to gaussian models
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)
        
        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )
        gs = self.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0]
        )
        # 重要！得到了GS球，合并了各种class

        # render gaussians
        outputs, _ = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.)
        )
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )
        
        return outputs,gs
        
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_losses(outputs, image_infos, cam_infos)
        
        return loss_dict
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metric_dict = super().compute_metrics(outputs, image_infos)
        
        return metric_dict