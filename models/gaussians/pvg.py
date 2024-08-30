"""
Filename: pvg.py

Author: Ziyu Chen (ziyu.sjtu@gmail.com)

Description:
Unofficial implementation of PVG based on the work by Yurui Chen, Chun Gu, Junzhe Jiang, Xiatian Zhu, Li Zhang.

Original paper: https://arxiv.org/abs/2311.18561
"""

from typing import Dict, List
import random
import logging
import torch
import torch.distributions.uniform as uniform
from torch.nn import Parameter

from models.gaussians.basics import *
from models.gaussians.vanilla import VanillaGaussians

logger = logging.getLogger()

class PeriodicVibrationGaussians(VanillaGaussians):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._taus = torch.zeros(1, 1, device=self.device)
        self._betas = torch.zeros(1, 1, device=self.device)
        self._velocity = torch.zeros(1, 3, device=self.device)
        
        self.T = self.ctrl_cfg.cycle_length
        self.t_grad_accum = None

    def set_cur_frame(self, frame_id: int):
        self.cur_frame = frame_id
    def register_normalized_timestamps(self, normalized_timestamps: int):
        """
        num_timesteps: the total number of timesteps of both train and test
        """
        self.normalized_timestamps = normalized_timestamps
        
        self.num_timestamps = len(normalized_timestamps)
        self.normalized_time_interval = 1.0 / (self.num_timestamps - 1)
        self.train_time_scale = self.ctrl_cfg.time_interval / self.normalized_time_interval
        
    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor, init_times: torch.Tensor) -> None:
        super().create_from_pcd(init_means, init_colors)
        
        # time related parameters
        self._taus = Parameter((init_times * self.train_time_scale).to(self.device))                      # life peak
        self._velocity = Parameter(torch.zeros(self.num_points, 3).to(self.device))                       # vibration direction
        betas_init = torch.sqrt(torch.ones(self.num_points, 1).to(self.device) * self.ctrl_cfg.betas_init)
        self._betas = Parameter(torch.log(betas_init))                                                    # life span

    @property
    def get_scaling_t(self):
        return torch.exp(self._betas)
    @property
    def vibr_dirs_norm(self):
        return self._velocity.norm(dim=-1)
    @property
    def temporal_means(self):
        a = 1/self.T * torch.pi * 2
        means = self._means + self._velocity * torch.sin(
            (self.cur_time - self._taus) * a
        ) / a
        if self.in_smooth:
            return means + self.velocity * self.delta_t
        else:
            return means
    @property
    def temporal_opacities(self):
        return self.get_opacity * torch.exp(
            -0.5 * (self.cur_time - self._taus)**2 / (self.get_scaling_t ** 2)
        )
    @property
    def get_marginal_t(self):
        return torch.exp(-0.5 * (self._taus - self.cur_time) ** 2 / self.get_scaling_t ** 2)
    @property
    def rho(self):
        """staticness coefficient"""
        return self.get_scaling_t / self.T
    @property
    def velocity(self):
        return self._velocity * torch.exp(-0.5 * self.rho)
    @property
    def gamma(self):
        """
        dynamic scale factor for Position-aware point adaptive control
        refer to PVG Section 3.3
        """
        with torch.no_grad():
            gamma = (self._means - self.scene_origin).norm(dim=-1) * self.scene_scale - 1
            gamma = torch.where(gamma<=1, 1, gamma) / self.scene_scale
        return gamma
    
    def after_train(
        self,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (radii > 0).flatten()
            full_mask = torch.zeros(self.num_points, device=radii.device, dtype=torch.bool)
            full_mask[self.filter_mask] = visible_mask
            
            grads = xys_grad.norm(dim=-1)
            t_grads = self._taus.grad.clone().abs()[self.filter_mask].squeeze()
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=grads.device, dtype=grads.dtype)
                self.xys_grad_norm[self.filter_mask] = grads
                
                self.t_grad_accum = torch.zeros(self.num_points, device=grads.device, dtype=grads.dtype)
                self.t_grad_accum[self.filter_mask] = t_grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[full_mask] = self.vis_counts[full_mask] + 1
                self.xys_grad_norm[full_mask] = grads[visible_mask] + self.xys_grad_norm[full_mask]
                self.t_grad_accum[full_mask] = t_grads[visible_mask] + self.t_grad_accum[full_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros(self.num_points, device=radii.device, dtype=torch.float32)
            newradii = radii[visible_mask]
            self.max_2Dsize[full_mask] = torch.maximum(
                self.max_2Dsize[full_mask], newradii / float(last_size)
            )

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            self.class_prefix+"xyz": [self._means],
            self.class_prefix+"sh_dc": [self._features_dc],
            self.class_prefix+"sh_rest": [self._features_rest],
            self.class_prefix+"opacity": [self._opacities],
            self.class_prefix+"scaling": [self._scales],
            self.class_prefix+"rotation": [self._quats],
            self.class_prefix+"velocity": [self._velocity],
            self.class_prefix+"life_peak": [self._taus],
            self.class_prefix+"life_span": [self._betas]
        }
        
    def refinement_after(self, step, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = (
                self.step < self.ctrl_cfg.stop_split_at
                and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
                and self.num_points < self.ctrl_cfg.densify_until_num_points
            )
            # split & duplicate
            print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
            if do_densification:
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_xyz_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                
                t_avg_grad = self.t_grad_accum / self.vis_counts
                high_t_grads = t_avg_grad > self.ctrl_cfg.densify_t_grad_thresh
                high_grads = high_xyz_grads | high_t_grads
                
                splits_xyz = (
                    self.get_scaling.max(dim=-1).values > \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale * self.gamma
                ).squeeze()
                splits_t = (torch.max(self.get_scaling_t, dim=1).values > self.ctrl_cfg.densify_t_size_thresh) & high_t_grads
                splits = splits_xyz | splits_t
                
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.ctrl_cfg.n_split_samples
                (
                    split_means,
                    split_feature_dc,
                    split_feature_rest,
                    split_opacities,
                    split_scales,
                    split_quats,
                    split_dirs,
                    split_taus,
                    split_betas,
                ) = self.split_gaussians(splits, nsamps)

                dups_xyz = (
                    self.get_scaling.max(dim=-1).values <= \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale * self.gamma
                ).squeeze()
                dups_t = (torch.max(self.get_scaling_t, dim=1).values <= self.ctrl_cfg.densify_t_size_thresh) & high_t_grads
                
                dups = dups_xyz | dups_t
                dups &= high_grads
                (
                    dup_means,
                    dup_feature_dc,
                    dup_feature_rest,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                    dup_dirs,
                    dup_taus,
                    dup_betas,
                ) = self.dup_gaussians(dups)
                
                self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                # self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))
                self._features_dc = Parameter(torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                self._features_rest = Parameter(torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                self._opacities = Parameter(torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))
                self._velocity = Parameter(torch.cat([self._velocity.detach(), split_dirs, dup_dirs], dim=0))
                self._taus = Parameter(torch.cat([self._taus.detach(), split_taus, dup_taus], dim=0))
                self._betas = Parameter(torch.cat([self._betas.detach(), split_betas, dup_betas], dim=0))
                
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                    dim=0,
                )
                
                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)

            # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
            if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_gaussian_param_groups()
                remove_from_optim(optimizer, deleted_mask, param_groups)
            print(f"Class {self.class_prefix} left points: {self.num_points}")

            # reset opacity
            if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                    # we align to original repo of gaussians spalting
                reset_value = torch.min(self.get_opacity.data,
                                        torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                self._opacities.data = torch.logit(reset_value)
                # reset the exp of optimizer
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix+"opacity":
                        old_params = group["params"][0]
                        param_state = optimizer.state[old_params]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.t_grad_accum = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > 
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale * self.gamma
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self._velocity = Parameter(self._velocity[~culls].detach())
        self._taus = Parameter(self._taus[~culls].detach())
        self._betas = Parameter(self._betas[~culls].detach())

        print(f"     Cull: {n_bef - self.num_points}")
        return culls
    
    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            self.get_scaling[split_mask].repeat(samps, 1) * centered_samples
            # torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        # step 6, sample new temporal properties
        # new_velocity = self._velocity[split_mask].repeat(samps, 1)
        # new_taus = self._taus[split_mask].repeat(samps, 1)
        # new_betas = self._betas[split_mask].repeat(samps, 1)
        stds_t = self.get_scaling_t[split_mask].repeat(samps, 1)
        means_t = torch.zeros((stds_t.size(0), 1), device="cuda")
        samples_t = torch.normal(mean=means_t, std=stds_t)
        new_taus = samples_t+self._taus[split_mask].repeat(samps, 1)
        
        new_betas = torch.log(self.get_scaling_t[split_mask].repeat(samps, 1) / size_fac)
        new_velocity = self._velocity[split_mask].repeat(samps, 1)
        new_means = new_means + self.velocity[split_mask].repeat(samps, 1) * (samples_t)
    
        not_split_xyz_mask = (
            self.get_scaling.max(dim=-1).values <= \
                self.ctrl_cfg.densify_size_thresh * self.scene_scale * self.gamma
        ).squeeze()[split_mask]
        new_scales[not_split_xyz_mask.repeat(samps)] = torch.log(
            self.get_scaling[split_mask].repeat(samps, 1)
        )[not_split_xyz_mask.repeat(samps)]
        
        not_split_t_mask = (torch.max(self.get_scaling_t, dim=1).values <= self.ctrl_cfg.densify_t_size_thresh)[split_mask]
        new_betas[not_split_t_mask.repeat(samps)] = torch.log(
            self.get_scaling_t[split_mask].repeat(samps, 1)
        )[not_split_t_mask.repeat(samps)]
        
        if self.ctrl_cfg.no_time_split:
            new_betas = torch.log(self.get_scaling_t[split_mask].repeat(samps, 1))
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats, new_velocity, new_taus, new_betas
    
    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        dup_velocity = self._velocity[dup_mask]
        dup_taus = self._taus[dup_mask]
        dup_betas = self._betas[dup_mask]
        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats, dup_velocity, dup_taus, dup_betas
    
    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        # set time and smooth strategy
        scaled_train_t = self.normalized_timestamps[self.cur_frame] * self.train_time_scale # t2 in paper
        if self.training and (
            self.ctrl_cfg.enable_temporal_smoothing and random.random() < self.ctrl_cfg.smooth_probability
        ):
            self.in_smooth = True
            bound = self.normalized_time_interval * self.ctrl_cfg.distribution_span * self.train_time_scale
            self.cur_time = scaled_train_t + uniform.Uniform(-bound, bound).sample((1,)).item() # t1 in paper
            self.delta_t = scaled_train_t - self.cur_time # t2 - t1
        else:
            self.in_smooth = False
            self.cur_time = scaled_train_t
            self.delta_t = 0.0
            
        filter_mask = (self.get_marginal_t > 0.05).squeeze()
        self.filter_mask = filter_mask
        
        means = self.temporal_means
        activated_opacities = self.temporal_opacities
        activated_scales = self.get_scaling
        activated_rotations = self.get_quats
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict

    def compute_reg_loss(self):
        loss_dict = super().compute_reg_loss()
        
        velocity_reg = self.reg_cfg.get("velocity_reg", None)
        # use per point velocity regularization to replace per image velocity regularization
        if velocity_reg:
            if (self.cur_radii > 0).sum():
                velocity_loss = self.velocity.norm(dim=-1)[self.filter_mask]
                loss_dict["velocity_reg"] = velocity_loss[self.cur_radii > 0].mean() * velocity_reg.w
        return loss_dict

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        N = state_dict["_means"].shape[0]
        self._means = Parameter(torch.zeros((N,) + self._means.shape[1:], device=self.device))
        self._scales = Parameter(torch.zeros((N,) + self._scales.shape[1:], device=self.device))
        self._quats = Parameter(torch.zeros((N,) + self._quats.shape[1:], device=self.device))
        self._features_dc = Parameter(torch.zeros((N,) + self._features_dc.shape[1:], device=self.device))
        self._features_rest = Parameter(torch.zeros((N,) + self._features_rest.shape[1:], device=self.device))
        self._opacities = Parameter(torch.zeros((N,) + self._opacities.shape[1:], device=self.device))
        self._taus = Parameter(torch.zeros((N,) + self._taus.shape[1:], device=self.device))
        self._betas = Parameter(torch.zeros((N,) + self._betas.shape[1:], device=self.device))
        self._velocity = Parameter(torch.zeros((N,) + self._velocity.shape[1:], device=self.device))
        msg = super().load_state_dict(state_dict, **kwargs)
        return msg