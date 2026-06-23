import math

import numpy as np
import torch
from modified_diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer as ModifiedGaussianRasterizer,
)

from models.gaussians.basics import dataclass_camera, dataclass_gs


def get_projection_matrix(znear, zfar, fov_x, fov_y, cx, cy):
    tan_half_fov_y = math.tan(fov_y / 2)
    tan_half_fov_x = math.tan(fov_x / 2)

    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right

    proj = torch.zeros(4, 4)
    z_sign = 1.0
    proj[0, 0] = 2.0 * znear / (right - left)
    proj[1, 1] = 2.0 * znear / (top - bottom)
    proj[0, 2] = cx
    proj[1, 2] = cy
    proj[3, 2] = z_sign
    proj[2, 2] = z_sign * zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    return proj


def focal_to_fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def modified_render(
    gs: dataclass_gs,
    cam: dataclass_camera,
    opacity_mask: torch.Tensor,
    is_train_set: bool,
    override_color=None,
    zfar=0.01,
    znear=100.0,
):
    screenspace_points = torch.zeros_like(
        gs.means, dtype=gs.means.dtype, requires_grad=True, device=gs.means.device
    )
    screenspace_points.retain_grad()

    fov_x = focal_to_fov(cam.Ks[0, 0], int(cam.W))
    fov_y = focal_to_fov(cam.Ks[1, 1], int(cam.H))
    cx = (cam.Ks[0, 2] - int(cam.W) / 2) / int(cam.W) * 2
    cy = (cam.Ks[1, 2] - int(cam.H) / 2) / int(cam.H) * 2

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=gs.means.device)
    world_view_transform = cam.camtoworlds.inverse().transpose(0, 1)
    projection_matrix = get_projection_matrix(
        znear=znear, zfar=zfar, fov_x=fov_x, fov_y=fov_y, cx=cx, cy=cy
    ).transpose(0, 1).to(gs.means.device)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam.H),
        image_width=int(cam.W),
        tanfovx=math.tan(fov_x * 0.5),
        tanfovy=math.tan(fov_y * 0.5),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=1,
        campos=camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = ModifiedGaussianRasterizer(raster_settings=raster_settings)

    means3d = gs.means
    means2d = screenspace_points
    opacity = gs.opacities.squeeze() * opacity_mask if opacity_mask is not None else gs.opacities.squeeze()
    opacity = opacity.unsqueeze(-1)
    scales = gs.scales
    rotations = gs.quats

    shs = None
    colors_precomp = None
    if is_train_set or override_color is None:
        shs = gs.shs
        if shs.requires_grad:
            shs.retain_grad()
    elif override_color is not None:
        colors_precomp = override_color
    else:
        colors_precomp = gs.rgbs

    if means3d.requires_grad:
        means3d.retain_grad()
    if opacity.requires_grad:
        opacity.retain_grad()
    if scales.requires_grad:
        scales.retain_grad()
    if rotations.requires_grad:
        rotations.retain_grad()

    rendered_image, depth, radii, pixel_gaussian_counter = rasterizer(
        means3D=means3d,
        means2D=means2d,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    params_output = {
        "means": means3d,
        "rotations": rotations,
        "scales": scales,
        "opacities": opacity,
        "shs": shs,
    }
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "pixel_gaussian_counter": pixel_gaussian_counter,
        "opacity": depth,
        "params_output": params_output,
    }
