# Miscellaneous utility functions for exporting point clouds.
import importlib
import logging
import os

import numpy as np
import open3d as o3d
import torch
import torch.distributed as dist

logger = logging.getLogger()

def import_str(string: str):
    """ Import a python module given string paths

    Args:
        string (str): The given paths

    Returns:
        Any: Imported python module / object
    """
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

def export_points_to_ply(
    positions: torch.tensor,
    colors: torch.tensor,
    save_path: str,
    normalize: bool = False,
    ):
    # normalize points
    if normalize:
        aabb_min = positions.min(0)[0]
        aabb_max = positions.max(0)[0]
        positions = (positions - aabb_min) / (aabb_max - aabb_min)
    if isinstance(colors, torch.Tensor):
        positions = positions.cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()
    
    # clamp colors
    colors = np.clip(colors, a_min=0., a_max=1.)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)

def export_gaussians_to_ply(model, path, name='point_cloud.ply', aabb=None):
    model.eval()
    filename = os.path.join(path, name)
    map_to_tensors = {}
    
    with torch.no_grad():
        positions = model.means
        if aabb is not None:
            aabb = aabb.to(positions.device)
            aabb_min, aabb_max = aabb[:3], aabb[3:]
            aabb_center = (aabb_min + aabb_max) / 2
            aabb_sacle_max = (aabb_max - aabb_min).max() / 2 * 1.1
            vis_mask = torch.logical_and(positions >= aabb_min, positions < aabb_max).all(-1)
        else:
            aabb_center = positions.mean(0)
            aabb_sacle_max = (positions - aabb_center).abs().max() * 1.1
            vis_mask = torch.ones_like(positions[:, 0], dtype=torch.bool)
            
        positions = ((positions[vis_mask] - aabb_center) / aabb_sacle_max).cpu().numpy()
        map_to_tensors["positions"] = o3d.core.Tensor(positions, o3d.core.float32)
        map_to_tensors["normals"] = o3d.core.Tensor(np.zeros_like(positions), o3d.core.float32)

        colors = model.colors[vis_mask].data.cpu().numpy()
        map_to_tensors["colors"] = (colors * 255).astype(np.uint8)
        for i in range(colors.shape[1]):
            map_to_tensors[f"f_dc_{i}"] = colors[:, i : i + 1]

        shs = model.shs_rest[vis_mask].data.cpu().numpy()
        if model.config.sh_degree > 0:
            shs = shs.reshape((colors.shape[0], -1, 1))
            for i in range(shs.shape[-1]):
                map_to_tensors[f"f_rest_{i}"] = shs[:, i]

        map_to_tensors["opacity"] = model.opacities[vis_mask].data.cpu().numpy()

        scales = model.scales[vis_mask].data.cpu().unsqueeze(-1).numpy()
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i]

        quats = model.quats[vis_mask].data.cpu().unsqueeze(-1).numpy()

        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i]

    pcd = o3d.t.geometry.PointCloud(map_to_tensors)
    o3d.t.io.write_point_cloud(str(filename), pcd)
    
    logger.info(f"Exported point cloud to {filename}, containing {vis_mask.sum().item()} points.")

def is_enabled() -> bool:
    """
    Returns:
        True if distributed training is enabled
    """
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    """
    Returns:
        The rank of the current process within the global process group.
    """
    return dist.get_rank() if is_enabled() else 0


def get_world_size():
    return dist.get_world_size() if is_enabled() else 1


def is_main_process() -> bool:
    """
    Returns:
        True if the current process is the main one.
    """
    return get_global_rank() == 0