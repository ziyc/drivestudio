"""
@brief Utilities for visualizing multi-camera data and depth

This module provides functions for:
- Combining multiple camera images into tiled layouts for various datasets
- Visualizing depth information on images
- Color mapping and depth visualization
- Utility functions for image processing and color manipulation
"""

import hashlib
from typing import List, Optional, Tuple

import cv2
import matplotlib.cm as cm
import numpy as np
import torch

def to8b(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def get_layout(dataset_type: str):
    if dataset_type == "waymo":
        layout = layout_waymo
    elif dataset_type == "pandaset":
        layout = layout_pandaset
    elif dataset_type == "argoverse":
        layout = layout_argoverse
    elif dataset_type == "nuscenes":
        layout = layout_nuscenes
    elif dataset_type == "kitti":
        layout = layout_kitti
    elif dataset_type == "nuplan":
        layout = layout_nuplan
    else:
        raise ValueError(f"dataset_type {dataset_type} not supported")
    return layout

def layout_nuplan(
    imgs: List[np.array], cam_names: List[str]
) -> np.array:
    """Combine cameras into a tiled image for NuPlan dataset.
    Layout:
    ##############################################
    #    CAM_L0    #    CAM_F0    #    CAM_R0    #
    ##############################################
    #    CAM_L1    #              #    CAM_R1    #
    ##############################################
    #    CAM_L2    #    CAM_B0    #    CAM_R2    #
    ##############################################
    """
    channel = imgs[0].shape[-1]
    height, width = imgs[0].shape[:2]

    # Create a canvas that's 3 times the height and 3 times the width of a single image
    tiled_height = height * 3
    tiled_width = width * 3
    tiled_img = np.zeros((tiled_height, tiled_width, channel), dtype=np.float32)
    filled_mask = np.zeros((tiled_height, tiled_width), dtype=np.uint8)
    
    for idx, cam_name in enumerate(cam_names):
        img = imgs[idx]
        if cam_name == "CAM_F0":
            tiled_img[:height, width:2*width] = img
            filled_mask[:height, width:2*width] = 1
        elif cam_name == "CAM_L0":
            tiled_img[:height, :width] = img
            filled_mask[:height, :width] = 1
        elif cam_name == "CAM_R0":
            tiled_img[:height, 2*width:] = img
            filled_mask[:height, 2*width:] = 1
        elif cam_name == "CAM_L1":
            tiled_img[height:2*height, :width] = img
            filled_mask[height:2*height, :width] = 1
        elif cam_name == "CAM_R1":
            tiled_img[height:2*height, 2*width:] = img
            filled_mask[height:2*height, 2*width:] = 1
        elif cam_name == "CAM_L2":
            tiled_img[2*height:, :width] = img
            filled_mask[2*height:, :width] = 1
        elif cam_name == "CAM_R2":
            tiled_img[2*height:, 2*width:] = img
            filled_mask[2*height:, 2*width:] = 1
        elif cam_name == "CAM_B0":
            tiled_img[2*height:, width:2*width] = img
            filled_mask[2*height:, width:2*width] = 1

    # Crop the image according to the largest filled area
    min_y, max_y = np.where(filled_mask)[0].min(), np.where(filled_mask)[0].max()
    min_x, max_x = np.where(filled_mask)[1].min(), np.where(filled_mask)[1].max()
    tiled_img = tiled_img[min_y:max_y+1, min_x:max_x+1]
    
    return tiled_img

def layout_waymo(
    imgs: List[np.array], cam_names: List[str]
) -> np.array:
    """Combine cameras into a tiled image.
    Layout:

        ######################################################################################
        # left_camera # front_left_camera # front_camera # front_right_camera # right_camera #
        ######################################################################################
    """
    channel = imgs[0].shape[-1]
    front_cam_idx = cam_names.index('front_camera')
    front_img = imgs[front_cam_idx]
    landscape_width, landscape_height = front_img.shape[1], front_img.shape[0]
    
    height = landscape_height
    width = landscape_width * 5
    tiled_img = np.zeros((height, width, channel), dtype=np.float32)
    filled_mask = np.zeros((height, width), dtype=np.uint8)
    
    for idx, cam_name in enumerate(cam_names):
        img = imgs[idx]
        if cam_name == "left_camera":
            tiled_img[landscape_height - img.shape[0]:, :landscape_width] = img
            filled_mask[landscape_height - img.shape[0]:, :landscape_width] = 1
        elif cam_name == "front_left_camera":
            tiled_img[:, landscape_width : 2 * landscape_width] = img
            filled_mask[:, landscape_width : 2 * landscape_width] = 1
        elif cam_name == "front_camera":
            tiled_img[:, 2 * landscape_width : 3 * landscape_width] = img
            filled_mask[:, 2 * landscape_width : 3 * landscape_width] = 1
        elif cam_name == "front_right_camera":
            tiled_img[:, 3 * landscape_width : 4 * landscape_width] = img
            filled_mask[:, 3 * landscape_width : 4 * landscape_width] = 1
        elif cam_name == "right_camera":
            tiled_img[landscape_height - img.shape[0]:, 4 * landscape_width :] = img
            filled_mask[landscape_height - img.shape[0]:, 4 * landscape_width :] = 1
    
    # crop the image according to the lagrest filled area
    min_y, max_y = np.where(filled_mask)[0].min(), np.where(filled_mask)[0].max()
    min_x, max_x = np.where(filled_mask)[1].min(), np.where(filled_mask)[1].max()
    tiled_img = tiled_img[min_y:max_y, min_x:max_x]
    return tiled_img

def layout_nuscenes(
    imgs: List[np.array], cam_names: List[str]
) -> np.array:
    """Combine cameras into a tiled image.
    Layout:

        ################################################################
        # CAM_FRONT_LEFT  #     CAM_FRONT      #     CAM_FRONT_RIGHT   #
        ################################################################
        #  CAM_BACK_LEFT  #     CAM_BACK       #     CAM_BACK_RIGHT    #
        ################################################################
    """
    channel = imgs[0].shape[-1]
    for img in imgs:
        landscape_width = max(img.shape[0], img.shape[1])
        landscape_height = min(img.shape[0], img.shape[1])
        break

    height = landscape_height * 2
    width = landscape_width * 3
    tiled_img = np.zeros((height, width, channel), dtype=np.float32)
    filled_mask = np.zeros((height, width), dtype=np.uint8)
    
    for idx, cam_name in enumerate(cam_names):
        img = imgs[idx]
        if cam_name == "CAM_FRONT_LEFT":
            tiled_img[:landscape_height, :landscape_width] = img
            filled_mask[:landscape_height, :landscape_width] = 1
        elif cam_name == "CAM_FRONT":
            tiled_img[:landscape_height, landscape_width : 2 * landscape_width] = img
            filled_mask[:landscape_height, landscape_width : 2 * landscape_width] = 1
        elif cam_name == "CAM_FRONT_RIGHT":
            tiled_img[:landscape_height, 2 * landscape_width :] = img
            filled_mask[:landscape_height, 2 * landscape_width :] = 1
        elif cam_name == "CAM_BACK_LEFT":
            tiled_img[landscape_height :, :landscape_width] = img
            filled_mask[landscape_height :, :landscape_width] = 1
        elif cam_name == "CAM_BACK":
            tiled_img[landscape_height :, landscape_width : 2 * landscape_width] = img
            filled_mask[landscape_height :, landscape_width : 2 * landscape_width] = 1
        elif cam_name == "CAM_BACK_RIGHT":
            tiled_img[landscape_height :, 2 * landscape_width :] = img
            filled_mask[landscape_height :, 2 * landscape_width :] = 1
    
    # crop the image according to the largest filled area
    min_y, max_y = np.where(filled_mask)[0].min(), np.where(filled_mask)[0].max()
    min_x, max_x = np.where(filled_mask)[1].min(), np.where(filled_mask)[1].max()
    tiled_img = tiled_img[min_y:max_y, min_x:max_x]
    return tiled_img

def layout_pandaset(
    imgs: List[np.array], cam_names: List[str]
) -> np.array:
    """Combine cameras into a tiled image.
    Layout:

        ################################################################
        # front_left_camera #    front_camera     # front_right_camera #
        ################################################################
        #    left_camera    #     back_camera     #     right_camera   #
        ################################################################
    """
    channel = imgs[0].shape[-1]
    for img in imgs:
        landscape_width = max(img.shape[0], img.shape[1])
        landscape_height = min(img.shape[0], img.shape[1])
        break

    height = landscape_height + landscape_height
    width = landscape_width + landscape_width + landscape_width
    tiled_img = np.zeros((height, width, channel), dtype=np.float32)
    filled_mask = np.zeros((height, width), dtype=np.uint8)
    
    for idx, cam_name in enumerate(cam_names):
        img = imgs[idx]
        if cam_name == "front_left_camera":
            tiled_img[:landscape_height, :landscape_width] = img
            filled_mask[:landscape_height, :landscape_width] = 1
        elif cam_name == "front_camera":
            tiled_img[:landscape_height, landscape_width : 2 * landscape_width] = img
            filled_mask[:landscape_height, landscape_width : 2 * landscape_width] = 1
        elif cam_name == "front_right_camera":
            tiled_img[:landscape_height, 2 * landscape_width :] = img
            filled_mask[:landscape_height, 2 * landscape_width :] = 1
        elif cam_name == "left_camera":
            tiled_img[landscape_height :, :landscape_width] = img
            filled_mask[landscape_height :, :landscape_width] = 1
        elif cam_name == "back_camera":
            tiled_img[landscape_height :, landscape_width : 2 * landscape_width] = img
            filled_mask[landscape_height :, landscape_width : 2 * landscape_width] = 1
        elif cam_name == "right_camera":
            tiled_img[landscape_height :, 2 * landscape_width :] = img
            filled_mask[landscape_height :, 2 * landscape_width :] = 1
    
    # crop the image according to the lagrest filled area
    min_y, max_y = np.where(filled_mask)[0].min(), np.where(filled_mask)[0].max()
    min_x, max_x = np.where(filled_mask)[1].min(), np.where(filled_mask)[1].max()
    tiled_img = tiled_img[min_y:max_y, min_x:max_x]
    return tiled_img

def layout_kitti(
    imgs: List[np.array], cam_names: List[str]
) -> np.array:
    """Combine cameras into a tiled image.
    Layout:

        ##############################
        #    CAM_LEFT  #  CAM_RIGHT  #
        ##############################
    """
    channel = imgs[0].shape[-1]
    height = imgs[0].shape[0]
    width = imgs[0].shape[1] * 2
    
    tiled_img = np.zeros((height, width, channel), dtype=np.float32)
    filled_mask = np.zeros((height, width), dtype=np.uint8)
    
    for idx, cam_name in enumerate(cam_names):
        img = imgs[idx]
        if cam_name == "CAM_LEFT":
            tiled_img[:, :img.shape[1]] = img
            filled_mask[:, :img.shape[1]] = 1
        elif cam_name == "CAM_RIGHT":
            tiled_img[:, img.shape[1]:] = img
            filled_mask[:, img.shape[1]:] = 1
    
    # crop the image according to the largest filled area
    min_y, max_y = np.where(filled_mask)[0].min(), np.where(filled_mask)[0].max()
    min_x, max_x = np.where(filled_mask)[1].min(), np.where(filled_mask)[1].max()
    tiled_img = tiled_img[min_y:max_y, min_x:max_x]
    
    return tiled_img

def layout_argoverse(
    imgs: List[np.array], cam_names: List[str]
) -> np.array:
    """Combine cameras into a tiled image.
    Layout:

        ##########################################################
        # ring_front_left # ring_front_center # ring_front_right #
        ##########################################################
        # ring_side_left  #                   #  ring_side_right #
        ##########################################################
        ############ ring_rear_left # ring_rear_right ############
        ##########################################################
    """
    channel = imgs[0].shape[-1]
    for img in imgs:
        landscape_width = max(img.shape[0], img.shape[1])
        landscape_height = min(img.shape[0], img.shape[1])
        break

    height = landscape_height + landscape_height + landscape_height
    width = landscape_width + landscape_height + landscape_width
    tiled_img = np.zeros((height, width, channel), dtype=np.float32)
    filled_mask = np.zeros((height, width), dtype=np.uint8)
    
    for idx, cam_name in enumerate(cam_names):
        img = imgs[idx]
        if cam_name == "ring_front_left":
            tiled_img[:landscape_height, :landscape_width] = img
            filled_mask[:landscape_height, :landscape_width] = 1
        elif cam_name == "ring_front_center":
            tiled_img[:landscape_height, landscape_width : landscape_width + landscape_height] = img[:landscape_height, :]
            filled_mask[:landscape_height, landscape_width : landscape_width + landscape_height] = 1
        elif cam_name == "ring_front_right":
            tiled_img[:landscape_height, landscape_width + landscape_height :] = img
            filled_mask[:landscape_height, landscape_width + landscape_height :] = 1
        elif cam_name == "ring_side_left":
            tiled_img[landscape_height : 2 * landscape_height, :landscape_width] = img
            filled_mask[landscape_height : 2 * landscape_height, :landscape_width] = 1
        elif cam_name == "ring_side_right":
            tiled_img[
                landscape_height : 2 * landscape_height,
                landscape_width + landscape_height :
            ] = img
            filled_mask[
                landscape_height : 2 * landscape_height,
                landscape_width + landscape_height :
            ] = 1
        elif cam_name == "ring_rear_left":
            tiled_img[
                2 * landscape_height : 3 * landscape_height, int(0.5 * landscape_height):int(landscape_width + 0.5 * landscape_height)
            ] = img
            filled_mask[
                2 * landscape_height : 3 * landscape_height, int(0.5 * landscape_height):int(landscape_width + 0.5 * landscape_height)
            ] = 1
        elif cam_name == "ring_rear_right":
            tiled_img[
                2 * landscape_height : 3 * landscape_height, int(landscape_width + 0.5 * landscape_height):int(2 * landscape_width + 0.5 * landscape_height)
            ] = img
            filled_mask[
                2 * landscape_height : 3 * landscape_height, int(landscape_width + 0.5 * landscape_height):int(2 * landscape_width + 0.5 * landscape_height)
            ] = 1
    
    # crop the image according to the lagrest filled area
    min_y, max_y = np.where(filled_mask)[0].min(), np.where(filled_mask)[0].max()
    min_x, max_x = np.where(filled_mask)[1].min(), np.where(filled_mask)[1].max()
    tiled_img = tiled_img[min_y:max_y, min_x:max_x]
    return tiled_img

def dump_3d_bbox_on_image(
    coords, img,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = img.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    coords = coords.astype(np.int32)
    for index in range(coords.shape[0]):
        if isinstance(color, tuple):
            c = color
        elif isinstance(color, list):
            c = color[index]
        projected_points2d = coords[index]
        bbox = projected_points2d.tolist()
        cv2.line(canvas, bbox[0], bbox[1], c, thickness)
        cv2.line(canvas, bbox[0], bbox[4], c, thickness)
        cv2.line(canvas, bbox[0], bbox[3], c, thickness)
        cv2.line(canvas, bbox[1], bbox[2], c, thickness)
        cv2.line(canvas, bbox[1], bbox[5], c, thickness)
        cv2.line(canvas, bbox[2], bbox[3], c, thickness)
        cv2.line(canvas, bbox[2], bbox[6], c, thickness)
        cv2.line(canvas, bbox[3], bbox[7], c, thickness)
        cv2.line(canvas, bbox[4], bbox[7], c, thickness)
        cv2.line(canvas, bbox[4], bbox[5], c, thickness)
        cv2.line(canvas, bbox[5], bbox[6], c, thickness)
        cv2.line(canvas, bbox[6], bbox[7], c, thickness)
    canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    return canvas

def color_mapper(id: str) -> tuple:
    # use SHA256 to hash the id
    hash_object = hashlib.sha256(id.encode())
    hash_hex = hash_object.hexdigest()
    
    r = int(hash_hex[0:2], 16)
    g = int(hash_hex[2:4], 16)
    b = int(hash_hex[4:6], 16)
    return (r, g, b)

def sinebow(h):
    """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
    f = lambda x: np.sin(np.pi * x) ** 2
    return np.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def matte(vis, acc, dark=0.8, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :],
    )
    bg = np.where(bg_mask, light, dark)
    return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]


def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)


def visualize_cmap(
    value,
    weight,
    colormap,
    lo=None,
    hi=None,
    percentile=99.0,
    curve_fn=lambda x: x,
    modulus=None,
    matte_background=True,
):
    """Visualize a 1D image and a 1D weighting according to some colormap.
    from mipnerf

    Args:
      value: A 1D image.
      weight: A weight map, in [0, 1].
      colormap: A colormap function.
      lo: The lower bound to use when rendering, if None then use a percentile.
      hi: The upper bound to use when rendering, if None then use a percentile.
      percentile: What percentile of the value map to crop to when automatically
        generating `lo` and `hi`. Depends on `weight` as well as `value'.
      curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
        before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
      modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
        `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
      matte_background: If True, matte the image over a checkerboard.

    Returns:
      A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    if lo is None or hi is None:
        lo_auto, hi_auto = weighted_percentile(
            value, weight, [50 - percentile / 2, 50 + percentile / 2]
        )
        # If `lo` or `hi` are None, use the automatically-computed bounds above.
        eps = np.finfo(np.float32).eps
        lo = lo or (lo_auto - eps)
        hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
            np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
        )
    if weight is not None:
        value *= weight
    else:
        weight = np.ones_like(value)
    if colormap:
        colorized = colormap(value)[..., :3]
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return matte(colorized, weight) if matte_background else colorized


def visualize_depth(
    x, acc=None, lo=None, hi=None, depth_curve_fn=lambda x: -np.log(x + 1e-6)
):
    """Visualizes depth maps."""
    return visualize_cmap(
        x,
        acc,
        cm.get_cmap("turbo"),
        curve_fn=depth_curve_fn,
        lo=lo,
        hi=hi,
        matte_background=False,
    )
    
depth_visualizer = lambda frame, opacity: visualize_depth(
    frame,
    opacity,
    lo=4.0,
    hi=120,
    depth_curve_fn=lambda x: -np.log(x + 1e-6),
)