# Utility functions for geometric transformations and projections.
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

def transform_points(points, transform_matrix):
    """
    Apply a 4x4 transformation matrix to 3D points.

    Args:
        points: (N, 3) tensor of 3D points
        transform_matrix: (4, 4) transformation matrix

    Returns:
        (N, 3) tensor of transformed 3D points
    """
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    homo_points = torch.cat([points, ones], dim=1)  # N x 4
    transformed_points = torch.matmul(homo_points, transform_matrix.T)
    return transformed_points[:, :3]

def get_corners(l: float, w: float, h: float):
    """
    Get 8 corners of a 3D bounding box centered at origin.

    Args:
        l, w, h: length, width, height of the box

    Returns:
        (3, 8) array of corner coordinates
    """
    return np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2],
    ])
    
def project_camera_points_to_image(points_cam, cam_intrinsics):
    """
    Project 3D points from camera space to 2D image space.

    Args:
        points_cam (np.ndarray): Shape (N, 3), points in camera space.
        cam_intrinsics (np.ndarray): Shape (3, 3), intrinsic matrix of the camera.

    Returns:
        tuple: (projected_points, depths)
            - projected_points (np.ndarray): Shape (N, 2), projected 2D points in image space.
            - depths (np.ndarray): Shape (N,), depth values of the projected points.
    """
    points_img = cam_intrinsics @ points_cam.T
    depths = points_img[2, :]
    projected_points = (points_img[:2, :] / (depths + 1e-6)).T
    
    return projected_points, depths

def cube_root(x):
    return torch.sign(x) * torch.abs(x) ** (1. / 3)

def spherical_to_cartesian(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)

def uniform_sample_sphere(num_samples, device, inverse=False):
    """
    refer to https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    sample points uniformly inside a sphere
    """
    if not inverse:
        dist = torch.rand((num_samples,)).to(device)
        dist = cube_root(dist)
    else:
        dist = torch.rand((num_samples,)).to(device)
        dist = 1 / dist.clamp_min(0.02)
    thetas = torch.arccos(2 * torch.rand((num_samples,)) - 1).to(device)
    phis = 2 * torch.pi * torch.rand((num_samples,)).to(device)
    pts = spherical_to_cartesian(dist, thetas, phis)
    return pts

def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def scale_ftheta_calibration(parameters, width, height, fit=False, scale=None):
    """Resize NCore calibration; stored principal points use integer pixel centers."""
    if parameters['reference_poly'] not in ('PIXELDIST_TO_ANGLE', 'ANGLE_TO_PIXELDIST'):
        raise ValueError('Unknown FTheta reference polynomial')
    for key, count in (
        ('resolution', 2),
        ('principal_point', 2),
        ('linear_cde', 3),
        ('pixeldist_to_angle_poly', 6),
        ('angle_to_pixeldist_poly', 6),
    ):
        values = np.asarray(parameters[key], dtype=np.float64)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError('Invalid FTheta calibration: ' + key)
    if min(parameters['resolution']) <= 0 or not 0 < float(parameters['max_angle']) < np.pi:
        raise ValueError('Invalid FTheta resolution or max_angle')
    sx, sy = width / parameters['resolution'][0], height / parameters['resolution'][1]
    # interpolate(scale_factor=...) rounds output sizes without changing the pixel scale.
    if scale is not None:
        sx = sy = float(scale)
    offset = [0.0, 0.0]
    if fit:
        sx = sy = min(sx, sy)
        offset = [
            (width - parameters["resolution"][0] * sx) / 2,
            (height - parameters["resolution"][1] * sx) / 2,
        ]
    if sx <= 0 or not np.isclose(sx, sy, rtol=1e-5):
        raise ValueError('FTheta requires positive, isotropic image scaling')
    p = dict(parameters)
    p['resolution'] = [int(width), int(height)]
    p['principal_point'] = [
        (float(v) + 0.5) * sx - 0.5 + offset[i] for i, v in enumerate(p['principal_point'])
    ]
    p['pixeldist_to_angle_poly'] = [
        float(v) / sx**i for i, v in enumerate(p['pixeldist_to_angle_poly'])
    ]
    p['angle_to_pixeldist_poly'] = [float(v) * sx for v in p['angle_to_pixeldist_poly']]
    return p


def _camera_polynomial(coefficients, x):
    y = torch.zeros_like(x)
    for c in reversed(coefficients):
        y = y * x + float(c)
    return y


def _invert_camera_polynomial(coefficients, target, initial):
    derivative = [i * float(c) for i, c in enumerate(coefficients)][1:]
    value = initial.clone()
    for _ in range(12):
        slope = _camera_polynomial(derivative, value)
        safe = torch.where(slope.abs() > 1e-10, slope, torch.ones_like(slope))
        value = value - (_camera_polynomial(coefficients, value) - target) / safe
    valid = torch.isfinite(value) & (
        (_camera_polynomial(coefficients, value) - target).abs() < 1e-4 * (1 + target.abs())
    )
    return value, valid


def project_camera_model(
    points, intrinsics, camera_model='pinhole', radial_coeffs=None, ftheta_parameters=None
):
    """Project camera-space points to pixel-center coordinates, with validity."""
    r = torch.linalg.vector_norm(points[..., :2], dim=-1)
    theta = torch.atan2(r, points[..., 2])
    valid = torch.isfinite(points).all(-1) & (points.norm(dim=-1) > 1e-8)
    if camera_model == 'pinhole':
        xy = points[..., :2] / points[..., 2:].clamp_min(1e-8)
        pixels = xy * intrinsics.diagonal()[:2] + intrinsics[:2, 2]
        return pixels, valid & (points[..., 2] > 0)
    if camera_model == 'fisheye':
        coefficients = [0.0, 1.0]
        for k in (radial_coeffs if radial_coeffs is not None else [0.0] * 4):
            coefficients.extend([0.0, float(k)])
        radius = _camera_polynomial(coefficients, theta)
        xy = points[..., :2] * (radius / r.clamp_min(1e-8))[..., None]
        pixels = xy * intrinsics.diagonal()[:2] + intrinsics[:2, 2]
        return pixels, valid & (theta < np.pi / 2)
    if camera_model != 'ftheta' or ftheta_parameters is None:
        raise ValueError('Expected pinhole, fisheye, or ftheta with calibration')
    p = ftheta_parameters
    radius = _camera_polynomial(p['angle_to_pixeldist_poly'], theta)
    if p['reference_poly'] == 'PIXELDIST_TO_ANGLE':
        radius, converged = _invert_camera_polynomial(p['pixeldist_to_angle_poly'], theta, radius)
        valid &= converged
    elif p['reference_poly'] != 'ANGLE_TO_PIXELDIST':
        raise ValueError('Unknown FTheta reference polynomial')
    xy = points[..., :2] * (radius / r.clamp_min(1e-8))[..., None]
    c, d, e = [float(v) for v in p['linear_cde']]
    pixels = torch.stack((c * xy[..., 0] + d * xy[..., 1], e * xy[..., 0] + xy[..., 1]), -1)
    pixels += points.new_tensor(p['principal_point']) + 0.5
    valid &= (theta <= float(p['max_angle'])) & (radius >= 0) & torch.isfinite(pixels).all(-1)
    return pixels, valid


def camera_model_rays(
    height, width, intrinsics, camera_model='pinhole', radial_coeffs=None, ftheta_parameters=None
):
    """Unit camera rays and valid pixels. Pixel centers are (x + .5, y + .5)."""
    y, x = torch.meshgrid(
        torch.arange(height, device=intrinsics.device, dtype=intrinsics.dtype) + 0.5,
        torch.arange(width, device=intrinsics.device, dtype=intrinsics.dtype) + 0.5,
        indexing='ij',
    )
    pixels = torch.stack((x, y), -1)
    valid = torch.ones((height, width), dtype=torch.bool, device=intrinsics.device)
    if camera_model == 'pinhole':
        xy = (pixels - intrinsics[:2, 2]) / intrinsics.diagonal()[:2]
        return F.normalize(torch.cat((xy, torch.ones_like(x[..., None])), -1), dim=-1), valid
    if camera_model == 'fisheye':
        xy = (pixels - intrinsics[:2, 2]) / intrinsics.diagonal()[:2]
        radius = xy.norm(dim=-1)
        coefficients = [0.0, 1.0]
        for k in (radial_coeffs if radial_coeffs is not None else [0.0] * 4):
            coefficients.extend([0.0, float(k)])
        theta, valid = _invert_camera_polynomial(coefficients, radius, radius)
        valid &= (theta >= 0) & (theta < np.pi / 2)
    elif camera_model == 'ftheta' and ftheta_parameters is not None:
        p = ftheta_parameters
        c, d, e = [float(v) for v in p['linear_cde']]
        if abs(c - d * e) < 1e-10:
            raise ValueError('Singular FTheta affine calibration')
        offset = pixels - (intrinsics.new_tensor(p['principal_point']) + 0.5)
        xy = torch.stack(
            (offset[..., 0] - d * offset[..., 1], c * offset[..., 1] - e * offset[..., 0]), -1
        ) / (c - d * e)
        radius = xy.norm(dim=-1)
        theta = _camera_polynomial(p['pixeldist_to_angle_poly'], radius)
        if p['reference_poly'] == 'ANGLE_TO_PIXELDIST':
            theta, valid = _invert_camera_polynomial(p['angle_to_pixeldist_poly'], radius, theta)
        elif p['reference_poly'] != 'PIXELDIST_TO_ANGLE':
            raise ValueError('Unknown FTheta reference polynomial')
        valid &= (theta >= 0) & (theta <= float(p['max_angle']))
    else:
        raise ValueError('Expected pinhole, fisheye, or calibrated ftheta camera')
    rays = torch.cat(
        (xy * (torch.sin(theta) / radius.clamp_min(1e-8))[..., None], torch.cos(theta)[..., None]),
        -1,
    )
    rays = torch.where((radius < 1e-8)[..., None], rays.new_tensor([0.0, 0.0, 1.0]), rays)
    valid &= torch.isfinite(rays).all(-1)
    return torch.nan_to_num(rays), valid


def camera_midpoint_pose(start, end):
    """One representative camera-to-world pose for a complete image."""
    from scipy.spatial.transform import Rotation, Slerp

    pose = np.eye(4)
    pose[:3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((start[:3, :3], end[:3, :3]))))(
        0.5
    ).as_matrix()
    pose[:3, 3] = (start[:3, 3] + end[:3, 3]) * 0.5
    return pose


def project_world_camera(
    points,
    pose,
    intrinsics,
    width,
    height,
    camera_model='pinhole',
    radial_coeffs=None,
    ftheta_parameters=None,
):
    """Project world points using one camera pose for the whole image."""
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    pixels, valid = project_camera_model(
        local, intrinsics, camera_model, radial_coeffs, ftheta_parameters
    )
    valid &= (
        (pixels[..., 0] >= 0)
        & (pixels[..., 0] < width)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < height)
    )
    return pixels, local[..., 2], valid