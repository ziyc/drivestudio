import math
import numpy as np

import torch
from numpy.typing import NDArray

from typing import Dict, List, Optional, Union
from dataclasses import dataclass, fields
from sklearn.neighbors import NearestNeighbors
from pytorch3d.transforms import matrix_to_quaternion

from gsplat.rendering import rasterization
from gsplat.cuda_legacy._wrapper import num_sh_bases
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from gsplat.cuda._wrapper import spherical_harmonics

def interpolate_quats(q1, q2, fraction=0.5):
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
    
    dot = (q1 * q2).sum(dim=-1)
    dot = torch.clamp(dot, -1, 1)
    
    neg_mask = dot < 0
    q2[neg_mask] = -q2[neg_mask]
    dot[neg_mask] = -dot[neg_mask]
    
    similar_mask = dot > 0.9995
    q_interp_similar = q1 + fraction * (q2 - q1)

    theta_0 = torch.acos(dot)
    theta = theta_0 * fraction
    
    sin_theta = torch.sin(theta)
    sin_theta_0 = torch.sin(theta_0)
    
    s1 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    
    q_interp = (s1[..., None] * q1) + (s2[..., None] * q2)
    
    final_q_interp = torch.zeros_like(q1)
    final_q_interp[similar_mask] = q_interp_similar[similar_mask]
    final_q_interp[~similar_mask] = q_interp[~similar_mask]
    return final_q_interp

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )

def quat_mult(q1, q2):
    # NOTE:
    # Q1 is the quaternion that rotates the vector from the original position to the final position
    # Q2 is the quaternion that been rotated
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T

def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def projection_matrix(znear, zfar, fovx, fovy, device:Union[str,torch.device]="cpu"):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )

@dataclass
class dataclass_camera:
    camtoworlds: torch.Tensor
    camtoworlds_gt: torch.Tensor
    Ks: torch.Tensor
    H: int
    W: int

@dataclass
class dataclass_gs:
    _opacities: torch.Tensor
    _means: torch.Tensor
    _rgbs: torch.Tensor
    _scales: torch.Tensor
    _quats: torch.Tensor
    detach_keys: List[str]
    extras: Optional[Dict[str, torch.Tensor]] = None
    def set_grad_controller(self, detach_keys):
        self.detach_keys = detach_keys
    @property
    def opacities(self):
        if "activated_opacities" in self.detach_keys:
            return self._opacities.detach()
        else:
            return self._opacities
    @property
    def means(self):
        if "means" in self.detach_keys:
            return self._means.detach()
        else:
            return self._means
    @property
    def rgbs(self):
        if "colors" in self.detach_keys:
            return self._rgbs.detach()
        else:
            return self._rgbs
    @property
    def scales(self):
        if "scales" in self.detach_keys:
            return self._scales.detach()
        else:
            return self._scales
    @property
    def quats(self):
        if "quats" in self.detach_keys:
            return self._quats.detach()
        else:
            return self._quats
        
def remove_from_optim(optimizer, deleted_mask, param_dict):
    """removes the deleted_mask from the optimizer provided"""
    for group_idx, group in enumerate(optimizer.param_groups):
        name = group["name"]
        if name in param_dict.keys():
            old_params = group["params"][0]
            new_params = param_dict[name]
            assert len(new_params) == 1
            param_state = optimizer.state[old_params]
            del optimizer.state[old_params]

            # Modify the state directly without deleting and reassigning.
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

            # Update the parameter in the optimizer's param group.
            del optimizer.param_groups[group_idx]["params"][0]
            del optimizer.param_groups[group_idx]["params"]
            optimizer.param_groups[group_idx]["params"] = new_params
            optimizer.state[new_params[0]] = param_state

def dup_in_optim(optimizer, dup_mask, param_dict, n=2):
    """adds the parameters to the optimizer"""
    for group_idx, group in enumerate(optimizer.param_groups):
        name = group["name"]
        if name in param_dict.keys():
            old_params = group["params"][0]
            new_params = param_dict[name]
            param_state = optimizer.state[old_params]
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [param_state["exp_avg"], torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims)],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            del optimizer.state[old_params]
            optimizer.state[new_params[0]] = param_state
            optimizer.param_groups[group_idx]["params"] = new_params
            del old_params
    
def k_nearest_sklearn(x: torch.Tensor, k: int):
    """
    Find k-nearest neighbors using sklearn's NearestNeighbors.
    x: The data tensor of shape [num_samples, num_features]
    k: The number of neighbors to retrieve
    """
    # Convert tensor to numpy array
    x_np = x.cpu().numpy()

    # Build the nearest neighbors model
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

    # Find the k-nearest neighbors
    distances, indices = nn_model.kneighbors(x_np)

    # Exclude the point itself from the result and return
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

if __name__ == "__main__":
    quats_prev_frame = torch.tensor([
        [ 4.3390e-02,  4.1600e-06, -9.5784e-05,  9.9906e-01],
        [ 1.1272e-04,  1.0807e-08,  9.5874e-05, -1.0000e+00],
        [ 1.7490e-04,  1.6769e-08, -9.5874e-05,  1.0000e+00]
    ], device='cuda:0')
    
    quats_next_frame = torch.tensor([
        [ 4.2516e-02,  4.0762e-06, -9.5787e-05,  9.9910e-01],
        [ 3.8867e-05,  3.7264e-09, -9.5874e-05,  1.0000e+00],
        [ 1.8267e-04,  1.7513e-08, -9.5874e-05,  1.0000e+00]
    ], device='cuda:0')
    
    quats_cur_frame = interpolate_quats(quats_prev_frame, quats_next_frame, 0.5)