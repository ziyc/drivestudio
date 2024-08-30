# Acknowledgement: https://github.com/JiahuiLei/GART
from typing import Optional
import os
import sys
import pickle
import trimesh

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    axis_angle_to_matrix
)

from models.modules import VoxelDeformer
from third_party.smplx.smplx import SMPLLayer
from third_party.smplx.smplx.utils import SMPLOutput
from third_party.smplx.smplx.lbs import vertices2joints, batch_rigid_transform

def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper

class SMPL(SMPLLayer):
    
    @blockPrinting
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        """
        Extension of the official SMPL implementation to support more joints.
        Args:
            Same as SMPLLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super(SMPL, self).__init__(*args, **kwargs)
        smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                            7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
            
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
        self.register_buffer('joint_map', torch.tensor(smpl_to_openpose, dtype=torch.long))

    def forward(self, *args, **kwargs) -> SMPLOutput:
        """
        Run forward pass. Same as SMPL and also append an extra set of joints if joint_regressor_extra is specified.
        """
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        joints = smpl_output.joints[:, self.joint_map, :]
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, smpl_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        smpl_output.joints = joints
        return smpl_output

def get_predefined_human_rest_pose(pose_type):
    print(f"Using predefined pose: {pose_type}")
    body_pose_t = torch.zeros((1, 69))
    if pose_type.lower() == "da_pose":
        body_pose_t[:, 2] = torch.pi / 6
        body_pose_t[:, 5] = -torch.pi / 6
    elif pose_type.lower() == "a_pose":
        body_pose_t[:, 2] = 0.2
        body_pose_t[:, 5] = -0.2
        body_pose_t[:, 47] = -0.8
        body_pose_t[:, 50] = 0.8
    elif pose_type.lower() == "t_pose":
        pass
    else:
        raise ValueError("Unknown cano_pose: {}".format(pose_type))
    return body_pose_t.reshape(23, 3)
    
class SMPLTemplate(nn.Module):
    def __init__(self, smpl_model_path, num_human, init_beta, cano_pose_type, voxel_deformer_res=64, use_voxel_deformer=False, is_resume=False):
        super().__init__()
        assert num_human == init_beta.shape[0], "num_human should be the same as the number of beta"
        self.num_human = num_human
        self.dim = 24
        self._template_layer = SMPLLayer(model_path=smpl_model_path)

        init_beta = torch.as_tensor(init_beta, dtype=torch.float32).cpu()
        self.register_buffer("init_beta", init_beta)
        self.cano_pose_type = cano_pose_type
        self.name = "smpl"

        can_pose = get_predefined_human_rest_pose(cano_pose_type)
        can_pose = axis_angle_to_matrix(torch.cat([torch.zeros(1, 3), can_pose], 0))
        self.register_buffer("canonical_pose", can_pose)

        init_smpl_output = self._template_layer(
            betas=init_beta,
            body_pose=can_pose[None, 1:].repeat(num_human, 1, 1, 1),
            global_orient=can_pose[None, 0].repeat(num_human, 1, 1, 1),
            return_full_pose=True,
        )
        J_canonical, A0 = init_smpl_output.J, init_smpl_output.A
        A0_inv = torch.inverse(A0)
        self.register_buffer("A0_inv", A0_inv)
        self.register_buffer("J_canonical", J_canonical)

        v_init = init_smpl_output.vertices  # [B, 6890, 3]
        W_init = self._template_layer.lbs_weights  # [6890, 24]
        self.register_buffer("W", W_init[None].repeat(num_human, 1, 1))

        self.use_voxel_deformer = use_voxel_deformer
        if self.use_voxel_deformer:
            self.voxel_deformer = VoxelDeformer(
                vtx=v_init,
                vtx_features=W_init.unsqueeze(0).repeat(num_human, 1, 1), # [B, 6890, 24]
                resolution_dhw=[
                    voxel_deformer_res // 4,
                    voxel_deformer_res,
                    voxel_deformer_res,
                ],
                is_resume=is_resume, # if is resume, the weight will be randomly initialized to save time
            )

        # * Important, record first joint position, because the global orientation is rotating using this joint position as center, so we can compute the action on later As
        j0_t = init_smpl_output.joints[:, 0]
        self.register_buffer("j0_t", j0_t)
        return

    def get_init_vf(self):
        init_smpl_output = self._template_layer(
            betas=self.init_beta,
            body_pose=self.canonical_pose[None, 1:].repeat(self.num_human, 1, 1, 1),
            global_orient=self.canonical_pose[None, 0].repeat(self.num_human, 1, 1, 1),
            return_full_pose=True,
        )
        v_init = init_smpl_output.vertices  # 1,6890,3
        v_init = v_init
        faces = self._template_layer.faces_tensor
        return v_init, faces

    def get_rot_action(self, axis_angle):
        # apply this action to canonical additional bones
        # axis_angle: B,3
        assert axis_angle.ndim == 2 and axis_angle.shape[-1] == 3
        B = len(axis_angle)
        R = axis_angle_to_matrix(axis_angle)  # B,3,3
        I = torch.eye(3).to(R)[None].expand(B, -1, -1)  # B,3,3
        t0 = self.j0_t[None].expand(B, -1)  # B,3
        T = torch.eye(4).to(R)[None].expand(B, -1, -1)  # B,4,4
        T[:, :3, :3] = R
        T[:, :3, 3] = torch.einsum("bij, bj -> bi", I - R, t0)
        return T  # B,4,4

    def forward(self, masked_theta=None, xyz_canonical=None, instances_mask=None):
        # skinning
        if masked_theta is None:
            A = None
        else:
            assert (
                masked_theta.ndim == 3 and masked_theta.shape[-1] == 4
            ), "pose should have shape Bx24x3, in axis-angle format"
            nB = len(masked_theta)
            _, A = batch_rigid_transform(
                quaternion_to_matrix(masked_theta),
                self.J_canonical[instances_mask],
                self._template_layer.parents,
            )
            A = torch.einsum("bnij, bnjk->bnik", A, self.A0_inv[instances_mask])  # B,24,4,4

        if xyz_canonical is None or not self.use_voxel_deformer:
            # forward theta only
            W = self.W
        else:
            W = self.voxel_deformer(xyz_canonical)  # B,N,24+K
        W = W[instances_mask]
        return W, A

    def remove_instance(self, ins_id):
        ins_mask = torch.ones(self.num_human, dtype=torch.bool)
        ins_mask[ins_id] = False
        self.J_canonical = self.J_canonical[ins_mask]
        self.W = self.W[ins_mask]
        self.num_human -= 1
        
        if self.use_voxel_deformer:
            self.voxel_deformer.lbs_voxel_base     = self.voxel_deformer.lbs_voxel_base[ins_mask]
            self.voxel_deformer.voxel_w_correction = nn.Parameter(self.voxel_deformer.voxel_w_correction[ins_mask])
            self.voxel_deformer.offset             = self.voxel_deformer.offset[ins_mask]
            self.voxel_deformer.scale              = self.voxel_deformer.scale[ins_mask]
        
    def add_instance(self, ins_id, new_dict):
        voxel_deformer_dict = new_dict["voxel_deformer"]
        self.J_canonical = torch.cat([self.J_canonical, new_dict['J_canonical'][None]], dim=0)
        self.W = torch.cat([self.W, new_dict['W'][None]], dim=0)
        self.num_human += 1
        
        if self.use_voxel_deformer:
            self.voxel_deformer.lbs_voxel_base     = torch.cat([self.voxel_deformer.lbs_voxel_base, voxel_deformer_dict['lbs_voxel_base'][None]], dim=0)
            self.voxel_deformer.voxel_w_correction = nn.Parameter(torch.cat([self.voxel_deformer.voxel_w_correction, voxel_deformer_dict['voxel_w_correction'][None]], dim=0))
            self.voxel_deformer.offset             = torch.cat([self.voxel_deformer.offset, voxel_deformer_dict['offset'][None]], dim=0)
            self.voxel_deformer.scale              = torch.cat([self.voxel_deformer.scale, voxel_deformer_dict['scale'][None]], dim=0)

def init_xyz_on_mesh(v_init, faces, subdivide_num):
    # * xyz
    denser_v, denser_f = v_init.detach().cpu().numpy(), faces
    for i in range(subdivide_num):
        denser_v, denser_f = trimesh.remesh.subdivide(denser_v, denser_f)
    body_mesh = trimesh.Trimesh(denser_v, denser_f, process=False)
    v_init = torch.as_tensor(denser_v, dtype=torch.float32)
    return v_init, body_mesh

def init_qso_on_mesh(
    body_mesh,
    scale_init_factor,
    thickness_init_factor,
    max_scale,
    min_scale,
    s_inv_act,
    opacity_base_logit,
):
    # * Quaternion
    # each column is a basis vector
    # the local frame is z to normal, xy on the disk
    normal = body_mesh.vertex_normals.copy()
    v_init = torch.as_tensor(body_mesh.vertices.copy())
    faces = torch.as_tensor(body_mesh.faces.copy())

    uz = torch.as_tensor(normal, dtype=torch.float32)
    rand_dir = torch.randn_like(uz)
    ux = F.normalize(torch.cross(uz, rand_dir, dim=-1), dim=-1)
    uy = F.normalize(torch.cross(uz, ux, dim=-1), dim=-1)
    frame = torch.stack([ux, uy, uz], dim=-1)  # N,3,3
    ret_q = matrix_to_quaternion(frame)

    # * Scaling
    xy = v_init[faces[:, 1]] - v_init[faces[:, 0]]
    xz = v_init[faces[:, 2]] - v_init[faces[:, 0]]
    area = torch.norm(torch.cross(xy, xz, dim=-1), dim=-1) / 2
    vtx_nn_area = torch.zeros_like(v_init[:, 0])
    for i in range(3):
        vtx_nn_area.scatter_add_(0, faces[:, i], area / 3.0)
    radius = torch.sqrt(vtx_nn_area / np.pi)
    # radius = torch.clamp(radius * scale_init_factor, max=max_scale, min=min_scale)
    # ! 2023.11.22, small eps
    radius = torch.clamp(
        radius * scale_init_factor, max=max_scale - 1e-4, min=min_scale + 1e-4
    )
    thickness = radius * thickness_init_factor
    # ! 2023.11.22, small eps
    thickness = torch.clamp(thickness, max=max_scale - 1e-4, min=min_scale + 1e-4)
    radius_logit = s_inv_act(radius)
    thickness_logit = s_inv_act(thickness)
    ret_s = torch.stack([radius_logit, radius_logit, thickness_logit], dim=-1)

    ret_o = torch.ones_like(v_init[:, :1]) * opacity_base_logit
    return ret_q, ret_s, ret_o

def get_on_mesh_init_geo_values(
    template,
    opacity_init_logit, 
    on_mesh_subdivide = 0,
    scale_init_factor = 1.0,
    thickness_init_factor = 0.5, 
    max_scale = 1.0,
    min_scale = 0.0,
    s_inv_act = torch.logit,
):
    v, f = template.get_init_vf()
    x_all, q_all, s_all, o_all = [], [], [], []
    for i in range(len(v)):
        x, mesh = init_xyz_on_mesh(v[i], f, on_mesh_subdivide)
        q, s, o = init_qso_on_mesh(
            mesh,
            scale_init_factor,
            thickness_init_factor,
            max_scale,
            min_scale,
            s_inv_act,
            opacity_init_logit,
        )
        
        x_all.append(x)
        q_all.append(q)
        s_all.append(s)
        o_all.append(o)
    
    x_all = torch.cat(x_all, dim=0)
    q_all = torch.cat(q_all, dim=0)
    s_all = torch.cat(s_all, dim=0)
    o_all = torch.cat(o_all, dim=0)
    return x_all, q_all, s_all, o_all
    
phalp_colors =[
    [213.0,255.0,0.0,],
    [255.0,0.0,86.0,],
    [158.0,0.0,142.0,],
    [14.0,76.0,161.0,],
    [255.0,229.0,2.0,],
    [0.0,95.0,57.0,],
    [0.0,255.0,0.0,],
    [149.0,0.0,58.0,],
    [255.0,147.0,126.0,],
    [164.0,36.0,0.0,],
    [0.0,21.0,68.0,],
    [145.0,208.0,203.0,],
    [98.0,14.0,0.0,],
    [107.0,104.0,130.0,],
    [0.0,0.0,255.0,],
    [0.0,125.0,181.0,],
    [106.0,130.0,108.0,],
    [0.0,174.0,126.0,],
    [194.0,140.0,159.0,],
    [190.0,153.0,112.0,],
    [0.0,143.0,156.0,],
    [95.0,173.0,78.0,],
    [255.0,0.0,0.0,],
    [255.0,0.0,246.0,],
    [255.0,2.0,157.0,],
    [104.0,61.0,59.0,],
    [255.0,116.0,163.0,],
    [150.0,138.0,232.0,],
    [152.0,255.0,82.0,],
    [167.0,87.0,64.0,],
    [1.0,255.0,254.0,],
    [255.0,238.0,232.0,],
    [254.0,137.0,0.0,],
    [189.0,198.0,255.0,],
    [1.0,208.0,255.0,],
    [187.0,136.0,0.0,],
    [117.0,68.0,177.0,],
    [165.0,255.0,210.0,],
    [255.0,166.0,254.0,],
    [119.0,77.0,0.0,],
    [122.0,71.0,130.0,],
    [38.0,52.0,0.0,],
    [0.0,71.0,84.0,],
    [67.0,0.0,44.0,],
    [181.0,0.0,255.0,],
    [255.0,177.0,103.0,],
    [255.0,219.0,102.0,],
    [144.0,251.0,146.0,],
    [126.0,45.0,210.0,],
    [189.0,211.0,147.0,],
    [229.0,111.0,254.0,],
    [222.0,255.0,116.0,],
    [0.0,255.0,120.0,],
    [0.0,155.0,255.0,],
    [0.0,100.0,1.0,],
    [0.0,118.0,255.0,],
    [133.0,169.0,0.0,],
    [0.0,185.0,23.0,],
    [120.0,130.0,49.0,],
    [0.0,255.0,198.0,],
    [255.0,110.0,65.0,],
    [232.0,94.0,190.0,],
    [0.0,0.0,0.0,],
    [1.0,0.0,103.0,],
]