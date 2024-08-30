import numpy as np
from typing import Literal, Union

import torch
import torch.nn.functional as F
from torch import autograd, nn, Tensor

def reduce(
    loss: Union[torch.Tensor, np.ndarray], 
    mask: Union[torch.Tensor, np.ndarray] = None, 
    reduction: Literal['mean', 'mean_in_mask', 'sum', 'max', 'min', 'none']='mean'):

    if mask is not None:
        if mask.dim() == loss.dim() - 1:
            mask = mask.view(*loss.shape[:-1], 1).expand_as(loss)
        assert loss.dim() == mask.dim(), f"Expects loss.dim={loss.dim()} to be equal to mask.dim()={mask.dim()}"
    
    if reduction == 'mean':
        return loss.mean() if mask is None else (loss * mask).mean()
    elif reduction == 'mean_in_mask':
        return loss.mean() if mask is None else (loss * mask).sum() / mask.sum().clip(1e-5)
    elif reduction == 'sum':
        return loss.sum() if mask is None else (loss * mask).sum()
    elif reduction == 'max':
        return loss.max() if mask is None else loss[mask].max()
    elif reduction == 'min':
        return loss.min() if mask is None else loss[mask].min()
    elif reduction == 'none':
        return loss if mask is None else loss * mask
    else:
        raise RuntimeError(f"Invalid reduction={reduction}")

class SafeBCE(autograd.Function):
    """ Perform clipped BCE without disgarding gradients (preserve clipped gradients)
        This function is equivalent to torch.clip(x, limit), 1-limit) before BCE, 
        BUT with grad existing on those clipped values.
        
    NOTE: pytorch original BCELoss implementation is equivalent to limit = np.exp(-100) here.
        see doc https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    """
    @staticmethod
    def forward(ctx, x, y, limit):
        assert (torch.where(y!=1, y+1, y)==1).all(), u'target must all be {0,1}'
        ln_limit = ctx.ln_limit = np.log(limit)
        # ctx.clip_grad = clip_grad
        
        # NOTE: for example, torch.log(1-torch.tensor([1.000001])) = nan
        x = torch.clip(x, 0, 1)
        y = torch.clip(y, 0, 1)
        ctx.save_for_backward(x, y)
        return -torch.where(y==0, torch.log(1-x).clamp_min_(ln_limit), torch.log(x).clamp_min_(ln_limit))
        # return -(y * torch.log(x).clamp_min_(ln_limit) + (1-y)*torch.log(1-x).clamp_min_(ln_limit))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        ln_limit = ctx.ln_limit
        
        # NOTE: for y==0, do not clip small x; for y==1, do not clip small (1-x)
        limit = np.exp(ln_limit)
        # x = torch.clip(x, eclip, 1-eclip)
        x = torch.where(y==0, torch.clip(x, 0, 1-limit), torch.clip(x, limit, 1))
        
        grad_x = grad_y = None
        if ctx.needs_input_grad[0]:
            # ttt = torch.where(y==0, 1/(1-x), -1/x) * grad_output * (~(x==y))
            # with open('grad.txt', 'a') as fp:
            #     fp.write(f"{ttt.min().item():.05f}, {ttt.max().item():.05f}\n")
            # NOTE: " * (~(x==y))" so that those already match will not generate gradients.
            grad_x = torch.where(y==0, 1/(1-x), -1/x) * grad_output * (~(x==y))
            # grad_x = ( (1-y)/(1-x) - y/x ) * grad_output
        if ctx.needs_input_grad[1]:
            grad_y = (torch.log(1-x) - torch.log(x)) * grad_output * (~(x==y))
        #---- x, y, limit
        return grad_x, grad_y, None

def safe_binary_cross_entropy(input: torch.Tensor, target: torch.Tensor, limit: float = 0.1, reduction="mean") -> torch.Tensor:
    loss = SafeBCE.apply(input, target, limit)
    return reduce(loss, None, reduction=reduction)

def binary_cross_entropy(input: torch.Tensor, target: torch.Tensor, reduction="mean") -> torch.Tensor:
    loss = F.binary_cross_entropy(input, target, reduction="none")
    return reduce(loss, None, reduction=reduction)

def normalize_depth(depth: Tensor, max_depth: float = 80.0):
    return torch.clamp(depth / max_depth, 0.0, 1.0)

def safe_normalize_depth(depth: Tensor, max_depth: float = 80.0):
    return torch.clamp(depth / max_depth, 1e-06, 1.0)

class DepthLoss(nn.Module):
    def __init__(
        self,
        loss_type: Literal["l1", "l2", "smooth_l1"] = "l2",
        normalize: bool = True,
        use_inverse_depth: bool = False,
        depth_error_percentile: float = None,
        upper_bound: float = 80,
        reduction: Literal["mean_on_hit", "mean_on_hw", "sum", "none"] = "mean_on_hit",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        self.use_inverse_depth = use_inverse_depth
        self.upper_bound = upper_bound
        self.depth_error_percentile = depth_error_percentile
        self.reduction = reduction

    def _compute_depth_loss(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor,
        max_depth: float = 80,
        hit_mask: Tensor = None,
    ):
        pred_depth = pred_depth.squeeze()
        gt_depth = gt_depth.squeeze()
        if hit_mask is not None:
            pred_depth = pred_depth * hit_mask
            gt_depth = gt_depth * hit_mask
        
        # cal valid mask to make sure gt_depth is valid
        valid_mask = (gt_depth > 0.01) & (gt_depth < max_depth) & (pred_depth > 0.0001)
        
        # normalize depth to (0, 1)
        if self.normalize:
            pred_depth = safe_normalize_depth(pred_depth[valid_mask], max_depth=max_depth)
            gt_depth = safe_normalize_depth(gt_depth[valid_mask], max_depth=max_depth)
        else:
            pred_depth = pred_depth[valid_mask]
            gt_depth = gt_depth[valid_mask]
        
        # inverse the depth map (0, 1) -> (1, +inf)
        if self.use_inverse_depth:
            pred_depth = 1./pred_depth
            gt_depth = 1./gt_depth
            
        # cal loss
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_depth, gt_depth, reduction="none")
        elif self.loss_type == "l1":
            return F.l1_loss(pred_depth, gt_depth, reduction="none")
        elif self.loss_type == "l2":
            return F.mse_loss(pred_depth, gt_depth, reduction="none")
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

    def __call__(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor,
        hit_mask: Tensor = None,
    ):
        depth_error = self._compute_depth_loss(pred_depth, gt_depth, self.upper_bound, hit_mask)
        if self.depth_error_percentile is not None:
            # to avoid outliers. not used for now
            depth_error = depth_error.flatten()
            depth_error = depth_error[
                depth_error.argsort()[
                    : int(len(depth_error) * self.depth_error_percentile)
                ]
            ]
        
        if self.reduction == "sum":
            depth_error = depth_error.sum()
        elif self.reduction == "none":
            depth_error = depth_error
        elif self.reduction == "mean_on_hit":
            depth_error = depth_error.mean()
        elif self.reduction == "mean_on_hw":
            n = gt_depth.shape[0]*gt_depth.shape[1]
            depth_error = depth_error.sum() / n
        else:
            raise NotImplementedError(f"Unknown reduction method: {self.reduction}")

        return depth_error