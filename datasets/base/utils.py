from typing import List, Tuple, Union
import numpy as np
import imageio
import numbers

import torch
import skimage
from skimage.transform import resize as cpu_resize
from torchvision.transforms.functional import resize as gpu_resize

def load_rgb(path: str, downscale: numbers.Number = 1) -> np.ndarray:
    """ Load image

    Args:
        path (str): Given image file path
        downscale (numbers.Number, optional): Optional downscale ratio. Defaults to 1.

    Returns:
        np.ndarray: [H, W, 3], in range [0,1]
    """
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    if downscale != 1:
        H, W, _ = img.shape
        img = cpu_resize(img, (int(H // downscale), int(W // downscale)), anti_aliasing=False)
    # [H, W, 3]
    return img
        
def img_to_torch_and_downscale(
    x: Union[np.ndarray, torch.Tensor], hw: Tuple[int, int],
    use_cpu_downscale=False, antialias=False, 
    dtype=None, device=None):
    """ Check, convert and apply downscale to input image `x`
    
    Args:
        x (Union[np.ndarray, torch.Tensor]): [H, W, (...)] Input image
        downscale (float, optional): Downscaling ratio. Defaults to 1.
        use_cpu_downscale (bool, optional): Whether use CPU downscaling algo (T), or use GPU (F). Defaults to False.
        antialias (bool, optional): Whether use anti-aliasing. Defaults to False.
        dtype (torch.dtype, optional): Output torch.dtype. Defaults to None.
        device (torch.device, optional): Output torch.device. Defaults to None.

    Returns:
        torch.Tensor: [new_H, new_W, (...)] Converted and downscaled torch.Tensor image
    """
    H_, W_ = hw
    if use_cpu_downscale:
        x_np = x if isinstance(x, np.ndarray) else x.data.cpu().numpy()
        x = torch.tensor(cpu_resize(x_np, (H_, W_), anti_aliasing=antialias),
                            dtype=dtype, device=device)
    else:
        x = check_to_torch(x, dtype=dtype, device=device)
        x = x.cuda() if not x.is_cuda else x
        if x.dim() == 2:
            x = gpu_resize(x.unsqueeze(0), (H_, W_), antialias=antialias).squeeze(0)
        else:
            x = gpu_resize(x.movedim(-1, 0), (H_, W_), antialias=antialias).movedim(0, -1)
    assert [H_, W_] == [*x.shape[:2]]
    return check_to_torch(x, dtype=dtype, device=device)

def check_to_torch(
    x: Union[np.ndarray, torch.Tensor, List, Tuple],
    ref: torch.Tensor=None, dtype: torch.dtype=None, device: torch.device=None) -> torch.Tensor:
    """ Check and convert input `x` to torch.Tensor

    Args:
        x (Union[np.ndarray, torch.Tensor, List, Tuple]): Input
        ref (torch.Tensor, optional): Reference tensor for dtype and device. Defaults to None.
        dtype (torch.dtype, optional): Target torch.dtype. Defaults to None.
        device (torch.device, optional): Target torch.device. Defaults to None.

    Returns:
        torch.Tensor: Converted torch.Tensor
    """
    if ref is not None:
        if dtype is None:
            dtype = ref.dtype
        if device is None:
            device = ref.device
    if x is None:
        return x
    elif isinstance(x, torch.Tensor):
        return x.to(dtype=dtype or x.dtype, device=device or x.device)
    else:
        return torch.tensor(x, dtype=dtype, device=device)