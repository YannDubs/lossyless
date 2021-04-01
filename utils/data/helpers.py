import logging
import math

import torch
from torchvision.transforms import functional as F_trnsf

logger = logging.getLogger(__name__)


def rotate(x, angle):
    """Rotate a 2D tensor by a certain angle (in degrees)."""
    angle = torch.as_tensor([angle * math.pi / 180])
    cos, sin = torch.cos(angle), torch.sin(angle)
    rot_mat = torch.as_tensor([[cos, sin], [-sin, cos]])
    return x @ rot_mat


def int_or_ratio(alpha, n):
    """Return an integer for alpha. If float, it's seen as ratio of `n`."""
    if isinstance(alpha, int):
        return alpha
    return int(alpha * n)


def npimg_resize(np_imgs, size):
    """Batchwise resizing numpy images."""
    if np_imgs.ndim == 3:
        np_imgs = np_imgs[:, :, :, None]

    torch_imgs = torch.from_numpy(np_imgs.transpose((0, 3, 1, 2))).contiguous()
    torch_imgs = F_trnsf.resize(torch_imgs, size=size)
    np_imgs = torch_imgs.numpy().transpose((0, 2, 3, 1))
    return np_imgs
