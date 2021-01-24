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


def sample_param_augment(rv, interval_trnsf):
    """
    Sample a parameter (and it's index) for transformations using an action r.v. `rv` and difference
    in parameter space between min and max transforms, i.e. `interval_trnsf`. The mean of `rv` will
    always correspond to 0 parameter. Eg for rotations interval_trnsf=360 and mean rv is 0 degrees.
    """
    support = rv.support()
    k_trnsf = support[1] + 1 - support[0]  # number of transforms
    delta_trnsf = interval_trnsf / k_trnsf
    i_sample = rv.rvs()
    theta_sample = i_sample - rv.mean()  # ensure mean is 0 degrees
    theta_sample *= delta_trnsf  # put in transforms
    return theta_sample, i_sample


class RotationAction(torch.nn.Module):
    """Rotate the image by a sampled angle.

    Parameters
    ----------
    rv : scipy.stats.rv_discrete
        Discrete distribution to sample the rotation. The mean will correspond to 0 degrees rotation
        and the rest will correspond to angles with a fixed interval.

    max_angle : float, optional
        Maximimum angle by which to rotate on one one side.

    kwargs :
        Additional arguments to `F.rotate`.
    """

    def __init__(self, rv, max_angle=90, **kwargs):
        super().__init__()
        self.rv = rv
        self.max_angle = max_angle
        self.kwargs = kwargs

    def forward(self, img):
        # also store last index you sampled
        angle, self.i = sample_param_augment(self.rv, self.max_angle * 2)
        return F_trnsf.rotate(img, angle, **self.kwargs)


class TranslationAction(torch.nn.Module):
    """Translate the image by a sampled amount.

    Parameters
    ----------
    rv : scipy.stats.rv_discrete
        Discrete distribution to sample the translate. The mean will correspond to 0 shift
        and the rest will correspond to shifts with a fixed interval.

    dim : int, optional
        Dim on which to translate. If `0` x axis, if `1` y axis.

    max_trnslt : int, optional
        Maximum pixels by which to translate (negatively and positively).

    kwargs :
        Additional arguments to `F.affine`.
    """

    def __init__(self, rv, dim=0, max_trnslt=4, **kwargs):
        super().__init__()
        self.rv = rv
        self.dim = dim
        self.max_trnslt = max_trnslt
        self.kwargs = kwargs

    def forward(self, img):
        trnslt = [0, 0]
        trnslt_val, self.i = sample_param_augment(self.rv, self.max_trnslt * 2)
        trnslt[self.dim] = trnslt_val
        return F_trnsf.affine(
            img, translate=trnslt, angle=0, scale=1, shear=(0, 0), **self.kwargs
        )


class ScalingAction(torch.nn.Module):
    """Scale the image by a sampled factor.

    Parameters
    ----------
    rv : scipy.stats.rv_discrete
        Discrete distribution to sample the scaling. The mean will correspond to no scaling
        and the rest will correspond to scales by a fixed delta.

    max_scale : int, optional
        Maximum amount by which to scale. Note: the min will be 2-max_scale, eg if 1.2 then min is 0.8.

    kwargs :
        Additional arguments to `F.rotate`.
    """

    def __init__(self, rv, max_scale=4, **kwargs):
        super().__init__()
        self.rv = rv
        self.max_scale = max_scale
        self.kwargs = kwargs

    def forward(self, img):
        scale_val, self.i = sample_param_augment(self.rv, (self.max_scale - 1) * 2)
        return F_trnsf.affine(
            img,
            translate=(0, 0),
            angle=0,
            scale=(1 + scale_val),
            shear=(0, 0),
            **self.kwargs,
        )


def int_or_ratio(alpha, n):
    """Return an integer for alpha. If float, it's seen as ratio of `n`."""
    if isinstance(alpha, int):
        return alpha
    return int(alpha * n)
