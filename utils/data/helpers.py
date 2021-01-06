import logging
import torch

import math
from scipy.spatial import cKDTree
from scipy.special import digamma, gamma
import scipy
import numpy as np
from torchvision import transforms as transform_lib
from torchvision.transforms import functional as F_trnsf

from lossyless.helpers import to_numpy

logger = logging.getLogger(__name__)


def differential_entropy(x, k=3, eps=1e-10, p=np.inf, base=2):
    """Kozachenko-Leonenko Estimator [1] of diffential entropy.
    
    Note
    ----
    - This is an improved (vectorized + additional norms) reimplementation 
    of https://github.com/gregversteeg/NPEET.
    
    Parameters
    ----------
    x : array-like, shape=(n,d)
        Samples from which to estimate the entropy.
        
    k : int, optional
        Nearest neigbour to use for estimation. Lower means less bias, 
        higher less variance.
        
    eps : float, otpional
        Additonal noise.
        
    p : {2, np.inf}, optional
        p-norm to use for comparing distances. 2 might give instabilities.

    base : int, optional
        Base for the logs.
    
    References
    ----------
    [1] Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating 
    mutual information. Physical review E, 69(6), 066138.
    """
    x = to_numpy(x)
    n, d = x.shape

    if p == 2:
        log_vol = (d / 2.0) * math.log(math.pi) - math.log(gamma(d / 2.0 + 1))
    elif not np.isfinite(p):
        log_vol = d * math.log(2)
    else:
        raise ValueError(f"p={p} but must be 2 or inf.")

    x = x + eps * np.random.rand(*x.shape)
    tree = cKDTree(x)
    nn_dist, _ = tree.query(x, [k + 1], p=p)

    const = digamma(n) - digamma(k) + log_vol
    h = const + d * np.log(nn_dist).mean()
    return h / math.log(base)


def discrete_entropy(x, base=2, is_plugin=False):
    """Estimate the discrete entropy even when sample space is much larger than number of samples.
    By using the Nemenman-Schafee-Bialek Bayesian estimator [1]. All credits: https://github.com/simomarsili/ndd.

    Parameters
    ---------
    x : array-like, shape=(n,d)
        Samples from which to estimate the entropy.

    base : int, optional
        Base for the logs.

    is_plugin : int, optional
        Whether to use the plugin computation instead of Bayesian estimation. This should only
        be used if you have access to the entire distribution.

    kwargs :
        Additional arguments to `ndd.entropy.`

    Reference
    ---------
    [1] Nemenman, I., Bialek, W., & Van Steveninck, R. D. R. (2004). Entropy and information in 
    neural spike trains: Progress on the sampling problem. Physical Review E, 69(5), 056111.
    """
    x = to_numpy(x)
    _, counts = np.unique(x, return_counts=True, axis=0)

    if is_plugin:
        return scipy.stats.entropy(counts, base=base)

    if max(counts) == 1:
        logging.warn("Examples are only seen once. Increase samples! We return -inf.")
        return -np.inf

    try:
        import ndd
    except ImportError:
        logging.warn("To compute discrete entropies you need to install `ndd`.")
        return -np.inf

    return ndd.entropy(counts) / math.log(base)


def rotate(x, angle):
    """Rotate a 2D tensor by a certain angle (in degrees)."""
    angle = torch.tensor([angle * math.pi / 180])
    cos, sin = torch.cos(angle), torch.sin(angle)
    rot_mat = torch.tensor([[cos, sin], [-sin, cos]])
    return x @ rot_mat


def sample_param_augment(rv, interval_trnsf):
    """
    Sample a parameter for transformations using an action r.v. `rv` and difference in parameter
    space between min and max transforms, i.e. `interval_trnsf`. The mean of `rv` will always correspond
    to 0 parameter. Eg for rotations interval_trnsf=360 and mean rv is 0 degress
    """
    support = rv.support()
    k_trnsf = support[1] + 1 - support[0]  # number of transforms
    delta_trnsf = interval_trnsf / k_trnsf
    sample = rv.rvs()
    sample -= rv.mean()  # ensure mean is 0 degrees
    sample *= delta_trnsf  # put in transforms
    return sample


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
        angle = sample_param_augment(self.rv, self.max_angle * 2)
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
        trnslt_val = sample_param_augment(self.rv, self.max_trnslt * 2)
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
        scale_val = sample_param_augment(self.rv, (self.max_scale - 1) * 2)
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
