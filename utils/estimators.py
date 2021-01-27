import logging
import math

import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.special import digamma, gamma

import torch
from lossyless.helpers import to_numpy

from .helpers import all_logging_disabled, at_least_ndim, cont_tuple_to_tuple_cont

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
    return float(h / math.log(base))


def discrete_entropy(x, base=2, is_plugin=False, **kwargs):
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

    try:
        import ndd
    except ImportError:
        logger.warn("To compute discrete entropies you need to install `ndd`.")
        return -np.inf

    return float(ndd.entropy(counts, **kwargs) / math.log(base))


def conditional_entropy(
    y, x, is_discrete=True, is_joint=True, k_y=None, k_x=None, **kwargs
):
    """Estimate the conditional (differential) entropy H[Y|X]
    
    Note
    ----
    - if y is continuous but x is discrete use `is_joint=False, is_discrete=False`.
    - if y is discrete but x is continuous use `is_joint=True, is_discrete=True` 
    but the results will likely be bad.

    Parameters
    ---------
    y : array-like, shape=(n,d)
        Samples from which to estimate the entropy.

    x : array-like, shape=(n,d)
        Conditioning samples from which to estimate the entropy.

    is_discrete : bool, optional
        Whether to treat the samples as from discrete r.v. For mixed r.v.s you should prefer
        `False` but the results might not be very good.
        
    is_joint : bool, optional
        Whether to estimate the conditional entropy by `H[Y|X] = H[Y,X] - H[X]` instead of
        directly by `E_x[H[Y|x]]`. Joint is nearly always prefered as it uses estimators. Besides
        when Y is vastly undersampled compared to X, typically when there's never two Ys but many
        replacates of X. This will use plugin estimate over X.

    k_y : int, optional
        Cardinality of the support of  sample space of y. Only if `is_discete`. Should only be given
        if you are sure. + `ndd` gives strange results when k is large.

    k_x : int, optional
        Cardinality of the support of sample space of y. Only if `is_discete`. Should only be given
        if you are sure. + `ndd` gives strange results when k is large.

    kwargs :
        Additional arguments to the entropy estimator.
    """
    y, x = to_numpy(y), to_numpy(x)
    entropy = discrete_entropy if is_discrete else differential_entropy

    if is_joint:
        yx = np.concatenate((y, x), axis=1)

        # cannot really use k_y, k_x by just doing k=k_y,k_x, because it's about the support
        # which might be much smaller.
        H_yx = entropy(yx, **kwargs)

        if is_discrete and k_x is not None:
            kwargs["k"] = k_x

        H_x = entropy(x, **kwargs)
        H_YlX = H_yx - H_x

    else:
        H_YlX = 0
        _, inverse, counts = np.unique(
            x, return_counts=True, return_inverse=True, axis=0
        )

        if is_discrete and k_y is not None:
            kwargs["k"] = k_y

        # when computing H[Y|x] for many x there will be no repeated Y
        # so will have to use plugin estimate. Disable logging that says so
        with all_logging_disabled():
            for i, count in enumerate(counts):
                p_i = count / sum(counts)
                H_Ylx = entropy(y[inverse == i], **kwargs)
                H_YlX += p_i * H_Ylx

    return H_YlX


def predict(trainer, dataloader):
    """Predict the data for all the dataloader. Return in numpy."""
    model = trainer.get_model()
    was_training = model.training
    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            x, target = batch
            pred = model(x)
            preds.append(to_numpy(pred))
            targets.append(target)

    if was_training:
        # restore
        model.train()

    if isinstance(targets[0], (tuple, list)):
        # if multiple targets want to return a tuple of concatenated, so goes from list
        # of tuple of torch to tuple of torch
        targets = tuple(
            to_numpy(torch.cat(t)) for t in cont_tuple_to_tuple_cont(targets)
        )
    else:
        targets = to_numpy(torch.cat(targets))

    # shape=[len(dataloader), *out_shape]
    preds = np.concatenate(preds, axis=0)

    return preds, targets


def estimate_entropies(
    trainer, datamodule, is_test=True, is_discrete_M=True, is_discrete_Y=True
):
    """Estimate the invariance distortion H[M(X)|Z] (upper bound on bayes risk) and current bayes risk 
    H[Y|Z] on train or test set. This should only be used if Z is discretized, i.e. using H bottleneck. 
    Trainer.model should be a CompressionModule.
    """
    # get the max invariant from the dataset
    dkwargs = {"additional_target": "max_inv"}
    if is_test:
        dataloader = datamodule.test_dataloader(dataset_kwargs=dkwargs)
    else:
        dataloader = datamodule.train_dataloader(dataset_kwargs=dkwargs)

    z_samples, (y_samples, Mx_samples) = predict(trainer, dataloader)

    z_samples = at_least_ndim(z_samples, 2, is_prefix_padded=False)
    y_samples = at_least_ndim(y_samples, 2, is_prefix_padded=False)
    Mx_samples = at_least_ndim(Mx_samples, 2, is_prefix_padded=False)

    if is_discrete_Y:
        # you will always see at least one of each label (test or train)
        k_y = len(np.unique(y_samples))
        H_YlZ = conditional_entropy(y_samples, z_samples, is_discrete=True, k_y=k_y)
    else:
        # z is discrete but m is not => directly estimate the conditional entropy
        H_YlZ = conditional_entropy(
            Mx_samples, z_samples, is_discrete=False, is_joint=False
        )

    # even if M is discrete there will usually only be sample per Mx so cannot compute joint H[Mx,Z]
    # => directly estimate conditional entropy
    H_MlZ = conditional_entropy(
        Mx_samples, z_samples, is_discrete=is_discrete_M, is_joint=False
    )

    H_Z = discrete_entropy(z_samples)
    return H_MlZ, H_YlZ, H_Z
