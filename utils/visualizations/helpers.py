import io
import logging

import matplotlib.pyplot as plt
import numpy as np

try:
    from pygifsicle import optimize
except ImportError:
    pass

try:
    import imageio
except ImportError:
    pass


logger = logging.getLogger(__name__)


def kwargs_log_scale(unique_val, mode="equidistant", base=None):
    """Return arguments to set log_scale as one would wish.

    Parameters
    ----------
    unique_val : np.array
        All unique values that will be plotted on the axis that should be put in log scale.

    axis : {"x","y"}
        Axis for which to use log_scales.

    mode : ["smooth","equidistant"], optional
        How to deal with the zero, which cannot be dealt with by default as log would give  -infitiy.
        The key is that we will use a region close to zero which is linear instead of log.
        In the case of `equidistant` we use ensure that the large tick at zero is at the same distanec
        of other ticks than if there was no linear. The problem is that this will give rise to
        inexistant kinks when the plot goes from linear to log scale. `Smooth` tries to deal
        with that by smoothly varying vetwen linear and log. For examples see
        https://github.com/matplotlib/matplotlib/issues/7008.

    base : int, optional
        Base to use for the log plot. If `None` automatically tries to find it. If `1` doeesn't use
        any log scale.
    """
    unique_val.sort()

    # automatically compute base
    if base is None:
        # take avg multiplier between each consecutive elements as base i.e 2,8,32 would be 4
        # but 0.1,1,10 would be 10
        diffs = unique_val[unique_val > 0][1:] / unique_val[unique_val > 0][:-1]
        base = int(diffs.mean().round())

    # if constant diff don't use logscale
    if base == 1 or np.diff(unique_val).var() == 0:
        return dict(value="linear")

    # only need to use symlog if there are negative values (i.e. need some linear region)
    if (unique_val <= 0).any():
        min_nnz = np.abs(unique_val[unique_val != 0]).min()
        if mode == "smooth":
            linscale = np.log(np.e) / np.log(base) * (1 - (1 / base))
        elif mode == "equidistant":
            linscale = 1 - (1 / base)
        else:
            raise ValueError(f"Unkown mode={mode}")

        return {
            "value": "symlog",
            "linthresh": min_nnz,
            "base": base,
            "subs": list(range(base)),
            "linscale": linscale,
        }
    else:
        return {
            "value": "log",
            "base": base,
            "subs": list(range(base)),
        }


def fig2img(fig, dpi=200, format="png", is_transparent=False):
    """Convert a Matplotlib figure to a imageio Image and return it"""
    buf = io.BytesIO()
    fig.savefig(
        buf, dpi=dpi, bbox_inches="tight", format=format, transparent=is_transparent
    )
    buf.seek(0)
    img = imageio.imread(buf)
    return img


def giffify(
    save_filename,
    gen_single_fig,
    sweep_parameter,
    sweep_values,
    fps=2,
    quality=70,
    is_transparent=False,
    **kwargs,
):
    """Make a gif by calling `single_fig` with varying parameters.

    Parameters
    ----------
    save_filename : str
        name fo the file for saving the gif.

    gen_single_fig : callable
        Function which returns a matplotlib figure.

    sweep_parameter : str
        Name of the parameter to `single_fig` that will be swept over.

    sweep_values : array-like
        Values to sweep over.

    fps : int, optional
        Number of frame per second. I.e. speed of gif.

    is_transparent: bool, optional
        Whether to use a transparent background.

    kwargs :
        Arguments to `single_fig` that should not be swept over.
    """
    figs = []
    for i, v in enumerate(sweep_values):
        fig = gen_single_fig(**{sweep_parameter: v}, **kwargs)
        plt.close()
        img = fig2img(fig, is_transparent=is_transparent)
        if i > 0:
            img = transform.resize(img, size)
        else:
            size = img.shape[:2]
        figs.append(img)

    imageio.mimsave(save_filename, figs, fps=fps)

    try:
        optimize(save_filename)
    except Exception as e:
        logger.info(f"Could not compress gif: {e}")
