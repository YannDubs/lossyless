import contextlib
import io
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cbook import MatplotlibDeprecationWarning

try:
    from pygifsicle import optimize
except ImportError:
    pass

try:
    import imageio
except ImportError:
    pass


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def plot_config(
    style="ticks",
    context="notebook",
    palette="colorblind",
    font_scale=1,
    font="sans-serif",
    is_ax_off=False,
    rc=dict(),
    set_kwargs=dict(),
    despine_kwargs=dict(),
):
    """Temporary seaborn and matplotlib figure style / context / limits / ....

    Parameters
    ----------
    style : dict, None, or one of {darkgrid, whitegrid, dark, white, ticks}
        A dictionary of parameters or the name of a preconfigured set.

    context : dict, None, or one of {paper, notebook, talk, poster}
        A dictionary of parameters or the name of a preconfigured set.

    palette : string or sequence
        Color palette, see :func:`color_palette`

    font : string
        Font family, see matplotlib font manager.

    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.

    is_ax_off : bool, optional
        Whether to turn off all axes.

    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        style dictionaries.

    set_kwargs : dict, optional
        kwargs for matplotlib axes. Such as xlim, ylim, ...

    despine_kwargs : dict, optional
        Arguments to `sns.despine`.
    """
    defaults = plt.rcParams.copy()

    try:
        rc["font.family"] = font
        plt.rcParams.update(rc)

        with sns.axes_style(style=style, rc=rc), sns.plotting_context(
            context=context, font_scale=font_scale, rc=rc
        ), sns.color_palette(palette):
            yield
            last_fig = plt.gcf()
            for i, ax in enumerate(last_fig.axes):
                ax.set(**set_kwargs)

                if is_ax_off:
                    ax.axis("off")

        sns.despine(**despine_kwargs)

    finally:
        with warnings.catch_warnings():
            # filter out depreciation warnings when resetting defaults
            warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
            # reset defaults
            plt.rcParams.update(defaults)


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
