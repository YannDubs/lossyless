import inspect

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from lossyless.helpers import to_numpy


def get_default_args(func):
    """Return the default arguments of a function.
    credit : https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def assert_sns_vary_only_param(data, sns_kwargs, param_vary_only):
    """
    Make sure that the only multiindices that have not been conditioned over for plotting and has non
    unique values are in `param_vary_only`.
    """
    if param_vary_only is not None:
        multi_idcs = data.index
        issues = []
        for idx in multi_idcs.levels:
            is_varying = len(idx.values) != 1
            is_conditioned = idx.name in sns_kwargs.values()
            is_can_vary = idx.name in param_vary_only
            if is_varying and not is_conditioned and not is_can_vary:
                issues.append(idx.name)

        if len(issues) > 0:
            raise ValueError(
                f"Not only varying {param_vary_only}. Also varying {issues}."
            )


def aggregate(table, cols_to_agg=[], aggregates=["mean", "sem"]):
    """Aggregate values of pandas dataframe over some columns.

    Parameters
    ----------
    table : pd.DataFrame or pd.Series
        Table to aggregate.

    cols_to_agg : list of str
        List of columns over which to aggregate. E.g. `["seed"]`.

    aggregates : list of str
        List of functions to use for aggregation. The aggregated columns will be called `{col}_{aggregate}`.
    """
    if len(cols_to_agg) == 0:
        return table

    if isinstance(table, pd.Series):
        table = table.to_frame()

    table_agg = table.groupby(
        by=[c for c in table.index.names if c not in cols_to_agg]
    ).agg(aggregates)
    table_agg.columns = ["_".join(col).rstrip("_") for col in table_agg.columns.values]
    return table_agg


def save_fig(fig, filename, dpi, is_tight=True):
    """General function for many different types of figures."""

    # order matters ! and don't use elif!
    if isinstance(fig, sns.FacetGrid):
        fig = fig.fig

    if isinstance(fig, torch.Tensor):
        x = fig.permute(1, 2, 0)
        if x.size(2) == 1:
            fig = plt.imshow(to_numpy(x.squeeze()), cmap="gray")
        else:
            fig = plt.imshow(to_numpy(x))
        plt.axis("off")

    if isinstance(fig, matplotlib.image.AxesImage):
        fig = fig.get_figure()

    if isinstance(fig, matplotlib.figure.Figure):

        plt_kwargs = {}
        if is_tight:
            plt_kwargs["bbox_inches"] = "tight"

        fig.savefig(filename, dpi=dpi, **plt_kwargs)
        plt.close(fig)
    else:
        raise ValueError(f"Unkown figure type {type(fig)}")
