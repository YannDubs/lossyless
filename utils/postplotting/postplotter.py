import abc
from pathlib import Path

from .pretty_renamer import PRETTY_RENAMER

__all__ = ["PostPlotter"]


class PostPlotter(abc.ABC):
    """Base class for a post training plotter.

    Parameters
    ----------
    is_return_plots : bool, optional
        Whether to return plots instead of saving them.

    prfx : str, optional
        Prefix for the filename to save.

    pretty_renamer : dict, optional
        Dictionary mapping string (keys) to human readable ones for nicer printing and plotting.

    dpi : int, optional
        Resolution of the figures

    plot_config_kwargs : dict, optional
        Default general config for plotting, e.g. arguments to matplotlib.rc, sns.plotting_context,
        color palettes, matplotlib.set...
    """

    def __init__(
        self,
        is_return_plots=False,
        prfx="",
        pretty_renamer=PRETTY_RENAMER,
        dpi=300,
        plot_config_kwargs={},
    ):
        self.is_return_plots = is_return_plots
        self.prfx = prfx
        self.pretty_renamer = pretty_renamer
        self.dpi = dpi
        self.plot_config_kwargs = plot_config_kwargs

    def prettify_(self, table):
        """Make the name and values in a dataframe prettier / human readable (inplace)."""
        idcs = table.index.names
        table = table.reset_index()  # also want to modify multiindex so tmp flatten
        table.columns = [self.pretty_renamer[c] for c in table.columns]
        table = table.applymap(self.pretty_renamer)
        table = table.set_index([self.pretty_renamer[c] for c in idcs])

        # replace `None` with "None" for string columns such that can see those
        str_col = table.select_dtypes(include=object).columns
        table[str_col] = table[str_col].fillna(value="None")

        return table

    def prettify_kwargs(self, table, **kwargs):
        """Change the kwargs of plotting function such that usable with `prettify(table)`."""
        cols_and_idcs = list(table.columns) + list(table.index.names)
        return {
            # only prettify if part of the columns (not arguments to seaborn)
            k: self.pretty_renamer[v]
            if isinstance(v, str) and self.pretty_renamer[v] in cols_and_idcs
            else v
            for k, v in kwargs.items()
        }
