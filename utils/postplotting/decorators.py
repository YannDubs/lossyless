import functools
from pathlib import Path

import matplotlib.pyplot as plt

from lossyless.helpers import plot_config

from .helpers import aggregate, assert_sns_vary_only_param, get_default_args, save_fig

__all__ = ["data_getter", "table_summarizer", "folder_split", "single_plot"]


def data_getter(fn):
    """Get the correct data."""
    dflt_kwargs = get_default_args(fn)

    @functools.wraps(fn)
    def helper(
        self, data=dflt_kwargs["data"], filename=dflt_kwargs["filename"], **kwargs
    ):
        if data is None:
            # if None run all tables
            return [
                helper(self, data=k, filename=filename, **kwargs)
                for k in self.tables.keys()
            ]

        if isinstance(data, str):
            # cannot use format because might be other other patterns (format cannot do partial format)
            filename = filename.replace("{table}", data)
            data = self.tables[data]

        data = data.copy()

        return fn(self, data=data, filename=filename, **kwargs)

    return helper


def table_summarizer(fn):
    """Get the data and save the summarized output to a csv if needed.."""
    dflt_kwargs = get_default_args(fn)

    @functools.wraps(fn)
    def helper(
        self, data=dflt_kwargs["data"], filename=dflt_kwargs["filename"], **kwargs
    ):

        summary = fn(self, data=data, **kwargs)

        if self.is_return_plots:
            return summary
        else:
            summary.to_csv(self.save_dir / f"{self.prfx}{filename}.csv")

    return helper


def folder_split(fn):
    """Split the dataset by the values in folder_col and call fn on each subfolder."""
    dflt_kwargs = get_default_args(fn)

    @functools.wraps(fn)
    def helper(
        self,
        *args,
        data=dflt_kwargs["data"],
        folder_col=dflt_kwargs["folder_col"],
        filename=dflt_kwargs["filename"],
        **kwargs,
    ):
        kws = ["folder_col"]
        for kw in kws:
            kwargs[kw] = eval(kw)

        if folder_col is None:
            processed_filename = self.save_dir / f"{self.prfx}{filename}"
            return fn(self, *args, data=data, filename=processed_filename, **kwargs)

        else:
            out = []
            flat = data.reset_index(drop=False)
            for curr_folder in flat[folder_col].unique():
                curr_data = flat[flat[folder_col] == curr_folder]

                sub_dir = self.save_dir / f"{folder_col}_{curr_folder}"
                sub_dir.mkdir(parents=True, exist_ok=True)

                processed_filename = sub_dir / f"{self.prfx}{filename}"

                out.append(
                    fn(
                        self,
                        *args,
                        data=curr_data.set_index(data.index.names),
                        filename=processed_filename,
                        **kwargs,
                    )
                )
            return out

    return helper


def single_plot(fn):
    """
    Wraps any of the aggregator function to produce a single figure. THis enables setting
    general seaborn and matplotlib parameters, saving the figure if needed, and aggregating the
    data over desired indices.
    """
    dflt_kwargs = get_default_args(fn)

    @functools.wraps(fn)
    def helper(
        self,
        x,
        y,
        *args,
        data=dflt_kwargs["data"],
        folder_col=dflt_kwargs["folder_col"],
        filename=dflt_kwargs["filename"],
        cols_vary_only=dflt_kwargs["cols_vary_only"],
        cols_to_agg=dflt_kwargs["cols_to_agg"],
        aggregates=dflt_kwargs["aggregates"],
        plot_config_kwargs=dflt_kwargs["plot_config_kwargs"],
        row_title=dflt_kwargs["row_title"],
        col_title=dflt_kwargs["col_title"],
        x_rotate=dflt_kwargs["x_rotate"],
        legend_out=dflt_kwargs["legend_out"],
        is_no_legend_title=dflt_kwargs["is_no_legend_title"],
        set_kwargs=dflt_kwargs["set_kwargs"],
        **kwargs,
    ):
        filename = Path(str(filename).format(x=x, y=y))

        kws = [
            "folder_col",
            "filename",
            "cols_vary_only",
            "cols_to_agg",
            "aggregates",
            "plot_config_kwargs",
            "row_title",
            "col_title",
            "x_rotate",
            "is_no_legend_title",
            "set_kwargs",
        ]
        for kw in kws:
            kwargs[kw] = eval(kw)  # put back in kwargs

        kwargs["x"] = x
        kwargs["y"] = y

        assert_sns_vary_only_param(data, kwargs, cols_vary_only)

        data = aggregate(data, cols_to_agg, aggregates)
        pretty_data = self.prettify_(data)
        pretty_kwargs = self.prettify_kwargs(pretty_data, **kwargs)
        used_plot_config = dict(self.plot_config_kwargs, **plot_config_kwargs)

        with plot_config(**used_plot_config):
            sns_plot = fn(self, *args, data=pretty_data, **pretty_kwargs)

        for ax in sns_plot.axes.flat:
            plt.setp(ax.texts, text="")
        sns_plot.set_titles(row_template=row_title, col_template=col_title)

        if x_rotate != 0:
            # calling directly `set_xticklabels` on FacetGrid removes the labels sometimes
            for axes in sns_plot.axes.flat:
                axes.set_xticklabels(axes.get_xticklabels(), rotation=x_rotate)

        if is_no_legend_title:
            #! not going to work well if is_legend_out (double legend)
            for ax in sns_plot.fig.axes:
                handles, labels = ax.get_legend_handles_labels()
                if len(handles) > 1:
                    ax.legend(handles=handles[1:], labels=labels[1:])

        sns_plot.set(**set_kwargs)

        if not legend_out:
            plt.legend()

        if self.is_return_plots:
            return sns_plot
        else:
            save_fig(sns_plot, f"{filename}.png", dpi=self.dpi)

    return helper
