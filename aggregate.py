"""Entropy point to aggregate a series of results obtained using `main.py` in a nice plot / table.

This should be called by `python aggregate.py <conf>` where <conf> sets all configs from the cli, see 
the file `config/aggregate.yaml` for details about the configs. or use `python aggregate.py -h`.
"""
import functools
import glob
import inspect
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import hydra
from lossyless.helpers import BASE_LOG, check_import
from main import COMPRESSOR_RES
from omegaconf import OmegaConf
from utils.helpers import StrFormatter, omegaconf2namespace
from utils.visualizations.helpers import kwargs_log_scale, plot_config

try:
    import sklearn.metrics
except:
    pass

logger = logging.getLogger(__name__)


PRETTY_RENAMER = StrFormatter(
    exact_match={},
    subtring_replace={
        # Math stuff
        "H_Q_Zls": r"$\mathrm{H}_{\theta}[Z|S]$",
        "H_Q_Tlz": r"$\mathrm{H}_{\theta}[T|Z]$",
        "H_Q_Z": r"$\mathrm{H}_{\theta}[Z]$",
        "H_Q_S": r"$\mathrm{H}_{\theta}[S]$",
        "H_Ylz": r"$\mathrm{H}[Y|Z]$",
        "H_Zlx": r"$\mathrm{H}[Z|X]$",
        "H_Mlz": r"$\mathrm{H}[M(X)|Z]$",
        "H_Z": r"$\mathrm{H}[Z]$",
        "I_Q_Zx": r"$\mathrm{I}_{\theta}[Z;X]$",
        "I_Q_Zm": r"$\mathrm{I}_{\theta}[Z;M]$",
        "beta": r"$\beta$",
        # General
        "_": " ",
        "Resnet": "ResNet",
        "Ivae": "Inv. VAE",
        "Ivib": "Inv. VIB",
        "Ince": "Inv. NCE",
        "Bananarot": "Rotation Inv. Banana",
        "Bananaxtrnslt": "X-axis Inv. Banana",
        "Bananaytrnslt": "Y-axis Inv. Banana",
        "Lr": "Learning Rate",
        "Online Loss": r"$\mathrm{H}_{\theta}[Y|Z]$",
    },
    to_upper=["Cifar10", "Mnist", "Mlp", "Vae", "Nce", "Vib", "Adam",],
)


@hydra.main(config_name="aggregate", config_path="config")
def main_cli(cfg):
    # uses main_cli sot that `main` can be called from notebooks.
    return main(cfg)


def main(cfg):

    begin(cfg)

    # make sure you are using primitive types from now on because omegaconf does not always work
    cfg = omegaconf2namespace(cfg)

    aggregator = Aggregator(pretty_renamer=PRETTY_RENAMER, **cfg.kwargs)

    logger.info(f"Recolting the data ..")
    for name, pattern in cfg.collect_data.items():
        if pattern is not None:
            aggregator.collect_data(pattern=pattern, table_name=name)

    aggregator.subset(cfg.col_val_subset)

    for f in cfg.agg_mode:

        logger.info(f"Mode {f} ...")

        if f is None:
            continue

        if f in cfg:
            kwargs = cfg[f]
        else:
            kwargs = {}

        getattr(aggregator, f)(**kwargs)

    logger.info("Finished.")


def begin(cfg):
    """Script initialization."""
    OmegaConf.set_struct(cfg, False)  # allow pop
    PRETTY_RENAMER.update(cfg.kwargs.pop("pretty_renamer"))
    OmegaConf.set_struct(cfg, True)

    logger.info(f"Aggregating {cfg.experiment} ...")


# DECORATORS
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

        if self.is_return_plots:
            return sns_plot
        else:
            sns_plot.fig.savefig(
                f"{filename}.png", dpi=self.dpi,
            )
            plt.close(sns_plot.fig)

    return helper


# MAIN CLASS
class Aggregator:
    """Result aggregator.

    Parameters
    ----------
    save_dir : str or Path
        Where to save all results.

    base_dir : str or Path
        Base folder from which all paths start.

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
        save_dir,
        base_dir=Path(__file__).parent,
        is_return_plots=False,
        prfx="",
        pretty_renamer=PRETTY_RENAMER,
        dpi=300,
        plot_config_kwargs={},
    ):
        self.base_dir = Path(base_dir)
        self.save_dir = self.base_dir / Path(save_dir)
        self.is_return_plots = is_return_plots
        self.prfx = prfx
        self.pretty_renamer = pretty_renamer
        self.dpi = dpi
        self.tables = dict()
        self.param_names = dict()
        self.plot_config_kwargs = plot_config_kwargs

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def collect_data(
        self,
        pattern=f"results/**/{COMPRESSOR_RES}",
        table_name="featurizer",
        params_to_rm=["jid"],
    ):
        """Collect all the data.

        Notes
        -----
        - Load all the results that are saved in csvs, such that the path name of the form
        `param1_value1/param2_value2/...` and the values in the csv are such that the columns are
        "train", "test", (and possibly other mode), while index shows parameter name.
        - The loaded data are a dataframe where each row is a different run, (multi)indices are the
        parameters and columns contain train_metrics and test_metrics.

        Parameters
        ----------
        pattern : str
            Pattern for globbing data.

        table_name : str, optional
            Name of the table under which to save the loaded data.

        params_to_rm : list of str, optional   
            Params to remove.
        """
        paths = glob.glob(str(self.base_dir / pattern), recursive=True)
        if len(paths) == 0:
            raise ValueError(f"No files found for your pattern={pattern}")

        results = []
        self.param_names[table_name] = set()
        for path in paths:
            # rm the last folder and file (filename)
            path_clean = path.rsplit("/", maxsplit=1)[0]
            # rm the first folder ("results")
            path_clean = path_clean.split("/", maxsplit=1)[-1]

            # make dict of params
            params = path_to_params(path_clean)

            for p in params_to_rm:
                params.pop(p)

            # looks like : DataFrame(param1:...,param2:..., param3:...)
            df_params = pd.DataFrame.from_dict(params, orient="index").T
            # looks like : dict(train={metric1:..., metric2:...}, test={metric1:..., metric2:...})
            dicts = pd.read_csv(path, index_col=0).to_dict()
            # flattens dicts and make dataframe :
            # DataFrame(train/metric1:...,train/metric2:..., test/metric1:..., test/metric2:...)
            df_metrics = pd.json_normalize(dicts, sep="/")

            results.append(pd.concat([df_params, df_metrics], axis=1))

        param_name = list(params.keys())
        self.tables[table_name] = pd.concat(results, axis=0).set_index(param_name)
        self.param_names[table_name] = param_name

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

    def subset(self, col_val):
        """Subset all tables by keeping only the given values in given columns.

        Parameters
        ----------
        col_val : dict
            A dictionary where the keys are the columns to subset and values are a list of values to keep.
        """
        for col, val in col_val.items():
            logger.debug("Keeping only val={val} for col={col}.")
            for k in self.table_names:
                self.tables[k] = self.tables[k][(self.tables[k][col]).isin(val)]
                if self.tables[k].empty:
                    logger.info(f"Empty table after filtering {col}={val}")

    @data_getter
    def plot_all_RD_curves(
        self,
        data=None,
        rate_cols=["test/feat/rate"],
        distortion_cols=["test/feat/distortion", "test/feat/online_loss"],
        logbase_x=None,
        cols_to_agg=["seed"],
        filename="all_RD_curves_{table}",
        **kwargs,
    ):
        """Main function for plotting different Rate distortion plots.

        Parameters
        ----------
        data : pd.DataFrame or str
            Dataframe to use for plotting. If str will use one of self.tables. If `None` runs all tables.

        rate_cols : list of str
            List of columns that can be considered as rates and for which we should generate RD curves.

        distortion_cols : list of str
            List of columns that can be considered as distortions and for which we should generate
            RD curves.

        logbase_x : int, optional
            Base of the x  axis. If 1 no logscale. if `None` will automatically chose.

        cols_to_agg : list of str, optional
            Paremeters over which to aggregate the RD curves and compute the standard errors.

        kwargs :
            Additional arguments to `plot_scatter_lines`.
        """
        data = merge_rate_distortions(data, rate_cols, distortion_cols)

        is_single_col = len(distortion_cols) == 1
        is_single_row = len(rate_cols) == 1

        return self.plot_scatter_lines(
            data=data,
            y="rate_val_mean",
            x="distortion_val_mean",
            kind="line",
            logbase_x=logbase_x,
            row=None if is_single_row else "rate_type",
            col=None if is_single_col else "distortion_type",
            cols_to_agg=cols_to_agg,
            is_x_errorbar=True,
            is_y_errorbar=True,
            sharey=is_single_row,
            sharex=is_single_col,
            filename=filename,
            **kwargs,
        )

    def plot_invariance_RD_curve(
        self,
        col_dist_param="dist",
        noninvariant="vae",
        rate_col="test/feat/rate",
        upper_distortion="test/feat/distortion",
        desirable_distortion="test/feat/online_loss",
        logbase_x=None,
        cols_to_agg=["seed"],
        filename="invariance_RD_curve",
        **kwargs,
    ):
        """Plot a specific rate distortion curve which where the distortion is the invaraince
        distortion H[M(X)|Z], but the non invariant model shows both H[M(X)|Z] and H[X|Z]. Where
        H[X|Z] is the distoriton used during training for the non invariant models, but is also a
        tight upper bound on the maximal H[M(X)|Z] for a (noninvariant) optimal Z.

        Parameters
        ----------
        col_dist_param : str, optional
            Name of the column that will distinguish the non invariant and the invariant model.

        noninvariant : str, optional
            Name of the non invariant model.

        desirable_distortion : str, optional
            Name fo the column containing the invariance distortion.

        kwargs :
            Additional arguments to `plot_scatter_lines`.
        """
        results = self.tables["featurizer"]
        results = merge_rate_distortions(
            results, [rate_col], [upper_distortion, desirable_distortion]
        )

        tmp = results.reset_index()
        tmp = tmp[
            (tmp[col_dist_param] == noninvariant)
            | (tmp["distortion_type"] == upper_distortion)
        ]
        tmp.loc[
            (tmp[col_dist_param] == noninvariant)
            & (tmp["distortion_type"] == upper_distortion),
            col_dist_param,
        ] = f"Worst {noninvariant}"
        tmp["distortion_type"] = "distortion"
        results = tmp.set_index(results.index.names)

        return self.plot_scatter_lines(
            data=results,
            y="rate_val_mean",
            x="distortion_val_mean",
            kind="line",
            hue=col_dist_param,
            logbase_x=logbase_x,
            cols_to_agg=cols_to_agg,
            is_x_errorbar=True,
            is_y_errorbar=True,
            sharey=False,
            sharex=False,
            filename=filename,
            **kwargs,
        )

    @data_getter
    @table_summarizer
    def summarize_RD_curves(
        self,
        data=None,
        rate_cols=["test/feat/rate"],
        distortion_cols=["test/feat/distortion", "test/feat/online_loss"],
        cols_to_agg=["seed"],
        cols_to_sweep=["beta"],
        mse_cols=["test/feat/distortion", "test/feat/online_loss"],
        compare_cols=["dist"],
        epsilon_close_distortion=0.01,
        filename="summarized_RD_curves_{table}",
    ):
        """Summarize RD curves by a table: area under the RD curve, average rate for (nearly)
        lossless prediction, ...

        Parameters
        ----------
        data : pd.DataFrame or str, optional
            Dataframe to summarize. If str will use one of self.tables. If `None` uses all data
                in self.tables.

        rate_cols : list of str, optional
            List of columns that can be considered as rates and for which we should generate RD curves.

        distortion_cols : list of str, optional
            List of columns that can be considered as distortions and for which we should generate
            RD curves.

        cols_to_agg : list of str, optional
            List of columns over which to aggregate the summarizes. Typically ["seed"].

        cols_to_sweep : list of str, optional
            Columns over which to sweep to generate different values on the RD curve. Typically ["beta"].

        mse_cols : list of str, optional
            List of columns that are distortions (subset of distortion_cols) but where the distortions
            are in terms of mean squared error (variance) instead of entropies. In that case the columns
            will first be processed to entropy (upper bounds) so that that the distortion and rate
            are both in terms of the same unit which makes it more meaningfull.

        compare_cols : list of str, optional
            List of columns that you whish to compare (typically some model hyperparameters). This
            is used to compute the compute the approx. rate needed for lossless prediction, because
            lossless would be defined as reaching the minimal distortion for all different models
            that only differ in terms of `compare_cols`.

        epsilon_close_distortion : float, optional
            Threshold from which you can be considered to have similar distortion. This is used to
            comptute all the rates for "losless" prediction, which means that you are delta close
            in terms of prediction to the best one.

        filename : str, optional
            Name of the file for saving to summarized RD curves. Can interpolate {table} if from 
            self.tables.
        """
        check_import("sklearn", "summarize_RD_curves")

        # to be meaningfull, both the disortion and the rate columns should be in bits / erntropies (also as approximately linar the
        # trapezoidal rule should be a very good approximation of integral)
        # h[X] = -1/2 log(2 pi e Var[X]) + KL...
        for mse_col in mse_cols:
            data[mse_col] = (
                0.5 * np.log(2 * np.pi * np.e * data[mse_col]) / np.log(BASE_LOG)
            )

        data = merge_rate_distortions(data, rate_cols, distortion_cols)
        data = data.reset_index(level=cols_to_sweep)

        # area under the curve => summary of how good rate over all distortions
        aurd = data.groupby(data.index.names).apply(apply_area_under_RD).rename("AURD")
        aurd = aggregate(aurd, cols_to_agg)

        # compute the avg rate for each model to have distortion than best for THAT model
        data_toagg = data.reset_index(level=cols_to_agg)
        rate_mindist_cur = data_toagg.groupby(data_toagg.index.names).apply(
            apply_rate_mindistortion,
            epsilon=epsilon_close_distortion,
            name="mindist_curr",
        )

        # compute the avg rate for each model to have distortion than best for All model
        dropped = data_toagg.reset_index(level=compare_cols)
        mindist_all = dropped.groupby(dropped.index.names).min()[
            "distortion_val"
        ]  # table of all minimum distortions
        rate_mindist_all = data_toagg.groupby(data_toagg.index.names).apply(
            apply_rate_mindistortion,
            epsilon=epsilon_close_distortion,
            name="mindist_all",
            min_distortion_df=mindist_all,
            to_drop=compare_cols,
        )

        summary = pd.concat([aurd, rate_mindist_cur, rate_mindist_all], axis=1)
        return summary

    @data_getter
    @table_summarizer
    def summarize_metrics(
        self,
        data=None,
        cols_to_agg=["seed"],
        aggregates=["mean", "sem"],
        filename="summarized_metrics_{table}",
    ):
        """Aggregate all the metrics and save them.

        Parameters
        ----------
        data : pd.DataFrame or str, optional
                Dataframe to summarize. If str will use one of self.tables. If `None` uses all data
                in self.tables.

        cols_to_agg : list of str
            List of columns over which to aggregate. E.g. `["seed"]`.

        aggregates : list of str
            List of functions to use for aggregation. The aggregated columns will be called `{col}_{aggregate}`.

        filename : str, optional
                Name of the file for saving the metrics. Can interpolate {table} if from self.tables.
        """
        return aggregate(data, cols_to_agg, aggregates)

    @data_getter
    def plot_superpose(
        self,
        x,
        to_superpose,
        value_name,
        data=None,
        filename="{table}_superposed_{value_name}",
        **kwargs,
    ):
        """Plot a single line figure with multiple superposed lineplots.

        Parameters
        ----------
        x : str
            Column name of x axis.

        to_superpose : dictionary
            Dictionary of column values that should be plotted on the figure. The keys
            correspond to the columns to plot and the values correspond to the name they should be given.

        value_name : str
            Name of the yaxis.

        data : pd.DataFrame or str, optional
            Dataframe used for plotting. If str will use one of self.tables. If `None` runs all tables.

        filename : str, optional
            Name of the figure when saving. Can use {value_name} for interpolation.

        kwargs :
            Additional arguments to `plot_scatter_lines`.
        """
        renamer = to_superpose
        key_to_plot = to_superpose.keys()

        data = data.melt(
            ignore_index=False,
            id_vars=[x],
            value_vars=[c for c in key_to_plot],
            value_name=value_name,
            var_name="mode",
        )

        data["mode"] = data["mode"].replace(renamer)
        kwargs["hue"] = "mode"

        return self.plot_scatter_lines(
            data=data,
            x=x,
            y=value_name,
            filename=filename.format(value_name=value_name),
            **kwargs,
        )

    @data_getter
    @folder_split
    @single_plot
    def plot_scatter_lines(
        self,
        x,
        y,
        data=None,
        filename="{table}_lines_{y}_vs_{x}",
        mode="relplot",
        folder_col=None,
        logbase_x=1,
        logbase_y=1,
        sharex=True,
        sharey=False,
        legend_out=True,
        is_no_legend_title=False,
        set_kwargs={},
        x_rotate=0,
        cols_vary_only=None,
        cols_to_agg=[],
        aggregates=["mean", "sem"],
        is_x_errorbar=False,
        is_y_errorbar=False,
        row_title="{row_name}",
        col_title="{col_name}",
        plot_config_kwargs={},
        **kwargs,
    ):
        """Plotting all combinations of scatter and line plots.

        Parameters
        ----------
        x : str
            Column name of x axis.

        y : str
            Column name for the y axis.

        data : pd.DataFrame or str, optional
            Dataframe used for plotting. If str will use one of self.tables. If `None` runs all tables.

        filename : str or Path, optional
            Path to the file to which to save the results to. Will start at `base_dir`.
            Can interpolate {x} and {y}.

        mode : {"relplot","lmplot"}, optional
            Underlying function to use from seaborn. `lmplot` can also plot the estimated regression
            line.

        folder_col : str, optional
            Name of a column that will be used to separate the plot into multiple subfolders.

        logbase_x, logbase_y : int, optional
            Base of the x (resp. y) axis. If 1 no logscale. if `None` will automatically chose.

        sharex,sharey : bool, optional
            Wether to share x (resp. y) axis.

        legend_out : bool, optional
            Whether to put the legend outside of the figure.

        is_no_legend_title : bool, optional
            Whether to remove the legend title. If `is_legend_out` then will actually duplicate the
            legend :/, the best in that case is to remove the test of the legend column .

        set_kwargs : dict, optional
            Additional arguments to `FacetGrid.set`. E.g.
            dict(xlim=(0,None),xticks=[0,1],xticklabels=["a","b"]).

        x_rotate : int, optional
            By how much to rotate the x labels.

        cols_vary_only : list of str, optional
            Name of the columns that can vary when plotting (e.g. over which to compute bootstrap CI).
            This ensures that you are not you are not taking averages over values that you don't want.
            If `None` does not check. This is especially useful for

        cols_to_agg : list of str
            List of columns over which to aggregate. E.g. `["seed"]`. In case the underlying data
            are given at uniform intervals X, this is probably not needed as seaborn's line plot will
            compute the bootstrap CI for you.

        aggregates : list of str
            List of functions to use for aggregation. The aggregated columns will be called
            `{col}_{aggregate}`.

        is_x_errorbar,is_y_errorbar : bool, optional
            Whether to standard error (over the aggregation of cols_to_agg) as error bar . If `True`,
            `cols_to_agg` should not be empty and `"sem"` should be in `aggregates`.

        row_title,col_title : str, optional
            Template for the titles of the Facetgrid. Can use `{row_name}` and `{col_name}`
            respectively.

        plot_config_kwargs : dict, optional
            General config for plotting, e.g. arguments to matplotlib.rc, sns.plotting_context,
            matplotlib.set ...

        kwargs :
            Additional arguments to underlying seaborn plotting function. E.g. `col`, `row`, `hue`,
            `style`, `size` ...
        """
        kwargs["x"] = x
        kwargs["y"] = y

        if is_x_errorbar or is_y_errorbar:
            if (len(cols_to_agg) == 0) or ("sem" not in aggregates):
                logger.warn(
                    f"Not plotting errorbars due to empty cols_to_agg={cols_to_agg} or 'sem' not in aggregates={aggregates}."
                )
                is_x_errorbar, is_y_errorbar = False, False

        if mode == "relplot":
            used_kwargs = dict(
                legend="full",
                kind="line",
                markers=True,
                facet_kws={
                    "sharey": sharey,
                    "sharex": sharex,
                    "legend_out": legend_out,
                },
                style=kwargs.get("hue", None),
            )
            used_kwargs.update(kwargs)

            sns_plot = sns.relplot(data=data, **used_kwargs)

        elif mode == "lmplot":
            used_kwargs = dict(
                legend="full", sharey=sharey, sharex=sharex, legend_out=legend_out,
            )
            used_kwargs.update(kwargs)

            sns_plot = sns.lmplot(data=data, **used_kwargs)

        else:
            raise ValueError(f"Unkown mode={mode}.")

        if is_x_errorbar or is_y_errorbar:
            xerr, yerr = None, None
            if is_x_errorbar:
                x_sem = x.rsplit(" ", maxsplit=1)[0] + " Sem"  # _mean -> _sem
                xerr = data[x_sem]

            if is_y_errorbar:
                y_sem = y.rsplit(" ", maxsplit=1)[0] + " Sem"  # _mean -> _sem
                yerr = data[y_sem]

            sns_plot.map_dataframe(add_errorbars, yerr=yerr, xerr=xerr)

        if logbase_x != 1 or logbase_y != 1:
            sns_plot.map_dataframe(set_log_scale, basex=logbase_x, basey=logbase_y)

        return sns_plot


# HELPERS


def path_to_params(path):
    """Take a path name of the form `param1_value1/param2_value2/...` and returns a dictionary."""
    params = {}

    for name in path.split("/"):
        if "_" in name:
            k, v = name.split("_", maxsplit=1)
            params[k] = v

    return params


def get_param_in_kwargs(data, **kwargs):
    """
    Return all arguments that are names of the multiindex (i.e. param) of the data. I.e. for plotting
    this means that you most probably conditioned over them.
    """
    return {
        n: col
        for n, col in kwargs.items()
        if (isinstance(col, str) and col in data.index.names)
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


def add_errorbars(data, yerr, xerr, **kwargs):
    """Add errorbar to each sns.facetplot."""
    datas = [data]
    if xerr is not None:
        datas += [xerr.rename("xerr")]
    if yerr is not None:
        datas += [yerr.rename("yerr")]

    df = pd.concat(datas, axis=1).set_index(["hue", "style"])

    for idx in df.index.unique():
        # error bars will be different for different hue and style
        df_curr = df.loc[idx, :] if len(df.index.unique()) > 1 else df
        errs = dict()
        if xerr is not None:
            errs["xerr"] = df_curr["xerr"]
        if yerr is not None:
            errs["yerr"] = df_curr["yerr"]

        plt.errorbar(
            df_curr["x"].values,
            df_curr["y"].values,
            fmt="none",
            ecolor="lightgray",
            **errs,
        )


def set_log_scale(data, basex, basey, **kwargs):
    """Set the log scales as desired."""
    x_data = data["x"].unique()
    y_data = data["y"].unique()
    plt.xscale(**kwargs_log_scale(x_data, base=basex))
    plt.yscale(**kwargs_log_scale(y_data, base=basey))


def merge_rate_distortions(results, rate_cols, distortion_cols):
    """
    Adds a `distortion_type` and `rate_type` index by melting over `distortion_cols` and `rate_cols`
    respectively. The values columns are resepectively `distortion_val`and `rate_val`.
    """
    results = results.melt(
        id_vars=rate_cols,
        value_vars=distortion_cols,
        ignore_index=False,
        var_name="distortion_type",
        value_name="distortion_val",
    ).set_index(["distortion_type"], append=True)

    results = results.melt(
        id_vars="distortion_val",
        value_vars=rate_cols,
        ignore_index=False,
        var_name="rate_type",
        value_name="rate_val",
    ).set_index(["rate_type"], append=True)
    return results


def apply_area_under_RD(df):
    """Compute the area under the rate distortion curve using trapezoidal rule."""
    df = df.sort_values(by="distortion_val")
    return sklearn.metrics.auc(df["distortion_val"], df["rate_val"])


def apply_rate_mindistortion(
    df, epsilon=0.01, min_distortion_df=None, to_drop=["dist"], name="mindistortion"
):
    """
    Compute the rate for (delta) close to lossless prediction. `name` is name of added column. `min_distortion_df`
    is a dataframe containing the minimal distortions for a set of parameters (all parameters besides `to_drop`).
    For example you might want to see the rate of different models that are close to the best possible distoriton
    over ALL models so `to_drop` will be the model column. If `min_distortion_df` is None you don't look at best disortion
    over ALL models but best over CURRENT models.
    """
    if min_distortion_df is None:
        min_distortion = df["distortion_val"].min()
    else:
        current_idcs = df.reset_index(level=to_drop).index.unique()
        assert len(current_idcs) == 1
        min_distortion = min_distortion_df.loc[current_idcs[0]]

    threshold = min_distortion + epsilon
    df = df[df["distortion_val"] <= threshold]
    # returning series in apply can be very slow
    return pd.Series(
        [df["rate_val"].mean(), df["rate_val"].sem(), threshold],
        index=[f"rate_val_{name}_mean", f"rate_val_{name}_sem", f"{name}_threshold"],
    )


def get_varying_levels(df, is_index=True):
    """Return the name of the levels that are varying in multi index / columns."""
    if is_index:
        levels, names = df.index.levels, df.index.names
    else:
        levels, names = df.columns.levels, df.columns.names
    return [n for l, n in zip(levels, names) if len(l) > 1]


if __name__ == "__main__":
    main_cli()
