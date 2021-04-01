"""Entry point to aggregate a series of results obtained using `main.py` in a nice plot / table.

This should be called by `python aggregate.py <conf>` where <conf> sets all configs from the cli, see 
the file `config/aggregate.yaml` for details about the configs. or use `python aggregate.py -h`.
"""

import glob
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import hydra
from lossyless.helpers import BASE_LOG, check_import
from main import COMPRESSOR_RES, CONFIG_FILE, get_stage_name
from omegaconf import OmegaConf
from utils.helpers import cfg_load, cfg_save, getattr_from_oneof, omegaconf2namespace
from utils.postplotting import (
    PRETTY_RENAMER,
    PostPlotter,
    data_getter,
    folder_split,
    single_plot,
    table_summarizer,
)
from utils.postplotting.helpers import aggregate, save_fig
from utils.visualizations.helpers import kwargs_log_scale

try:
    import sklearn.metrics
except:
    pass

try:
    import optuna

    #! waiting for https://github.com/optuna/optuna/pull/2450
    # from optuna.visualization.matplotlib import plot_pareto_front
    from utils.visualizations.pareto_front import plot_pareto_front
except:
    pass

logger = logging.getLogger(__name__)


@hydra.main(config_name="aggregate", config_path="config")
def main_cli(cfg):
    # uses main_cli sot that `main` can be called from notebooks.
    return main(cfg)


def main(cfg):

    begin(cfg)

    # make sure you are using primitive types from now on because omegaconf does not always work
    cfg = omegaconf2namespace(cfg)

    aggregator = ResultAggregator(pretty_renamer=PRETTY_RENAMER, **cfg.kwargs)

    logger.info(f"Collecting the data ..")
    for name, pattern in cfg.collect_data.items():
        if pattern is not None:
            aggregator.collect_data(pattern=pattern, table_name=name)

    if len(aggregator.tables) > 1:
        # if multiple tables also add "merged" that contains all
        aggregator.merge_tables(list(cfg.collect_data.keys()))

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


# MAIN CLASS
class ResultAggregator(PostPlotter):
    """Aggregates batches of results (multirun)

    Parameters
    ----------
    save_dir : str or Path
        Where to save all results.

    base_dir : str or Path
        Base folder from which all paths start.

    kwargs :
        Additional arguments to `PostPlotter`.
    """

    def __init__(self, save_dir, base_dir=Path(__file__).parent, **kwargs):
        super().__init__(**kwargs)
        self.base_dir = Path(base_dir)
        self.save_dir = self.base_dir / Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.tables = dict()
        self.param_names = dict()
        self.cfgs = dict()

    def merge_tables(self, to_merge=["featurizer", "predictor"]):
        """Add one large table called `"merge"` that concatenates other tables."""
        merged = self.tables[to_merge[0]]
        for table in to_merge[1:]:
            merged = pd.merge(
                merged, self.tables[table], left_index=True, right_index=True
            )
        self.param_names["merged"] = list(merged.index.names)
        self.tables["merged"] = merged

    def collect_data(
        self,
        pattern=f"results/**/{COMPRESSOR_RES}",
        table_name="featurizer",
        params_to_rm=["jid"],
        params_to_add={},
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

        params_to_add : dict, optional
            Parameters to add. Those will be added from the `config.yaml` files. The key should be 
            the name of the paramter that you weant to add and the value should be the config key 
            (using dots). E.g. {"lr": "optimizer.lr"}. The config file should be saved at the same
            place as the results file.
        """
        # TODO test params_to_add
        paths = list(self.base_dir.glob(pattern))
        if len(paths) == 0:
            raise ValueError(f"No files found for your pattern={pattern}")

        results = []
        self.param_names[table_name] = set()
        for path in paths:
            folder = path.parent

            # select everything from "exp_"
            path_clean = "exp_" + str(path.resolve()).split("/exp_")[-1]
            # make dict of params
            params = path_to_params(path_clean)

            for p in params_to_rm:
                params.pop(p)

            try:
                cfg = cfg_load(folder / f"{get_stage_name(table_name)}_{CONFIG_FILE}")
                for name, param_key in params_to_add.items():
                    params[name] = cfg.select(param_key)
                self.cfgs[table_name] = cfg  # will ony save last
            except FileNotFoundError:
                if len(params_to_add) > 0:
                    logger.exception(
                        "Cannot use `params_to_add` as config file was not found:"
                    )
                    raise

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
            xlabel="Distortion",
            ylabel="Rate (bits)",
            **kwargs,
        )

    @data_getter
    def plot_invariance_RD_curve(
        self,
        data="featurizer",
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
        data : pd.DataFrame or str
            Dataframe to use for plotting. If str will use one of self.tables. If `None` runs all tables.

        col_dist_param : str, optional
            Name of the column that will distinguish the non invariant and the invariant model.

        noninvariant : str, optional
            Name of the non invariant model.

        desirable_distortion : str, optional
            Name fo the column containing the invariance distortion.

        kwargs :
            Additional arguments to `plot_scatter_lines`.
        """
        results = data
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
            xlabel="Distortion",
            ylabel="Rate",
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
        xlabel="",
        ylabel="",
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
                logger.warning(
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

        #! waiting for https://github.com/mwaskom/seaborn/issues/2456
        if xlabel != "":
            for ax in sns_plot.fig.axes:
                ax.set_xlabel(xlabel)

        if ylabel != "":
            for ax in sns_plot.fig.axes:
                ax.set_ylabel(ylabel)

        sns_plot.tight_layout()

        return sns_plot

    def plot_optuna_hypopt(
        self,
        storage,
        study_name="main",
        filename="hypopt",
        plot_functions_str=[
            "plot_param_importances",
            "plot_parallel_coordinate",
            "plot_optimization_history",
        ],
    ):
        """Plot a summary of Optuna study"""
        check_import("optuna", "plot_optuna_hypopt")
        study = optuna.load_study(study_name, storage)
        cfg = self.cfgs[list(self.cfgs.keys())[-1]]  # which cfg shouldn't matter

        best_trials = study.best_trials
        to_save = {
            "solutions": [{"values": t.values, "params": t.params} for t in best_trials]
        }
        cfg_save(to_save, self.save_dir / f"{self.prfx}{filename}.yaml")

        for i, monitor in enumerate(cfg.monitor_return):
            for plot_f_str in plot_functions_str:
                if (
                    plot_f_str == "plot_optimization_history"
                    and len(cfg.monitor_return) > 1
                ):
                    #! waiting for https://github.com/optuna/optuna/issues/2531
                    continue

                # plotting
                plt_modules = [optuna.visualization.matplotlib]
                plot_f = getattr_from_oneof(plt_modules, plot_f_str)
                out = plot_f(
                    study, target=lambda trial: trial.values[i], target_name=monitor
                )

                # saving
                nice_monitor = monitor.replace("/", "_")
                filename = self.save_dir / f"{plot_f_str}_{nice_monitor}"
                save_fig(out, filename, self.dpi)

        if len(cfg.monitor_return) > 1:
            out = plot_pareto_front(
                study, target_names=cfg.monitor_return, include_dominated_trials=False
            )
            filename = self.save_dir / "plot_pareto_front"
            save_fig(out, filename, self.dpi)


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
