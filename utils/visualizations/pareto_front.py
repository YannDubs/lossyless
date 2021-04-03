#! copy pasted from https://github.com/optuna/optuna/pull/2450 until gets merged

from typing import List, Optional

import optuna
from optuna._experimental import experimental
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
from optuna.visualization.matplotlib._matplotlib_imports import _imports

if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = optuna.logging.get_logger(__name__)


@experimental("2.7.0")
def plot_pareto_front(
    study: Study,
    *,
    target_names: Optional[List[str]] = None,
    include_dominated_trials: bool = True,
    axis_order: Optional[List[int]] = None,
) -> "Axes":
    """Plot the Pareto front of a study.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_pareto_front` for an example.

    Example:

        The following code snippet shows how to plot the Pareto front of a study.

        .. plot::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", 0, 5)
                y = trial.suggest_float("y", 0, 3)

                v0 = 4 * x ** 2 + 4 * y ** 2
                v1 = (x - 5) ** 2 + (y - 5) ** 2
                return v0, v1


            study = optuna.create_study(directions=["minimize", "minimize"])
            study.optimize(objective, n_trials=50)

            optuna.visualization.matplotlib.plot_pareto_front(study)

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values.
        target_names:
            Objective name list used as the axis titles. If :obj:`None` is specified,
            "Objective {objective_index}" is used instead.
        include_dominated_trials:
            A flag to include all dominated trial's objective values.
        axis_order:
            A list of indices indicating the axis order. If :obj:`None` is specified,
            default order is used.

    Returns:
        A :class:`matplotlib.axes.Axes` object.

    Raises:
        :exc:`ValueError`:
            If the number of objectives of ``study`` isn't 2 or 3.
    """

    _imports.check()

    if len(study.directions) == 2:
        return _get_pareto_front_2d(
            study, target_names, include_dominated_trials, axis_order
        )
    elif len(study.directions) == 3:
        return _get_pareto_front_3d(
            study, target_names, include_dominated_trials, axis_order
        )
    else:
        raise ValueError(
            "`plot_pareto_front` function only supports 2 or 3 objective studies."
        )


def _get_non_pareto_front_trials(
    study: Study, pareto_trials: List[FrozenTrial]
) -> List[FrozenTrial]:

    non_pareto_trials = []
    for trial in study.get_trials():
        if trial.state == TrialState.COMPLETE and trial not in pareto_trials:
            non_pareto_trials.append(trial)
    return non_pareto_trials


def _get_pareto_front_2d(
    study: Study,
    target_names: Optional[List[str]],
    include_dominated_trials: bool = False,
    axis_order: Optional[List[int]] = None,
) -> "Axes":

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Pareto-front Plot")
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    if target_names is None:
        target_names = ["Objective 0", "Objective 1"]
    elif len(target_names) != 2:
        raise ValueError("The length of `target_names` is supposed to be 2.")

    ax.set_xlabel(target_names[0])
    ax.set_ylabel(target_names[1])

    # Prepare data for plotting.
    trials = study.best_trials
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        return ax

    if include_dominated_trials:
        non_pareto_trials = _get_non_pareto_front_trials(study, trials)
        trials += non_pareto_trials

    if axis_order is None:
        axis_order = list(range(2))
    else:
        if len(axis_order) != 2:
            raise ValueError(
                f"Size of `axis_order` {axis_order}. Expect: 2, Actual: {len(axis_order)}."
            )
        if len(set(axis_order)) != 2:
            raise ValueError(
                f"Elements of given `axis_order` {axis_order} are not unique!"
            )
        if max(axis_order) > 1:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {max(axis_order)} "
                "higher than 1."
            )
        if min(axis_order) < 0:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {min(axis_order)} "
                "lower than 0."
            )

    ax.scatter(
        x=[t.values[axis_order[0]] for t in trials[len(study.best_trials) :]],
        y=[t.values[axis_order[1]] for t in trials[len(study.best_trials) :]],
        color=cmap(0),
        label="Trial",
    )
    ax.scatter(
        x=[t.values[axis_order[0]] for t in trials[: len(study.best_trials)]],
        y=[t.values[axis_order[1]] for t in trials[: len(study.best_trials)]],
        color=cmap(3),
        label="Best Trial",
    )

    if include_dominated_trials:
        ax.legend()

    return ax


def _get_pareto_front_3d(
    study: Study,
    target_names: Optional[List[str]],
    include_dominated_trials: bool = False,
    axis_order: Optional[List[int]] = None,
) -> "Axes":

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Pareto-front Plot")
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    if target_names is None:
        target_names = ["Objective 0", "Objective 1", "Objective 2"]
    elif len(target_names) != 3:
        raise ValueError("The length of `target_names` is supposed to be 3.")

    ax.set_xlabel(target_names[0])
    ax.set_ylabel(target_names[1])
    ax.set_zlabel(target_names[2])

    trials = study.best_trials
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        return ax

    if include_dominated_trials:
        non_pareto_trials = _get_non_pareto_front_trials(study, trials)
        trials += non_pareto_trials

    if axis_order is None:
        axis_order = list(range(3))
    else:
        if len(axis_order) != 3:
            raise ValueError(
                f"Size of `axis_order` {axis_order}. Expect: 3, Actual: {len(axis_order)}."
            )
        if len(set(axis_order)) != 3:
            raise ValueError(
                f"Elements of given `axis_order` {axis_order} are not unique!."
            )
        if max(axis_order) > 2:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {max(axis_order)} "
                "higher than 2."
            )
        if min(axis_order) < 0:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {min(axis_order)} "
                "lower than 0."
            )

    ax.scatter(
        xs=[t.values[axis_order[0]] for t in trials[len(study.best_trials) :]],
        ys=[t.values[axis_order[1]] for t in trials[len(study.best_trials) :]],
        zs=[t.values[axis_order[2]] for t in trials[len(study.best_trials) :]],
        color=cmap(0),
        label="Trial",
    )
    ax.scatter(
        xs=[t.values[axis_order[0]] for t in trials[: len(study.best_trials)]],
        ys=[t.values[axis_order[1]] for t in trials[: len(study.best_trials)]],
        zs=[t.values[axis_order[2]] for t in trials[: len(study.best_trials)]],
        color=cmap(3),
        label="Best Trial",
    )

    if include_dominated_trials:
        ax.legend()

    return ax
