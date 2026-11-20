"""IGD+ as an additional multi-objective metric, computed on top of CARP-S.

CARP-S's `carps.analysis.gather_data.normalize_logs` only computes hypervolume. This module
adds IGD+ without touching CARP-S itself, by reusing the `reference_point`/`minimum_cost`/
`trial_value__cost_raw` columns it already attaches to multi-objective rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def _get_normalized_costs(df: pd.DataFrame) -> np.ndarray:
    costs = np.array(df["trial_value__cost_raw"].to_list())
    cost_min = df["minimum_cost"].iloc[0]
    cost_max = df["reference_point"].iloc[0]
    return (costs - cost_min) / (cost_max - cost_min)


def _best_known_pareto_front(task_df: pd.DataFrame) -> np.ndarray:
    costs_normed = _get_normalized_costs(task_df)
    front_ids = NonDominatedSorting().do(costs_normed, only_non_dominated_front=True)
    return np.unique(costs_normed[front_ids], axis=0)


def _igd_plus_for_run(run_df: pd.DataFrame, best_known_front: np.ndarray) -> pd.Series:
    run_df = run_df.sort_values("n_trials")
    costs_normed = _get_normalized_costs(run_df)
    ind = IGDPlus(best_known_front)
    igds = [float(ind(costs_normed[: n + 1])) for n in range(len(run_df))]
    return pd.Series(igds, index=run_df.index)


def add_igd_plus(logs: pd.DataFrame) -> pd.DataFrame:
    """Add IGD+ to logs already processed by `carps.analysis.gather_data.normalize_logs`.

    Adds "igd_plus" (raw, lower is better) and "igd_plus_inc_norm" (per-task normalized,
    cumulative-incumbent), the IGD+ analogue of "trial_value__cost_inc_norm" for hypervolume.

    Args:
        logs (pd.DataFrame): Logs returned by `normalize_logs`.

    Returns:
        pd.DataFrame: Logs with IGD+ columns added.
    """
    logs = logs.copy()
    logs["igd_plus"] = np.nan
    ids_mo = logs["reference_point"].notna()

    for task_id, task_df in logs.loc[ids_mo].groupby("task_id"):
        best_known_front = _best_known_pareto_front(task_df)
        for _, run_df in task_df.groupby(["optimizer_id", "seed"]):
            igds = _igd_plus_for_run(run_df, best_known_front)
            logs.loc[igds.index, "igd_plus"] = igds

    logs["igd_plus_norm"] = logs.groupby("task_id")["igd_plus"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    )
    logs["igd_plus_inc_norm"] = logs.groupby(["task_id", "optimizer_id", "seed"])["igd_plus_norm"].transform("cummin")
    return logs
