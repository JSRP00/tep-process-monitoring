"""Run monitors over every TEP fault and collect the metrics in a table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import FAULTS, load_test, load_train
from .metrics import detection_delay, detection_rate, false_alarm_rate
from .monitors import Monitor


def run(monitors: dict[str, Monitor], faults: list[int] = FAULTS, run_length: int = 1) -> pd.DataFrame:
    """Fit each monitor on the normal training data and evaluate it per fault.

    Returns a long table with one row per (monitor, statistic, fault). The
    statistic "any" combines all of a monitor's statistics (alarm if any fires).
    False alarm rates are measured on the normal test set (fault 0).
    """
    X_train = load_train()
    X_normal, y_normal = load_test(0)
    rows = []
    for name, monitor in monitors.items():
        monitor.fit(X_train)
        stat_names = list(monitor.limits_)
        choices = stat_names + (["any"] if len(stat_names) > 1 else [])
        normal_stats = monitor.statistics(X_normal)
        test_stats = {fault: (monitor.statistics(load_test(fault)[0]), load_test(fault)[1]) for fault in faults}
        for stat in choices:
            which = None if stat == "any" else stat
            far = false_alarm_rate(monitor.alarms(X_normal, which), y_normal)
            # Evaluation-only threshold giving exactly 1% false alarms on the normal test set,
            # so every method is compared at the same false alarm rate.
            oracle = np.quantile(normal_stats[stat], 0.99) if which else None
            for fault in faults:
                X, y = load_test(fault)
                alarms = monitor.alarms(X, which)
                values, _ = test_stats[fault]
                faulty = values[stat][y == 1] if which else None
                rows.append({
                    "monitor": name, "statistic": stat, "fault": fault,
                    "FDR": detection_rate(alarms, y),
                    "FAR_normal_test": far,
                    "delay_samples": detection_delay(alarms, y, run=run_length),
                    "FDR_at_1pct_FAR": float(np.mean(faulty > oracle)) if which else float("nan"),
                    "AUC": float(roc_auc_score(np.r_[np.zeros(len(normal_stats[stat])), np.ones(len(faulty))],
                                               np.r_[normal_stats[stat], faulty])) if which else float("nan"),
                })
    return pd.DataFrame(rows)


def summary(results: pd.DataFrame, metric: str = "FDR") -> pd.DataFrame:
    """Wide table of a metric (in %) per fault, one column per monitor/statistic."""
    table = results.assign(col=results["monitor"] + " " + results["statistic"]).pivot(
        index="fault", columns="col", values=metric) * 100
    return table.round(1)


def overview(results: pd.DataFrame) -> pd.DataFrame:
    """One row per monitor/statistic with the averages over all faults."""
    g = results.groupby(["monitor", "statistic"]).agg(
        mean_AUC=("AUC", "mean"), mean_FDR_at_1pct_FAR=("FDR_at_1pct_FAR", "mean"),
        FDR_own_limit=("FDR", "mean"), FAR_own_limit=("FAR_normal_test", "first"),
        median_delay_samples=("delay_samples", "median"))
    g[["mean_AUC", "mean_FDR_at_1pct_FAR", "FDR_own_limit", "FAR_own_limit"]] *= 100
    return g.round(1).sort_values("mean_AUC", ascending=False)
