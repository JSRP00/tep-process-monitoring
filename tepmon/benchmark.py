"""Run monitors over every TEP fault and collect the metrics in a table."""

from __future__ import annotations

import pandas as pd

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
        for stat in choices:
            which = None if stat == "any" else stat
            far = false_alarm_rate(monitor.alarms(X_normal, which), y_normal)
            for fault in faults:
                X, y = load_test(fault)
                alarms = monitor.alarms(X, which)
                rows.append({
                    "monitor": name, "statistic": stat, "fault": fault,
                    "FDR": detection_rate(alarms, y),
                    "FAR_normal_test": far,
                    "delay_samples": detection_delay(alarms, y, run=run_length),
                })
    return pd.DataFrame(rows)


def summary(results: pd.DataFrame) -> pd.DataFrame:
    """Wide table of FDR (%) per fault, one column per monitor/statistic."""
    table = results.assign(col=results["monitor"] + " " + results["statistic"]).pivot(
        index="fault", columns="col", values="FDR") * 100
    return table.round(1)
