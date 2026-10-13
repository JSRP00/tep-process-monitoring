"""Standard fault-detection metrics for process monitoring."""

from __future__ import annotations

import numpy as np


def detection_rate(alarms: np.ndarray, y: np.ndarray) -> float:
    """Fault detection rate (FDR): share of faulty samples that raise an alarm."""
    faulty = np.asarray(y) == 1
    return float(np.mean(np.asarray(alarms)[faulty])) if faulty.any() else float("nan")


def false_alarm_rate(alarms: np.ndarray, y: np.ndarray) -> float:
    """False alarm rate (FAR): share of normal samples that raise an alarm."""
    normal = np.asarray(y) == 0
    return float(np.mean(np.asarray(alarms)[normal])) if normal.any() else float("nan")


def detection_delay(alarms: np.ndarray, y: np.ndarray, run: int = 1) -> float:
    """Samples from fault onset to the first alarm.

    With ``run > 1`` the alarm must persist for ``run`` consecutive samples,
    which filters isolated false alarms. Returns NaN if the fault is never detected.
    Multiply by 3 to convert to minutes (TEP samples every 3 minutes).
    """
    alarms = np.asarray(alarms, dtype=bool)
    onset = int(np.argmax(np.asarray(y) == 1))
    after = alarms[onset:]
    if run > 1:
        window = np.convolve(after.astype(int), np.ones(run, dtype=int), mode="valid")
        hits = np.flatnonzero(window == run)
    else:
        hits = np.flatnonzero(after)
    return float(hits[0]) if hits.size else float("nan")
