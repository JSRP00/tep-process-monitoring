"""Ways of setting a control limit, and how well each one holds its false alarm rate.

A monitor is a model plus a control limit. The limit is supposed to give a known
false alarm rate (e.g. 1%) on new normal data. This module keeps the model fixed
and changes only how the limit is set:

- ``own``: the monitor's default limit, fitted on all training data. For PCA this
  is the theoretical limit (F distribution for T², Jackson–Mudholkar for SPE);
  for ML monitors, the empirical quantile on a random 30% hold-out.
- ``in_sample``: empirical (1 - alpha) quantile of the statistic on the same data
  the model was trained on.
- ``split``: the model is trained on the first part of the normal run and the
  limit is the split-conformal quantile of the statistic on the last part, which
  the model has not seen. The split is contiguous, so neighbouring
  (autocorrelated) samples do not end up on both sides.
- ``cv``: like ``split`` but with K contiguous folds: each fold is scored by a
  model trained on the others, the limit is the conformal quantile of all these
  out-of-fold statistics, and the final model is trained on all the data.

Conformal quantile: the ceil((n + 1)(1 - alpha))-th smallest of n calibration
values. Under exchangeability this guarantees a false alarm rate of at most
alpha on a new sample; time series are not exchangeable, so here the guarantee
is only approximate and is checked empirically.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .monitors import Monitor

METHODS = ["own", "in_sample", "split", "cv"]


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    """Split-conformal (1 - alpha) quantile of calibration values."""
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(values[min(k, n) - 1])


def fit_with_limits(make: Callable[[float], Monitor], X: np.ndarray, method: str, alpha: float = 0.01,
                    split_share: float = 0.3, n_folds: int = 5) -> Monitor:
    """Fit a monitor and set its control limits with the given method.

    ``make(calibration_share)`` must return a new, unfitted monitor; it is called
    with 0.3 for ``own`` (the monitor's default) and 0 otherwise, so that ML
    monitors train on all the data they are given.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)

    if method == "own":
        return make(0.3).fit(X)

    if method == "in_sample":
        monitor = make(0.0).fit(X)
        values = monitor.statistics(X)
        monitor.limits_ = {s: float(np.quantile(v, 1 - alpha)) for s, v in values.items()}
        return monitor

    if method == "split":
        cut = n - int(round(split_share * n))
        monitor = make(0.0).fit(X[:cut])
        # Statistics on the full run so lagged monitors keep their history at the cut.
        values = monitor.statistics(X)
        monitor.limits_ = {s: conformal_quantile(v[cut:], alpha) for s, v in values.items()}
        return monitor

    if method == "cv":
        folds = np.array_split(np.arange(n), n_folds)
        out_of_fold: dict[str, list[np.ndarray]] = {}
        for fold in folds:
            train = np.setdiff1d(np.arange(n), fold)
            # Train on the other folds, joined in time order.
            values = make(0.0).fit(X[train]).statistics(X)
            for s, v in values.items():
                out_of_fold.setdefault(s, []).append(v[fold])
        monitor = make(0.0).fit(X)
        monitor.limits_ = {s: conformal_quantile(np.concatenate(v), alpha) for s, v in out_of_fold.items()}
        return monitor

    raise ValueError(f"method must be one of {METHODS}, got {method!r}")
