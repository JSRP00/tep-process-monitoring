"""Process monitors.

Every monitor follows the same small interface so that classical and ML
methods can be compared on equal terms:

- ``fit(X)`` learns from normal-operation data only.
- ``statistics(X)`` returns a dict ``{name: array}`` of monitoring statistics.
- ``limits_`` is a dict ``{name: control limit}`` set during ``fit``.
- ``alarms(X)`` returns a boolean array: True where any statistic exceeds its limit.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class Monitor:
    """Base class. Subclasses implement ``fit`` and ``statistics``."""

    limits_: dict[str, float]

    def fit(self, X: np.ndarray) -> "Monitor":
        raise NotImplementedError

    def statistics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def alarms(self, X: np.ndarray, which: str | None = None) -> np.ndarray:
        """Alarms from one statistic (``which``) or from any of them (default)."""
        values = self.statistics(X)
        names = [which] if which else list(values)
        return np.any([values[n] > self.limits_[n] for n in names], axis=0)


class PCAMonitor(Monitor):
    """PCA-based MSPC with Hotelling's T² and SPE (Q) statistics.

    Parameters
    ----------
    n_components : int or float
        Number of principal components, or the share of variance to retain
        (e.g. 0.9 keeps the smallest number of components explaining 90%).
    alpha : float
        False-alarm probability used for both control limits (0.01 = 99% limits).

    Control limits follow the standard results: the F-distribution limit for
    T² and the Jackson–Mudholkar approximation for SPE.
    """

    def __init__(self, n_components: int | float = 0.9, alpha: float = 0.01):
        self.n_components = n_components
        self.alpha = alpha

    def fit(self, X: np.ndarray) -> "PCAMonitor":
        X = np.asarray(X, dtype=float)
        n, m = X.shape
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=1)
        self.std_[self.std_ == 0] = 1.0
        Z = (X - self.mean_) / self.std_

        cov = np.cov(Z, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        order = np.argsort(eigval)[::-1]
        eigval, eigvec = np.clip(eigval[order], 0, None), eigvec[:, order]

        if isinstance(self.n_components, float):
            share = np.cumsum(eigval) / eigval.sum()
            a = int(np.searchsorted(share, self.n_components) + 1)
        else:
            a = int(self.n_components)
        if not 1 <= a < m:
            raise ValueError(f"n_components must give between 1 and {m - 1} components, got {a}")

        self.n_components_ = a
        self.loadings_ = eigvec[:, :a]
        self.eigenvalues_ = eigval[:a]

        t2_limit = a * (n - 1) * (n + 1) / (n * (n - a)) * stats.f.ppf(1 - self.alpha, a, n - a)

        residual = eigval[a:]
        th1, th2, th3 = (np.sum(residual ** i) for i in (1, 2, 3))
        h0 = 1 - 2 * th1 * th3 / (3 * th2 ** 2)
        c = stats.norm.ppf(1 - self.alpha)
        spe_limit = th1 * (c * np.sqrt(2 * th2 * h0 ** 2) / th1 + 1 + th2 * h0 * (h0 - 1) / th1 ** 2) ** (1 / h0)

        self.limits_ = {"T2": float(t2_limit), "SPE": float(spe_limit)}
        return self

    def statistics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        scores = Z @ self.loadings_
        t2 = np.sum(scores ** 2 / self.eigenvalues_, axis=1)
        residual = Z - scores @ self.loadings_.T
        spe = np.sum(residual ** 2, axis=1)
        return {"T2": t2, "SPE": spe}


class ScoreMonitor(Monitor):
    """Wraps any anomaly-scoring model (e.g. scikit-learn) as a monitor.

    The model is trained on standardised normal data. Its anomaly score is
    ``-model.score_samples(X)`` by default (higher = more anomalous), and the
    control limit is the empirical (1 - alpha) quantile of the scores on a
    held-out part of the normal data, so it is not fitted on the same samples.

    Example
    -------
    >>> from sklearn.ensemble import IsolationForest
    >>> monitor = ScoreMonitor(IsolationForest(random_state=0), name="IF")
    """

    def __init__(self, model, name: str = "score", alpha: float = 0.01,
                 calibration_share: float = 0.3, score=None, random_state: int = 0):
        self.model = model
        self.name = name
        self.alpha = alpha
        self.calibration_share = calibration_share
        self.score = score or (lambda m, Z: -m.score_samples(Z))
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "ScoreMonitor":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=1)
        self.std_[self.std_ == 0] = 1.0
        Z = (X - self.mean_) / self.std_

        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(Z))
        n_cal = int(round(self.calibration_share * len(Z)))
        cal, train = Z[idx[:n_cal]], Z[idx[n_cal:]]

        self.model.fit(train)
        self.limits_ = {self.name: float(np.quantile(self.score(self.model, cal), 1 - self.alpha))}
        return self

    def statistics(self, X: np.ndarray) -> dict[str, np.ndarray]:
        Z = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        return {self.name: np.asarray(self.score(self.model, Z))}
