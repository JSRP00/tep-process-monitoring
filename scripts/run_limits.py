"""Compare ways of setting control limits: do they give the false alarm rate they promise?

Usage:  python scripts/run_limits.py

Each monitor keeps the same model; only the control limit changes (see
tepmon/limits.py). False alarms are measured on 22 independent normal runs: the
normal test file and the first 160 (fault-free) samples of each fault file.
Detection is the mean over the 21 faults of the share of faulty samples flagged.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from tepmon import FAULT_ONSET, FAULTS, AutoencoderMonitor, LaggedMonitor, PCAMonitor, ScoreMonitor, load_test, load_train
from tepmon.limits import METHODS, fit_with_limits

ALPHA = 0.01

# Same settings as scripts/run_benchmark.py; the argument is the calibration share.
MONITORS = {
    "PCA": lambda share: PCAMonitor(n_components=11, alpha=ALPHA),
    "DPCA": lambda share: LaggedMonitor(PCAMonitor(n_components=0.54, alpha=ALPHA), lags=2),
    "IsolationForest": lambda share: ScoreMonitor(IsolationForest(n_estimators=300, random_state=0), name="IF",
                                                  alpha=ALPHA, calibration_share=share),
    "OneClassSVM": lambda share: ScoreMonitor(OneClassSVM(nu=0.05, gamma="scale"), name="OCSVM",
                                              alpha=ALPHA, calibration_share=share),
    "LOF": lambda share: ScoreMonitor(LocalOutlierFactor(n_neighbors=20, novelty=True), name="LOF",
                                      alpha=ALPHA, calibration_share=share),
    "Autoencoder": lambda share: AutoencoderMonitor(hidden=(32, 11, 32), alpha=ALPHA, calibration_share=share),
    "DynamicAutoencoder": lambda share: LaggedMonitor(
        AutoencoderMonitor(hidden=(64, 20, 64), alpha=ALPHA, calibration_share=share), lags=2),
}


def evaluate(monitor) -> list[dict]:
    """False alarm and detection rates per statistic for a fitted monitor."""
    normal, faulty = {}, {}
    for fault in [0, *FAULTS]:
        X, _ = load_test(fault)
        for stat, values in monitor.statistics(X).items():
            alarms = values > monitor.limits_[stat]
            normal.setdefault(stat, []).append(alarms if fault == 0 else alarms[:FAULT_ONSET])
            if fault:
                faulty.setdefault(stat, []).append(alarms[FAULT_ONSET:].mean())
    return [{"statistic": s, "FAR": float(np.concatenate(normal[s]).mean()),
             "FAR_worst_run": float(max(a.mean() for a in normal[s])),
             "FDR": float(np.mean(faulty[s]))} for s in normal]


if __name__ == "__main__":
    X_train = load_train()
    rows = []
    for name, make in MONITORS.items():
        for method in METHODS:
            monitor = fit_with_limits(make, X_train, method, alpha=ALPHA)
            for row in evaluate(monitor):
                rows.append({"monitor": name, "method": method, **row})
            print(name, method, "done", flush=True)

    results = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    results.to_csv(out / "limits.csv", index=False)

    results["row"] = results["monitor"] + " " + results["statistic"]
    with open(out / "limits.md", "w") as f:
        for metric, title in [("FAR", "False alarm rate on 22 normal runs (%), target 1%"),
                              ("FAR_worst_run", "False alarm rate in the worst normal run (%)"),
                              ("FDR", "Mean detection rate over the 21 faults (%)")]:
            table = (results.pivot(index="row", columns="method", values=metric)[METHODS] * 100).round(1)
            f.write(f"## {title}\n\n{table.to_markdown()}\n\n")
    print((results.pivot(index="row", columns="method", values="FAR")[METHODS] * 100).round(1))
    print((results.pivot(index="row", columns="method", values="FDR")[METHODS] * 100).round(1))
