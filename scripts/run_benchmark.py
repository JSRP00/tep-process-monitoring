"""Run the benchmark and save the results.

Usage:  python scripts/run_benchmark.py

Add your own monitors to MONITORS. Any model with ``fit`` and ``score_samples``
(scikit-learn style) can be wrapped with ScoreMonitor.
"""

from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from tepmon import AutoencoderMonitor, LaggedMonitor, PCAMonitor, ScoreMonitor
from tepmon.benchmark import overview, run, summary

# Settings fixed before looking at test results:
# - PCA keeps 11 components (54% of variance); dynamic PCA keeps the same variance share.
# - Autoencoder bottlenecks match the number of PCA components (11 static, 20 dynamic).
# - Dynamic models use 2 lags (each sample plus the 2 previous ones).
MONITORS = {
    "PCA": PCAMonitor(n_components=11, alpha=0.01),
    "DPCA": LaggedMonitor(PCAMonitor(n_components=0.54, alpha=0.01), lags=2),
    "IsolationForest": ScoreMonitor(IsolationForest(n_estimators=300, random_state=0), name="IF", alpha=0.01),
    "OneClassSVM": ScoreMonitor(OneClassSVM(nu=0.05, gamma="scale"), name="OCSVM", alpha=0.01),
    "LOF": ScoreMonitor(LocalOutlierFactor(n_neighbors=20, novelty=True), name="LOF", alpha=0.01),
    "Autoencoder": AutoencoderMonitor(hidden=(32, 11, 32), alpha=0.01),
    "DynamicAutoencoder": LaggedMonitor(AutoencoderMonitor(hidden=(64, 20, 64), alpha=0.01), lags=2),
}

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    results = run(MONITORS)
    results.to_csv(out / "results.csv", index=False)
    with open(out / "summary.md", "w") as f:
        f.write("## Overview (averages over the 21 faults, %)\n\n")
        f.write(overview(results).to_markdown())
        for metric, title in [("AUC", "AUC per fault (%)"),
                              ("FDR_at_1pct_FAR", "Detection rate at 1% false alarms (%)"),
                              ("FDR", "Detection rate with each method's own control limit (%)")]:
            f.write(f"\n\n## {title}\n\n")
            f.write(summary(results, metric).dropna(axis=1, how="all").to_markdown())
        f.write("\n")
    print(overview(results))
