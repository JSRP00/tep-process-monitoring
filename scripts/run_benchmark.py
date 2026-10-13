"""Run the benchmark and save the results.

Usage:  python scripts/run_benchmark.py

Add your own monitors to MONITORS. Any model with ``fit`` and ``score_samples``
(scikit-learn style) can be wrapped with ScoreMonitor.
"""

from pathlib import Path

from sklearn.ensemble import IsolationForest

from tepmon import PCAMonitor, ScoreMonitor
from tepmon.benchmark import run, summary

MONITORS = {
    "PCA": PCAMonitor(n_components=11, alpha=0.01),
    "IsolationForest": ScoreMonitor(IsolationForest(n_estimators=300, random_state=0), name="IF", alpha=0.01),
}

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    results = run(MONITORS)
    results.to_csv(out / "results.csv", index=False)
    table = summary(results)
    far = (results.groupby(["monitor", "statistic"])["FAR_normal_test"].first() * 100).round(1)
    with open(out / "summary.md", "w") as f:
        f.write("Fault detection rate (%) per fault\n\n")
        f.write(table.to_markdown())
        f.write("\n\nFalse alarm rate (%) on the normal test set\n\n")
        f.write(far.to_frame("FAR").to_markdown())
        f.write("\n")
    print(table)
    print(far)
