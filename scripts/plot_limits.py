"""Plot the false alarm rate of each monitor under each way of setting its control limit.

Reads results/limits.csv (written by run_limits.py) and saves results/limits.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]

ROWS = [  # (monitor, statistic, label), top to bottom
    ("PCA", "T2", "PCA T²"),
    ("PCA", "SPE", "PCA SPE"),
    ("DPCA", "T2", "Dynamic PCA T²"),
    ("DPCA", "SPE", "Dynamic PCA SPE"),
    ("Autoencoder", "RE", "Autoencoder"),
    ("DynamicAutoencoder", "RE", "Dynamic autoencoder"),
    ("OneClassSVM", "OCSVM", "One-class SVM"),
    ("LOF", "LOF", "LOF"),
    ("IsolationForest", "IF", "Isolation Forest"),
]
METHODS = [  # (key, label, colour, marker)
    ("in_sample", "In-sample", "#eb6834", "o"),
    ("own", "Own", "#2a78d6", "s"),
    ("split", "Split", "#1baf7a", "D"),
    ("cv", "CV", "#eda100", "^"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"


def main():
    df = pd.read_csv(ROOT / "results" / "limits.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    fig.patch.set_facecolor("white")

    offsets = [-0.24, -0.08, 0.08, 0.24]
    for (key, label, colour, marker), dy in zip(METHODS, offsets):
        ys, xs = [], []
        for i, (mon, stat, _) in enumerate(ROWS):
            row = df[(df.monitor == mon) & (df.statistic == stat) & (df.method == key)]
            xs.append(100 * row.FAR.iloc[0])
            ys.append(i + dy)
        ax.scatter(xs, ys, s=38, color=colour, marker=marker, label=label,
                   edgecolors="white", linewidths=0.8, zorder=3)

    ax.axvline(1, color=INK, lw=1.2, zorder=2)
    ax.text(1.06, -0.75, "1% target", color=INK, fontsize=8.5, va="center")

    ax.set_xscale("log")
    ax.set_xlim(0.4, 45)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 40]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_ylim(len(ROWS) - 0.5, -1.1)
    ax.set_yticks(range(len(ROWS)), [r[2] for r in ROWS])
    for i in range(len(ROWS)):
        ax.axhline(i + 0.5, color=GRID, lw=0.6, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=1)

    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    ax.set_xlabel("False alarm rate on 22 independent normal runs (log scale)", color=MUTED, fontsize=9)
    ax.legend(ncols=4, loc="lower left", bbox_to_anchor=(-0.01, 1.0),
              frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1.2)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.text(0.02, 0.965, "A 99% control limit should give 1% false alarms",
             color=INK, fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.915, "False alarm rate of each monitor, by how its control limit was set",
             color=MUTED, fontsize=9, va="top")
    fig.savefig(ROOT / "results" / "limits.png", facecolor="white")
    print("saved results/limits.png")


if __name__ == "__main__":
    main()
