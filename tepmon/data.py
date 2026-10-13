"""Download and load the Tennessee Eastman Process (TEP) benchmark data.

Data: the classic Braatz dataset (Chiang, Russell & Braatz, 2000), 52 variables
(41 measured + 11 manipulated), sampled every 3 minutes.

- ``d00.dat``: 500 normal-operation samples used for training.
- ``dXX_te.dat`` (XX = 00..21): 960 test samples. In the fault files the fault
  is introduced after 8 hours, so samples 0..159 are normal and 160..959 faulty.

The files are downloaded from a public mirror pinned to a fixed commit, so
results are reproducible.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

MIRROR = "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz"
COMMIT = "0643318858ce7c87292bd59820b459e12a13d009"
N_VARIABLES = 52
FAULT_ONSET = 160  # index of the first faulty sample in the test files
FAULTS = list(range(1, 22))


def data_dir() -> Path:
    """Folder where the raw files are stored (override with TEPMON_DATA)."""
    return Path(os.environ.get("TEPMON_DATA", Path.home() / ".cache" / "tepmon"))


def _fetch(name: str) -> Path:
    path = data_dir() / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{MIRROR}/{COMMIT}/{name}"
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def download_all() -> Path:
    """Download every training and test file. Returns the data folder."""
    _fetch("d00.dat")
    for fault in [0, *FAULTS]:
        _fetch(f"d{fault:02d}_te.dat")
    return data_dir()


def load_train() -> np.ndarray:
    """Normal-operation training data, shape (500, 52)."""
    x = np.loadtxt(_fetch("d00.dat"))
    # d00.dat is stored transposed (52 x 500) in the original distribution.
    if x.shape[0] == N_VARIABLES:
        x = x.T
    return x


def load_test(fault: int) -> tuple[np.ndarray, np.ndarray]:
    """Test data for a fault (0 = normal operation).

    Returns ``(X, y)`` with X of shape (960, 52) and y a 0/1 label per sample
    (1 = the fault is active). For fault 0 every label is 0.
    """
    if fault not in [0, *FAULTS]:
        raise ValueError(f"fault must be between 0 and 21, got {fault}")
    x = np.loadtxt(_fetch(f"d{fault:02d}_te.dat"))
    y = np.zeros(len(x), dtype=int)
    if fault != 0:
        y[FAULT_ONSET:] = 1
    return x, y
