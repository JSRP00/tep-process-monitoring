# tep-process-monitoring

[![tests](https://github.com/JSRP00/tep-process-monitoring/actions/workflows/tests.yml/badge.svg)](https://github.com/JSRP00/tep-process-monitoring/actions/workflows/tests.yml)

Benchmark of classical multivariate statistical process control (MSPC) and machine-learning methods for fault detection on the Tennessee Eastman Process (TEP).

All methods are trained on normal operation only and evaluated on the 21 standard TEP faults with the same metrics, so they can be compared on equal terms.

## Data

The classic Braatz dataset: 52 process variables sampled every 3 minutes, 500 normal training samples and, for each fault, 960 test samples with the fault introduced after sample 160. The files are downloaded automatically on first use from a public mirror pinned to a fixed commit.

## Methods

| Monitor | Statistics | Control limit |
| --- | --- | --- |
| `PCAMonitor` | Hotelling's T², SPE (Q) | F distribution (T²), Jackson–Mudholkar (SPE) |
| `ScoreMonitor` | Anomaly score of any scikit-learn style model | Empirical quantile on held-out normal data |

## Metrics

- **FDR** (fault detection rate): share of faulty samples that raise an alarm.
- **FAR** (false alarm rate): share of samples that raise an alarm on the normal test set.
- **Detection delay**: samples from fault onset to the first alarm (×3 for minutes).

## Baseline: PCA (11 components, 99% limits)

Fault detection rate (%). The "any" column raises an alarm when either statistic exceeds its limit.

| Fault | SPE | T² | any |
|---:|---:|---:|---:|
| 1 | 99.8 | 99.2 | 99.8 |
| 2 | 98.9 | 98.0 | 98.9 |
| 3 | 9.6 | 2.6 | 12.0 |
| 4 | 99.6 | 8.8 | 99.6 |
| 5 | 34.9 | 24.6 | 37.1 |
| 6 | 100.0 | 99.2 | 100.0 |
| 7 | 100.0 | 96.0 | 100.0 |
| 8 | 98.0 | 97.0 | 98.0 |
| 9 | 8.1 | 2.9 | 10.5 |
| 10 | 56.2 | 40.1 | 63.1 |
| 11 | 77.0 | 28.2 | 77.8 |
| 12 | 98.6 | 97.4 | 99.0 |
| 13 | 95.5 | 94.0 | 95.5 |
| 14 | 100.0 | 88.4 | 100.0 |
| 15 | 12.0 | 5.0 | 16.2 |
| 16 | 50.0 | 22.2 | 56.9 |
| 17 | 94.2 | 77.2 | 94.4 |
| 18 | 91.4 | 89.1 | 91.4 |
| 19 | 36.4 | 1.1 | 37.2 |
| 20 | 59.8 | 35.9 | 62.9 |
| 21 | 54.2 | 30.4 | 54.4 |

False alarm rate on the normal test set: 1.7% (T²), 7.1% (SPE), 8.8% (any). Faults 3, 9 and 15 are known to be very hard to detect with any method.

Full results, including detection delays: [`results/`](results).

## Usage

```bash
git clone https://github.com/JSRP00/tep-process-monitoring.git
cd tep-process-monitoring
pip install -e ".[test]"
python scripts/run_benchmark.py
```

Adding a model takes one line in `scripts/run_benchmark.py`:

```python
from sklearn.ensemble import IsolationForest
from tepmon import ScoreMonitor

MONITORS["IsolationForest"] = ScoreMonitor(IsolationForest(random_state=0), name="IF")
```

When comparing methods, check the false alarm rate as well as the detection rate: a method with a higher FAR will look better on FDR for that reason alone.

## Tests

```bash
pytest
```

## License

MIT. The TEP data are © The Board of Trustees of the University of Illinois (Braatz group) and are downloaded, not redistributed.
