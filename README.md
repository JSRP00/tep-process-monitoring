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

## Results

Fault detection rate (%) with 99% control limits. PCA uses 11 components; "PCA any" raises an alarm when either T² or SPE exceeds its limit. Isolation Forest uses 300 trees, with its limit set on held-out normal data.

| Fault | PCA SPE | PCA T² | PCA any | Isolation Forest |
|---:|---:|---:|---:|---:|
| 1 | 99.8 | 99.2 | 99.8 | 99.2 |
| 2 | 98.9 | 98.0 | 98.9 | 98.1 |
| 3 | 9.6 | 2.6 | 12.0 | 20.0 |
| 4 | 99.6 | 8.8 | 99.6 | 18.5 |
| 5 | 34.9 | 24.6 | 37.1 | 38.2 |
| 6 | 100.0 | 99.2 | 100.0 | 99.6 |
| 7 | 100.0 | 96.0 | 100.0 | 55.6 |
| 8 | 98.0 | 97.0 | 98.0 | 98.6 |
| 9 | 8.1 | 2.9 | 10.5 | 20.4 |
| 10 | 56.2 | 40.1 | 63.1 | 53.5 |
| 11 | 77.0 | 28.2 | 77.8 | 33.5 |
| 12 | 98.6 | 97.4 | 99.0 | 99.0 |
| 13 | 95.5 | 94.0 | 95.5 | 95.6 |
| 14 | 100.0 | 88.4 | 100.0 | 21.4 |
| 15 | 12.0 | 5.0 | 16.2 | 20.4 |
| 16 | 50.0 | 22.2 | 56.9 | 40.4 |
| 17 | 94.2 | 77.2 | 94.4 | 47.0 |
| 18 | 91.4 | 89.1 | 91.4 | 90.2 |
| 19 | 36.4 | 1.1 | 37.2 | 16.0 |
| 20 | 59.8 | 35.9 | 62.9 | 53.2 |
| 21 | 54.2 | 30.4 | 54.4 | 42.6 |
| **False alarm rate** | 7.1 | 1.7 | 8.8 | 11.0 |

**Reading the results**

- Isolation Forest does not outperform PCA. It matches it on faults with large, widespread deviations (1, 2, 6, 8, 12, 13, 18) but misses most of faults 4, 7, 11, 14 and 17, which PCA's SPE statistic catches through a change in the correlation structure rather than in the magnitude of individual variables.
- Its apparent gain on faults 3, 9 and 15 (about 20% vs 10–16%) is not real detection: it is close to its own false alarm rate on normal data (11%). These faults remain essentially undetectable for both methods.
- Methods should be compared at similar false alarm rates. With the limit estimated on a few hundred normal samples, the empirical 99% quantile is noisy, which partly explains the 11% FAR.

Full results, including detection delays: [`results/`](results).

## Usage

```bash
git clone https://github.com/JSRP00/tep-process-monitoring.git
cd tep-process-monitoring
pip install -e ".[test]"
python scripts/run_benchmark.py
```

Adding a model takes one line in `MONITORS` in `scripts/run_benchmark.py`. Any model with `fit` and `score_samples` (scikit-learn style) can be wrapped with `ScoreMonitor`, as done for Isolation Forest.

When comparing methods, check the false alarm rate as well as the detection rate: a method with a higher FAR will look better on FDR for that reason alone.

## Tests

```bash
pytest
```

## License

MIT. The TEP data are © The Board of Trustees of the University of Illinois (Braatz group) and are downloaded, not redistributed.
