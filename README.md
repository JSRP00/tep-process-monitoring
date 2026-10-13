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
| `AutoencoderMonitor` | Reconstruction error of a 32–11–32 autoencoder | Empirical quantile on held-out normal data |

## Metrics

- **FDR** (fault detection rate): share of faulty samples that raise an alarm.
- **FAR** (false alarm rate): share of samples that raise an alarm on the normal test set.
- **Detection delay**: samples from fault onset to the first alarm (×3 for minutes).

## Results

Fault detection rate (%) with 99% control limits. PCA uses 11 components; "PCA any" raises an alarm when either T² or SPE exceeds its limit. Isolation Forest uses 300 trees. The autoencoder has layers 32–11–32, so its bottleneck matches the 11 PCA components; its statistic is the reconstruction error. For both ML models the limit is set on held-out normal data. All settings were fixed before looking at the test results.

| Fault | PCA SPE | PCA T² | PCA any | Isolation Forest | Autoencoder |
|---:|---:|---:|---:|---:|---:|
| 1 | 99.8 | 99.2 | 99.8 | 99.2 | 99.8 |
| 2 | 98.9 | 98.0 | 98.9 | 98.1 | 98.8 |
| 3 | 9.6 | 2.6 | 12.0 | 20.0 | 6.8 |
| 4 | 99.6 | 8.8 | 99.6 | 18.5 | 99.5 |
| 5 | 34.9 | 24.6 | 37.1 | 38.2 | 33.6 |
| 6 | 100.0 | 99.2 | 100.0 | 99.6 | 100.0 |
| 7 | 100.0 | 96.0 | 100.0 | 55.6 | 100.0 |
| 8 | 98.0 | 97.0 | 98.0 | 98.6 | 98.0 |
| 9 | 8.1 | 2.9 | 10.5 | 20.4 | 5.1 |
| 10 | 56.2 | 40.1 | 63.1 | 53.5 | 58.0 |
| 11 | 77.0 | 28.2 | 77.8 | 33.5 | 72.2 |
| 12 | 98.6 | 97.4 | 99.0 | 99.0 | 99.0 |
| 13 | 95.5 | 94.0 | 95.5 | 95.6 | 95.5 |
| 14 | 100.0 | 88.4 | 100.0 | 21.4 | 100.0 |
| 15 | 12.0 | 5.0 | 16.2 | 20.4 | 10.6 |
| 16 | 50.0 | 22.2 | 56.9 | 40.4 | 49.6 |
| 17 | 94.2 | 77.2 | 94.4 | 47.0 | 93.2 |
| 18 | 91.4 | 89.1 | 91.4 | 90.2 | 90.2 |
| 19 | 36.4 | 1.1 | 37.2 | 16.0 | 27.0 |
| 20 | 59.8 | 35.9 | 62.9 | 53.2 | 60.9 |
| 21 | 54.2 | 30.4 | 54.4 | 42.6 | 47.8 |
| **Mean detection rate** | 70.2 | 54.2 | 71.7 | 55.3 | 68.8 |
| **False alarm rate** | 7.1 | 1.7 | 8.8 | 11.0 | 4.3 |

**Reading the results**

- **The autoencoder performs on par with PCA's SPE, not better.** Both measure how far a sample departs from the structure learnt on normal data, and they detect the same faults at similar rates. Across 5 random seeds the autoencoder's false alarm rate ranges from 3.9% to 7.3% and its mean detection rate from 68.7% to 71.2%, against 7.1% and 70.2% for PCA's SPE. On this dataset the non-linearity adds little over linear PCA.
- **Isolation Forest is clearly worse.** It misses most of faults 4, 7, 11, 14 and 17, which change the correlation between variables rather than the size of individual variables; scoring samples one by one does not capture that structure.
- **Faults 3, 9 and 15 remain undetectable for every method.** Detection rates there are close to each method's own false alarm rate on normal data, so they reflect chance alarms rather than detection. Isolation Forest's ~20% on these faults comes with an 11% false alarm rate.
- **Compare at similar false alarm rates.** A method with a higher FAR looks better on FDR for that reason alone. With limits estimated from a few hundred normal samples, the empirical 99% quantile is noisy.
- **None of these methods uses the time order of the data.** TEP measurements are autocorrelated; dynamic methods (dynamic PCA, LSTM autoencoders) are the natural next step.

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
