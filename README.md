# tep-process-monitoring

[![tests](https://github.com/JSRP00/tep-process-monitoring/actions/workflows/tests.yml/badge.svg)](https://github.com/JSRP00/tep-process-monitoring/actions/workflows/tests.yml)

Benchmark of classical multivariate statistical process control (MSPC) and machine-learning methods for fault detection on the Tennessee Eastman Process (TEP): PCA, dynamic PCA, autoencoders, one-class SVM, LOF and Isolation Forest.

All methods are trained on normal operation only and evaluated on the 21 standard TEP faults with the same metrics, so they can be compared on equal terms. A second experiment checks whether their control limits give the false alarm rate they promise.

## Data

The classic Braatz dataset: 52 process variables sampled every 3 minutes, 500 normal training samples and, for each fault, 960 test samples with the fault introduced after sample 160. The files are downloaded automatically on first use from a public mirror pinned to a fixed commit.

## Methods

| Monitor | Type | Statistic |
| --- | --- | --- |
| PCA | Linear, static | Hotelling's T² and SPE, with F-distribution and Jackson–Mudholkar limits |
| Dynamic PCA | Linear, uses the 2 previous samples | Same as PCA, on time-lagged data (Ku et al., 1995) |
| Autoencoder | Non-linear, static | Reconstruction error of a 32–11–32 network |
| Dynamic autoencoder | Non-linear, uses the 2 previous samples | Reconstruction error of a 64–20–64 network on time-lagged data |
| One-class SVM | Non-linear boundary, static | Distance to the learnt boundary |
| LOF | Local density, static | Local outlier factor |
| Isolation Forest | Tree ensemble, static | Isolation-based anomaly score |

All settings were fixed before looking at the test results. Autoencoder bottlenecks match the number of PCA components (11 static, 20 dynamic, the same 54% of variance). ML models set their control limit as the 99% quantile of their score on held-out normal training data.

## Metrics

Each method ends up with a different false alarm rate under its own control limit, so raw detection rates cannot be compared directly. The benchmark therefore reports three views:

- **AUC**: how well the statistic separates normal from faulty samples, independent of any threshold.
- **Detection at 1% FAR**: detection rate when every method's threshold is set to give exactly 1% false alarms on the normal test set. This is an evaluation-only threshold that puts all methods on equal terms.
- **Detection and false alarms with the method's own limit**: what each method would actually do if deployed as configured.

Detection delays are measured from fault onset to the first alarm with each method's own limit (TEP samples every 3 minutes).

## Results

Averages over the 21 faults (%), ordered by detection at 1% FAR:

| Method | Mean AUC | Detection at 1% FAR | Detection with own limit | False alarms with own limit | Median delay (min) |
|---|---:|---:|---:|---:|---:|
| Dynamic autoencoder | 87.6 | 65.1 | 72.3 | 6.2 | 33 |
| Dynamic PCA SPE | 87.0 | 64.8 | 78.1 | 19.3 | 15 |
| Autoencoder | 86.9 | 64.0 | 68.8 | 4.3 | 30 |
| PCA SPE | 86.3 | 63.3 | 70.2 | 7.1 | 9 |
| LOF | 86.4 | 59.7 | 72.8 | 11.8 | 15 |
| One-class SVM | 86.1 | 57.5 | 71.9 | 11.2 | 15 |
| PCA T² | 81.5 | 52.3 | 54.2 | 1.7 | 48 |
| Isolation Forest | 79.2 | 38.1 | 55.3 | 11.0 | 21 |

Detection rate (%) per fault at 1% false alarms:

| Fault | PCA T² | PCA SPE | Dynamic PCA SPE | Autoencoder | Dynamic autoencoder | One-class SVM | LOF | Isolation Forest |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 99 | 100 | 100 | 100 | 100 | 100 | 100 | 80 |
| 2 | 98 | 99 | 99 | 99 | 98 | 98 | 98 | 96 |
| 3 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 2 |
| 4 | 6 | 97 | 100 | 95 | 100 | 27 | 59 | 1 |
| 5 | 23 | 26 | 26 | 27 | 26 | 23 | 23 | 22 |
| 6 | 99 | 100 | 100 | 100 | 100 | 100 | 100 | 99 |
| 7 | 94 | 100 | 100 | 100 | 100 | 100 | 100 | 40 |
| 8 | 97 | 98 | 98 | 98 | 97 | 97 | 98 | 96 |
| 9 | 1 | 2 | 1 | 1 | 1 | 0 | 0 | 1 |
| 10 | 36 | 37 | 37 | 47 | 52 | 44 | 43 | 26 |
| 11 | 23 | 65 | 79 | 63 | 75 | 46 | 52 | 5 |
| 12 | 97 | 98 | 98 | 99 | 99 | 99 | 99 | 96 |
| 13 | 94 | 96 | 95 | 95 | 95 | 94 | 95 | 92 |
| 14 | 86 | 100 | 100 | 100 | 100 | 100 | 100 | 1 |
| 15 | 2 | 4 | 2 | 3 | 4 | 4 | 5 | 3 |
| 16 | 19 | 26 | 29 | 34 | 39 | 24 | 25 | 13 |
| 17 | 75 | 90 | 94 | 89 | 91 | 85 | 87 | 12 |
| 18 | 89 | 90 | 90 | 90 | 89 | 89 | 89 | 88 |
| 19 | 1 | 14 | 22 | 10 | 6 | 0 | 1 | 0 |
| 20 | 32 | 46 | 50 | 50 | 52 | 41 | 43 | 12 |
| 21 | 27 | 44 | 42 | 43 | 41 | 36 | 38 | 13 |

**Reading the results**

- **At equal false alarm rate, the best methods are within about 2 points of each other.** The dynamic autoencoder (65.1%), dynamic PCA (64.8%), autoencoder (64.0%) and PCA's SPE (63.3%) are practically tied: across random seeds the autoencoder alone varies by about ±1.3 points. Non-linear models add little over linear PCA on this dataset.
- **Using time order helps on specific faults, not across the board.** Dynamic models improve faults 10, 11, 16 and 19, which develop slowly or affect dynamics, but not the rest.
- **Own control limits can be badly miscalibrated.** Dynamic PCA looks best with its own limits (78.1% detection) only because it raises 19.3% false alarms. This is why methods must be compared at the same false alarm rate; the next section looks at the cause.
- **Generic anomaly detectors fall behind.** One-class SVM and LOF have similar AUC but detect less at 1% false alarms, and miss much of fault 4. Isolation Forest is clearly the weakest: scoring samples one by one misses faults that change the correlation between variables (4, 11, 14, 17).
- **Faults 3, 9 and 15 remain undetectable** for every method (≤ 5% at 1% false alarms).
- **Statistical methods react faster.** PCA's SPE has the shortest median delay (9 minutes), against about 30 minutes for the autoencoders.

Full results per fault and metric: [`results/summary.md`](results/summary.md).

## Control limits

A 99% control limit promises 1% false alarms on new normal data. This experiment keeps each model fixed and changes only how its limit is set:

| Method | How the limit is set |
| --- | --- |
| Own | The monitor's default: theoretical limits for PCA, a random 30% hold-out for ML models |
| In-sample | 99% quantile of the statistic on the training data |
| Split | Model trained on the first 70% of the normal run; conformal quantile on the last 30% |
| CV | 5 contiguous folds, each scored by a model trained on the others; conformal quantile of these out-of-fold values |

False alarms are measured on 22 independent normal runs (4,320 samples): the normal test file and the fault-free first 160 samples of each fault file.

False alarm rate (%), target 1%:

| Monitor | Own | In-sample | Split | CV |
|---|---:|---:|---:|---:|
| PCA T² | 1.2 | 3.0 | 2.2 | 2.6 |
| PCA SPE | 5.8 | 11.2 | 3.8 | 0.6 |
| Dynamic PCA T² | 0.6 | 1.5 | 2.7 | 4.9 |
| Dynamic PCA SPE | 15.2 | 16.6 | 1.9 | 0.6 |
| Autoencoder | 2.9 | 10.7 | 3.2 | 0.8 |
| Dynamic autoencoder | 4.5 | 17.3 | 1.2 | 1.2 |
| One-class SVM | 7.7 | 32.1 | 4.2 | 2.8 |
| LOF | 7.2 | 7.0 | 2.6 | 4.0 |
| Isolation Forest | 8.2 | 8.0 | 1.6 | 5.4 |

**Reading the results**

- **Limits set on the training data are optimistic.** In-sample limits give 1.5–32% false alarms instead of 1%, and above 7% for most monitors. Models fit their training data more closely than new data: on average, SPE on new normal data is 25–28% higher than on the training data, for both PCA and dynamic PCA.
- **The theoretical T² limit holds; the theoretical SPE limit does not.** T² stays close to 1% (1.2% and 0.6%). The SPE limit is computed from the training residuals, so it inherits the same optimism: 5.8% for PCA and 15.2% for dynamic PCA.
- **Calibrating on normal data the model has not seen helps, but is not exact.** Split and CV give 0.6–5.4%. With only 25 hours of normal data, the result depends on where the data are cut: moving the split from 20% to 40% changes PCA SPE from 5.0% to 1.9%. A 1% limit from 150 calibration samples rests on one or two observations.
- **Part of the apparent detection advantage is false alarms.** With split limits, dynamic PCA SPE drops from 78.1% to 68.1% detection, and its false alarms on the 22 normal runs from 15.2% to 1.9%.
- **Normal runs differ.** In the worst of the 22 normal runs, false alarms exceed 10% for most combinations of monitor and limit, so a limit set on a single normal run cannot anticipate every new run.

In practice: report the false alarm rate on independent normal data together with the detection rate, and set SPE and ML limits on data the model was not trained on.

Full tables, including detection rates and the worst run: [`results/limits.md`](results/limits.md).

## Usage

```bash
git clone https://github.com/JSRP00/tep-process-monitoring.git
cd tep-process-monitoring
pip install -e ".[test]"
python scripts/run_benchmark.py   # method comparison
python scripts/run_limits.py      # control limit comparison
```

Adding a model takes one line in `MONITORS` in `scripts/run_benchmark.py`. Any model with `fit` and `score_samples` (scikit-learn style) can be wrapped with `ScoreMonitor`, as done for Isolation Forest.

When comparing methods, check the false alarm rate as well as the detection rate: a method with a higher FAR will look better on FDR for that reason alone.

## Tests

```bash
pytest
```

## License

MIT. The TEP data are © The Board of Trustees of the University of Illinois (Braatz group) and are downloaded, not redistributed.
