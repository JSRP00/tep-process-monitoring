## False alarm rate on 22 normal runs (%), target 1%

| row                   |   own |   in_sample |   split |   cv |
|:----------------------|------:|------------:|--------:|-----:|
| Autoencoder RE        |   2.9 |        10.7 |     3.2 |  0.8 |
| DPCA SPE              |  15.2 |        16.6 |     1.9 |  0.6 |
| DPCA T2               |   0.6 |         1.5 |     2.7 |  4.9 |
| DynamicAutoencoder RE |   4.5 |        17.3 |     1.2 |  1.2 |
| IsolationForest IF    |   8.2 |         8   |     1.6 |  5.4 |
| LOF LOF               |   7.2 |         7   |     2.6 |  4   |
| OneClassSVM OCSVM     |   7.7 |        32.1 |     4.2 |  2.8 |
| PCA SPE               |   5.8 |        11.2 |     3.8 |  0.6 |
| PCA T2                |   1.2 |         3   |     2.2 |  2.6 |

## False alarm rate in the worst normal run (%)

| row                   |   own |   in_sample |   split |   cv |
|:----------------------|------:|------------:|--------:|-----:|
| Autoencoder RE        |   8.8 |        23.1 |    16.9 |  2.5 |
| DPCA SPE              |  22.5 |        23.8 |     8.8 |  1.9 |
| DPCA T2               |   6.2 |        15.6 |    27.5 | 34.4 |
| DynamicAutoencoder RE |  18.8 |        35   |    13.1 |  5.6 |
| IsolationForest IF    |  43.8 |        41.9 |    16.2 | 30   |
| LOF LOF               |  33.1 |        31.9 |    19.4 | 21.2 |
| OneClassSVM OCSVM     |  45   |        68.8 |    31.9 | 23.8 |
| PCA SPE               |   9.4 |        16.9 |    10.6 |  1.2 |
| PCA T2                |   9.4 |        21.9 |    20.6 | 19.4 |

## Mean detection rate over the 21 faults (%)

| row                   |   own |   in_sample |   split |   cv |
|:----------------------|------:|------------:|--------:|-----:|
| Autoencoder RE        |  68.8 |        75.4 |    69.4 | 65.6 |
| DPCA SPE              |  78.1 |        78.8 |    68.1 | 64.9 |
| DPCA T2               |  51.3 |        53.7 |    56.8 | 59.8 |
| DynamicAutoencoder RE |  72.3 |        81.2 |    67.6 | 68.5 |
| IsolationForest IF    |  55.3 |        53.7 |    41.2 | 49.3 |
| LOF LOF               |  72.8 |        72.7 |    67.4 | 69.7 |
| OneClassSVM OCSVM     |  71.9 |        86.1 |    66.8 | 65.8 |
| PCA SPE               |  70.2 |        74.1 |    68.8 | 63.4 |
| PCA T2                |  54.2 |        57.2 |    54.4 | 56.8 |

