"""Unit tests on synthetic data (no download needed)."""

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from tepmon import AutoencoderMonitor, PCAMonitor, ScoreMonitor, detection_delay, detection_rate, false_alarm_rate


@pytest.fixture
def normal_data():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(2000, 3))
    mixing = rng.normal(size=(3, 10))
    return latent @ mixing + 0.3 * rng.normal(size=(2000, 10))


def test_pca_false_alarm_rate_close_to_alpha(normal_data):
    monitor = PCAMonitor(n_components=3, alpha=0.01).fit(normal_data[:1000])
    far = monitor.alarms(normal_data[1000:]).mean()
    assert far < 0.05


def test_pca_variance_share_selects_components(normal_data):
    monitor = PCAMonitor(n_components=0.9).fit(normal_data)
    assert 1 <= monitor.n_components_ < normal_data.shape[1]


def test_pca_detects_mean_shift(normal_data):
    monitor = PCAMonitor(n_components=3).fit(normal_data[:1000])
    shifted = normal_data[1000:1200] + 3.0 * normal_data.std(axis=0)
    assert monitor.alarms(shifted).mean() > 0.9


def test_pca_rejects_invalid_components(normal_data):
    with pytest.raises(ValueError):
        PCAMonitor(n_components=10).fit(normal_data)


def test_score_monitor_wraps_sklearn_model(normal_data):
    monitor = ScoreMonitor(IsolationForest(random_state=0), name="IF").fit(normal_data[:1000])
    assert set(monitor.limits_) == {"IF"}
    shifted = normal_data[1000:1200] + 5.0 * normal_data.std(axis=0)
    assert monitor.alarms(shifted).mean() > monitor.alarms(normal_data[1000:]).mean()


def test_metrics():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    alarms = np.array([0, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert detection_rate(alarms, y) == pytest.approx(0.75)
    assert false_alarm_rate(alarms, y) == pytest.approx(1 / 3)
    assert detection_delay(alarms, y) == 1
    assert detection_delay(alarms, y, run=3) == 1
    assert np.isnan(detection_delay(np.zeros(7, dtype=bool), y))


def test_autoencoder_detects_mean_shift(normal_data):
    monitor = AutoencoderMonitor(hidden=(8, 3, 8), max_iter=500).fit(normal_data[:1000])
    assert set(monitor.limits_) == {"RE"}
    shifted = normal_data[1000:1200] + 3.0 * normal_data.std(axis=0)
    assert monitor.alarms(shifted).mean() > monitor.alarms(normal_data[1000:]).mean()
