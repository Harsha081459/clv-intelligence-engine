import numpy as np
import pytest

from src.monitoring.drift import DriftMonitor


@pytest.fixture()
def monitor():
    return DriftMonitor(n_bins=10)


def test_identical_distributions_near_zero(monitor):
    rng = np.random.RandomState(42)
    x = rng.normal(100.0, 15.0, 2000)
    assert monitor.compute_psi(x, x.copy()) == pytest.approx(0.0, abs=1e-6)


def test_constant_feature_returns_zero(monitor):
    x = np.full(500, 7.0)
    assert monitor.compute_psi(x, x) == 0.0


def test_shifted_distribution_positive(monitor):
    rng = np.random.RandomState(0)
    baseline = rng.normal(100.0, 10.0, 3000)
    shifted = rng.normal(130.0, 10.0, 3000)
    assert monitor.compute_psi(baseline, shifted) > 0.0


def test_psi_monotonic_in_shift(monitor):
    rng = np.random.RandomState(1)
    baseline = rng.normal(100.0, 10.0, 3000)
    small_shift = rng.normal(105.0, 10.0, 3000)
    large_shift = rng.normal(150.0, 10.0, 3000)
    psi_small = monitor.compute_psi(baseline, small_shift)
    psi_large = monitor.compute_psi(baseline, large_shift)
    assert psi_large > psi_small
