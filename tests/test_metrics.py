import numpy as np
import pytest

from src.evaluation.metrics import (
    calibration_coverage,
    decile_lift,
    mae,
    mape,
    pearson_correlation,
    rmse,
)

Y_TRUE = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
Y_PRED = np.array([1.0, 2.0, 3.0, 4.0, 6.0])


def test_mae():
    assert mae(Y_TRUE, Y_PRED) == pytest.approx(0.2)


def test_rmse():
    assert rmse(Y_TRUE, Y_PRED) == pytest.approx(np.sqrt(0.2))


def test_mape():
    # only the last element is off by 1/5 -> mean 0.2/5 -> 4%
    assert mape(Y_TRUE, Y_PRED) == pytest.approx(4.0)


def test_mape_all_zero_true_returns_zero():
    assert mape([0, 0, 0], [1, 2, 3]) == 0.0


def test_pearson():
    # hand-computed: 12 / sqrt(10 * 14.8)
    assert pearson_correlation(Y_TRUE, Y_PRED) == pytest.approx(
        12.0 / np.sqrt(148.0), abs=1e-4
    )


def test_perfect_prediction_metrics():
    y = np.array([10.0, 20.0, 30.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert pearson_correlation(y, y) == pytest.approx(1.0)


def test_empty_inputs_return_nan():
    assert np.isnan(mae([], []))
    assert np.isnan(rmse([], []))
    assert np.isnan(mape([], []))
    assert np.isnan(pearson_correlation([], []))


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1, 2], [1, 2, 3])


def test_decile_lift_top_decile_beats_bottom():
    n = 100
    y_true = np.arange(1, n + 1, dtype=float)
    y_pred = y_true + np.random.RandomState(0).normal(0, 3, n)
    lift = decile_lift(y_true, y_pred)
    assert lift.iloc[0]["lift"] > lift.iloc[-1]["lift"]
    assert lift["n_customers"].sum() == n


def test_calibration_coverage():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.0, 0.0, 5.0, 0.0])
    upper = np.array([2.0, 5.0, 9.0, 10.0])
    # points 1,2,4 inside; point 3 below its lower bound -> 3/4
    assert calibration_coverage(y_true, lower, upper) == pytest.approx(0.75)
