"""
Evaluation Metrics for CLV Models.

Provides regression accuracy metrics, ranking / lift metrics, uplift
evaluation, and prediction-interval diagnostics.

All functions are designed to be robust against edge-cases including
zeros in the denominator, NaN values, and empty inputs.
"""

import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup & config import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import CURRENCY_SYMBOL

logger = logging.getLogger(__name__)


# ======================================================================
# Regression accuracy metrics
# ======================================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted values.

    Returns
    -------
    float
        MAE value, or ``np.nan`` if inputs are empty.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return np.nan
    return float(np.nanmean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted values.

    Returns
    -------
    float
        RMSE value, or ``np.nan`` if inputs are empty.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return np.nan
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (%).

    Zero-valued ground-truth entries are excluded to avoid division by
    zero.  Returns 0.0 if all entries are zero, and ``np.nan`` for
    empty inputs.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted values.

    Returns
    -------
    float
        MAPE in percent (e.g. 12.5 means 12.5 %).
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return np.nan

    mask = y_true != 0
    if mask.sum() == 0:
        logger.warning("MAPE: all y_true values are zero – returning 0.0")
        return 0.0

    return float(
        np.nanmean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


# ======================================================================
# Correlation
# ======================================================================

def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pearson correlation coefficient between true and predicted values.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted values.

    Returns
    -------
    float
        Pearson *r* in [-1, 1], or ``np.nan`` when computation is
        impossible (e.g. constant inputs, empty arrays).
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if len(y_true) < 2:
        return np.nan

    # Remove NaN pairs
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 2:
        return np.nan

    r, _ = stats.pearsonr(y_true[valid], y_pred[valid])
    return float(r)


# ======================================================================
# Ranking / lift metrics
# ======================================================================

def decile_lift(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Compute lift in each predicted-CLV decile.

    Customers are ranked by ``y_pred`` into 10 equal-sized deciles
    (decile 1 = highest predicted CLV).  For each decile we report
    the mean actual CLV, overall mean, and lift = decile_mean / overall_mean.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted values.

    Returns
    -------
    pd.DataFrame
        Columns: ``decile``, ``n_customers``, ``mean_actual_clv``,
        ``mean_predicted_clv``, ``lift``.
        Returns an empty DataFrame if inputs are empty.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return pd.DataFrame(
            columns=["decile", "n_customers", "mean_actual_clv",
                      "mean_predicted_clv", "lift"]
        )

    df = pd.DataFrame({"actual": y_true, "predicted": y_pred})

    # Rank into deciles (1 = highest predicted CLV)
    df["decile"] = pd.qcut(
        df["predicted"], q=10, labels=False, duplicates="drop"
    )
    # Invert so that decile 1 = top predicted
    df["decile"] = df["decile"].max() - df["decile"] + 1

    overall_mean = df["actual"].mean()
    if overall_mean == 0:
        overall_mean = np.finfo(float).eps  # avoid division by zero

    summary = (
        df.groupby("decile")
        .agg(
            n_customers=("actual", "size"),
            mean_actual_clv=("actual", "mean"),
            mean_predicted_clv=("predicted", "mean"),
        )
        .reset_index()
        .sort_values("decile")
    )
    summary["lift"] = summary["mean_actual_clv"] / overall_mean

    logger.info(
        "Decile lift – top-decile lift=%.2fx, bottom-decile lift=%.2fx",
        summary.iloc[0]["lift"],
        summary.iloc[-1]["lift"],
    )
    return summary


# ======================================================================
# Uplift evaluation
# ======================================================================

def qini_coefficient(
    y_true: np.ndarray,
    uplift_scores: np.ndarray,
    treatment: np.ndarray,
) -> float:
    """
    Qini coefficient for uplift model evaluation.

    The Qini coefficient is the area between the uplift model's Qini
    curve and the random-targeting diagonal, normalised by the area
    under the ideal Qini curve.

    Parameters
    ----------
    y_true : array-like
        Binary outcome (1 = converted / purchased).
    uplift_scores : array-like
        Predicted uplift scores (higher ⇒ more incremental impact).
    treatment : array-like
        Binary treatment indicator (1 = treated, 0 = control).

    Returns
    -------
    float
        Qini coefficient in [0, 1], or ``np.nan`` if computation fails.
    """
    y_true = np.asarray(y_true, dtype=float)
    uplift_scores = np.asarray(uplift_scores, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    if len(y_true) == 0 or len(y_true) != len(uplift_scores) != len(treatment):
        return np.nan

    n = len(y_true)
    n_t = treatment.sum()
    n_c = n - n_t

    if n_t == 0 or n_c == 0:
        logger.warning("Qini: no treatment or control observations – returning NaN")
        return np.nan

    # Sort by descending uplift score
    order = np.argsort(-uplift_scores)
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    # Build cumulative Qini curve
    cum_t_outcomes = np.cumsum(y_sorted * t_sorted)
    cum_c_outcomes = np.cumsum(y_sorted * (1 - t_sorted))
    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1 - t_sorted)

    # Avoid divide-by-zero
    cum_t_safe = np.where(cum_t == 0, np.finfo(float).eps, cum_t)
    cum_c_safe = np.where(cum_c == 0, np.finfo(float).eps, cum_c)

    qini_curve = cum_t_outcomes / (n_t) - cum_c_outcomes / (n_c)

    # Area under the Qini curve (trapezoidal rule, normalised by n)
    qini_area = np.trapz(qini_curve, dx=1.0 / n)

    # Random targeting baseline
    random_area = qini_curve[-1] / 2.0

    # Qini coefficient
    if random_area == 0:
        return 0.0

    coef = float((qini_area - random_area) / abs(random_area)) if random_area != 0 else 0.0
    logger.info("Qini coefficient: %.4f", coef)
    return coef


# ======================================================================
# Prediction-interval diagnostics
# ======================================================================

def calibration_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Empirical coverage of prediction intervals.

    Parameters
    ----------
    y_true : array-like
        Observed values.
    y_lower, y_upper : array-like
        Lower and upper bounds of the prediction interval.

    Returns
    -------
    float
        Fraction of observations falling within [y_lower, y_upper],
        or ``np.nan`` for empty inputs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)

    if len(y_true) == 0:
        return np.nan

    # Remove NaN entries
    valid = ~(np.isnan(y_true) | np.isnan(y_lower) | np.isnan(y_upper))
    if valid.sum() == 0:
        return np.nan

    covered = (y_true[valid] >= y_lower[valid]) & (y_true[valid] <= y_upper[valid])
    coverage = float(covered.mean())

    logger.info(
        "Calibration coverage: %.2f%% (%d / %d samples)",
        coverage * 100,
        covered.sum(),
        valid.sum(),
    )
    return coverage


# ======================================================================
# Model comparison formatting
# ======================================================================

def model_comparison_table(
    results_dict: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Format a dictionary of model results into a pretty comparison table.

    Parameters
    ----------
    results_dict : dict
        ``{model_name: {metric_name: value, ...}, ...}``

        Example::

            {
                "BG/NBD": {"MAE": 1200, "RMSE": 1800, "MAPE": 35.2},
                "LightGBM": {"MAE": 900, "RMSE": 1400, "MAPE": 28.1},
            }

    Returns
    -------
    pd.DataFrame
        Comparison table with models as rows and metrics as columns.
        An empty DataFrame is returned if *results_dict* is empty.
    """
    if not results_dict:
        logger.warning("model_comparison_table: empty results_dict")
        return pd.DataFrame()

    df = pd.DataFrame(results_dict).T
    df.index.name = "Model"

    # Round numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].round(4)

    # Highlight best (lowest) value per column
    logger.info("Model comparison table:\n%s", df.to_string())
    return df


# ======================================================================
# Internal helpers
# ======================================================================

def _validate_inputs(
    y_true: Any, y_pred: Any
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert inputs to float64 arrays and validate shapes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]

    Raises
    ------
    ValueError
        If array lengths differ.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_pred has {len(y_pred)}."
        )
    return y_true, y_pred


# ======================================================================
# Standalone smoke-test
# ======================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
    )

    np.random.seed(42)
    n = 500

    y_true = np.random.lognormal(7, 1.5, n)
    y_pred = y_true * np.random.uniform(0.7, 1.3, n)

    print("=== Regression Metrics ===")
    print(f"  MAE  : {CURRENCY_SYMBOL}{mae(y_true, y_pred):,.2f}")
    print(f"  RMSE : {CURRENCY_SYMBOL}{rmse(y_true, y_pred):,.2f}")
    print(f"  MAPE : {mape(y_true, y_pred):.2f}%")
    print(f"  Pearson r: {pearson_correlation(y_true, y_pred):.4f}")

    print("\n=== Decile Lift ===")
    lift_df = decile_lift(y_true, y_pred)
    print(lift_df.to_string(index=False))

    print("\n=== Calibration Coverage ===")
    y_lower = y_pred - np.abs(y_pred) * 0.3
    y_upper = y_pred + np.abs(y_pred) * 0.3
    cov = calibration_coverage(y_true, y_lower, y_upper)
    print(f"  Coverage: {cov:.2%}")

    print("\n=== Qini Coefficient ===")
    treatment = np.random.binomial(1, 0.3, n)
    uplift = np.random.randn(n)
    qini = qini_coefficient(y_true > np.median(y_true), uplift, treatment)
    print(f"  Qini: {qini:.4f}")

    print("\n=== Model Comparison ===")
    comp = model_comparison_table(
        {
            "Model A": {"MAE": 1200.0, "RMSE": 1800.0, "MAPE": 35.2},
            "Model B": {"MAE": 900.0, "RMSE": 1400.0, "MAPE": 28.1},
        }
    )
    print(comp.to_string())

    # --- Edge-case tests ---
    print("\n=== Edge Cases ===")
    assert np.isnan(mae([], []))
    assert np.isnan(rmse([], []))
    assert np.isnan(mape([], []))
    assert np.isnan(pearson_correlation([], []))
    assert mape([0, 0, 0], [1, 2, 3]) == 0.0
    assert np.isnan(calibration_coverage([], [], []))
    print("  ✓ All edge-case tests passed.")
