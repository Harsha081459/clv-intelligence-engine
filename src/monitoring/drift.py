"""
Population Stability Index (PSI) Drift Monitor.

Detects distribution drift between a baseline (training) period and
subsequent periods using PSI.  Supports per-feature PSI computation,
quarterly time-series tracking, heatmap visualisation, and automated
alerting.

PSI interpretation (standard thresholds):
    * < 0.10  → **Stable** — no significant shift
    * 0.10–0.20 → **Moderate** — investigate
    * > 0.20  → **Significant** — likely drift, retrain recommended
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    PSI_THRESHOLD_LOW,
    PSI_THRESHOLD_HIGH,
)

logger = logging.getLogger(__name__)

# Small constant to avoid log(0)
_EPS = 1e-8


class DriftMonitor:
    """Population Stability Index (PSI) computation and drift monitoring.

    Attributes:
        n_bins: Number of quantile-based bins used for PSI computation.
    """

    def __init__(self, n_bins: int = 10) -> None:
        """Configure the drift monitor.

        Args:
            n_bins: Default number of equal-frequency bins for PSI.
        """
        self.n_bins = n_bins
        logger.info("DriftMonitor initialised with n_bins=%d.", n_bins)

    # ------------------------------------------------------------------
    # Core PSI
    # ------------------------------------------------------------------
    def compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: Optional[int] = None,
    ) -> float:
        """Compute the Population Stability Index between two distributions.

        PSI = Σ (actual_pct − expected_pct) × ln(actual_pct / expected_pct)

        Bins are derived from the *expected* distribution's quantiles so
        that each expected bin has roughly equal mass.

        Edge cases:
            * Empty bin in actual or expected → smoothed with ``_EPS``.
            * Constant feature → PSI = 0.0.

        Args:
            expected: Baseline (reference) distribution values.
            actual: Current distribution values to compare.
            n_bins: Number of bins (overrides instance default).

        Returns:
            PSI value (float ≥ 0).
        """
        n_bins = n_bins or self.n_bins
        expected = np.asarray(expected, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)

        # Handle constant features
        if expected.std() == 0 and actual.std() == 0:
            return 0.0

        # Derive bin edges from the expected distribution
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.unique(np.percentile(expected, quantiles))

        # If all values collapse into a single bin, return 0
        if len(bin_edges) < 2:
            return 0.0

        # Ensure extreme edges capture all values
        bin_edges[0] = min(expected.min(), actual.min()) - 1
        bin_edges[-1] = max(expected.max(), actual.max()) + 1

        # Count observations per bin
        expected_counts = np.histogram(expected, bins=bin_edges)[0].astype(np.float64)
        actual_counts = np.histogram(actual, bins=bin_edges)[0].astype(np.float64)

        # Convert to proportions with smoothing
        expected_pct = (expected_counts + _EPS) / (expected_counts.sum() + _EPS * len(expected_counts))
        actual_pct = (actual_counts + _EPS) / (actual_counts.sum() + _EPS * len(actual_counts))

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    # ------------------------------------------------------------------
    # Feature-level PSI
    # ------------------------------------------------------------------
    def compute_feature_psi(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: List[str],
    ) -> Dict[str, float]:
        """Compute PSI for each feature between baseline and current data.

        Args:
            baseline_df: Reference-period DataFrame.
            current_df: Current-period DataFrame.
            features: Column names to evaluate.

        Returns:
            Dict mapping feature name → PSI value.
        """
        psi_values: Dict[str, float] = {}
        for feat in features:
            if feat not in baseline_df.columns or feat not in current_df.columns:
                logger.warning("Feature '%s' missing from one of the DataFrames.", feat)
                continue

            baseline_vals = baseline_df[feat].dropna().values
            current_vals = current_df[feat].dropna().values

            if len(baseline_vals) == 0 or len(current_vals) == 0:
                logger.warning("Feature '%s' has no valid values in one period.", feat)
                psi_values[feat] = np.nan
                continue

            psi_values[feat] = self.compute_psi(baseline_vals, current_vals)

        logger.info("Feature PSI: %s", {k: f"{v:.4f}" for k, v in psi_values.items()})
        return psi_values

    # ------------------------------------------------------------------
    # Drift classification
    # ------------------------------------------------------------------
    def detect_drift(
        self,
        psi_values: Dict[str, float],
    ) -> pd.DataFrame:
        """Classify each feature's drift status based on PSI thresholds.

        Classification:
            * ``stable``     — PSI < 0.10
            * ``moderate``   — 0.10 ≤ PSI < 0.20
            * ``significant``— PSI ≥ 0.20

        Args:
            psi_values: Dict mapping feature name → PSI value.

        Returns:
            DataFrame with columns: ``feature``, ``psi``, ``status``,
            ``alert``.
        """
        records: List[Dict] = []
        for feat, psi in psi_values.items():
            if np.isnan(psi):
                status = "unknown"
                alert = False
            elif psi < PSI_THRESHOLD_LOW:
                status = "stable"
                alert = False
            elif psi < PSI_THRESHOLD_HIGH:
                status = "moderate"
                alert = True
            else:
                status = "significant"
                alert = True

            records.append({
                "feature": feat,
                "psi": round(psi, 4) if not np.isnan(psi) else np.nan,
                "status": status,
                "alert": alert,
            })

        df = pd.DataFrame(records)
        n_alerts = df["alert"].sum()
        logger.info(
            "Drift detection: %d features evaluated, %d alerts triggered.",
            len(df),
            n_alerts,
        )
        return df

    # ------------------------------------------------------------------
    # Quarterly PSI timeline
    # ------------------------------------------------------------------
    def compute_quarterly_psi(
        self,
        df: pd.DataFrame,
        date_col: str,
        features: List[str],
    ) -> pd.DataFrame:
        """Compute PSI between consecutive quarters.

        The first quarter serves as the baseline; PSI is computed for
        each subsequent quarter relative to the immediately preceding
        quarter.

        Args:
            df: Transaction-level or customer-level DataFrame with a
                datetime column.
            date_col: Name of the datetime column.
            features: Feature columns to monitor.

        Returns:
            DataFrame with columns: ``quarter``, ``feature``, ``psi``.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df["_quarter"] = df[date_col].dt.to_period("Q").astype(str)

        quarters = sorted(df["_quarter"].unique())
        if len(quarters) < 2:
            logger.warning("Fewer than 2 quarters — cannot compute quarterly PSI.")
            return pd.DataFrame(columns=["quarter", "feature", "psi"])

        records: List[Dict] = []
        for i in range(1, len(quarters)):
            baseline = df[df["_quarter"] == quarters[i - 1]]
            current = df[df["_quarter"] == quarters[i]]

            psi_vals = self.compute_feature_psi(baseline, current, features)
            for feat, psi in psi_vals.items():
                records.append({
                    "quarter": quarters[i],
                    "baseline_quarter": quarters[i - 1],
                    "feature": feat,
                    "psi": psi,
                })

        result = pd.DataFrame(records)
        logger.info(
            "Quarterly PSI computed: %d quarter-pairs × %d features.",
            len(quarters) - 1,
            len(features),
        )
        return result

    # ------------------------------------------------------------------
    # Visualisation — Heatmap
    # ------------------------------------------------------------------
    def plot_psi_heatmap(
        self,
        quarterly_psi_df: pd.DataFrame,
    ) -> go.Figure:
        """Plotly heatmap of PSI over quarters × features.

        Args:
            quarterly_psi_df: Output of ``compute_quarterly_psi``.

        Returns:
            Plotly Figure.
        """
        pivot = quarterly_psi_df.pivot_table(
            index="feature",
            columns="quarter",
            values="psi",
            aggfunc="mean",
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[
                    [0.0, "#2ecc71"],        # stable — green
                    [PSI_THRESHOLD_LOW / 0.4, "#f1c40f"],  # moderate — yellow
                    [PSI_THRESHOLD_HIGH / 0.4, "#e74c3c"],  # significant — red
                    [1.0, "#8e44ad"],         # extreme — purple
                ],
                colorbar=dict(title="PSI"),
                text=np.round(pivot.values, 3),
                texttemplate="%{text}",
            )
        )
        fig.update_layout(
            title="Feature Drift Heatmap (PSI by Quarter)",
            xaxis_title="Quarter",
            yaxis_title="Feature",
            template="plotly_white",
            width=900,
            height=max(400, 50 * len(pivot)),
        )
        logger.info("PSI heatmap plot created.")
        return fig

    # ------------------------------------------------------------------
    # Visualisation — Distribution comparison
    # ------------------------------------------------------------------
    def plot_distribution_comparison(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        feature_name: str,
    ) -> go.Figure:
        """Overlaid distribution plot of baseline vs current values.

        Args:
            baseline: Reference-period feature values.
            current: Current-period feature values.
            feature_name: Human-readable feature name (for labels).

        Returns:
            Plotly Figure.
        """
        psi = self.compute_psi(baseline, current)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=baseline,
            name="Baseline",
            opacity=0.55,
            marker_color="#636EFA",
            histnorm="probability density",
        ))
        fig.add_trace(go.Histogram(
            x=current,
            name="Current",
            opacity=0.55,
            marker_color="#EF553B",
            histnorm="probability density",
        ))
        fig.update_layout(
            title=f"Distribution Comparison — {feature_name}  (PSI = {psi:.4f})",
            xaxis_title=feature_name,
            yaxis_title="Density",
            barmode="overlay",
            template="plotly_white",
            width=800,
            height=450,
        )
        logger.info(
            "Distribution comparison plot for '%s' (PSI=%.4f).",
            feature_name,
            psi,
        )
        return fig

    # ------------------------------------------------------------------
    # Alert generation
    # ------------------------------------------------------------------
    def generate_alerts(
        self,
        psi_values: Dict[str, float],
    ) -> List[str]:
        """Generate alert messages for drifted features.

        An alert is raised when PSI exceeds ``PSI_THRESHOLD_LOW`` (0.10).

        Args:
            psi_values: Dict mapping feature name → PSI.

        Returns:
            List of human-readable alert strings.
        """
        alerts: List[str] = []
        for feat, psi in psi_values.items():
            if np.isnan(psi):
                continue
            if psi >= PSI_THRESHOLD_HIGH:
                alerts.append(
                    f"🔴 SIGNIFICANT DRIFT on '{feat}': PSI = {psi:.4f} "
                    f"(threshold {PSI_THRESHOLD_HIGH}). Retraining recommended."
                )
            elif psi >= PSI_THRESHOLD_LOW:
                alerts.append(
                    f"🟡 MODERATE DRIFT on '{feat}': PSI = {psi:.4f} "
                    f"(threshold {PSI_THRESHOLD_LOW}). Investigate."
                )

        if not alerts:
            logger.info("No drift alerts — all features stable.")
        else:
            for msg in alerts:
                logger.warning(msg)

        return alerts


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info("=== DriftMonitor — smoke test ===")

    np.random.seed(42)
    n = 2000

    # Simulate baseline and drifted distributions
    baseline_monetary = np.random.exponential(3000, n)
    current_monetary = np.random.exponential(3500, n)  # shifted

    baseline_freq = np.random.poisson(5, n).astype(float)
    current_freq = np.random.poisson(5, n).astype(float)  # stable

    monitor = DriftMonitor(n_bins=10)

    # Single PSI
    psi_monetary = monitor.compute_psi(baseline_monetary, current_monetary)
    psi_freq = monitor.compute_psi(baseline_freq, current_freq)
    print(f"PSI monetary: {psi_monetary:.4f}")
    print(f"PSI frequency: {psi_freq:.4f}")

    # Feature PSI
    df_base = pd.DataFrame({"monetary": baseline_monetary, "frequency": baseline_freq})
    df_curr = pd.DataFrame({"monetary": current_monetary, "frequency": current_freq})
    feature_psi = monitor.compute_feature_psi(df_base, df_curr, ["monetary", "frequency"])

    # Detect drift
    drift_df = monitor.detect_drift(feature_psi)
    print("\nDrift detection:\n", drift_df)

    # Alerts
    alerts = monitor.generate_alerts(feature_psi)
    for a in alerts:
        print(a)

    # Quarterly PSI (synthetic time series)
    dates = pd.date_range("2010-01-01", periods=n * 2, freq="D")[:n * 2]
    df_ts = pd.DataFrame({
        "date": dates,
        "monetary": np.concatenate([baseline_monetary, current_monetary]),
        "frequency": np.concatenate([baseline_freq, current_freq]),
    })
    quarterly_psi = monitor.compute_quarterly_psi(df_ts, "date", ["monetary", "frequency"])
    print("\nQuarterly PSI:\n", quarterly_psi)

    # Plots
    fig1 = monitor.plot_distribution_comparison(baseline_monetary, current_monetary, "Monetary Value")
    fig1.show()

    if not quarterly_psi.empty:
        fig2 = monitor.plot_psi_heatmap(quarterly_psi)
        fig2.show()

    logger.info("=== Smoke test passed ===")
