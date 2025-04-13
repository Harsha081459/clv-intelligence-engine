"""
BG/NBD + Gamma-Gamma Probabilistic CLV Model.

Wraps the `lifetimes` library to provide a clean interface for fitting
probabilistic CLV models, generating predictions, and producing
diagnostic visualizations.

All monetary values are in INR (₹).
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import (
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix,
    plot_calibration_purchases_vs_holdout_purchases,
)

# ---------------------------------------------------------------------------
# Path setup & config import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    GBP_TO_INR,
    GROSS_MARGIN,
    PENALIZER_COEF,
    CLV_PREDICTION_MONTHS,
    MONTHLY_DISCOUNT_RATE,
    MODELS_DIR,
    CURRENCY_SYMBOL,
)

logger = logging.getLogger(__name__)


class ProbabilisticCLV:
    """
    BG/NBD + Gamma-Gamma Customer Lifetime Value model.

    This model estimates:
    - Expected number of future transactions (BG/NBD)
    - Probability that a customer is still "alive" (BG/NBD)
    - Expected average transaction value (Gamma-Gamma)
    - Customer Lifetime Value combining the above

    Parameters
    ----------
    penalizer_coef : float, optional
        L2 regularisation penalty for both BG/NBD and Gamma-Gamma fitters.
        Defaults to ``config.PENALIZER_COEF``.
    margin : float, optional
        Gross-margin multiplier applied when computing CLV.
        Defaults to ``config.GROSS_MARGIN``.
    """

    def __init__(
        self,
        penalizer_coef: float = PENALIZER_COEF,
        margin: float = GROSS_MARGIN,
    ) -> None:
        self.penalizer_coef = penalizer_coef
        self.margin = margin

        self.bgf = BetaGeoFitter(penalizer_coef=self.penalizer_coef)
        self.ggf = GammaGammaFitter(penalizer_coef=self.penalizer_coef)

        self._bgf_fitted = False
        self._ggf_fitted = False

        logger.info(
            "ProbabilisticCLV initialised  (penalizer=%.4f, margin=%.2f)",
            self.penalizer_coef,
            self.margin,
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, rfm_data: pd.DataFrame) -> "ProbabilisticCLV":
        """
        Fit the BG/NBD model on all customers and the Gamma-Gamma model
        on customers with ``frequency > 0``.

        Parameters
        ----------
        rfm_data : pd.DataFrame
            Must contain columns: ``frequency``, ``recency``, ``T``,
            ``monetary_value``.

        Returns
        -------
        self
        """
        required_cols = {"frequency", "recency", "T", "monetary_value"}
        missing = required_cols - set(rfm_data.columns)
        if missing:
            raise ValueError(f"rfm_data is missing columns: {missing}")

        # --- BG/NBD ---
        logger.info("Fitting BG/NBD on %d customers …", len(rfm_data))
        self.bgf.fit(
            rfm_data["frequency"],
            rfm_data["recency"],
            rfm_data["T"],
        )
        self._bgf_fitted = True
        logger.info(
            "BG/NBD fitted – params: %s",
            {k: round(v, 4) for k, v in self.bgf.summary.to_dict()["coef"].items()},
        )

        # --- Gamma-Gamma (requires frequency > 0) ---
        ggf_data = rfm_data[rfm_data["frequency"] > 0].copy()
        logger.info(
            "Fitting Gamma-Gamma on %d repeat customers …", len(ggf_data)
        )
        self.ggf.fit(
            ggf_data["frequency"],
            ggf_data["monetary_value"],
        )
        self._ggf_fitted = True
        logger.info(
            "Gamma-Gamma fitted – params: %s",
            {k: round(v, 4) for k, v in self.ggf.summary.to_dict()["coef"].items()},
        )

        return self

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def predict_purchases(
        self, rfm_data: pd.DataFrame, t: int = 365
    ) -> pd.Series:
        """
        Predict the expected number of purchases in the next *t* days.

        Parameters
        ----------
        rfm_data : pd.DataFrame
            Must contain ``frequency``, ``recency``, ``T``.
        t : int
            Prediction horizon in days.

        Returns
        -------
        pd.Series
            Expected purchase counts, indexed like *rfm_data*.
        """
        self._check_bgf()
        preds = self.bgf.conditional_expected_number_of_purchases_up_to_time(
            t,
            rfm_data["frequency"],
            rfm_data["recency"],
            rfm_data["T"],
        )
        preds = pd.Series(preds, index=rfm_data.index)
        logger.info(
            "Predicted purchases (t=%d days): mean=%.2f, median=%.2f",
            t,
            preds.mean(),
            preds.median(),
        )
        return preds

    def predict_alive_probability(self, rfm_data: pd.DataFrame) -> pd.Series:
        """
        Compute the probability that each customer is still "alive"
        (i.e. has not permanently churned).

        Returns
        -------
        pd.Series
            P(alive) values in [0, 1], indexed like *rfm_data*.
        """
        self._check_bgf()
        p_alive = self.bgf.conditional_probability_alive(
            rfm_data["frequency"],
            rfm_data["recency"],
            rfm_data["T"],
        )
        p_alive = pd.Series(p_alive, index=rfm_data.index)
        logger.info(
            "P(alive): mean=%.4f, min=%.4f, max=%.4f",
            p_alive.mean(),
            p_alive.min(),
            p_alive.max(),
        )
        return p_alive

    def predict_clv(
        self,
        rfm_data: pd.DataFrame,
        months: int = CLV_PREDICTION_MONTHS,
        discount_rate: float = MONTHLY_DISCOUNT_RATE,
    ) -> pd.Series:
        """
        Predict Customer Lifetime Value in INR using the Gamma-Gamma
        ``customer_lifetime_value`` method.

        Only customers with ``frequency > 0`` receive a CLV estimate;
        the rest are assigned 0.

        Parameters
        ----------
        rfm_data : pd.DataFrame
            Full RFM table.
        months : int
            Prediction horizon in months.
        discount_rate : float
            Monthly discount rate.

        Returns
        -------
        pd.Series
            CLV values in INR (₹).
        """
        self._check_bgf()
        self._check_ggf()

        # Filter to repeat customers
        mask = rfm_data["frequency"] > 0
        repeat = rfm_data.loc[mask]

        clv = self.ggf.customer_lifetime_value(
            self.bgf,
            repeat["frequency"],
            repeat["recency"],
            repeat["T"],
            repeat["monetary_value"],
            time=months,
            discount_rate=discount_rate,
            freq="D",  # RFM periods are in days
        )

        # Reindex to full customer base, fill non-repeat with 0
        clv_full = clv.reindex(rfm_data.index, fill_value=0.0)
        # Apply gross margin
        clv_full = clv_full * self.margin

        logger.info(
            "CLV (months=%d): mean=%s%.2f, total=%s%.2f",
            months,
            CURRENCY_SYMBOL,
            clv_full.mean(),
            CURRENCY_SYMBOL,
            clv_full.sum(),
        )
        return clv_full

    def predict_all(
        self,
        rfm_data: pd.DataFrame,
        months: int = CLV_PREDICTION_MONTHS,
    ) -> pd.DataFrame:
        """
        Generate a comprehensive predictions DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``predicted_purchases``, ``p_alive``,
            ``predicted_clv``, ``clv_inr``.
        """
        t_days = months * 30  # approximate

        preds = pd.DataFrame(index=rfm_data.index)
        preds["predicted_purchases"] = self.predict_purchases(rfm_data, t=t_days)
        preds["p_alive"] = self.predict_alive_probability(rfm_data)
        preds["predicted_clv"] = self.predict_clv(rfm_data, months=months)
        preds["clv_inr"] = preds["predicted_clv"]  # already in INR

        logger.info("predict_all complete – shape %s", preds.shape)
        return preds

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    def plot_frequency_recency_matrix(self) -> plt.Figure:
        """
        Heatmap of expected purchases by frequency & recency.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._check_bgf()
        plt.figure(figsize=(10, 8))
        plot_frequency_recency_matrix(self.bgf)
        plt.title("Expected Purchases – Frequency / Recency Matrix")
        plt.tight_layout()
        return plt.gcf()

    def plot_probability_alive_matrix(self) -> plt.Figure:
        """
        Heatmap of P(alive) by frequency & recency.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._check_bgf()
        plt.figure(figsize=(10, 8))
        plot_probability_alive_matrix(self.bgf)
        plt.title("Probability Alive – Frequency / Recency Matrix")
        plt.tight_layout()
        return plt.gcf()

    def plot_calibration(
        self,
        rfm_observation: pd.DataFrame,
        holdout_transactions: pd.DataFrame,
    ) -> plt.Figure:
        """
        Plot expected vs actual repeat transactions (calibration chart).

        Parameters
        ----------
        rfm_observation : pd.DataFrame
            RFM data from the observation period.
        holdout_transactions : pd.DataFrame
            Actual repeat-transaction counts in the holdout period.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._check_bgf()
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_calibration_purchases_vs_holdout_purchases(
            self.bgf,
            rfm_observation,
            holdout_transactions,
            ax=ax,
        )
        ax.set_title("BG/NBD Calibration – Expected vs Actual Purchases")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, models_dir: Path | str | None = None) -> Path:
        """
        Persist fitted model parameters to disk.

        Saves only the fitted parameters (not the full fitter objects)
        to avoid PicklingError from internal lambdas in lifetimes.

        Parameters
        ----------
        models_dir : Path or str, optional
            Directory to save into. Defaults to ``config.MODELS_DIR``.

        Returns
        -------
        Path
            Path to the saved file.
        """
        models_dir = Path(models_dir) if models_dir else MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)

        filepath = models_dir / "probabilistic_clv.pkl"
        payload = {
            "bgf_params": {k: float(v) for k, v in self.bgf.params_.items()},
            "ggf_params": {k: float(v) for k, v in self.ggf.params_.items()},
            "penalizer_coef": self.penalizer_coef,
            "margin": self.margin,
            "_bgf_fitted": self._bgf_fitted,
            "_ggf_fitted": self._ggf_fitted,
        }
        joblib.dump(payload, filepath)
        logger.info("ProbabilisticCLV saved to %s", filepath)
        return filepath

    @classmethod
    def load(cls, models_dir: Path | str | None = None) -> "ProbabilisticCLV":
        """
        Load a previously-saved ProbabilisticCLV model.

        Reconstructs the fitters and injects saved parameters.

        Parameters
        ----------
        models_dir : Path or str, optional
            Directory to load from. Defaults to ``config.MODELS_DIR``.

        Returns
        -------
        ProbabilisticCLV
        """
        models_dir = Path(models_dir) if models_dir else MODELS_DIR
        filepath = models_dir / "probabilistic_clv.pkl"

        payload = joblib.load(filepath)
        instance = cls(
            penalizer_coef=payload["penalizer_coef"],
            margin=payload["margin"],
        )
        # Reconstruct BG/NBD
        instance.bgf.params_ = payload["bgf_params"]
        instance._bgf_fitted = payload["_bgf_fitted"]
        # Reconstruct Gamma-Gamma
        instance.ggf.params_ = payload["ggf_params"]
        instance._ggf_fitted = payload["_ggf_fitted"]

        logger.info("ProbabilisticCLV loaded from %s", filepath)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_bgf(self) -> None:
        """Raise if BG/NBD has not been fitted."""
        if not self._bgf_fitted:
            raise RuntimeError(
                "BG/NBD model has not been fitted. Call .fit() first."
            )

    def _check_ggf(self) -> None:
        """Raise if Gamma-Gamma has not been fitted."""
        if not self._ggf_fitted:
            raise RuntimeError(
                "Gamma-Gamma model has not been fitted. Call .fit() first."
            )


# ======================================================================
# Standalone smoke-test
# ======================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
    )

    # --- Generate synthetic RFM data for quick validation ---
    np.random.seed(42)
    n = 500
    synthetic_rfm = pd.DataFrame(
        {
            "frequency": np.random.poisson(3, n),
            "recency": np.random.uniform(1, 365, n),
            "T": np.random.uniform(180, 730, n),
            "monetary_value": np.random.lognormal(6, 1, n) * GBP_TO_INR,
        }
    )
    # Ensure recency <= T
    synthetic_rfm["recency"] = synthetic_rfm[["recency", "T"]].min(axis=1)

    model = ProbabilisticCLV()
    model.fit(synthetic_rfm)

    preds = model.predict_all(synthetic_rfm)
    print("\n--- Sample Predictions ---")
    print(preds.head(10).to_string())
    print(f"\nMean CLV: {CURRENCY_SYMBOL}{preds['clv_inr'].mean():,.2f}")
    print(f"Total CLV: {CURRENCY_SYMBOL}{preds['clv_inr'].sum():,.2f}")

    # Save & reload
    saved_path = model.save()
    reloaded = ProbabilisticCLV.load()
    preds2 = reloaded.predict_all(synthetic_rfm)
    assert np.allclose(preds["predicted_clv"], preds2["predicted_clv"]), \
        "Reload parity check failed!"
    print("\n✓ Save / load round-trip passed.")
