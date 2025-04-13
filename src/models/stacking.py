"""
Stacking Ensemble for CLV Prediction.

Combines BG/NBD probabilistic predictions and LightGBM gradient-boosted
predictions via a Ridge-regression meta-learner.  Out-of-fold (OOF)
predictions are used during fitting to prevent data leakage.

All monetary values are in INR (₹).
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# Path setup & config import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    OPTUNA_CV_FOLDS,
    CURRENCY_SYMBOL,
)

logger = logging.getLogger(__name__)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (safe for zeros)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class StackedCLVModel:
    """
    Two-layer stacking ensemble for CLV prediction.

    **Level-0 models** (base learners):
    - BG/NBD + Gamma-Gamma probabilistic model
    - LightGBM (or XGBoost) gradient-boosted model

    **Level-1 model** (meta-learner):
    - Ridge regression trained on out-of-fold predictions from the
      two base learners.

    Parameters
    ----------
    alpha : float
        Regularisation strength for Ridge meta-learner.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.meta_learner = Ridge(alpha=self.alpha)
        self._fitted = False

        logger.info("StackedCLVModel initialised (alpha=%.4f)", self.alpha)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        bgf_predictions: np.ndarray | pd.Series,
        lgbm_predictions: np.ndarray | pd.Series,
        cv_folds: int = OPTUNA_CV_FOLDS,
    ) -> "StackedCLVModel":
        """
        Fit the meta-learner using out-of-fold (OOF) predictions.

        To avoid leakage the training data is split into *cv_folds*
        folds.  For each fold the OOF predictions from both base models
        are collected and concatenated.  The meta-learner is then trained
        on the full set of OOF prediction pairs.

        .. note::

           The base models themselves are **not** re-trained here.
           ``bgf_predictions`` and ``lgbm_predictions`` should already
           be computed (e.g. via cross-validated loops in the training
           pipeline) before calling this method.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (used only for indexing / alignment).
        y : array-like
            Ground-truth CLV values.
        bgf_predictions : array-like
            Predictions from the BG/NBD model (same length as *y*).
        lgbm_predictions : array-like
            Predictions from the LightGBM model (same length as *y*).
        cv_folds : int
            Number of folds for OOF prediction collection.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=float)
        bgf_predictions = np.asarray(bgf_predictions, dtype=float)
        lgbm_predictions = np.asarray(lgbm_predictions, dtype=float)

        if not (len(y) == len(bgf_predictions) == len(lgbm_predictions)):
            raise ValueError(
                "y, bgf_predictions, and lgbm_predictions must have the same length."
            )

        n = len(y)
        oof_stack = np.zeros((n, 2), dtype=float)
        oof_y = np.zeros(n, dtype=float)

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            # Collect OOF predictions for the validation fold
            oof_stack[val_idx, 0] = bgf_predictions[val_idx]
            oof_stack[val_idx, 1] = lgbm_predictions[val_idx]
            oof_y[val_idx] = y[val_idx]

            logger.debug(
                "Fold %d: train=%d, val=%d",
                fold_idx + 1,
                len(train_idx),
                len(val_idx),
            )

        # Train meta-learner on the complete set of OOF predictions
        self.meta_learner.fit(oof_stack, oof_y)
        self._fitted = True

        # Quick in-sample diagnostic
        oof_pred = self.meta_learner.predict(oof_stack)
        oof_mae = mean_absolute_error(oof_y, oof_pred)
        logger.info(
            "Meta-learner fitted on %d OOF samples – OOF MAE=%s%.2f",
            n,
            CURRENCY_SYMBOL,
            oof_mae,
        )
        logger.info(
            "Meta-learner coefficients: bgf=%.4f, lgbm=%.4f, intercept=%.4f",
            self.meta_learner.coef_[0],
            self.meta_learner.coef_[1],
            self.meta_learner.intercept_,
        )

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        bgf_pred: np.ndarray | pd.Series,
        lgbm_pred: np.ndarray | pd.Series,
    ) -> np.ndarray:
        """
        Stack base-learner predictions and produce final CLV forecast.

        Parameters
        ----------
        bgf_pred : array-like
            BG/NBD model predictions.
        lgbm_pred : array-like
            LightGBM model predictions.

        Returns
        -------
        np.ndarray
            Stacked CLV predictions (INR).
        """
        self._check_fitted()

        stacked = np.column_stack(
            [np.asarray(bgf_pred, dtype=float), np.asarray(lgbm_pred, dtype=float)]
        )
        preds = self.meta_learner.predict(stacked)
        logger.info(
            "Stacked prediction: mean=%s%.2f, std=%.2f",
            CURRENCY_SYMBOL,
            np.mean(preds),
            np.std(preds),
        )
        return preds

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred_bgnbd: np.ndarray | pd.Series,
        y_pred_lgbm: np.ndarray | pd.Series,
        y_pred_stacked: np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """
        Compare MAE, RMSE, and MAPE for all three model variants.

        Parameters
        ----------
        y_true : array-like
            Ground-truth CLV values.
        y_pred_bgnbd : array-like
            BG/NBD predictions.
        y_pred_lgbm : array-like
            LightGBM predictions.
        y_pred_stacked : array-like
            Stacked ensemble predictions.

        Returns
        -------
        pd.DataFrame
            Comparison table with models as rows and metrics as columns.
        """
        y_true = np.asarray(y_true, dtype=float)

        models = {
            "BG/NBD + Gamma-Gamma": np.asarray(y_pred_bgnbd, dtype=float),
            "LightGBM": np.asarray(y_pred_lgbm, dtype=float),
            "Stacked Ensemble": np.asarray(y_pred_stacked, dtype=float),
        }

        rows = []
        for name, y_pred in models.items():
            mae = mean_absolute_error(y_true, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mape = _mape(y_true, y_pred)
            rows.append(
                {
                    "Model": name,
                    f"MAE ({CURRENCY_SYMBOL})": round(mae, 2),
                    f"RMSE ({CURRENCY_SYMBOL})": round(rmse, 2),
                    "MAPE (%)": round(mape, 2),
                }
            )

        comparison = pd.DataFrame(rows).set_index("Model")
        logger.info("Model comparison:\n%s", comparison.to_string())
        return comparison

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, models_dir: Path | str | None = None) -> Path:
        """
        Persist the meta-learner to disk.

        Parameters
        ----------
        models_dir : Path or str, optional
            Directory to save into.  Defaults to ``config.MODELS_DIR``.

        Returns
        -------
        Path
            Path to the saved file.
        """
        self._check_fitted()
        models_dir = Path(models_dir) if models_dir else MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)

        filepath = models_dir / "stacked_clv_meta.pkl"
        payload = {
            "meta_learner": self.meta_learner,
            "alpha": self.alpha,
        }
        joblib.dump(payload, filepath)
        logger.info("StackedCLVModel saved to %s", filepath)
        return filepath

    @classmethod
    def load(cls, models_dir: Path | str | None = None) -> "StackedCLVModel":
        """
        Load a previously-saved StackedCLVModel.

        Parameters
        ----------
        models_dir : Path or str, optional
            Directory to load from.  Defaults to ``config.MODELS_DIR``.

        Returns
        -------
        StackedCLVModel
        """
        models_dir = Path(models_dir) if models_dir else MODELS_DIR
        filepath = models_dir / "stacked_clv_meta.pkl"

        payload = joblib.load(filepath)
        instance = cls(alpha=payload["alpha"])
        instance.meta_learner = payload["meta_learner"]
        instance._fitted = True

        logger.info("StackedCLVModel loaded from %s", filepath)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        """Raise if meta-learner has not been fitted."""
        if not self._fitted:
            raise RuntimeError(
                "Meta-learner has not been fitted. Call .fit() first."
            )


# ======================================================================
# Standalone smoke-test
# ======================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
    )

    np.random.seed(RANDOM_STATE)
    n = 300

    # --- Synthetic data ---
    y_true = np.random.lognormal(8, 1.5, n)
    bgf_pred = y_true * np.random.uniform(0.6, 1.4, n)
    lgbm_pred = y_true * np.random.uniform(0.7, 1.3, n)
    X_dummy = pd.DataFrame({"idx": np.arange(n)})

    # --- Fit ---
    stack = StackedCLVModel()
    stack.fit(X_dummy, y_true, bgf_pred, lgbm_pred)

    # --- Predict ---
    stacked_pred = stack.predict(bgf_pred, lgbm_pred)

    # --- Evaluate ---
    comp = stack.evaluate(y_true, bgf_pred, lgbm_pred, stacked_pred)
    print("\n--- Model Comparison ---")
    print(comp.to_string())

    # --- Save / load round-trip ---
    saved_path = stack.save()
    reloaded = StackedCLVModel.load()
    stacked_pred2 = reloaded.predict(bgf_pred, lgbm_pred)
    assert np.allclose(stacked_pred, stacked_pred2), "Reload parity check failed!"
    print("\n✓ Save / load round-trip passed.")
