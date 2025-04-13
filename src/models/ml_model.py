"""
Gradient-Boosting CLV Regression Model (LightGBM / XGBoost).

Supports Optuna-based Bayesian hyperparameter tuning with k-fold
cross-validation, SHAP-based feature-importance explanations, and
full model persistence.

All monetary values are in INR (₹).
"""

import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import optuna
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score

# ---------------------------------------------------------------------------
# Path setup & config import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    LGBM_DEFAULT_PARAMS,
    OPTUNA_N_TRIALS,
    OPTUNA_CV_FOLDS,
    RANDOM_STATE,
    MODELS_DIR,
    CURRENCY_SYMBOL,
)

logger = logging.getLogger(__name__)

# Silence Optuna's default logging (we log summaries ourselves)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class CLVBoostingModel:
    """
    Gradient-boosted-tree CLV regressor.

    Supports LightGBM and XGBoost back-ends with a unified API for
    tuning, training, prediction, and SHAP-based interpretation.

    Parameters
    ----------
    model_type : str
        ``'lightgbm'`` or ``'xgboost'``.
    """

    SUPPORTED_TYPES = {"lightgbm", "xgboost"}

    def __init__(self, model_type: str = "lightgbm") -> None:
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"model_type must be one of {self.SUPPORTED_TYPES}, "
                f"got '{model_type}'"
            )
        self.model_type = model_type
        self.model = None
        self.best_params: Dict[str, Any] = {}
        self.feature_names: list[str] = []
        self._shap_explainer = None

        logger.info("CLVBoostingModel initialised (type=%s)", self.model_type)

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------
    @staticmethod
    def prepare_features(
        customer_features: pd.DataFrame,
        probabilistic_predictions: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Merge RFM / behavioural features with BG/NBD outputs to create
        the feature matrix for the boosting model.

        Parameters
        ----------
        customer_features : pd.DataFrame
            Customer-level features.  Expected to contain an RFM core
            (``frequency``, ``recency``, ``T``, ``monetary_value``) plus
            any additional engineered columns.  If a column named
            ``holdout_revenue`` exists it is extracted as the target.
        probabilistic_predictions : pd.DataFrame
            Output of ``ProbabilisticCLV.predict_all``.  Expected columns:
            ``predicted_purchases``, ``p_alive``, ``predicted_clv``.

        Returns
        -------
        X : pd.DataFrame
            Merged feature matrix.
        y : pd.Series or None
            Target variable (``holdout_revenue``) if present; else *None*.
        """
        # Align on index
        merged = customer_features.join(
            probabilistic_predictions[["predicted_purchases", "p_alive", "predicted_clv"]],
            how="inner",
        )

        # Separate target if available
        y = None
        if "holdout_revenue" in merged.columns:
            y = merged.pop("holdout_revenue")

        # Drop non-numeric / identifier columns (keep only features)
        X = merged.select_dtypes(include=[np.number]).copy()

        logger.info(
            "Feature matrix ready: X.shape=%s, target present=%s",
            X.shape,
            y is not None,
        )
        return X, y

    # ------------------------------------------------------------------
    # Hyper-parameter tuning
    # ------------------------------------------------------------------
    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = OPTUNA_N_TRIALS,
    ) -> Dict[str, Any]:
        """
        Bayesian optimisation of hyper-parameters via Optuna.

        Uses 5-fold cross-validated negative MAE as the objective.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target values.
        n_trials : int
            Number of Optuna trials.

        Returns
        -------
        dict
            Best hyper-parameter dict.
        """
        logger.info(
            "Starting Optuna tuning (%d trials, %d-fold CV) …",
            n_trials,
            OPTUNA_CV_FOLDS,
        )

        def _objective(trial: optuna.Trial) -> float:
            if self.model_type == "lightgbm":
                params = self._suggest_lgbm_params(trial)
            else:
                params = self._suggest_xgb_params(trial)

            estimator = self._build_estimator(params)
            kf = KFold(
                n_splits=OPTUNA_CV_FOLDS,
                shuffle=True,
                random_state=RANDOM_STATE,
            )
            scores = cross_val_score(
                estimator, X, y, cv=kf, scoring="neg_mean_absolute_error"
            )
            return -scores.mean()  # minimise MAE

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        logger.info(
            "Optuna tuning complete – best MAE=%.4f, params=%s",
            study.best_value,
            self.best_params,
        )
        return self.best_params

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ) -> "CLVBoostingModel":
        """
        Train the boosting model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target values.
        params : dict, optional
            Hyper-parameters.  Falls back to ``self.best_params`` (from
            tuning) or library defaults.

        Returns
        -------
        self
        """
        params = params or self.best_params or {}
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        elif not self.feature_names:
            self.feature_names = [f"f{i}" for i in range(X.shape[1])]

        self.model = self._build_estimator(params)
        self.model.fit(X, y)
        self._shap_explainer = None  # invalidate cached explainer

        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        n_feats = X.shape[1] if hasattr(X, 'shape') else 0
        logger.info(
            "Model trained on %d samples x %d features", n_samples, n_feats
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate CLV predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same schema as training).

        Returns
        -------
        np.ndarray
            Predicted CLV values (INR).
        """
        self._check_fitted()
        preds = self.model.predict(X)
        logger.info(
            "Predictions: mean=%s%.2f, std=%.2f",
            CURRENCY_SYMBOL,
            np.mean(preds),
            np.std(preds),
        )
        return preds

    # ------------------------------------------------------------------
    # SHAP explanations
    # ------------------------------------------------------------------
    def compute_shap_values(self, X: pd.DataFrame) -> shap.Explanation:
        """
        Compute SHAP values for the trained model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        shap.Explanation
        """
        self._check_fitted()

        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model)

        shap_values = self._shap_explainer(X)
        logger.info("SHAP values computed for %d samples", len(X))
        return shap_values

    def plot_shap_waterfall(
        self,
        X: pd.DataFrame,
        customer_idx: int = 0,
        feature_names: Optional[list[str]] = None,
    ) -> plt.Figure:
        """
        SHAP waterfall plot for a single customer.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        customer_idx : int
            Row index of the customer to explain.
        feature_names : list[str], optional
            Override feature names for display.

        Returns
        -------
        matplotlib.figure.Figure
        """
        shap_values = self.compute_shap_values(X)
        if feature_names:
            shap_values.feature_names = feature_names

        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[customer_idx], show=False)
        plt.title(f"SHAP Waterfall – Customer index {customer_idx}")
        plt.tight_layout()
        return fig

    def plot_shap_summary(self, X: pd.DataFrame) -> plt.Figure:
        """
        SHAP beeswarm (summary) plot for all customers.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        matplotlib.figure.Figure
        """
        shap_values = self.compute_shap_values(X)

        fig = plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, show=False)
        plt.title("SHAP Feature Importance (Beeswarm)")
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, filepath: Path | str | None = None) -> Path:
        """
        Save the trained model and metadata.

        Parameters
        ----------
        filepath : Path or str, optional
            Destination file.  Defaults to
            ``config.MODELS_DIR / 'clv_boosting_{model_type}.pkl'``.

        Returns
        -------
        Path
            Path to the saved file.
        """
        self._check_fitted()
        if filepath is None:
            filepath = MODELS_DIR / f"clv_boosting_{self.model_type}.pkl"
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "model_type": self.model_type,
            "best_params": self.best_params,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, filepath)
        logger.info("CLVBoostingModel saved to %s", filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Path | str) -> "CLVBoostingModel":
        """
        Load a previously-saved CLVBoostingModel.

        Parameters
        ----------
        filepath : Path or str
            Path to the saved pickle.

        Returns
        -------
        CLVBoostingModel
        """
        filepath = Path(filepath)
        payload = joblib.load(filepath)

        instance = cls(model_type=payload["model_type"])
        instance.model = payload["model"]
        instance.best_params = payload["best_params"]
        instance.feature_names = payload["feature_names"]

        logger.info("CLVBoostingModel loaded from %s", filepath)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_estimator(self, params: Dict[str, Any]):
        """Instantiate a scikit-learn-compatible estimator."""
        merged = dict(LGBM_DEFAULT_PARAMS)  # start from defaults
        merged.update(params)

        if self.model_type == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMRegressor(**merged)
        else:
            import xgboost as xgb

            # Translate any LightGBM-specific keys to XGBoost equivalents
            xgb_params = {
                "objective": "reg:squarederror",
                "n_estimators": merged.get("n_estimators", 500),
                "learning_rate": merged.get("learning_rate", 0.05),
                "max_depth": max(merged.get("max_depth", 6), 1),
                "subsample": merged.get("subsample", 0.8),
                "colsample_bytree": merged.get("colsample_bytree", 0.8),
                "reg_alpha": merged.get("reg_alpha", 0.1),
                "reg_lambda": merged.get("reg_lambda", 0.1),
                "random_state": merged.get("random_state", RANDOM_STATE),
                "verbosity": 0,
            }
            return xgb.XGBRegressor(**xgb_params)

    @staticmethod
    def _suggest_lgbm_params(trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest LightGBM hyper-parameters for an Optuna trial."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.3, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    @staticmethod
    def _suggest_xgb_params(trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest XGBoost hyper-parameters for an Optuna trial."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.3, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }

    def _check_fitted(self) -> None:
        """Raise if model has not been trained."""
        if self.model is None:
            raise RuntimeError(
                "Model has not been trained. Call .train() first."
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
    n = 400

    # --- Synthetic feature matrix ---
    X = pd.DataFrame(
        {
            "frequency": np.random.poisson(3, n),
            "recency": np.random.uniform(1, 365, n),
            "T": np.random.uniform(180, 730, n),
            "monetary_value": np.random.lognormal(6, 1, n),
            "predicted_purchases": np.random.uniform(0, 10, n),
            "p_alive": np.random.uniform(0.1, 1.0, n),
            "predicted_clv": np.random.uniform(100, 50000, n),
        }
    )
    y = (
        X["monetary_value"] * X["predicted_purchases"] * np.random.uniform(0.5, 1.5, n)
    )

    # --- Quick train (no Optuna for speed) ---
    model = CLVBoostingModel(model_type="lightgbm")
    model.train(X, y)
    preds = model.predict(X)
    print(f"\nTrain MAE: {CURRENCY_SYMBOL}{np.mean(np.abs(y - preds)):,.2f}")

    # --- Save / load round-trip ---
    path = model.save()
    reloaded = CLVBoostingModel.load(path)
    preds2 = reloaded.predict(X)
    assert np.allclose(preds, preds2), "Reload parity check failed!"
    print("✓ Save / load round-trip passed.")
