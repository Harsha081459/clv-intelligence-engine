"""
Conformal Prediction Module
============================
Provides calibrated prediction intervals for CLV estimates
using MAPIE's conformal prediction framework.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    CONFORMAL_ALPHA, CONFORMAL_COVERAGE,
    MODELS_DIR, CURRENCY_SYMBOL
)

logger = logging.getLogger(__name__)


class ConformalCLVPredictor:
    """
    Wraps a CLV prediction model with conformal prediction intervals.
    
    Uses MAPIE's SplitConformalRegressor (v1.0+ API) to provide
    distribution-free prediction intervals with guaranteed coverage.
    
    The key insight: unlike Bayesian credible intervals, conformal
    intervals have provable finite-sample coverage guarantees.
    """
    
    def __init__(self, base_estimator, alpha: float = CONFORMAL_ALPHA):
        """
        Parameters
        ----------
        base_estimator : sklearn-compatible estimator
            The base CLV prediction model (e.g., LightGBM, stacked model).
        alpha : float
            Significance level. alpha=0.1 → 90% coverage interval.
        """
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.coverage = 1 - alpha
        self.conformal_model = None
        self._is_fitted = False
    
    def fit_and_conformalize(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_calib: np.ndarray, 
        y_calib: np.ndarray
    ):
        """
        Fit the base model and calibrate conformal scores.
        
        MAPIE v1.4.0 with prefit=True (default):
          1. Fit the base estimator ourselves.
          2. Wrap it in SplitConformalRegressor (prefit=True).
          3. Call .conformalize() on the calibration set.
        
        Parameters
        ----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training targets.
        X_calib : array-like
            Calibration features (held out from training).
        y_calib : array-like
            Calibration targets.
        """
        # Step 1: Fit the base estimator on training data
        logger.info("Fitting base estimator on %d training samples...", len(X_train))
        self.base_estimator.fit(X_train, y_train)
        
        try:
            # MAPIE v1.0+ API
            from mapie.regression import SplitConformalRegressor
            
            logger.info("Using MAPIE v1.4+ API (SplitConformalRegressor, prefit=True)")
            self.conformal_model = SplitConformalRegressor(
                estimator=self.base_estimator,
                prefit=True,
                confidence_level=self.coverage,
            )
            # No .fit() — estimator is already fitted (prefit=True)
            self.conformal_model.conformalize(X_calib, y_calib)
            self._api_version = "v1"
            
        except ImportError:
            # Fall back to MAPIE v0.x API
            from mapie.regression import MapieRegressor
            
            logger.info("Using MAPIE v0.x API (MapieRegressor)")
            self.conformal_model = MapieRegressor(
                estimator=self.base_estimator,
                method="plus",
                cv="prefit"
            )
            self.conformal_model.fit(X_calib, y_calib)
            self._api_version = "v0"
        
        self._is_fitted = True
        logger.info(f"Conformal model fitted (coverage={self.coverage:.0%})")
        
        # Validate on calibration set
        _, y_pis = self.predict(X_calib)
        coverage = self._compute_coverage(y_calib, y_pis[:, 0], y_pis[:, 1])
        logger.info(f"Calibration set coverage: {coverage:.1%} (target: {self.coverage:.0%})")
    
    def predict(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict CLV with conformal intervals.
        
        Parameters
        ----------
        X : array-like
            Features for prediction.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (y_pred, y_intervals) where y_intervals has shape (n, 2)
            with columns [lower_bound, upper_bound].
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit_and_conformalize first.")
        
        if self._api_version == "v1":
            # v1.4+: predict() for point, predict_interval() for intervals
            y_pred = self.conformal_model.predict(X)
            _, y_pis = self.conformal_model.predict_interval(X)
            # y_pis shape: (n_samples, 2, n_confidence_levels)
            if y_pis.ndim == 3:
                intervals = y_pis[:, :, 0]
            else:
                intervals = y_pis
        else:
            # v0.x: predict returns (y_pred, y_pis)
            y_pred, y_pis = self.conformal_model.predict(X, alpha=self.alpha)
            if y_pis.ndim == 3:
                intervals = y_pis[:, :, 0]
            else:
                intervals = y_pis
        
        # Ensure lower bound is non-negative (CLV can't be negative)
        intervals[:, 0] = np.maximum(intervals[:, 0], 0)
        
        return y_pred, intervals
    
    def predict_with_labels(
        self, X: np.ndarray, customer_ids: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Predict CLV with intervals and return a formatted DataFrame.
        
        Parameters
        ----------
        X : array-like
            Features.
        customer_ids : array-like, optional
            Customer IDs for the index.
            
        Returns
        -------
        pd.DataFrame
            Columns: predicted_clv, clv_lower, clv_upper, interval_width
        """
        y_pred, intervals = self.predict(X)
        
        result = pd.DataFrame({
            'predicted_clv': y_pred,
            'clv_lower': intervals[:, 0],
            'clv_upper': intervals[:, 1],
            'interval_width': intervals[:, 1] - intervals[:, 0],
        })
        
        if customer_ids is not None:
            result.index = customer_ids
            result.index.name = 'CustomerID'
        
        return result
    
    def evaluate_coverage(
        self, y_true: np.ndarray, X: np.ndarray
    ) -> dict:
        """
        Evaluate the empirical coverage on a test set.
        
        Parameters
        ----------
        y_true : array-like
            True CLV values.
        X : array-like
            Features.
            
        Returns
        -------
        dict
            Coverage metrics: empirical_coverage, target_coverage,
            avg_interval_width, median_interval_width
        """
        y_pred, intervals = self.predict(X)
        
        coverage = self._compute_coverage(y_true, intervals[:, 0], intervals[:, 1])
        widths = intervals[:, 1] - intervals[:, 0]
        
        metrics = {
            'empirical_coverage': coverage,
            'target_coverage': self.coverage,
            'coverage_gap': abs(coverage - self.coverage),
            'avg_interval_width': widths.mean(),
            'median_interval_width': np.median(widths),
            'avg_relative_width': (widths / (np.abs(y_pred) + 1e-6)).mean(),
        }
        
        logger.info(f"Coverage evaluation:")
        logger.info(f"  Empirical: {coverage:.1%} (target: {self.coverage:.0%})")
        logger.info(f"  Avg interval width: {CURRENCY_SYMBOL}{widths.mean():,.0f}")
        logger.info(f"  Median interval width: {CURRENCY_SYMBOL}{np.median(widths):,.0f}")
        
        return metrics
    
    def format_prediction(self, customer_id, predicted_clv, lower, upper) -> str:
        """
        Format a prediction as a business-readable string.
        
        Example: "Customer 12345: expected CLV ₹4,200 (₹2,800–₹5,900 at 90% confidence)"
        """
        return (
            f"Customer {customer_id}: expected CLV "
            f"{CURRENCY_SYMBOL}{predicted_clv:,.0f} "
            f"({CURRENCY_SYMBOL}{lower:,.0f}–{CURRENCY_SYMBOL}{upper:,.0f} "
            f"at {self.coverage:.0%} confidence)"
        )
    
    @staticmethod
    def _compute_coverage(y_true, lower, upper):
        """Compute empirical coverage."""
        covered = ((y_true >= lower) & (y_true <= upper)).mean()
        return covered
    
    def save(self, filepath: Optional[Path] = None):
        """Save the conformal model."""
        if filepath is None:
            filepath = MODELS_DIR / "conformal_model.pkl"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'conformal_model': self.conformal_model,
            'alpha': self.alpha,
            'coverage': self.coverage,
            'api_version': self._api_version,
        }, filepath)
        logger.info(f"Saved conformal model to {filepath}")
    
    @classmethod
    def load(cls, filepath: Optional[Path] = None):
        """Load a saved conformal model."""
        if filepath is None:
            filepath = MODELS_DIR / "conformal_model.pkl"
        data = joblib.load(filepath)
        instance = cls.__new__(cls)
        instance.conformal_model = data['conformal_model']
        instance.alpha = data['alpha']
        instance.coverage = data['coverage']
        instance._api_version = data['api_version']
        instance._is_fitted = True
        logger.info(f"Loaded conformal model from {filepath}")
        return instance
