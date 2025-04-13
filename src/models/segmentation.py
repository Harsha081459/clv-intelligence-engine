"""
Customer Segmentation using Gaussian Mixture Models (GMM).

Segments customers based on predicted CLV, probability of being alive,
and recency. Uses BIC-based model selection and auto-labels clusters
with business-meaningful names via centroid analysis.
"""

import sys
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    MODELS_DIR,
    GMM_K_RANGE,
    SEGMENT_NAMES,
    RANDOM_STATE,
    CURRENCY_SYMBOL,
)

logger = logging.getLogger(__name__)

# Persistence paths
_SEGMENTATION_MODEL_PATH = MODELS_DIR / "segmentation_gmm.pkl"
_SEGMENTATION_SCALER_PATH = MODELS_DIR / "segmentation_scaler.pkl"
_SEGMENTATION_META_PATH = MODELS_DIR / "segmentation_meta.pkl"


class CustomerSegmentation:
    """Gaussian Mixture Model segmentation on CLV predictions.

    Features used for clustering:
        - log1p(predicted_clv)
        - p_alive  (probability of being active)
        - recency_days

    The optimal number of components is selected via BIC, and each
    cluster is automatically assigned a business-meaningful label based
    on its centroid characteristics.
    """

    def __init__(self) -> None:
        """Initialise an empty segmentation model."""
        self.gmm: Optional[GaussianMixture] = None
        self.scaler: Optional[StandardScaler] = None
        self.optimal_k: Optional[int] = None
        self.bic_scores: Optional[Dict[int, float]] = None
        self.cluster_to_label: Optional[Dict[int, str]] = None
        self._feature_names = ["log1p_clv", "p_alive", "recency_days"]
        logger.info("CustomerSegmentation initialised.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_features(
        predicted_clv: np.ndarray,
        p_alive: np.ndarray,
        recency_days: np.ndarray,
    ) -> np.ndarray:
        """Build the feature matrix from raw inputs.

        Args:
            predicted_clv: Array of predicted customer lifetime values.
            p_alive: Array of customer alive probabilities (0-1).
            recency_days: Array of days since last purchase.

        Returns:
            2-D array of shape (n_customers, 3).
        """
        predicted_clv = np.asarray(predicted_clv, dtype=np.float64)
        p_alive = np.asarray(p_alive, dtype=np.float64)
        recency_days = np.asarray(recency_days, dtype=np.float64)
        
        predicted_clv = np.nan_to_num(predicted_clv, nan=0.0)
        p_alive = np.nan_to_num(p_alive, nan=0.0)
        recency_days = np.nan_to_num(recency_days, nan=0.0)
        
        predicted_clv_non_negative = np.maximum(predicted_clv, 0.0)
        
        return np.column_stack([np.log1p(predicted_clv_non_negative), p_alive, recency_days])

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    def find_optimal_k(
        self,
        X: np.ndarray,
        k_range: range = GMM_K_RANGE,
    ) -> Tuple[Dict[int, float], int]:
        """Evaluate GMM for each *k* in ``k_range`` and pick the best by BIC.

        Args:
            X: Scaled feature matrix (n_samples, n_features).
            k_range: Candidate component counts (default from config).

        Returns:
            Tuple of (bic_scores dict, optimal_k).
        """
        bic_scores: Dict[int, float] = {}
        for k in k_range:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=RANDOM_STATE,
                n_init=3,
                max_iter=300,
            )
            gmm.fit(X)
            bic_scores[k] = gmm.bic(X)
            logger.debug("k=%d  BIC=%.2f", k, bic_scores[k])

        optimal_k = min(bic_scores, key=bic_scores.get)  # type: ignore[arg-type]
        logger.info(
            "Optimal k=%d selected (BIC=%.2f). All BIC: %s",
            optimal_k,
            bic_scores[optimal_k],
            bic_scores,
        )
        self.bic_scores = bic_scores
        self.optimal_k = optimal_k
        return bic_scores, optimal_k

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        predicted_clv: np.ndarray,
        p_alive: np.ndarray,
        recency_days: np.ndarray,
    ) -> "CustomerSegmentation":
        """Fit the segmentation model.

        Steps:
            1. Build features  (log1p(clv), p_alive, recency_days).
            2. StandardScaler normalisation.
            3. Find optimal k via BIC (if not already set).
            4. Fit final GMM with optimal k.

        Args:
            predicted_clv: Predicted CLV values per customer.
            p_alive: P(alive) per customer (0-1).
            recency_days: Days since last purchase per customer.

        Returns:
            self (for chaining).
        """
        X_raw = self._build_features(predicted_clv, p_alive, recency_days)

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        # Optimal k
        if self.optimal_k is None:
            self.find_optimal_k(X_scaled)

        # Fit final model
        self.gmm = GaussianMixture(
            n_components=self.optimal_k,
            covariance_type="full",
            random_state=RANDOM_STATE,
            n_init=5,
            max_iter=500,
        )
        self.gmm.fit(X_scaled)
        logger.info(
            "GMM fitted with k=%d on %d customers.",
            self.optimal_k,
            X_scaled.shape[0],
        )

        # Auto-label using the training data
        labels = self.gmm.predict(X_scaled)
        self.cluster_to_label = self._assign_labels(X_raw, labels)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(
        self,
        predicted_clv: np.ndarray,
        p_alive: np.ndarray,
        recency_days: np.ndarray,
    ) -> np.ndarray:
        """Assign segment cluster IDs to customers.

        Args:
            predicted_clv: Predicted CLV values per customer.
            p_alive: P(alive) per customer.
            recency_days: Days since last purchase per customer.

        Returns:
            Integer array of cluster IDs.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self.gmm is None or self.scaler is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_raw = self._build_features(predicted_clv, p_alive, recency_days)
        X_scaled = self.scaler.transform(X_raw)
        return self.gmm.predict(X_scaled)

    # ------------------------------------------------------------------
    # Label segments
    # ------------------------------------------------------------------
    def _assign_labels(
        self,
        X_raw: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, str]:
        """Map cluster IDs → meaningful names via centroid analysis.

        The centroid of each cluster (in *unscaled* feature space) is
        ranked along the CLV and p_alive dimensions.  The mapping logic:

        * **Champions** – highest CLV *and* high p_alive
        * **At-Risk High-Value** – high CLV *but* low p_alive
        * **Promising** – medium CLV *and* high p_alive
        * **Hibernating** – low CLV *and* low p_alive
        * **Lost** – very low across all dimensions

        When k < 5 some labels are dropped; when k > 5 the extra
        clusters are named *Segment-{i}*.
        """
        unique_labels = np.unique(labels)
        k = len(unique_labels)

        # Compute centroids in raw feature space
        centroids = np.zeros((k, X_raw.shape[1]))
        for i, lbl in enumerate(unique_labels):
            centroids[i] = X_raw[labels == lbl].mean(axis=0)

        # centroids columns: [log1p_clv, p_alive, recency_days]
        clv_col = centroids[:, 0]
        alive_col = centroids[:, 1]

        # Composite score for ranking
        composite = clv_col * 0.6 + alive_col * 0.4
        rank_order = np.argsort(-composite)  # descending

        canonical_names = [
            SEGMENT_NAMES["champions"],
            SEGMENT_NAMES["at_risk_high_value"],
            SEGMENT_NAMES["promising"],
            SEGMENT_NAMES["hibernating"],
            SEGMENT_NAMES["lost"],
        ]

        mapping: Dict[int, str] = {}
        for rank_pos, cluster_id in enumerate(rank_order):
            cluster_id_int = int(unique_labels[cluster_id])
            if rank_pos < len(canonical_names):
                name = canonical_names[rank_pos]
            else:
                name = f"Segment-{cluster_id_int}"

            # Refine At-Risk: high CLV but low p_alive
            if rank_pos == 0:
                # Already champion (best composite)
                pass
            elif rank_pos == 1:
                # Second-best composite — check if p_alive is low
                alive_rank = np.argsort(alive_col)
                # If this cluster is in bottom half by p_alive → At-Risk High-Value
                if np.where(alive_rank == cluster_id)[0][0] < k // 2:
                    name = SEGMENT_NAMES["at_risk_high_value"]
                else:
                    name = SEGMENT_NAMES["promising"]

            mapping[cluster_id_int] = name

        logger.info("Cluster-to-label mapping: %s", mapping)
        return mapping

    def label_segments(self, segment_data: pd.DataFrame) -> pd.DataFrame:
        """Apply human-readable labels to segment cluster IDs.

        Args:
            segment_data: DataFrame that contains a ``segment_id``
                column with integer cluster IDs.

        Returns:
            The input DataFrame with an added ``segment_name`` column.

        Raises:
            RuntimeError: If the label mapping is unavailable.
        """
        if self.cluster_to_label is None:
            raise RuntimeError(
                "Label mapping not available. Call fit() first."
            )

        segment_data = segment_data.copy()
        segment_data["segment_name"] = (
            segment_data["segment_id"]
            .map(self.cluster_to_label)
            .fillna("Unknown")
        )
        return segment_data

    # ------------------------------------------------------------------
    # Segment profiles
    # ------------------------------------------------------------------
    def get_segment_profiles(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Build a summary profile for each segment.

        Args:
            customer_data: DataFrame with at least columns
                ``segment_name``, ``predicted_clv``, ``p_alive``,
                ``recency_days``.

        Returns:
            DataFrame with columns: segment_name, count, avg_clv,
            avg_p_alive, avg_recency, recommended_action.
        """
        action_map = {
            SEGMENT_NAMES["champions"]: "Loyalty rewards & VIP perks",
            SEGMENT_NAMES["at_risk_high_value"]: "Urgent win-back campaign",
            SEGMENT_NAMES["promising"]: "Upsell & cross-sell offers",
            SEGMENT_NAMES["hibernating"]: "Re-engagement email sequence",
            SEGMENT_NAMES["lost"]: "Low-cost reactivation or suppress",
        }

        profiles = (
            customer_data.groupby("segment_name")
            .agg(
                count=("predicted_clv", "size"),
                avg_clv=("predicted_clv", "mean"),
                avg_p_alive=("p_alive", "mean"),
                avg_recency=("recency_days", "mean"),
            )
            .reset_index()
        )

        profiles["recommended_action"] = profiles["segment_name"].map(action_map).fillna("Review manually")
        profiles = profiles.sort_values("avg_clv", ascending=False).reset_index(drop=True)

        # Pretty-format CLV
        profiles["avg_clv"] = profiles["avg_clv"].round(2)
        profiles["avg_p_alive"] = profiles["avg_p_alive"].round(4)
        profiles["avg_recency"] = profiles["avg_recency"].round(1)

        logger.info("Segment profiles:\n%s", profiles.to_string())
        return profiles

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def plot_segments_scatter(
        self,
        predicted_clv: np.ndarray,
        p_alive: np.ndarray,
        segment_labels: np.ndarray,
    ) -> go.Figure:
        """Create a Plotly scatter plot of CLV vs p_alive coloured by segment.

        Args:
            predicted_clv: Predicted CLV values.
            p_alive: P(alive) per customer.
            segment_labels: String segment names for each customer.

        Returns:
            Plotly Figure object.
        """
        df_plot = pd.DataFrame(
            {
                "Predicted CLV": np.asarray(predicted_clv),
                "P(Alive)": np.asarray(p_alive),
                "Segment": np.asarray(segment_labels),
            }
        )

        fig = px.scatter(
            df_plot,
            x="P(Alive)",
            y="Predicted CLV",
            color="Segment",
            title="Customer Segments — CLV vs Probability Alive",
            labels={
                "Predicted CLV": f"Predicted CLV ({CURRENCY_SYMBOL})",
                "P(Alive)": "Probability Alive",
            },
            opacity=0.6,
            template="plotly_white",
        )
        fig.update_layout(
            legend_title_text="Segment",
            width=900,
            height=600,
        )
        logger.info("Segment scatter plot created.")
        return fig

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Persist the fitted GMM, scaler, and metadata to disk."""
        if self.gmm is None:
            raise RuntimeError("Nothing to save — model not fitted.")

        with open(_SEGMENTATION_MODEL_PATH, "wb") as f:
            pickle.dump(self.gmm, f)
        with open(_SEGMENTATION_SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)

        meta = {
            "optimal_k": self.optimal_k,
            "bic_scores": self.bic_scores,
            "cluster_to_label": self.cluster_to_label,
            "feature_names": self._feature_names,
        }
        with open(_SEGMENTATION_META_PATH, "wb") as f:
            pickle.dump(meta, f)

        logger.info(
            "Segmentation model saved to %s", _SEGMENTATION_MODEL_PATH
        )

    @classmethod
    def load(cls) -> "CustomerSegmentation":
        """Load a previously saved segmentation model.

        Returns:
            A fully initialised ``CustomerSegmentation`` instance.

        Raises:
            FileNotFoundError: If model artefacts are missing.
        """
        for path in (
            _SEGMENTATION_MODEL_PATH,
            _SEGMENTATION_SCALER_PATH,
            _SEGMENTATION_META_PATH,
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing artefact: {path}")

        instance = cls()
        with open(_SEGMENTATION_MODEL_PATH, "rb") as f:
            instance.gmm = pickle.load(f)
        with open(_SEGMENTATION_SCALER_PATH, "rb") as f:
            instance.scaler = pickle.load(f)
        with open(_SEGMENTATION_META_PATH, "rb") as f:
            meta = pickle.load(f)

        instance.optimal_k = meta["optimal_k"]
        instance.bic_scores = meta["bic_scores"]
        instance.cluster_to_label = meta["cluster_to_label"]
        instance._feature_names = meta.get("feature_names", instance._feature_names)

        logger.info(
            "Segmentation model loaded (k=%d).", instance.optimal_k
        )
        return instance


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info("=== Segmentation module — smoke test ===")

    np.random.seed(RANDOM_STATE)
    n = 500
    clv = np.random.exponential(5000, n)
    alive = np.random.beta(5, 2, n)
    recency = np.random.exponential(60, n)

    seg = CustomerSegmentation()
    seg.fit(clv, alive, recency)
    cluster_ids = seg.predict(clv, alive, recency)

    df_test = pd.DataFrame(
        {
            "predicted_clv": clv,
            "p_alive": alive,
            "recency_days": recency,
            "segment_id": cluster_ids,
        }
    )
    df_test = seg.label_segments(df_test)

    profiles = seg.get_segment_profiles(df_test)
    print(profiles)

    fig = seg.plot_segments_scatter(clv, alive, df_test["segment_name"].values)
    fig.show()

    seg.save()
    seg_loaded = CustomerSegmentation.load()
    logger.info("Reload test — optimal_k = %d", seg_loaded.optimal_k)
    logger.info("=== Smoke test passed ===")
