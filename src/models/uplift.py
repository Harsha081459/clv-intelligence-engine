"""
Uplift Modelling — T-Learner with Synthetic Treatment Simulation.

Estimates the *incremental* effect of a marketing treatment on each
customer's revenue.  Because true randomised-trial data is not available,
treatment assignment and uplift effects are **synthetically simulated**
for demonstration purposes.

Model:
    Two XGBRegressors (T-Learner):
        * model_treatment  — trained on the treated sub-population
        * model_control    — trained on the control sub-population
    CATE (uplift) = E[Y|X, T=1] − E[Y|X, T=0]
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
from xgboost import XGBRegressor

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    MODELS_DIR,
    TREATMENT_FRACTION,
    SYNTHETIC_UPLIFT_EFFECT,
    RANDOM_STATE,
    CURRENCY_SYMBOL,
)

logger = logging.getLogger(__name__)

# Persistence paths
_UPLIFT_TREATMENT_PATH = MODELS_DIR / "uplift_model_treatment.pkl"
_UPLIFT_CONTROL_PATH = MODELS_DIR / "uplift_model_control.pkl"
_UPLIFT_META_PATH = MODELS_DIR / "uplift_meta.pkl"


class UpliftModel:
    """T-Learner uplift model with synthetic treatment simulation.

    **Important**: Treatment assignment is *simulated*, not based on a
    real A/B test.  The ``synthetic_uplift`` parameter controls the
    artificial revenue boost applied to the treatment group.  All
    downstream uplift estimates should be interpreted with this caveat.
    """

    def __init__(
        self,
        treatment_fraction: float = TREATMENT_FRACTION,
        synthetic_uplift: float = SYNTHETIC_UPLIFT_EFFECT,
    ) -> None:
        """Initialise the uplift model.

        Args:
            treatment_fraction: Proportion of customers assigned to the
                synthetic treatment group (default from config).
            synthetic_uplift: Multiplicative uplift factor applied to
                treated customers' outcomes (default from config).
        """
        self.treatment_fraction = treatment_fraction
        self.synthetic_uplift = synthetic_uplift
        self.model_treatment: Optional[XGBRegressor] = None
        self.model_control: Optional[XGBRegressor] = None
        logger.info(
            "UpliftModel initialised (treatment_frac=%.2f, uplift=%.2f).",
            self.treatment_fraction,
            self.synthetic_uplift,
        )

    # ------------------------------------------------------------------
    # Simulate treatment
    # ------------------------------------------------------------------
    def simulate_treatment(
        self,
        customer_features: pd.DataFrame,
        holdout_revenue: np.ndarray,
        random_state: int = RANDOM_STATE,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Create a synthetic treatment / control split.

        **This is a simulation.**  In a production setting, this step
        would be replaced by data from a genuine randomised controlled
        trial or quasi-experiment.

        Procedure:
            1. Randomly assign ``treatment_fraction`` of customers to the
               treatment group (T=1); the rest are control (T=0).
            2. For treated customers, multiply their holdout revenue by
               ``(1 + synthetic_uplift)`` and add a small random purchase-
               probability boost so the treated outcome is systematically
               higher.

        Args:
            customer_features: Feature matrix (n_customers × p).
            holdout_revenue: Observed holdout-period revenue per customer.
            random_state: Seed for reproducibility.

        Returns:
            Tuple of (X, treatment_flag, y_outcome) where:
                - X: same as ``customer_features``
                - treatment_flag: binary array (1 = treated)
                - y_outcome: synthetic outcome revenue array
        """
        rng = np.random.RandomState(random_state)
        n = len(customer_features)
        holdout_revenue = np.asarray(holdout_revenue, dtype=np.float64)

        # Random treatment assignment
        treatment = np.zeros(n, dtype=int)
        n_treat = int(n * self.treatment_fraction)
        treat_idx = rng.choice(n, size=n_treat, replace=False)
        treatment[treat_idx] = 1

        # Synthetic outcome: treated customers get uplift
        y_outcome = holdout_revenue.copy()

        # Boost purchase probability (customers with zero revenue may
        # now have positive revenue) and existing revenue.
        purchase_boost = rng.binomial(1, self.synthetic_uplift, size=n_treat)
        baseline_mean = holdout_revenue[holdout_revenue > 0].mean() if (holdout_revenue > 0).any() else 500.0

        y_outcome[treat_idx] = (
            y_outcome[treat_idx] * (1 + self.synthetic_uplift)
            + purchase_boost * baseline_mean * rng.uniform(0.2, 0.5, size=n_treat)
        )

        logger.info(
            "Simulated treatment: %d treated / %d control, "
            "avg outcome treated=%.2f, control=%.2f",
            treatment.sum(),
            n - treatment.sum(),
            y_outcome[treatment == 1].mean(),
            y_outcome[treatment == 0].mean(),
        )

        return customer_features, treatment, y_outcome

    # ------------------------------------------------------------------
    # Fit T-Learner
    # ------------------------------------------------------------------
    def fit_t_learner(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> "UpliftModel":
        """Train the T-Learner (two separate regressors).

        Args:
            X: Feature matrix.
            treatment: Binary treatment indicator (0/1).
            y: Outcome variable (revenue).

        Returns:
            self (for chaining).
        """
        treatment = np.asarray(treatment)
        y = np.asarray(y, dtype=np.float64)

        mask_t = treatment == 1
        mask_c = treatment == 0

        X_treat = X[mask_t] if isinstance(X, pd.DataFrame) else X[mask_t]
        X_ctrl = X[mask_c] if isinstance(X, pd.DataFrame) else X[mask_c]

        self.model_treatment = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        self.model_control = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0,
        )

        logger.info("Training treatment model on %d samples …", mask_t.sum())
        self.model_treatment.fit(X_treat, y[mask_t])

        logger.info("Training control model on %d samples …", mask_c.sum())
        self.model_control.fit(X_ctrl, y[mask_c])

        logger.info("T-Learner fitted successfully.")
        return self

    # ------------------------------------------------------------------
    # Predict uplift
    # ------------------------------------------------------------------
    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Predict individual-level uplift (CATE).

        Args:
            X: Feature matrix (same schema as training).

        Returns:
            Array of uplift scores per customer.

        Raises:
            RuntimeError: If models are not fitted.
        """
        if self.model_treatment is None or self.model_control is None:
            raise RuntimeError("Models not fitted. Call fit_t_learner() first.")

        y_treat = self.model_treatment.predict(X)
        y_ctrl = self.model_control.predict(X)
        uplift = y_treat - y_ctrl
        logger.debug(
            "Uplift stats — mean=%.2f, median=%.2f, std=%.2f",
            uplift.mean(),
            np.median(uplift),
            uplift.std(),
        )
        return uplift

    # ------------------------------------------------------------------
    # Rank customers
    # ------------------------------------------------------------------
    def rank_by_uplift(
        self,
        X: pd.DataFrame,
        customer_ids: np.ndarray,
    ) -> pd.DataFrame:
        """Rank customers by uplift score and assign targeting segments.

        Segments:
            * **Persuadable** — top 10 % by uplift
            * **Favourable** — next 20 %
            * **Neutral / Negative** — remainder

        Args:
            X: Feature matrix.
            customer_ids: Customer ID array.

        Returns:
            DataFrame sorted descending by uplift score with columns
            ``customer_id``, ``uplift_score``, ``uplift_rank``,
            ``segment``.
        """
        uplift = self.predict_uplift(X)

        df = pd.DataFrame(
            {
                "customer_id": np.asarray(customer_ids),
                "uplift_score": uplift,
            }
        )
        df = df.sort_values("uplift_score", ascending=False).reset_index(drop=True)
        df["uplift_rank"] = range(1, len(df) + 1)

        n = len(df)
        top_10 = int(n * 0.10)
        top_30 = int(n * 0.30)

        conditions = [
            df["uplift_rank"] <= top_10,
            df["uplift_rank"] <= top_30,
        ]
        choices = ["Persuadable", "Favourable"]
        df["segment"] = np.select(conditions, choices, default="Neutral/Negative")

        logger.info(
            "Uplift ranking: %d persuadable, %d favourable, %d neutral/negative",
            (df["segment"] == "Persuadable").sum(),
            (df["segment"] == "Favourable").sum(),
            (df["segment"] == "Neutral/Negative").sum(),
        )
        return df

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def plot_uplift_distribution(self, uplift_scores: np.ndarray) -> go.Figure:
        """Plot the distribution of uplift scores.

        Args:
            uplift_scores: Array of predicted CATE values.

        Returns:
            Plotly Figure.
        """
        fig = px.histogram(
            x=uplift_scores,
            nbins=60,
            title="Distribution of Predicted Uplift Scores",
            labels={"x": f"Uplift ({CURRENCY_SYMBOL})", "y": "Count"},
            template="plotly_white",
            opacity=0.75,
            color_discrete_sequence=["#636EFA"],
        )
        fig.add_vline(
            x=0, line_dash="dash", line_color="red",
            annotation_text="Zero uplift",
        )
        fig.update_layout(width=800, height=450)
        logger.info("Uplift distribution plot created.")
        return fig

    def plot_qini_curve(
        self,
        y_true: np.ndarray,
        uplift_scores: np.ndarray,
        treatment: np.ndarray,
    ) -> go.Figure:
        """Plot a Qini curve comparing the model to random targeting.

        The Qini curve measures cumulative incremental revenue when
        customers are targeted in order of predicted uplift.

        Args:
            y_true: Observed outcome.
            uplift_scores: Model-predicted uplift.
            treatment: Binary treatment indicator.

        Returns:
            Plotly Figure.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        uplift_scores = np.asarray(uplift_scores, dtype=np.float64)
        treatment = np.asarray(treatment, dtype=int)

        # Sort by uplift descending
        order = np.argsort(-uplift_scores)
        y_sorted = y_true[order]
        t_sorted = treatment[order]

        n = len(y_true)
        n_treat_total = treatment.sum()
        n_ctrl_total = n - n_treat_total

        qini_model: List[float] = [0.0]
        qini_random: List[float] = [0.0]
        fractions: List[float] = [0.0]

        cumsum_treat = 0.0
        cumsum_ctrl = 0.0
        n_treat_cum = 0
        n_ctrl_cum = 0

        total_gain = (
            y_true[treatment == 1].sum() / max(n_treat_total, 1)
            - y_true[treatment == 0].sum() / max(n_ctrl_total, 1)
        ) * n_treat_total

        for i in range(n):
            if t_sorted[i] == 1:
                cumsum_treat += y_sorted[i]
                n_treat_cum += 1
            else:
                cumsum_ctrl += y_sorted[i]
                n_ctrl_cum += 1

            if (i + 1) % max(n // 100, 1) == 0 or i == n - 1:
                fraction = (i + 1) / n
                # Qini = cumulative uplift revenue
                gain_treat = cumsum_treat / max(n_treat_cum, 1) * n_treat_cum if n_treat_cum else 0
                gain_ctrl = cumsum_ctrl / max(n_ctrl_cum, 1) * n_ctrl_cum if n_ctrl_cum else 0
                qini_val = (
                    cumsum_treat / max(n_treat_cum, 1)
                    - cumsum_ctrl / max(n_ctrl_cum, 1)
                ) * n_treat_cum * fraction
                qini_model.append(qini_val)
                qini_random.append(total_gain * fraction)
                fractions.append(fraction)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fractions, y=qini_model,
            mode="lines", name="Model",
            line=dict(color="#636EFA", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=fractions, y=qini_random,
            mode="lines", name="Random",
            line=dict(color="grey", width=2, dash="dash"),
        ))
        fig.update_layout(
            title="Qini Curve — Model vs Random Targeting",
            xaxis_title="Fraction of Customers Targeted",
            yaxis_title=f"Cumulative Incremental Revenue ({CURRENCY_SYMBOL})",
            template="plotly_white",
            width=800,
            height=500,
        )
        logger.info("Qini curve plot created.")
        return fig

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Persist the fitted T-Learner models and metadata."""
        if self.model_treatment is None:
            raise RuntimeError("Nothing to save — models not fitted.")

        with open(_UPLIFT_TREATMENT_PATH, "wb") as f:
            pickle.dump(self.model_treatment, f)
        with open(_UPLIFT_CONTROL_PATH, "wb") as f:
            pickle.dump(self.model_control, f)

        meta = {
            "treatment_fraction": self.treatment_fraction,
            "synthetic_uplift": self.synthetic_uplift,
        }
        with open(_UPLIFT_META_PATH, "wb") as f:
            pickle.dump(meta, f)

        logger.info("Uplift models saved to %s", MODELS_DIR)

    @classmethod
    def load(cls) -> "UpliftModel":
        """Load a previously saved T-Learner model.

        Returns:
            A fully initialised ``UpliftModel`` instance.

        Raises:
            FileNotFoundError: If model artefacts are missing.
        """
        for path in (_UPLIFT_TREATMENT_PATH, _UPLIFT_CONTROL_PATH, _UPLIFT_META_PATH):
            if not path.exists():
                raise FileNotFoundError(f"Missing artefact: {path}")

        with open(_UPLIFT_META_PATH, "rb") as f:
            meta = pickle.load(f)

        instance = cls(
            treatment_fraction=meta["treatment_fraction"],
            synthetic_uplift=meta["synthetic_uplift"],
        )
        with open(_UPLIFT_TREATMENT_PATH, "rb") as f:
            instance.model_treatment = pickle.load(f)
        with open(_UPLIFT_CONTROL_PATH, "rb") as f:
            instance.model_control = pickle.load(f)

        logger.info("Uplift models loaded from %s", MODELS_DIR)
        return instance


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info("=== Uplift module — smoke test ===")

    np.random.seed(RANDOM_STATE)
    n = 600

    # Synthetic features & revenue
    features = pd.DataFrame({
        "frequency": np.random.poisson(5, n),
        "recency": np.random.exponential(50, n),
        "monetary": np.random.exponential(3000, n),
        "tenure": np.random.uniform(30, 365, n),
    })
    holdout_rev = np.random.exponential(2000, n)

    model = UpliftModel()
    X, treatment, y = model.simulate_treatment(features, holdout_rev)
    model.fit_t_learner(X, treatment, y)

    uplift_scores = model.predict_uplift(X)
    ranking = model.rank_by_uplift(X, np.arange(n))
    print(ranking.head(15))

    fig1 = model.plot_uplift_distribution(uplift_scores)
    fig1.show()

    fig2 = model.plot_qini_curve(y, uplift_scores, treatment)
    fig2.show()

    model.save()
    model_loaded = UpliftModel.load()
    logger.info("Reload test passed.")
    logger.info("=== Smoke test passed ===")
