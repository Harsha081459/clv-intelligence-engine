"""
Marketing Budget Allocation Optimiser.

Allocates a fixed marketing budget across customers by ranking them on
their expected incremental revenue per cost (uplift ROI). Provides
greedy allocation, scenario analysis under multiple uplift multipliers,
and segment-level breakdowns.  All monetary values are in **INR (₹)**.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    COST_PER_CONTACT_INR,
    TOTAL_QUARTERLY_BUDGET_INR,
    CURRENCY_SYMBOL,
)

logger = logging.getLogger(__name__)


class BudgetAllocator:
    """Marketing budget optimisation engine.

    Given per-customer uplift scores and expected CLV values, the
    allocator ranks customers by expected incremental revenue per unit
    cost and allocates budget top-down (greedy).

    Args:
        cost_per_contact: Cost to contact a single customer (₹).
        uplift_scores: Predicted uplift (CATE) per customer.
        expected_clv: Expected CLV for each customer (₹).
        segment_labels: Segment name per customer (e.g. "Champions").
    """

    def __init__(
        self,
        cost_per_contact: float,
        uplift_scores: np.ndarray,
        expected_clv: np.ndarray,
        segment_labels: np.ndarray,
    ) -> None:
        self.cost_per_contact = cost_per_contact
        self.uplift_scores = np.asarray(uplift_scores, dtype=np.float64)
        self.expected_clv = np.asarray(expected_clv, dtype=np.float64)
        self.segment_labels = np.asarray(segment_labels)

        # Pre-compute expected incremental revenue per customer
        # uplift_score represents the predicted lift in revenue
        self.expected_incremental = self.uplift_scores.copy()
        self.roi_ratio = self.expected_incremental / self.cost_per_contact

        # Sort index by roi_ratio descending
        self._sorted_idx = np.argsort(-self.roi_ratio)

        logger.info(
            "BudgetAllocator initialised: %d customers, "
            "cost/contact=%s%.0f, mean uplift=%s%.2f",
            len(self.uplift_scores),
            CURRENCY_SYMBOL,
            self.cost_per_contact,
            CURRENCY_SYMBOL,
            self.expected_incremental.mean(),
        )

    # ------------------------------------------------------------------
    # Greedy optimiser
    # ------------------------------------------------------------------
    def optimize_greedy(
        self,
        total_budget: float,
    ) -> Dict:
        """Greedy top-down budget allocation.

        Customers are sorted by (expected_uplift_revenue / cost) in
        descending order.  We iterate and allocate until the budget is
        exhausted.

        Args:
            total_budget: Total available marketing budget (₹).

        Returns:
            Dict with keys:
                - ``selected_customers``: DataFrame with index, uplift,
                  segment, cost, expected_revenue.
                - ``total_cost``: Actual spend (₹).
                - ``expected_incremental_revenue``: Sum of expected
                  incremental revenue from targeted customers (₹).
                - ``roi``: expected_incremental_revenue / total_cost.
        """
        remaining = total_budget
        selected_indices: List[int] = []

        for idx in self._sorted_idx:
            if remaining < self.cost_per_contact:
                break
            # Only target customers with positive expected uplift
            if self.expected_incremental[idx] <= 0:
                break
            selected_indices.append(int(idx))
            remaining -= self.cost_per_contact

        selected = np.array(selected_indices)
        total_cost = len(selected) * self.cost_per_contact
        total_incr_rev = self.expected_incremental[selected].sum() if len(selected) else 0.0
        roi = total_incr_rev / total_cost if total_cost > 0 else 0.0

        df_selected = pd.DataFrame({
            "customer_index": selected,
            "uplift_score": self.uplift_scores[selected] if len(selected) else [],
            "expected_clv": self.expected_clv[selected] if len(selected) else [],
            "segment": self.segment_labels[selected] if len(selected) else [],
            "cost": self.cost_per_contact,
            "expected_incremental_revenue": self.expected_incremental[selected] if len(selected) else [],
        })

        logger.info(
            "Greedy allocation: %d customers targeted, "
            "cost=%s%.0f, incr_rev=%s%.0f, ROI=%.2fx",
            len(selected),
            CURRENCY_SYMBOL,
            total_cost,
            CURRENCY_SYMBOL,
            total_incr_rev,
            roi,
        )

        return {
            "selected_customers": df_selected,
            "total_cost": total_cost,
            "expected_incremental_revenue": total_incr_rev,
            "roi": roi,
        }

    # ------------------------------------------------------------------
    # ROI curve
    # ------------------------------------------------------------------
    def roi_curve(
        self,
        budget_range: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Generate ROI curve data across a range of budget levels.

        For each budget level the greedy allocator is run and the
        expected incremental revenue and ROI are recorded.

        Args:
            budget_range: Array of budget values to evaluate.
                Defaults to 50 evenly-spaced values from 0 to
                ``config.TOTAL_QUARTERLY_BUDGET_INR``.

        Returns:
            DataFrame with columns: ``budget``, ``n_targeted``,
            ``total_cost``, ``incremental_revenue``, ``roi``.
        """
        if budget_range is None:
            budget_range = np.linspace(0, TOTAL_QUARTERLY_BUDGET_INR, 50)

        records: List[Dict] = []
        for budget in budget_range:
            result = self.optimize_greedy(budget)
            records.append({
                "budget": budget,
                "n_targeted": len(result["selected_customers"]),
                "total_cost": result["total_cost"],
                "incremental_revenue": result["expected_incremental_revenue"],
                "roi": result["roi"],
            })

        df = pd.DataFrame(records)
        logger.info("ROI curve computed for %d budget levels.", len(df))
        return df

    # ------------------------------------------------------------------
    # Scenario analysis
    # ------------------------------------------------------------------
    def scenario_analysis(
        self,
        total_budget: float,
        scenarios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict]:
        """Run allocation under multiple uplift multiplier scenarios.

        Default scenarios:
            * **conservative** (0.7×) — pessimistic uplift estimate
            * **base** (1.0×) — point estimate
            * **optimistic** (1.3×) — optimistic uplift estimate

        Args:
            total_budget: Marketing budget (₹).
            scenarios: Dict mapping scenario name → uplift multiplier.

        Returns:
            Dict mapping scenario name → allocation result dict.
        """
        if scenarios is None:
            scenarios = {
                "conservative": 0.7,
                "base": 1.0,
                "optimistic": 1.3,
            }

        results: Dict[str, Dict] = {}
        original_incremental = self.expected_incremental.copy()
        original_roi_ratio = self.roi_ratio.copy()
        original_sorted = self._sorted_idx.copy()

        for name, multiplier in scenarios.items():
            # Temporarily adjust uplift
            self.expected_incremental = original_incremental * multiplier
            self.roi_ratio = self.expected_incremental / self.cost_per_contact
            self._sorted_idx = np.argsort(-self.roi_ratio)

            result = self.optimize_greedy(total_budget)
            result["multiplier"] = multiplier
            results[name] = result

            logger.info(
                "Scenario '%s' (%.1fx): %d targeted, ROI=%.2fx",
                name,
                multiplier,
                len(result["selected_customers"]),
                result["roi"],
            )

        # Restore original values
        self.expected_incremental = original_incremental
        self.roi_ratio = original_roi_ratio
        self._sorted_idx = original_sorted

        return results

    # ------------------------------------------------------------------
    # Segment-level breakdown
    # ------------------------------------------------------------------
    def segment_allocation_breakdown(
        self,
        total_budget: float,
    ) -> pd.DataFrame:
        """Show how budget is distributed across customer segments.

        Args:
            total_budget: Marketing budget (₹).

        Returns:
            DataFrame with columns: ``segment``, ``n_targeted``,
            ``budget_allocated``, ``expected_revenue``,
            ``pct_of_budget``.
        """
        result = self.optimize_greedy(total_budget)
        df = result["selected_customers"]

        if df.empty:
            logger.warning("No customers selected — budget may be zero.")
            return pd.DataFrame(
                columns=["segment", "n_targeted", "budget_allocated",
                         "expected_revenue", "pct_of_budget"]
            )

        breakdown = (
            df.groupby("segment")
            .agg(
                n_targeted=("customer_index", "count"),
                expected_revenue=("expected_incremental_revenue", "sum"),
            )
            .reset_index()
        )
        breakdown["budget_allocated"] = breakdown["n_targeted"] * self.cost_per_contact
        total_alloc = breakdown["budget_allocated"].sum()
        breakdown["pct_of_budget"] = (
            (breakdown["budget_allocated"] / total_alloc * 100).round(1)
            if total_alloc > 0 else 0.0
        )
        breakdown = breakdown.sort_values("expected_revenue", ascending=False).reset_index(drop=True)

        logger.info("Segment allocation breakdown:\n%s", breakdown.to_string())
        return breakdown

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def plot_roi_curve(
        self,
        budget_range: Optional[np.ndarray] = None,
    ) -> go.Figure:
        """Plotly line chart of budget vs incremental revenue with 3 scenarios.

        Args:
            budget_range: Budget values to evaluate (default from config).

        Returns:
            Plotly Figure.
        """
        if budget_range is None:
            budget_range = np.linspace(0, TOTAL_QUARTERLY_BUDGET_INR, 50)

        scenarios = {"conservative": 0.7, "base": 1.0, "optimistic": 1.3}
        colours = {"conservative": "#EF553B", "base": "#636EFA", "optimistic": "#00CC96"}

        original_incremental = self.expected_incremental.copy()
        original_roi_ratio = self.roi_ratio.copy()
        original_sorted = self._sorted_idx.copy()

        fig = go.Figure()
        for name, multiplier in scenarios.items():
            self.expected_incremental = original_incremental * multiplier
            self.roi_ratio = self.expected_incremental / self.cost_per_contact
            self._sorted_idx = np.argsort(-self.roi_ratio)

            curve = self.roi_curve(budget_range)
            fig.add_trace(go.Scatter(
                x=curve["budget"],
                y=curve["incremental_revenue"],
                mode="lines",
                name=f"{name.title()} ({multiplier:.1f}×)",
                line=dict(color=colours[name], width=2.5),
            ))

        # Restore
        self.expected_incremental = original_incremental
        self.roi_ratio = original_roi_ratio
        self._sorted_idx = original_sorted

        fig.update_layout(
            title="Budget vs Expected Incremental Revenue (3 Scenarios)",
            xaxis_title=f"Marketing Budget ({CURRENCY_SYMBOL})",
            yaxis_title=f"Expected Incremental Revenue ({CURRENCY_SYMBOL})",
            template="plotly_white",
            width=900,
            height=550,
            legend_title_text="Scenario",
        )
        logger.info("ROI curve plot (3 scenarios) created.")
        return fig

    # ------------------------------------------------------------------
    # Business summary
    # ------------------------------------------------------------------
    def generate_summary_text(self, total_budget: float) -> str:
        """Generate a business-readable allocation summary.

        Example output::

            Targeting the top 3,200 customers at ₹150 each (₹4,80,000
            total) is expected to generate ₹18,50,000 in incremental
            revenue — 3.9x ROI.

        Args:
            total_budget: Budget for the campaign (₹).

        Returns:
            Human-readable summary string.
        """
        result = self.optimize_greedy(total_budget)
        n_selected = len(result["selected_customers"])
        cost = result["total_cost"]
        rev = result["expected_incremental_revenue"]
        roi = result["roi"]

        summary = (
            f"Targeting the top {n_selected:,} customers at "
            f"{CURRENCY_SYMBOL}{self.cost_per_contact:,.0f} each "
            f"({CURRENCY_SYMBOL}{cost:,.0f} total) is expected to "
            f"generate {CURRENCY_SYMBOL}{rev:,.0f} in incremental "
            f"revenue — {roi:.1f}x ROI."
        )

        logger.info("Budget summary: %s", summary)
        return summary


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info("=== BudgetAllocator — smoke test ===")

    np.random.seed(42)
    n = 1000

    uplift = np.random.normal(800, 400, n)
    clv = np.random.exponential(5000, n)
    segments = np.random.choice(
        ["Champions", "At-Risk High-Value", "Promising", "Hibernating", "Lost"],
        size=n,
    )

    allocator = BudgetAllocator(
        cost_per_contact=COST_PER_CONTACT_INR,
        uplift_scores=uplift,
        expected_clv=clv,
        segment_labels=segments,
    )

    # Greedy
    result = allocator.optimize_greedy(TOTAL_QUARTERLY_BUDGET_INR)
    print(f"\nTargeted: {len(result['selected_customers'])} customers")
    print(f"Total cost: {CURRENCY_SYMBOL}{result['total_cost']:,.0f}")
    print(f"Incr. revenue: {CURRENCY_SYMBOL}{result['expected_incremental_revenue']:,.0f}")
    print(f"ROI: {result['roi']:.2f}x")

    # Summary
    print("\n" + allocator.generate_summary_text(TOTAL_QUARTERLY_BUDGET_INR))

    # Scenario analysis
    scenarios = allocator.scenario_analysis(TOTAL_QUARTERLY_BUDGET_INR)
    for name, res in scenarios.items():
        print(f"  {name}: ROI={res['roi']:.2f}x")

    # Segment breakdown
    breakdown = allocator.segment_allocation_breakdown(TOTAL_QUARTERLY_BUDGET_INR)
    print("\n", breakdown)

    # Plot
    fig = allocator.plot_roi_curve()
    fig.show()

    logger.info("=== Smoke test passed ===")
