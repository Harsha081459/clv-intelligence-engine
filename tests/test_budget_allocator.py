import numpy as np
import pytest

from src.optimization.budget_allocator import BudgetAllocator


@pytest.fixture()
def allocator():
    return BudgetAllocator(
        cost_per_contact=100.0,
        uplift_scores=np.array([400.0, 300.0, 200.0, -50.0]),
        expected_clv=np.array([1000.0, 900.0, 800.0, 700.0]),
        segment_labels=np.array(["A", "B", "C", "D"]),
    )


def test_budget_respected(allocator):
    result = allocator.optimize_greedy(total_budget=250.0)
    assert result["total_cost"] <= 250.0
    assert result["total_cost"] == 200.0


def test_higher_roi_customers_chosen_first(allocator):
    result = allocator.optimize_greedy(total_budget=350.0)
    selected = list(result["selected_customers"]["customer_index"])
    assert selected == [0, 1, 2]  # sorted by descending uplift/cost
    # customer 3 has negative uplift and is never selected
    assert 3 not in selected


def test_roi_computation(allocator):
    result = allocator.optimize_greedy(total_budget=250.0)
    # (400 + 300) / 200 = 3.5
    assert result["roi"] == pytest.approx(3.5)
    assert result["expected_incremental_revenue"] == pytest.approx(700.0)


def test_zero_budget_selects_nobody(allocator):
    result = allocator.optimize_greedy(total_budget=0.0)
    assert result["total_cost"] == 0.0
    assert len(result["selected_customers"]) == 0
    assert result["roi"] == 0.0


def test_negative_uplift_never_selected(allocator):
    # even with budget for all 4 contacts, only the 3 positive-uplift
    # customers are targeted
    result = allocator.optimize_greedy(total_budget=400.0)
    assert result["total_cost"] == 300.0
