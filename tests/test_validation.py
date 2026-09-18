import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from src.models.validation import conformal_radius, customer_splits, out_of_fold_predictions
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessor import DataPreprocessor
from src.config import COL_CUSTOMER, COL_DATE, COL_INVOICE, COL_TOTAL_PRICE


class Memorizer(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.seen = set(X.index)
        return self

    def predict(self, X):
        return np.array([float(i in self.seen) for i in X.index])


def test_oof_rows_are_never_seen_during_fit():
    X = pd.DataFrame({"feature": range(100)})
    values = out_of_fold_predictions(Memorizer(), X, pd.Series(range(100)))
    assert np.all(values == 0)


def test_customer_partitions_are_disjoint_and_stable():
    index = pd.Index(range(100))
    splits = customer_splits(index)
    assert [len(x) for x in splits.values()] == [70, 15, 15]
    assert len(set().union(*(set(x) for x in splits.values()))) == 100
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["calibration"]).isdisjoint(splits["test"])
    for name, ids in customer_splits(index[::-1]).items():
        assert ids.equals(splits[name])


def test_conformal_uses_finite_sample_rank():
    assert conformal_radius(np.arange(1, 11), np.zeros(10), alpha=0.1) == 10
    with pytest.raises(ValueError):
        conformal_radius([1], [0], alpha=0.1)


def test_single_purchase_customers_have_monetary_feature():
    df = pd.DataFrame({COL_CUSTOMER: [1, 2], COL_INVOICE: ["a", "b"],
                       COL_DATE: pd.to_datetime(["2010-01-01", "2010-02-01"]), COL_TOTAL_PRICE: [10, 20]})
    rfm = FeatureEngineer().build_rfm_summary(df)
    assert rfm["monetary_value"].tolist() == [10, 20]


def test_monetary_feature_excludes_first_purchase_for_repeat_customers():
    df = pd.DataFrame({COL_CUSTOMER: [1, 1, 1], COL_INVOICE: ["a", "b", "c"],
                       COL_DATE: pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01"]), COL_TOTAL_PRICE: [1000, 10, 20]})
    assert FeatureEngineer().build_rfm_summary(df).loc[1, "monetary_value"] == 15


def test_temporal_split_has_no_gap_at_boundary():
    df = pd.DataFrame({COL_CUSTOMER: [1, 1, 1], COL_DATE: pd.to_datetime(["2010-11-30 23:59", "2010-12-01 00:00", "2010-12-01 12:00"])})
    obs, target = DataPreprocessor().temporal_split(df)
    assert len(obs) == 1
    assert len(target) == 2
