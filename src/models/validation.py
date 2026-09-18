import math

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


def customer_splits(index, seed=42):
    index = pd.Index(index).sort_values()
    if not index.is_unique or len(index) < 60:
        raise ValueError("Validation requires at least 60 unique customers")
    shuffled = index.to_numpy()[np.random.default_rng(seed).permutation(len(index))]
    train_end = int(0.7 * len(index))
    calibration_end = train_end + int(0.15 * len(index))
    return {
        "train": pd.Index(shuffled[:train_end]),
        "calibration": pd.Index(shuffled[train_end:calibration_end]),
        "test": pd.Index(shuffled[calibration_end:]),
    }


def out_of_fold_predictions(estimator, X, y, folds=5, seed=42):
    predictions = np.empty(len(X), dtype=float)
    for train, validation in KFold(n_splits=folds, shuffle=True, random_state=seed).split(X):
        model = clone(estimator)
        model.fit(X.iloc[train], y.iloc[train])
        predictions[validation] = model.predict(X.iloc[validation])
    return predictions


def conformal_radius(actual, predicted, alpha=0.1):
    errors = np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))
    if not 0 < alpha < 1 or not np.isfinite(errors).all():
        raise ValueError("Finite calibration errors and alpha in (0, 1) are required")
    rank = math.ceil((len(errors) + 1) * (1 - alpha))
    if rank > len(errors) or not len(errors):
        raise ValueError("Calibration set is too small for a finite interval at this alpha")
    return float(np.partition(errors, rank - 1)[rank - 1])


def split_labels(index, splits):
    labels = pd.Series(index=index, dtype="str")
    for name, ids in splits.items():
        labels.loc[ids] = name
    if labels.isna().any():
        raise ValueError("Every customer must belong to a partition")
    return labels
