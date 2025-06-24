"""Run conformal prediction phase standalone."""
import sys, logging
sys.path.insert(0, 'd:/ML_Project')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from src.monitoring.conformal import ConformalCLVPredictor
from src.config import *

features = pd.read_parquet(CUSTOMER_FEATURES_FILE)
holdout_actuals = pd.read_parquet(OUTPUT_DATA_DIR / 'holdout_actuals.parquet')
final_predictions = pd.read_parquet(CLV_PREDICTIONS_FILE)

common = features.index.intersection(holdout_actuals.index)
prob_preds = final_predictions.reindex(features.index)
exclude_cols = ['cohort_month', 'cohort_index']
feature_cols = [c for c in features.columns if c not in exclude_cols]

X_all = features.loc[common, feature_cols].fillna(0)
for col in ['p_alive', 'predicted_clv_bgnbd', 'predicted_clv_lgbm']:
    if col in prob_preds.columns:
        X_all[col] = prob_preds.reindex(common)[col].fillna(0)

y_all = holdout_actuals.loc[common, 'holdout_revenue']
X_np = X_all.values
y_np = y_all.values

X_train, X_temp, y_train, y_temp = train_test_split(X_np, y_np, test_size=0.4, random_state=RANDOM_STATE)
X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

base_model = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, verbosity=-1)
conformal = ConformalCLVPredictor(base_model, alpha=CONFORMAL_ALPHA)
conformal.fit_and_conformalize(X_train, y_train, X_calib, y_calib)

metrics = conformal.evaluate_coverage(y_test, X_test)

X_full = features[feature_cols].fillna(0)
for col in ['p_alive', 'predicted_clv_bgnbd', 'predicted_clv_lgbm']:
    if col in prob_preds.columns:
        X_full[col] = prob_preds.reindex(features.index)[col].fillna(0)

result = conformal.predict_with_labels(X_full.values, customer_ids=features.index)
for col in ['clv_lower', 'clv_upper', 'interval_width']:
    if col in result.columns:
        final_predictions[col] = result[col]

final_predictions.to_parquet(CLV_PREDICTIONS_FILE)
conformal.save()
print("CONFORMAL PREDICTION COMPLETE")
print(f"  Coverage: {metrics['empirical_coverage']:.1%}")
print(f"  Avg interval: {CURRENCY_SYMBOL}{metrics['avg_interval_width']:,.0f}")
