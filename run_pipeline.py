"""
CLV Intelligence Engine — Main Pipeline
=========================================
Orchestrates the entire CLV modeling pipeline from raw data to predictions.

Usage:
    python run_pipeline.py --phase all        # Run everything
    python run_pipeline.py --phase data       # Data cleaning + features only
    python run_pipeline.py --phase models     # Fit all models
    python run_pipeline.py --phase dashboard  # Launch Streamlit dashboard
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_FILE, CLEAN_TRANSACTIONS_FILE, CUSTOMER_FEATURES_FILE,
    RFM_SUMMARY_FILE, CLV_PREDICTIONS_FILE, CUSTOMER_SEGMENTS_FILE,
    OUTPUT_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR,
    OBSERVATION_END, HOLDOUT_END, GROSS_MARGIN,
    CLV_PREDICTION_MONTHS, MONTHLY_DISCOUNT_RATE,
    COST_PER_CONTACT_INR, RANDOM_STATE, CURRENCY_SYMBOL,
    CONFORMAL_ALPHA
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CLV_Pipeline")


def run_data_pipeline():
    """Phase 1: Load, clean, and engineer features."""
    from src.data.loader import load_raw_data
    from src.data.preprocessor import DataPreprocessor
    from src.data.feature_engineering import (
        FeatureEngineer, build_cohort_retention_matrix
    )
    
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA PIPELINE")
    logger.info("=" * 60)
    
    # Load raw data
    raw_df = load_raw_data()
    
    # Clean
    preprocessor = DataPreprocessor(convert_to_inr=True)
    clean_df = preprocessor.clean(raw_df)
    preprocessor.save_clean_data(clean_df)
    
    # Temporal split
    obs_df, holdout_df = preprocessor.temporal_split(clean_df)
    
    # Feature engineering
    fe = FeatureEngineer(observation_end=OBSERVATION_END)
    features = fe.build_all_features(obs_df)
    rfm = fe.build_rfm_summary(obs_df)
    holdout_actuals = fe.build_holdout_actuals(holdout_df)
    
    # Save features
    fe.save_features(features, rfm)
    
    # Save holdout actuals
    holdout_actuals.to_parquet(OUTPUT_DATA_DIR / "holdout_actuals.parquet")
    
    # Save observation and holdout data separately
    obs_df.to_parquet(PROCESSED_DATA_DIR / "transactions_observation.parquet", index=False)
    holdout_df.to_parquet(PROCESSED_DATA_DIR / "transactions_holdout.parquet", index=False)
    
    # Build and save cohort retention matrix
    retention_matrix = build_cohort_retention_matrix(obs_df)
    retention_matrix.to_parquet(OUTPUT_DATA_DIR / "cohort_retention.parquet")
    
    logger.info(f"\nData pipeline complete!")
    logger.info(f"  Clean transactions: {CLEAN_TRANSACTIONS_FILE}")
    logger.info(f"  Customer features: {CUSTOMER_FEATURES_FILE}")
    logger.info(f"  RFM summary: {RFM_SUMMARY_FILE}")
    logger.info(f"  Customers: {len(features):,}")
    
    return features, rfm, holdout_actuals, obs_df, holdout_df


def run_probabilistic_models(rfm=None, holdout_actuals=None):
    """Phase 2: Fit BG/NBD and Gamma-Gamma models."""
    import pandas as pd
    
    logger.info("=" * 60)
    logger.info("PHASE 2: PROBABILISTIC MODELS (BG/NBD + Gamma-Gamma)")
    logger.info("=" * 60)
    
    from src.models.probabilistic import ProbabilisticCLV
    
    if rfm is None:
        rfm = pd.read_parquet(RFM_SUMMARY_FILE)
    if holdout_actuals is None:
        holdout_actuals = pd.read_parquet(OUTPUT_DATA_DIR / "holdout_actuals.parquet")
    
    # Fit probabilistic models
    prob_model = ProbabilisticCLV()
    prob_model.fit(rfm)
    
    # Predict
    predictions = prob_model.predict_all(rfm, months=CLV_PREDICTION_MONTHS)
    
    # Validate against holdout
    common_customers = predictions.index.intersection(holdout_actuals.index)
    if len(common_customers) > 0:
        from src.evaluation.metrics import mae, rmse, mape, pearson_correlation
        
        actual_rev = holdout_actuals.loc[common_customers, 'holdout_revenue']
        pred_clv = predictions.loc[common_customers, 'predicted_clv']
        
        logger.info(f"\nProbabilistic Model Validation (n={len(common_customers):,}):")
        logger.info(f"  MAE:  {CURRENCY_SYMBOL}{mae(actual_rev, pred_clv):,.0f}")
        logger.info(f"  RMSE: {CURRENCY_SYMBOL}{rmse(actual_rev, pred_clv):,.0f}")
        logger.info(f"  MAPE: {mape(actual_rev, pred_clv):.1f}%")
        logger.info(f"  Pearson r: {pearson_correlation(actual_rev, pred_clv):.4f}")
    
    # Save
    prob_model.save()
    predictions.to_parquet(OUTPUT_DATA_DIR / "probabilistic_predictions.parquet")
    
    logger.info("Probabilistic models complete!")
    
    return prob_model, predictions


def run_ml_models(features=None, predictions_prob=None, holdout_actuals=None):
    """Phase 3: Train LightGBM + stacking."""
    import pandas as pd
    import numpy as np
    
    logger.info("=" * 60)
    logger.info("PHASE 3: ML MODELS (LightGBM + Stacking)")
    logger.info("=" * 60)
    
    from src.models.ml_model import CLVBoostingModel
    from src.models.stacking import StackedCLVModel
    from src.evaluation.metrics import mae, rmse, mape, pearson_correlation
    
    if features is None:
        features = pd.read_parquet(CUSTOMER_FEATURES_FILE)
    if predictions_prob is None:
        predictions_prob = pd.read_parquet(OUTPUT_DATA_DIR / "probabilistic_predictions.parquet")
    if holdout_actuals is None:
        holdout_actuals = pd.read_parquet(OUTPUT_DATA_DIR / "holdout_actuals.parquet")
    
    # Prepare features: merge customer features with probabilistic predictions
    ml_features = features.copy()
    for col in ['predicted_purchases', 'p_alive', 'predicted_clv']:
        if col in predictions_prob.columns:
            ml_features[f'prob_{col}'] = predictions_prob[col]
    
    # Target: holdout revenue (fill 0 for customers not in holdout)
    ml_features['target_revenue'] = holdout_actuals.reindex(ml_features.index)['holdout_revenue'].fillna(0)
    
    # Prepare X, y
    exclude_cols = ['target_revenue', 'total_revenue', 'cohort_month', 'cohort_index']
    feature_cols = [c for c in ml_features.columns if c not in exclude_cols]
    
    X = ml_features[feature_cols].fillna(0)
    y = ml_features['target_revenue']
    feature_names = feature_cols
    
    # Train LightGBM
    lgbm_model = CLVBoostingModel(model_type='lightgbm')
    logger.info("Tuning LightGBM hyperparameters with Optuna...")
    best_params = lgbm_model.tune_hyperparameters(X, y, n_trials=30)
    lgbm_model.train(X, y, params=best_params)
    lgbm_predictions = lgbm_model.predict(X)
    
    # Train stacking model
    bgf_pred = predictions_prob.reindex(ml_features.index)['predicted_clv'].fillna(0).values
    
    stacked_model = StackedCLVModel()
    stacked_model.fit(X, y, bgf_pred, lgbm_predictions)
    stacked_predictions = stacked_model.predict(bgf_pred, lgbm_predictions)
    
    # Compare models
    logger.info(f"\n{'='*60}")
    logger.info("MODEL COMPARISON")
    logger.info(f"{'='*60}")
    
    results = {}
    for name, pred in [('BG/NBD', bgf_pred), ('LightGBM', lgbm_predictions), ('Stacked', stacked_predictions)]:
        results[name] = {
            'MAE': mae(y, pred),
            'RMSE': rmse(y, pred),
            'MAPE': mape(y, pred),
            'Pearson_r': pearson_correlation(y, pred),
        }
        logger.info(f"  {name:12s}: MAE={CURRENCY_SYMBOL}{results[name]['MAE']:,.0f}, "
                    f"RMSE={CURRENCY_SYMBOL}{results[name]['RMSE']:,.0f}, "
                    f"MAPE={results[name]['MAPE']:.1f}%, "
                    f"r={results[name]['Pearson_r']:.4f}")
    
    # Save comparison
    comparison_df = pd.DataFrame(results).T
    comparison_df.to_parquet(OUTPUT_DATA_DIR / "model_comparison.parquet")
    
    # Save best predictions
    final_predictions = pd.DataFrame({
        'predicted_clv': stacked_predictions,
        'predicted_clv_bgnbd': bgf_pred,
        'predicted_clv_lgbm': lgbm_predictions,
        'p_alive': predictions_prob.reindex(ml_features.index)['p_alive'].fillna(0).values,
    }, index=ml_features.index)
    
    # SHAP values
    logger.info("Computing SHAP values...")
    try:
        shap_values = lgbm_model.compute_shap_values(X)
        import joblib
        joblib.dump({'shap_values': shap_values, 'feature_names': feature_names}, 
                    MODELS_DIR / "shap_values.pkl")
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
    
    # Save models
    lgbm_model.save(MODELS_DIR / "lgbm_model.pkl")
    stacked_model.save()
    
    logger.info("ML models complete!")
    
    return lgbm_model, stacked_model, final_predictions, feature_names


def run_segmentation(final_predictions=None):
    """Phase 4a: Customer segmentation."""
    import pandas as pd
    
    logger.info("=" * 60)
    logger.info("PHASE 4a: CUSTOMER SEGMENTATION")
    logger.info("=" * 60)
    
    from src.models.segmentation import CustomerSegmentation
    
    if final_predictions is None:
        final_predictions = pd.read_parquet(CLV_PREDICTIONS_FILE)
    
    features = pd.read_parquet(CUSTOMER_FEATURES_FILE)
    
    seg_model = CustomerSegmentation()
    
    predicted_clv = final_predictions['predicted_clv']
    p_alive = final_predictions['p_alive']
    recency = features.reindex(predicted_clv.index)['recency'].fillna(0)
    
    seg_model.fit(predicted_clv, p_alive, recency)
    segments = seg_model.predict(predicted_clv, p_alive, recency)
    
    # Create full segment DataFrame
    segment_df = final_predictions.copy()
    segment_df['segment_id'] = segments
    
    # Label segments
    segment_data = pd.DataFrame({
        'predicted_clv': predicted_clv,
        'p_alive': p_alive,
        'recency_days': recency,
        'segment_id': segments,
    })
    labeled_data = seg_model.label_segments(segment_data)
    segment_df['segment_name'] = labeled_data['segment_name']
    
    # Save
    segment_df.to_parquet(CUSTOMER_SEGMENTS_FILE)
    seg_model.save()
    
    # Log segment profiles
    profiles = seg_model.get_segment_profiles(labeled_data)
    logger.info(f"\nSegment Profiles:\n{profiles}")
    
    logger.info("Segmentation complete!")
    
    return seg_model, segment_df


def run_uplift_model(features=None, holdout_actuals=None):
    """Phase 4b: Uplift modeling."""
    import pandas as pd
    
    logger.info("=" * 60)
    logger.info("PHASE 4b: UPLIFT MODELING")
    logger.info("=" * 60)
    
    from src.models.uplift import UpliftModel
    
    if features is None:
        features = pd.read_parquet(CUSTOMER_FEATURES_FILE)
    if holdout_actuals is None:
        holdout_actuals = pd.read_parquet(OUTPUT_DATA_DIR / "holdout_actuals.parquet")
    
    uplift_model = UpliftModel()
    
    # Prepare features
    exclude_cols = ['cohort_month', 'cohort_index']
    feature_cols = [c for c in features.columns if c not in exclude_cols]
    X = features[feature_cols].fillna(0)
    
    holdout_revenue = holdout_actuals.reindex(features.index)['holdout_revenue'].fillna(0)
    
    # Simulate treatment and fit
    X_np, treatment, y = uplift_model.simulate_treatment(X, holdout_revenue)
    uplift_model.fit_t_learner(X_np, treatment, y)
    
    # Predict uplift
    uplift_scores = uplift_model.predict_uplift(X_np)
    
    # Rank customers
    uplift_df = uplift_model.rank_by_uplift(X_np, features.index)
    
    # Save
    uplift_df.to_parquet(OUTPUT_DATA_DIR / "uplift_scores.parquet")
    uplift_model.save()
    
    logger.info(f"Uplift modeling complete!")
    logger.info(f"  Top 10% persuadable: {(uplift_df['uplift_rank'] <= 0.1).sum():,} customers")
    
    return uplift_model, uplift_df


def run_conformal_prediction(features=None, holdout_actuals=None, final_predictions=None):
    """Phase 6: Conformal prediction intervals."""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    logger.info("=" * 60)
    logger.info("PHASE 6: CONFORMAL PREDICTION INTERVALS")
    logger.info("=" * 60)
    
    from src.monitoring.conformal import ConformalCLVPredictor
    from src.evaluation.metrics import mae
    
    if features is None:
        features = pd.read_parquet(CUSTOMER_FEATURES_FILE)
    if holdout_actuals is None:
        holdout_actuals = pd.read_parquet(OUTPUT_DATA_DIR / "holdout_actuals.parquet")
    if final_predictions is None:
        final_predictions = pd.read_parquet(CLV_PREDICTIONS_FILE)
    
    # Prepare data
    common = features.index.intersection(holdout_actuals.index)
    
    prob_preds = final_predictions.reindex(features.index)
    exclude_cols = ['cohort_month', 'cohort_index']
    feature_cols = [c for c in features.columns if c not in exclude_cols]
    
    X_all = features.loc[common, feature_cols].fillna(0)
    # Add probabilistic features
    for col in ['p_alive', 'predicted_clv_bgnbd', 'predicted_clv_lgbm']:
        if col in prob_preds.columns:
            X_all[col] = prob_preds.reindex(common)[col].fillna(0)
    
    y_all = holdout_actuals.loc[common, 'holdout_revenue']
    
    X_np = X_all.values
    y_np = y_all.values
    
    # Split into train, calibration, test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_np, y_np, test_size=0.4, random_state=RANDOM_STATE
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )
    
    # Fit conformal model
    from lightgbm import LGBMRegressor
    base_model = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        random_state=RANDOM_STATE, verbosity=-1,
    )
    
    conformal = ConformalCLVPredictor(base_model, alpha=CONFORMAL_ALPHA)
    conformal.fit_and_conformalize(X_train, y_train, X_calib, y_calib)
    
    # Evaluate
    metrics = conformal.evaluate_coverage(y_test, X_test)
    
    # Predict for all customers
    X_full = features[feature_cols].fillna(0)
    for col in ['p_alive', 'predicted_clv_bgnbd', 'predicted_clv_lgbm']:
        if col in prob_preds.columns:
            X_full[col] = prob_preds.reindex(features.index)[col].fillna(0)
    
    result = conformal.predict_with_labels(X_full.values, customer_ids=features.index)
    
    # Merge with existing predictions
    if final_predictions is not None:
        for col in ['clv_lower', 'clv_upper', 'interval_width']:
            if col in result.columns:
                final_predictions[col] = result[col]
    
    # Save
    final_predictions.to_parquet(CLV_PREDICTIONS_FILE)
    conformal.save()
    
    logger.info("Conformal prediction complete!")
    logger.info(f"  Coverage: {metrics['empirical_coverage']:.1%}")
    logger.info(f"  Avg interval: {CURRENCY_SYMBOL}{metrics['avg_interval_width']:,.0f}")
    
    return conformal, final_predictions


def run_drift_monitoring(clean_df=None):
    """Phase 6b: PSI drift monitoring."""
    import pandas as pd
    
    logger.info("=" * 60)
    logger.info("PHASE 6b: DRIFT MONITORING (PSI)")
    logger.info("=" * 60)
    
    from src.monitoring.drift import DriftMonitor
    from src.config import COL_DATE
    
    if clean_df is None:
        clean_df = pd.read_parquet(CLEAN_TRANSACTIONS_FILE)
    
    monitor = DriftMonitor()
    
    # Features to monitor
    features_to_monitor = ['recency', 'frequency', 'monetary_value', 'T']
    
    # Build RFM per quarter
    from src.data.feature_engineering import FeatureEngineer
    
    quarters = pd.to_datetime(clean_df[COL_DATE]).dt.to_period('Q').unique()
    quarters = sorted(quarters)
    
    quarterly_rfm = {}
    for q in quarters:
        q_mask = pd.to_datetime(clean_df[COL_DATE]).dt.to_period('Q') == q
        q_df = clean_df[q_mask]
        if len(q_df) > 100:
            fe = FeatureEngineer(observation_end=q.end_time.to_pydatetime())
            try:
                rfm = fe.build_rfm_summary(q_df)
                quarterly_rfm[str(q)] = rfm
            except Exception as e:
                logger.warning(f"Skipping quarter {q}: {e}")
    
    quarter_keys = sorted(quarterly_rfm.keys())
    
    if len(quarter_keys) >= 2:
        # Compute PSI between consecutive quarters
        records = []
        for i in range(1, len(quarter_keys)):
            baseline_rfm = quarterly_rfm[quarter_keys[i - 1]]
            current_rfm = quarterly_rfm[quarter_keys[i]]
            
            psi_vals = monitor.compute_feature_psi(
                baseline_rfm, current_rfm, features_to_monitor
            )
            for feat, psi in psi_vals.items():
                records.append({
                    "quarter": quarter_keys[i],
                    "baseline_quarter": quarter_keys[i - 1],
                    "feature": feat,
                    "psi": psi,
                })
        
        quarterly_psi = pd.DataFrame(records)
        quarterly_psi.to_parquet(OUTPUT_DATA_DIR / "quarterly_psi.parquet")
        
        # Generate alerts for the latest quarter
        latest_psi = {
            r["feature"]: r["psi"] 
            for _, r in quarterly_psi[
                quarterly_psi["quarter"] == quarter_keys[-1]
            ].iterrows()
        }
        alerts = monitor.generate_alerts(latest_psi)
        for alert in alerts:
            logger.warning(f"  DRIFT ALERT: {alert}")
        
        if not alerts:
            logger.info("  No significant drift detected.")
    else:
        logger.warning("Not enough quarters for PSI computation.")
    
    logger.info("Drift monitoring complete!")


def run_all():
    """Run the complete pipeline."""
    start_time = time.time()
    
    logger.info("CLV Intelligence Engine - Full Pipeline")
    logger.info("=" * 60)
    
    # Phase 1: Data
    features, rfm, holdout_actuals, obs_df, holdout_df = run_data_pipeline()
    
    # Phase 2: Probabilistic models
    prob_model, prob_predictions = run_probabilistic_models(rfm, holdout_actuals)
    
    # Phase 3: ML models
    lgbm_model, stacked_model, final_predictions, feature_names = run_ml_models(
        features, prob_predictions, holdout_actuals
    )
    
    # Save final predictions
    final_predictions.to_parquet(CLV_PREDICTIONS_FILE)
    
    # Phase 4a: Segmentation
    seg_model, segment_df = run_segmentation(final_predictions)
    
    # Phase 4b: Uplift
    uplift_model, uplift_df = run_uplift_model(features, holdout_actuals)
    
    # Phase 6a: Conformal intervals
    conformal, final_predictions = run_conformal_prediction(
        features, holdout_actuals, final_predictions
    )
    
    # Phase 6b: Drift monitoring
    run_drift_monitoring()
    
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
    logger.info(f"{'='*60}")
    logger.info(f"\nTo launch the dashboard:")
    logger.info(f"  streamlit run dashboard/app.py")


def main():
    parser = argparse.ArgumentParser(
        description="CLV Intelligence Engine Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase", 
        choices=["all", "data", "probabilistic", "ml", "segmentation", 
                 "uplift", "conformal", "drift", "dashboard"],
        default="all",
        help="Which phase to run (default: all)"
    )
    
    args = parser.parse_args()
    
    if args.phase == "all":
        run_all()
    elif args.phase == "data":
        run_data_pipeline()
    elif args.phase == "probabilistic":
        run_probabilistic_models()
    elif args.phase == "ml":
        run_ml_models()
    elif args.phase == "segmentation":
        run_segmentation()
    elif args.phase == "uplift":
        run_uplift_model()
    elif args.phase == "conformal":
        run_conformal_prediction()
    elif args.phase == "drift":
        run_drift_monitoring()
    elif args.phase == "dashboard":
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(PROJECT_ROOT / "dashboard" / "app.py"),
            "--server.headless", "true",
        ])


if __name__ == "__main__":
    main()
