import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

# Configure plot style
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Set up directories
PROJECT_ROOT = 'd:/ML_Project'
sys.path.insert(0, PROJECT_ROOT)
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

from src.config import OUTPUT_DATA_DIR, MODELS_DIR

print("Loading data...")
comparison = pd.read_parquet(OUTPUT_DATA_DIR / 'model_comparison.parquet')
clv_preds = pd.read_parquet(OUTPUT_DATA_DIR / 'clv_predictions.parquet')
holdout = pd.read_parquet(OUTPUT_DATA_DIR / 'holdout_actuals.parquet')
segments = pd.read_parquet(OUTPUT_DATA_DIR / 'customer_segments.parquet')
uplift = pd.read_parquet(OUTPUT_DATA_DIR / 'uplift_scores.parquet')
shap_data = joblib.load(MODELS_DIR / 'shap_values.pkl')

print("1. Generating Model Comparison Plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2196F3', '#4CAF50', '#FF9800']
comparison['MAE'].plot(kind='bar', ax=axes[0], color=colors, edgecolor='white')
axes[0].set_title('MAE Comparison (Lower is Better)', fontsize=13)
axes[0].set_ylabel('MAE (INR)')
axes[0].tick_params(axis='x', rotation=0)

comparison['Pearson_r'].plot(kind='bar', ax=axes[1], color=colors, edgecolor='white')
axes[1].set_title('Pearson Correlation (Higher is Better)', fontsize=13)
axes[1].set_ylabel('Pearson r')
axes[1].set_ylim(0.85, 1.0)
axes[1].tick_params(axis='x', rotation=0)

plt.suptitle('Model Performance Comparison', fontsize=15, y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print("2. Generating Predicted vs Actual Plot...")
valid_idx = clv_preds.index.intersection(holdout.index)
y_true = holdout.loc[valid_idx, 'holdout_revenue'].fillna(0)
y_pred = clv_preds.loc[valid_idx, 'predicted_clv']

plt.figure(figsize=(8, 8))
plt.scatter(y_pred, y_true, alpha=0.3, s=15, color='#4CAF50')
max_val = max(y_pred.quantile(0.99), y_true.quantile(0.99))
plt.plot([0, max_val], [0, max_val], 'r--', lw=2)
plt.xlabel('Predicted CLV (INR)', fontsize=12)
plt.ylabel('Actual Holdout Revenue (INR)', fontsize=12)
plt.title('Stacked Ensemble: Predicted vs Actual', fontsize=14)
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'predicted_vs_actual.png'), dpi=300, bbox_inches='tight')
plt.close()

print("3. Generating SHAP Feature Importance Plot...")
shap_values = shap_data['shap_values']
shap_values.feature_names = shap_data['feature_names']
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title('SHAP Feature Importance (LightGBM)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

print("4. Generating Segmentation Plot...")
plot_df = segments[segments['predicted_clv'] < segments['predicted_clv'].quantile(0.98)]
plt.figure(figsize=(10, 7))
sns.scatterplot(data=plot_df, x='p_alive', y='predicted_clv', hue='segment_name', palette='tab10', alpha=0.6)
plt.title('Customer Segments: P(Alive) vs CLV', fontsize=14)
plt.xlabel('Probability of being Alive')
plt.ylabel('Predicted CLV (INR)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'segmentation_scatter.png'), dpi=300, bbox_inches='tight')
plt.close()

print("5. Generating Conformal Intervals Plot...")
sample = clv_preds.nlargest(40, 'predicted_clv').iloc[10:30].reset_index()
plt.figure(figsize=(12, 6))
plt.plot(range(len(sample)), sample['predicted_clv'], 'ko', label='Point Estimate')
plt.plot(range(len(sample)), sample['clv_upper'], '-', color='red', alpha=0.5, label='Upper Bound (90%)')
plt.plot(range(len(sample)), sample['clv_lower'], '-', color='red', alpha=0.5, label='Lower Bound (90%)')
plt.fill_between(range(len(sample)), sample['clv_lower'], sample['clv_upper'], color='red', alpha=0.1)
plt.title('Conformal Prediction Intervals (Sample High-Value Customers)', fontsize=14)
plt.xlabel('Customer Index')
plt.ylabel('CLV (INR)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'conformal_intervals.png'), dpi=300, bbox_inches='tight')
plt.close()

print("6. Generating Uplift Distribution Plot...")
plt.figure(figsize=(10, 6))
sns.histplot(data=uplift, x='uplift_score', hue='segment', bins=50, multiple="stack", palette='viridis')
plt.title('Uplift Score Distribution by Targeting Segment', fontsize=14)
plt.xlabel('Predicted Incremental Revenue (INR)')
plt.ylabel('Count')
plt.xlim(uplift['uplift_score'].quantile(0.01), uplift['uplift_score'].quantile(0.99))
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'uplift_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

print("All plots generated successfully in d:/ML_Project/plots/")
