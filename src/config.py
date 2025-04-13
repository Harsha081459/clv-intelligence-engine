"""
Central configuration for the CLV Intelligence Engine.
All paths, constants, and hyperparameters in one place.
"""

from pathlib import Path
from datetime import datetime

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "output"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Dataset Configuration
# ============================================================
RAW_DATA_FILE = RAW_DATA_DIR / "online_retail_II.xlsx"
CLEAN_TRANSACTIONS_FILE = PROCESSED_DATA_DIR / "transactions_clean.parquet"
CUSTOMER_FEATURES_FILE = PROCESSED_DATA_DIR / "customer_features.parquet"
RFM_SUMMARY_FILE = PROCESSED_DATA_DIR / "rfm_summary.parquet"

# Output files
CLV_PREDICTIONS_FILE = OUTPUT_DATA_DIR / "clv_predictions.parquet"
CUSTOMER_SEGMENTS_FILE = OUTPUT_DATA_DIR / "customer_segments.parquet"
UPLIFT_SCORES_FILE = OUTPUT_DATA_DIR / "uplift_scores.parquet"

# ============================================================
# Column Name Mapping (actual file → standardized)
# ============================================================
# The Excel file uses different names than UCI documentation
COLUMN_RENAME_MAP = {
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
}

# Standardized column names (after renaming)
COL_INVOICE = "InvoiceNo"
COL_STOCK_CODE = "StockCode"
COL_DESCRIPTION = "Description"
COL_QUANTITY = "Quantity"
COL_DATE = "InvoiceDate"
COL_PRICE = "UnitPrice"
COL_CUSTOMER = "CustomerID"
COL_COUNTRY = "Country"
COL_TOTAL_PRICE = "TotalPrice"

# ============================================================
# Data Cleaning Parameters
# ============================================================
# Non-product stock codes to filter out
NON_PRODUCT_CODES = [
    "POST", "DOT", "M", "BANK CHARGES", "PADS", "C2", "D",
    "CRUK", "S", "AMAZONFEE", "B", "Adjust bad debt",
    "Manual", "Discount",
]

# Outlier threshold (percentile for capping)
OUTLIER_PERCENTILE = 99.9

# ============================================================
# Temporal Split Configuration
# ============================================================
# Observation window: first ~12 months (for fitting models)
OBSERVATION_START = datetime(2009, 12, 1)
OBSERVATION_END = datetime(2010, 12, 1)

# Holdout window: next ~12 months (for validation)
HOLDOUT_START = datetime(2010, 12, 2)
HOLDOUT_END = datetime(2011, 12, 9)

# ============================================================
# Currency Configuration
# ============================================================
# Original data is in GBP (£), we convert to INR (₹)
GBP_TO_INR = 105.0  # Approximate exchange rate
CURRENCY_SYMBOL = "₹"
CURRENCY_NAME = "INR"

# ============================================================
# Business Parameters
# ============================================================
# Gross margin assumption
GROSS_MARGIN = 0.60  # 60%

# Marketing campaign parameters
COST_PER_CONTACT_INR = 150  # ₹150 per customer (email + offer)
TOTAL_QUARTERLY_BUDGET_INR = 50_00_000  # ₹50 Lakhs = ₹50,00,000
EXPECTED_REVENUE_LIFT_INR = 800  # Expected per persuadable customer

# Discount rate for CLV calculation (monthly)
MONTHLY_DISCOUNT_RATE = 0.01

# ============================================================
# Model Hyperparameters
# ============================================================
# BG/NBD & Gamma-Gamma
PENALIZER_COEF = 0.01
CLV_PREDICTION_MONTHS = 12

# LightGBM defaults (will be tuned by Optuna)
LGBM_DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
}

# Optuna tuning
OPTUNA_N_TRIALS = 50
OPTUNA_CV_FOLDS = 5

# ============================================================
# Segmentation Parameters
# ============================================================
GMM_K_RANGE = range(5, 6)  # Force exactly k=5 segments
SEGMENT_NAMES = {
    "champions": "Champions",
    "at_risk_high_value": "At-Risk High-Value",
    "promising": "Promising",
    "hibernating": "Hibernating",
    "lost": "Lost",
}

# ============================================================
# Uplift Modeling Parameters
# ============================================================
TREATMENT_FRACTION = 0.30  # 30% of customers treated
SYNTHETIC_UPLIFT_EFFECT = 0.15  # 15% boost in purchase probability

# ============================================================
# Monitoring Thresholds
# ============================================================
PSI_THRESHOLD_LOW = 0.10    # Below this = stable
PSI_THRESHOLD_HIGH = 0.20   # Above this = significant drift

# Conformal prediction
CONFORMAL_COVERAGE = 0.90   # 90% prediction interval
CONFORMAL_ALPHA = 1 - CONFORMAL_COVERAGE  # 0.10

# ============================================================
# Random State
# ============================================================
RANDOM_STATE = 42
