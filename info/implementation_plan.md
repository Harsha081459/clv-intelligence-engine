# CLV Intelligence Engine — Implementation Plan

> **Probabilistic CLV Forecasting + Marketing Spend Optimizer + Cohort Risk Dashboard**

## Overview

Build a production-grade Customer Lifetime Value prediction system that goes far beyond binary churn classification. The engine combines probabilistic models (BG/NBD + Gamma-Gamma), ML stacking (LightGBM), uplift-based marketing optimization, conformal prediction intervals, and drift monitoring — all served through a 4-tab Streamlit dashboard.

**Dataset:** UCI Online Retail II (~1M transactions, UK retailer, 2009–2011)

---

## User Review Required

> [!IMPORTANT]
> **Dataset Download:** The UCI Online Retail II dataset needs to be downloaded from Kaggle. I can either:
> - (A) Download it programmatically using `kagglehub` / `opendatasets` (requires Kaggle API credentials)
> - (B) You download it manually and place it in `d:\ML_Project\data\raw\`
>
> Please confirm which approach you prefer, or if you already have the dataset.

> [!IMPORTANT]
> **Currency:** The plan uses **£ (GBP)** throughout since the dataset is from a UK retailer. Your brief uses **₹ (INR)**. Should I:
> - (A) Keep everything in £ since that's the actual data currency
> - (B) Convert to ₹ using a fixed exchange rate for the dashboard display
> - (C) Make the currency configurable in the dashboard

> [!IMPORTANT]
> **Probabilistic model library:** Two options:
> - (A) **`lifetimes`** — Simple, fast (MLE-based), well-documented, but **archived/unmaintained**. Great for learning, simpler API.
> - (B) **`pymc-marketing`** — Actively maintained successor, Bayesian MCMC-based, gives full posterior uncertainty, supports covariates. Slower fitting, steeper learning curve.
> 
> **My recommendation:** Start with `lifetimes` for simplicity and speed. The API is nearly identical conceptually, and for a portfolio project the results will be the same. We can always migrate later.

> [!IMPORTANT]
> **Streamlit vs. Web Dashboard:** Your brief specifies Streamlit. Would you also want a separate polished HTML/CSS/JS dashboard for the portfolio, or is Streamlit sufficient?

---

## Open Questions

1. **Compute environment:** Are you running this on a local machine? Any GPU available for LightGBM/XGBoost? (CPU is fine, just affects training time)
2. **Python version:** Which Python version do you have? (3.9+ recommended)
3. **Virtual environment preference:** `venv`, `conda`, or `poetry`?
4. **MLflow:** Do you want local MLflow tracking, or skip it for simplicity?

---

## Technology Stack

| Component | Library | Rationale |
|-----------|---------|-----------|
| Probabilistic CLV | `lifetimes 0.11.3` | Stable, well-documented, perfect for BG/NBD + Gamma-Gamma |
| ML Models | `lightgbm` + `xgboost` | LightGBM primary (faster), XGBoost for comparison |
| Hyperparameter Tuning | `optuna` | Bayesian optimization, way better than grid search |
| Explainability | `shap` | SHAP waterfall plots for individual customer explanations |
| Conformal Intervals | `mapie` | Distribution-free prediction intervals |
| Uplift Modeling | Manual T-Learner | Simpler than `causalml`, demonstrates deeper understanding |
| Segmentation | `scikit-learn` (GaussianMixture) | GMM for soft clustering |
| Budget Optimization | Greedy + `scipy.optimize` | Greedy approach is more interpretable than LP for this use case |
| Dashboard | `streamlit` | Fast prototyping, interactive widgets, tab layout |
| Visualization | `plotly` + `seaborn` + `matplotlib` | Plotly for interactive charts, seaborn for EDA |
| Experiment Tracking | `mlflow` | Optional but nice for documenting model runs |
| Data | `pandas` + `numpy` | Standard |

---

## Project Structure

```
d:\ML_Project\
├── README.md
├── requirements.txt
├── setup.py                          # Optional package setup
├── .gitignore
│
├── data/
│   ├── raw/                          # Original dataset (gitignored)
│   │   └── online_retail_II.xlsx
│   ├── processed/                    # Cleaned data
│   │   ├── transactions_clean.parquet
│   │   └── customer_features.parquet
│   └── output/                       # Model predictions
│       ├── clv_predictions.parquet
│       └── customer_segments.parquet
│
├── notebooks/                        # EDA & exploration
│   ├── 01_eda_and_data_cleaning.ipynb
│   ├── 02_bgnbd_gamma_gamma.ipynb
│   ├── 03_ml_augmentation.ipynb
│   ├── 04_segmentation_uplift.ipynb
│   └── 05_uncertainty_monitoring.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Project configuration & constants
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Data loading & raw ingestion
│   │   ├── preprocessor.py           # Cleaning, filtering, deduplication
│   │   └── feature_engineering.py    # RFM features + behavioral features
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── probabilistic.py          # BG/NBD + Gamma-Gamma wrapper
│   │   ├── ml_model.py               # LightGBM / XGBoost training
│   │   ├── stacking.py               # Meta-learner (stacked ensemble)
│   │   ├── segmentation.py           # GMM clustering
│   │   └── uplift.py                 # T-Learner uplift model
│   │
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── budget_allocator.py       # Greedy budget allocation + ROI curves
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── conformal.py              # MAPIE conformal prediction intervals
│   │   └── drift.py                  # PSI computation + drift alerts
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                # MAE, RMSE, MAPE, Qini, calibration
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── plots.py                  # Reusable plot functions
│       └── cohort_heatmap.py         # Cohort retention heatmap
│
├── dashboard/
│   ├── app.py                        # Main Streamlit app
│   ├── tabs/
│   │   ├── __init__.py
│   │   ├── clv_explorer.py           # Tab 1: CLV Explorer
│   │   ├── segment_intelligence.py   # Tab 2: Segment Intelligence
│   │   ├── budget_optimizer.py       # Tab 3: Budget Optimizer
│   │   └── model_health.py           # Tab 4: Model Health Monitor
│   └── components/
│       ├── __init__.py
│       ├── kpi_cards.py              # Metric card components
│       └── filters.py                # Sidebar filter components
│
├── models/                           # Saved model artifacts
│   ├── bgnbd_model.pkl
│   ├── gamma_gamma_model.pkl
│   ├── lgbm_model.pkl
│   └── stacked_model.pkl
│
├── tests/
│   ├── test_preprocessor.py
│   ├── test_feature_engineering.py
│   └── test_metrics.py
│
└── mlruns/                           # MLflow tracking (gitignored)
```

---

## Proposed Changes — Phase by Phase

---

### Phase 1: Project Setup & Data Foundation

> Data loading, cleaning, EDA, feature engineering

#### [NEW] [requirements.txt](file:///d:/ML_Project/requirements.txt)
All project dependencies with pinned versions:
```
pandas>=2.0
numpy>=1.24,<2.0
scikit-learn>=1.3
lifetimes>=0.11.3
lightgbm>=4.0
xgboost>=2.0
shap>=0.44
mapie>=0.9
optuna>=3.5
mlflow>=2.10
streamlit>=1.30
plotly>=5.18
seaborn>=0.13
matplotlib>=3.8
scipy>=1.11
openpyxl>=3.1
pyarrow>=14.0
```

#### [NEW] [config.py](file:///d:/ML_Project/src/config.py)
Central configuration:
- File paths, date boundaries
- Observation window: `2009-12-01` to `2010-12-01` (12 months)
- Holdout window: `2010-12-02` to `2011-12-09` (~12 months)
- Margin assumption: 60%
- Cost-per-contact: £3 (≈₹150 equivalent)
- Budget range for scenarios

#### [NEW] [loader.py](file:///d:/ML_Project/src/data/loader.py)
- Load from Excel (both sheets) or CSV
- Combine into single DataFrame
- Parse `InvoiceDate` to datetime
- Initial type casting (`Customer ID` → int)

#### [NEW] [preprocessor.py](file:///d:/ML_Project/src/data/preprocessor.py)
Data cleaning pipeline:

> [!NOTE]
> **Column naming gotcha:** The actual Excel file uses `Invoice` (not `InvoiceNo`), `Price` (not `UnitPrice`), and `Customer ID` (with a space, not `CustomerID`). We'll rename columns early to standardize.

1. Load both Excel sheets (`Year 2009-2010` and `Year 2010-2011`) and concatenate
2. Rename columns: `Invoice` → `InvoiceNo`, `Price` → `UnitPrice`, `Customer ID` → `CustomerID`
3. Drop rows with missing `CustomerID` (~25% of data, ~240K+ rows)
4. Convert `CustomerID` from float to int
5. Remove cancellations (InvoiceNo starting with 'C')
6. Filter `Quantity > 0` and `UnitPrice > 0`
7. Remove non-product StockCodes: `['POST', 'DOT', 'M', 'BANK CHARGES', 'PADS', 'C2', 'D', 'CRUK', 'S', 'Adjust bad debt', 'Manual', 'Discount']`
8. Create `TotalPrice = Quantity × UnitPrice`
9. Drop exact duplicate rows
10. Cap extreme outliers (>99.9th percentile on TotalPrice)

**Expected result:** ~530K rows, ~4,300–5,900 unique customers

#### [NEW] [feature_engineering.py](file:///d:/ML_Project/src/data/feature_engineering.py)
Build customer-level feature matrix:

**Core RFM features:**
- `recency`: Days since last purchase (relative to observation end)
- `frequency`: Number of repeat purchases (total orders - 1, per lifetimes convention)
- `monetary_value`: Average order value
- `T`: Customer age in days (first purchase to observation end)
- `total_revenue`: Sum of all order values

**Behavioral features:**
- `return_rate`: Fraction of orders that were later cancelled (matched by customer)
- `category_diversity`: Number of unique StockCodes purchased
- `avg_items_per_order`: Average quantity per invoice
- `avg_basket_size`: Average number of distinct products per invoice
- `weekend_ratio`: Fraction of purchases on weekends
- `purchase_regularity`: Std dev of inter-purchase intervals (lower = more regular)
- `days_between_purchases`: Mean inter-purchase interval
- `country_is_uk`: Binary flag

**Cohort features:**
- `cohort_month`: First purchase month (for cohort analysis)
- `cohort_index`: Months since first purchase

#### [NEW] [01_eda_and_data_cleaning.ipynb](file:///d:/ML_Project/notebooks/01_eda_and_data_cleaning.ipynb)
EDA notebook covering:
- Transaction frequency distribution (expect power-law / heavy-tail)
- Inter-purchase time distribution
- Revenue concentration (Pareto analysis: what % of customers = 80% of revenue?)
- **Cohort retention heatmap** — the key visual
- RFM distribution plots
- Country breakdown
- Monthly revenue trend

---

### Phase 2: Probabilistic CLV Models (BG/NBD + Gamma-Gamma)

> The unique core of the project — probabilistic purchase modeling

#### [NEW] [probabilistic.py](file:///d:/ML_Project/src/models/probabilistic.py)
Wrapper class `ProbabilisticCLV`:

```python
class ProbabilisticCLV:
    def __init__(self, penalizer_coef=0.01, margin=0.6):
        self.bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
        self.ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
        self.margin = margin

    def fit(self, rfm_data):
        """Fit BG/NBD on (frequency, recency, T) and Gamma-Gamma on monetary"""

    def predict_purchases(self, t=365):
        """Predict expected number of purchases in next t days"""

    def predict_alive_probability(self):
        """Compute P(alive) for each customer"""

    def predict_clv(self, months=12, discount_rate=0.01):
        """Combined CLV = E[purchases] × E[avg_value] × margin"""

    def plot_calibration(self, holdout_data):
        """Expected vs actual transactions by frequency group"""

    def plot_frequency_recency_matrix(self):
        """Heatmap: expected purchases by frequency & recency"""
```

**Key validation:**
- Split at `2010-12-01`: observation (before) vs holdout (after)
- Compare predicted vs actual purchases in holdout
- Pearson correlation target: r > 0.65
- MAE on transaction count

#### [NEW] [02_bgnbd_gamma_gamma.ipynb](file:///d:/ML_Project/notebooks/02_bgnbd_gamma_gamma.ipynb)
- Fit BG/NBD, inspect parameters (r, α, a, b)
- Probability alive distribution
- Frequency-recency matrix
- Calibration: predicted vs actual in holdout
- Gamma-Gamma fit and validation
- Combined CLV prediction
- Top 20 most valuable customers table

---

### Phase 3: ML Augmentation (LightGBM Stacking)

> Boost probabilistic predictions with gradient boosting

#### [NEW] [ml_model.py](file:///d:/ML_Project/src/models/ml_model.py)
- Feature matrix: RFM + behavioral + BG/NBD outputs (`p_alive`, `expected_purchases`) as features
- Target: actual holdout revenue (regression)
- LightGBM with Optuna hyperparameter tuning (5-fold CV)
- XGBoost as comparison baseline
- SHAP value computation and storage

#### [NEW] [stacking.py](file:///d:/ML_Project/src/models/stacking.py)
Meta-learner stacking:
- Level-0: BG/NBD CLV prediction + LightGBM prediction
- Level-1: Ridge regression (simple, avoids overfitting on 2 features)
- Out-of-fold predictions to avoid data leakage
- **Key result:** Stacked model should beat both individual models by >10% MAE

#### [NEW] [metrics.py](file:///d:/ML_Project/src/evaluation/metrics.py)
- MAE, RMSE, MAPE for CLV regression
- Pearson correlation
- Decile lift chart (top 10% predicted CLV vs actual)
- Qini coefficient for uplift model
- Calibration metrics for conformal intervals

#### [NEW] [03_ml_augmentation.ipynb](file:///d:/ML_Project/notebooks/03_ml_augmentation.ipynb)
- Feature importance analysis
- Optuna optimization log
- Model comparison table: BG/NBD vs LightGBM vs Stacked
- SHAP waterfall plots for 3 representative customers:
  - Champion (high CLV, high p_alive)
  - At-Risk (high CLV, low p_alive)
  - Low-Value (low CLV)
- SHAP summary (beeswarm) plot

---

### Phase 4: Segmentation + Uplift Modeling

> Actionable customer tiers + causal marketing effect estimation

#### [NEW] [segmentation.py](file:///d:/ML_Project/src/models/segmentation.py)
- Features for clustering: `(log_predicted_CLV, p_alive, recency_days)`
- StandardScaler normalization
- Gaussian Mixture Model with BIC for k selection (try k=3,4,5,6,7)
- Segment naming logic based on centroid characteristics:
  - **Champions:** High CLV, high p_alive
  - **At-Risk High-Value:** High CLV, low p_alive ← **most actionable**
  - **Promising:** Medium CLV, high p_alive
  - **Hibernating:** Low CLV, low p_alive
  - **Lost:** Very low p_alive, very low CLV

#### [NEW] [uplift.py](file:///d:/ML_Project/src/models/uplift.py)
Manual T-Learner implementation:
1. **Simulate treatment assignment:** Randomly assign 30% of customers as "treated"
2. **Simulate treatment effect:** Treated customers get a synthetic 15% boost in 30-day purchase probability (clearly documented as simulation)
3. **T-Learner:**
   - Model 1 (treatment): XGBRegressor trained on treated group
   - Model 0 (control): XGBRegressor trained on control group
   - `uplift_score = model_1.predict(X) - model_0.predict(X)`
4. **Rank by uplift** — identify "persuadables" (top decile)
5. **Key insight to highlight:** High CLV ≠ high uplift. Some high-CLV customers are "always-buyers" who don't need campaigns.

#### [NEW] [04_segmentation_uplift.ipynb](file:///d:/ML_Project/notebooks/04_segmentation_uplift.ipynb)
- BIC curve for GMM k selection
- 2D scatter: p_alive vs predicted CLV, colored by segment
- Segment profile table (count, avg CLV, avg recency, recommended action)
- Uplift distribution by segment
- Qini curve: uplift model vs random
- Persuadable vs always-buyer analysis

---

### Phase 5: Budget Optimization Engine

> Business impact quantification — the "so what?" of the project

#### [NEW] [budget_allocator.py](file:///d:/ML_Project/src/optimization/budget_allocator.py)
```python
class BudgetAllocator:
    def __init__(self, cost_per_contact, uplift_scores, expected_revenue):
        ...

    def optimize_greedy(self, total_budget):
        """Greedy allocation: sort by expected_uplift_revenue / cost, allocate top-down"""
        # Returns: selected customers, expected ROI, incremental revenue

    def roi_curve(self, budget_range):
        """Generate ROI curve: budget spent vs expected incremental revenue"""
        # For each budget level, compute optimal allocation and expected return

    def scenario_analysis(self, total_budget, scenarios={'conservative': 0.7, 'base': 1.0, 'optimistic': 1.3}):
        """Run allocation under different uplift multipliers"""
```

**Key output:** "Targeting the top 8,000 customers at £3 each (£24K budget) is expected to generate £136K in incremental revenue — 5.6× ROI"

---

### Phase 6: Uncertainty Quantification + Monitoring

> Production readiness — calibrated intervals + drift detection

#### [NEW] [conformal.py](file:///d:/ML_Project/src/monitoring/conformal.py)

> [!WARNING]
> **MAPIE v1.0 breaking change (May 2025):** The old `MapieRegressor` API is deprecated. We must use the new `SplitConformalRegressor` class with separate `.fit()` and `.conformalize()` steps.

```python
from mapie.regression import SplitConformalRegressor

# New API: fit + conformalize are separate steps
estimator = SplitConformalRegressor(estimator=stacked_model)
estimator.fit(X_train, y_train)
estimator.conformalize(X_calib, y_calib)  # Separate calibration set needed
y_pred, y_pis = estimator.predict(X_test, alpha=0.1)  # 90% interval
```

- Split data into train/calibration/test (need explicit calibration set)
- 90% coverage target → validate on holdout (expect 88–92%)
- Output: `(predicted_clv, clv_lower, clv_upper)` for each customer
- Business framing: "Customer A: expected CLV £420 (£280–£590 at 90% confidence)"

#### [NEW] [drift.py](file:///d:/ML_Project/src/monitoring/drift.py)
- **PSI (Population Stability Index)** computation:
  ```
  PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)
  ```
- Compute PSI per feature across quarterly cohorts (Q1 vs Q2, Q2 vs Q3, etc.)
- Thresholds: PSI < 0.1 (stable), 0.1–0.2 (moderate drift), >0.2 (significant drift)
- CLV distribution comparison across cohorts
- Alert generation for dashboard

#### [NEW] [05_uncertainty_monitoring.ipynb](file:///d:/ML_Project/notebooks/05_uncertainty_monitoring.ipynb)
- Conformal interval calibration plot
- Coverage check on holdout
- PSI heatmap across features and quarters
- CLV distribution shift visualization

---

### Phase 7: Streamlit Dashboard

> The showcase — 4-tab interactive demo

#### [NEW] [app.py](file:///d:/ML_Project/dashboard/app.py)
Main app with `st.tabs()`:
- Page config, theming (dark mode)
- Sidebar with data upload option + filters
- Cache model loading with `@st.cache_resource`

#### [NEW] [clv_explorer.py](file:///d:/ML_Project/dashboard/tabs/clv_explorer.py)
**Tab 1: CLV Explorer**
- Interactive table: customer ID, segment, predicted CLV, p_alive, CLV interval
- Filter by segment, sort by any column
- Click a customer → SHAP waterfall explanation
- KPI cards: total predicted revenue, avg CLV, % at risk

#### [NEW] [segment_intelligence.py](file:///d:/ML_Project/dashboard/tabs/segment_intelligence.py)
**Tab 2: Segment Intelligence**
- Plotly scatter: p_alive vs CLV colored by segment
- Segment summary cards with recommended actions
- Cohort retention heatmap (Plotly heatmap)
- Segment migration Sankey diagram (stretch goal)

#### [NEW] [budget_optimizer.py](file:///d:/ML_Project/dashboard/tabs/budget_optimizer.py)
**Tab 3: Budget Optimizer**
- Slider: total marketing budget (£0 to £100K)
- Output: optimal allocation table, expected ROI
- ROI curve chart (Plotly)
- Scenario toggle: conservative / base / optimistic
- Segment-level allocation breakdown

#### [NEW] [model_health.py](file:///d:/ML_Project/dashboard/tabs/model_health.py)
**Tab 4: Model Health Monitor**
- PSI bar chart per feature over quarters
- CLV distribution overlay: current vs. baseline cohort
- Alert banner if drift detected (PSI > 0.2)
- Model performance metrics table

---

### Phase 8: Documentation & Polish

#### [NEW] [README.md](file:///d:/ML_Project/README.md)
Professional GitHub README with:
- Project title + tagline
- Architecture diagram (Mermaid)
- Model comparison results table
- Key features list
- Screenshots of each dashboard tab
- Setup instructions
- Dataset info
- Key findings (3 interesting discoveries)
- Future work

#### [NEW] [.gitignore](file:///d:/ML_Project/.gitignore)
Standard Python .gitignore + data files + mlruns

---

## Verification Plan

### Automated Tests

```bash
# Run unit tests
python -m pytest tests/ -v

# Validate model performance
python -m pytest tests/test_model_performance.py -v
```

**Test coverage:**
- `test_preprocessor.py`: Verify cleaning removes cancellations, null IDs, non-product codes
- `test_feature_engineering.py`: Verify RFM calculations on synthetic data
- `test_metrics.py`: Verify metric functions against known values

### Model Validation Checks

| Check | Method | Target |
|-------|--------|--------|
| BG/NBD calibration | Predicted vs actual transactions (holdout) | Pearson r > 0.65 |
| CLV regression | MAE, MAPE on holdout revenue | MAPE < 30% |
| Stacking improvement | MAE comparison vs individual models | >10% improvement |
| Segmentation quality | Silhouette score | > 0.45 |
| Uplift model | Qini coefficient | > 0.15 |
| Conformal intervals | Empirical coverage at 90% | 88–92% |
| PSI drift detection | Correctly flags synthetic drift | 100% (rule-based) |

### Dashboard Verification
- Launch `streamlit run dashboard/app.py`
- Verify all 4 tabs render without errors
- Test interactive widgets (sliders, filters, table sorting)
- Verify SHAP waterfall renders for selected customer
- Test budget optimizer with different inputs

### Manual Verification
- Review cohort retention heatmap for visual correctness
- Verify segment names align with centroid characteristics
- Check ROI numbers make business sense
- Confirm conformal intervals are not too wide (useful) or too narrow (miscalibrated)

---

## Execution Order

| Phase | Description | Estimated Effort |
|-------|-------------|-----------------|
| 1 | Project setup, data loading, cleaning, EDA, feature engineering | First |
| 2 | BG/NBD + Gamma-Gamma probabilistic models | After Phase 1 |
| 3 | LightGBM + stacking + SHAP | After Phase 2 |
| 4 | GMM segmentation + T-Learner uplift | After Phase 3 |
| 5 | Budget optimization engine | After Phase 4 |
| 6 | Conformal intervals + PSI monitoring | After Phase 3 (parallel with 4-5) |
| 7 | Streamlit dashboard | After Phases 4-6 |
| 8 | README, documentation, polish | Last |
