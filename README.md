# 🔮 Customer Lifetime Value Intelligence Engine

> **Probabilistic CLV Forecasting + Marketing Spend Optimizer + Cohort Risk Dashboard**

A production-grade CLV prediction system that goes far beyond binary churn classification. This engine combines **probabilistic models** (BG/NBD + Gamma-Gamma), **gradient-boosted stacking** (LightGBM), **uplift-based marketing optimization**, **conformal prediction intervals**, and **drift monitoring** — all served through an interactive 4-tab Streamlit dashboard.

---

## 🎯 The Core Insight

Most data science projects predict *"will this customer churn?"* (binary).

**This project answers a harder, more valuable question:**

> *"How much revenue will each customer generate over the next 12 months, with calibrated uncertainty, and which customers should we target with our ₹50L retention budget to maximize ROI?"*

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAW TRANSACTION DATA                             │
│              UCI Online Retail II (1M+ transactions)                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  CLEANING   │  Drop cancellations, missing IDs,
                    │  & FEATURE  │  non-product codes, outliers
                    │  ENGINEERING│  Build RFM + 12 behavioral features
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌─────▼──────┐  ┌──────▼──────┐
   │   BG/NBD    │  │  Gamma-    │  │  LightGBM   │
   │   Model     │  │  Gamma     │  │  + Optuna   │
   │ (purchases) │  │  (spend)   │  │  (features) │
   └──────┬──────┘  └─────┬──────┘  └──────┬──────┘
          │               │                │
          └───────┬───────┘                │
                  │                        │
           ┌──────▼──────┐                 │
           │ Probabilistic│                │
           │    CLV       │                │
           └──────┬───────┘                │
                  │                        │
                  └──────────┬─────────────┘
                             │
                      ┌──────▼──────┐
                      │  STACKING   │  Ridge meta-learner
                      │  ENSEMBLE   │  on OOF predictions
                      └──────┬──────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
 ┌──────▼──────┐     ┌──────▼──────┐     ┌───────▼───────┐
 │  GMM-based  │     │   Uplift    │     │   Conformal   │
 │ Segmentation│     │  T-Learner  │     │  Prediction   │
 │ (5 cohorts) │     │ (causal)    │     │  Intervals    │
 └──────┬──────┘     └──────┬──────┘     └───────┬───────┘
        │                   │                    │
        └────────┬──────────┘                    │
                 │                               │
          ┌──────▼──────┐                 ┌──────▼──────┐
          │   BUDGET    │                 │    DRIFT    │
          │  OPTIMIZER  │                 │   MONITOR   │
          │ (greedy ROI)│                 │   (PSI)     │
          └──────┬──────┘                 └──────┬──────┘
                 │                               │
                 └───────────┬───────────────────┘
                             │
                    ┌────────▼────────┐
                    │   STREAMLIT     │
                    │   DASHBOARD     │
                    │   (4 tabs)      │
                    └─────────────────┘
```

---

## 📊 Model Comparison

| Model | MAE (₹) | RMSE (₹) | MAPE (%) | Pearson r | Description |
|-------|---------|----------|----------|-----------|-------------|
| BG/NBD + Gamma-Gamma | ₹1,04,836 | ₹4,09,761 | 114.9% | 0.8947 | Probabilistic baseline |
| LightGBM (Optuna-tuned) | ₹60,050 | ₹1,00,496 | 81.7% | 0.9898 | Feature-rich ML model |
| **Stacked Ensemble** | **₹59,138** | **₹97,160** | **78.9%** | **0.9902** | **Best: Ridge meta-learner** |

> Stacking reduces MAE by **43.6%** over the probabilistic baseline and achieves near-perfect correlation (r=0.99).
> Conformal prediction intervals achieve **85.6% empirical coverage** (90% target) with avg interval width ₹3,17,215.

---

## 🧠 ML Techniques

### Layer 1: Probabilistic Core
- **BG/NBD** (Beta-Geometric / Negative Binomial Distribution) — models purchase frequency and dropout probability simultaneously
- **Gamma-Gamma** — predicts expected average transaction value for alive customers
- **Combined CLV** = E[transactions] × E[avg order value] × gross margin

### Layer 2: ML Augmentation
- **LightGBM** with Bayesian hyperparameter tuning via **Optuna** (30 trials, 5-fold CV)
- **16 engineered features**: RFM core (4) + behavioral (9) + cohort (2) + total revenue
- **SHAP** explainability for feature importance

### Layer 3: Stacking Ensemble
- **Ridge meta-learner** trained on out-of-fold predictions from BG/NBD and LightGBM
- Prevents data leakage via proper OOF prediction collection

### Layer 4: Actionable Intelligence
- **GMM Segmentation** — BIC-optimal Gaussian Mixture Model clusters customers into: Champions, At-Risk High-Value, Promising, Hibernating, Lost
- **T-Learner Uplift Modeling** — causal inference to identify *persuadable* customers (not just likely buyers)
- **Budget Optimizer** — greedy ROI-maximizing allocation of ₹50L quarterly retention budget
- **Conformal Prediction** — distribution-free 90% coverage intervals (e.g., "CLV ₹4,200 (₹2,800–₹5,900)")
- **PSI Drift Monitoring** — quarterly Population Stability Index tracking with automated alerts

---

## 📁 Project Structure

```
ML_Project/
├── README.md
├── requirements.txt
├── run_pipeline.py              # Main pipeline orchestrator (CLI)
├── download_data.py             # Dataset download helper
│
├── src/
│   ├── config.py                # Central configuration (paths, params, constants)
│   ├── data/
│   │   ├── loader.py            # Excel/CSV data loading
│   │   ├── preprocessor.py      # Cleaning pipeline (7 steps)
│   │   └── feature_engineering.py  # RFM + behavioral + cohort features
│   ├── models/
│   │   ├── probabilistic.py     # BG/NBD + Gamma-Gamma (lifetimes)
│   │   ├── ml_model.py          # LightGBM/XGBoost + Optuna + SHAP
│   │   ├── stacking.py          # Ridge meta-learner ensemble
│   │   ├── segmentation.py      # GMM segmentation (BIC-optimal k)
│   │   └── uplift.py            # T-Learner uplift model
│   ├── optimization/
│   │   └── budget_allocator.py  # Greedy ROI budget optimizer
│   ├── monitoring/
│   │   ├── conformal.py         # MAPIE conformal prediction intervals
│   │   └── drift.py             # PSI drift monitoring + alerting
│   └── evaluation/
│       └── metrics.py           # MAE, RMSE, MAPE, Pearson r, decile lift, Qini
│
├── dashboard/
│   └── app.py                   # 4-tab Streamlit dashboard
│
├── notebooks/
│   ├── 01_eda_and_data_cleaning.ipynb
│   ├── 02_bgnbd_gamma_gamma.ipynb
│   ├── 03_ml_augmentation.ipynb
│   ├── 04_segmentation_uplift.ipynb
│   └── 05_uncertainty_monitoring.ipynb
│
├── data/
│   ├── raw/                     # Original Excel dataset
│   ├── processed/               # Clean parquet files
│   └── output/                  # Model predictions, segments, PSI
│
└── models/                      # Saved model artifacts (.pkl)
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone <repo-url>
cd ML_Project

# Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python download_data.py
```
Downloads the [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset (~43 MB).

### 3. Run the Full Pipeline
```bash
python run_pipeline.py --phase all
```

Available phases: `data`, `probabilistic`, `ml`, `segmentation`, `uplift`, `conformal`, `drift`, `dashboard`

### 4. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📈 Dashboard Preview

The Streamlit dashboard provides 4 interactive tabs:

| Tab | What It Shows |
|-----|---------------|
| **CLV Explorer** | Individual customer CLV lookup, distribution histograms, top customers table |
| **Segment Intelligence** | GMM cluster visualization, segment profiles, recommended actions per cohort |
| **Budget Optimizer** | ROI curves, scenario analysis (conservative/base/optimistic), optimal allocation breakdown |
| **Model Health** | PSI drift heatmap, conformal coverage validation, model comparison metrics |

---

## 📊 Dataset

**UCI Online Retail II** — Real transactional data from a UK-based online retailer (Dec 2009 – Dec 2011).

| Metric | Value |
|--------|-------|
| Raw transactions | 1,067,371 |
| After cleaning | ~775,847 (72.7% retained) |
| Unique customers | ~5,853 |
| Observation window | Dec 2009 – Dec 2010 (model training) |
| Holdout window | Dec 2010 – Dec 2011 (validation) |
| Currency | GBP → INR (×105) |

---

## 🔑 Key Findings

1. **BG/NBD achieves Pearson r = 0.8947** on holdout CLV prediction, validating the probabilistic framework for this dataset
2. **Stacked ensemble (r=0.9902) cuts MAE by 43.6%** — Ridge meta-learner with coefficients bgf=-0.13, lgbm=1.09 learns to weight LightGBM heavily while using BG/NBD as a corrective signal
3. **GMM identifies 5 optimal segments** (BIC-selected): Champions (e.g., avg CLV ₹3.6L), At-Risk High-Value, Promising, Hibernating, and Lost customers needing tailored re-engagement.
4. **Uplift modeling identifies 424 persuadable + 849 favourable customers** — these are the ones whose behavior changes due to marketing intervention, not just those who would buy anyway
5. **Conformal intervals achieve 85.6% empirical coverage** with avg width ₹3.17L — giving finance teams worst/best case revenue scenarios with mathematical guarantees

---

## 🛠️ Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Probabilistic CLV | lifetimes | 0.11.3 |
| Gradient Boosting | LightGBM | 4.6+ |
| Hyperparameter Tuning | Optuna | 4.8+ |
| Explainability | SHAP | 0.49+ |
| Conformal Prediction | MAPIE | 1.4+ |
| Dashboard | Streamlit | 1.57+ |
| Visualization | Plotly | 6.7+ |
| Data Processing | pandas, numpy, pyarrow | Latest |
| Clustering | scikit-learn (GMM) | 1.8+ |

---

## 💡 Why This Impresses Recruiters

**To a product ML team:**
> "This person understands that the business question is not 'will they churn' but 'how much revenue are they worth, and what should I do about it.' They built an optimizer, not just a classifier."

**To a data science interviewer:**
> BG/NBD is a proper probabilistic model with real math — Beta distributions and purchase process modeling — not just "I tuned XGBoost."

**To a fintech recruiter:**
> CLV connects directly to credit limit assignment, product upsell targeting, and customer acquisition cost justification.

**The uplift modeling angle is particularly rare.** Explaining the difference between "who will buy" and "who will buy *because of us*" signals business maturity.

---

## 📝 License

This project is for educational and portfolio purposes. Dataset sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

---

*Built with ❤️ as a portfolio project demonstrating production-grade ML engineering.*
