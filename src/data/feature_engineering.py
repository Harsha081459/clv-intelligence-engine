"""
Feature Engineering Module
==========================
Builds customer-level features for CLV modeling.
Includes RFM features, behavioral features, and cohort assignment.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    COL_INVOICE, COL_STOCK_CODE, COL_QUANTITY, COL_PRICE,
    COL_CUSTOMER, COL_DATE, COL_TOTAL_PRICE, COL_COUNTRY,
    CUSTOMER_FEATURES_FILE, RFM_SUMMARY_FILE,
    OBSERVATION_END, GBP_TO_INR
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Constructs customer-level features from transaction data.
    
    Features built:
    - Core RFM: recency, frequency, monetary_value, T (customer age)
    - Behavioral: return_rate, category_diversity, basket_size, etc.
    - Temporal: weekend_ratio, purchase_regularity
    - Cohort: cohort_month, cohort_index
    """
    
    def __init__(self, observation_end: Optional[datetime] = None):
        """
        Parameters
        ----------
        observation_end : datetime, optional
            End of observation window. Used to compute recency and T.
            Defaults to config.OBSERVATION_END.
        """
        self.observation_end = observation_end or OBSERVATION_END
    
    def build_rfm_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build RFM summary table compatible with the lifetimes library.
        
        The lifetimes library expects:
        - frequency: number of REPEAT purchases (total orders - 1)
        - recency: time between first and last purchase (in days)
        - T: customer age = time between first purchase and observation end (in days)
        - monetary_value: average order value (across repeat purchases)
        
        Parameters
        ----------
        df : pd.DataFrame
            Cleaned transaction data (observation window only).
            
        Returns
        -------
        pd.DataFrame
            RFM summary indexed by CustomerID.
        """
        logger.info("Building RFM summary table...")
        
        # Aggregate to order level first (one row per invoice per customer)
        orders = (
            df.groupby([COL_CUSTOMER, COL_INVOICE, COL_DATE])
            [COL_TOTAL_PRICE].sum()
            .reset_index()
        )
        
        obs_end = pd.Timestamp(self.observation_end)
        
        # Customer-level aggregations
        customer_agg = orders.groupby(COL_CUSTOMER).agg(
            first_purchase=(COL_DATE, 'min'),
            last_purchase=(COL_DATE, 'max'),
            total_orders=(COL_INVOICE, 'nunique'),
            total_revenue=(COL_TOTAL_PRICE, 'sum'),
        )
        
        # RFM features (lifetimes convention)
        rfm = pd.DataFrame(index=customer_agg.index)
        
        # Frequency = number of REPEAT purchases (total - 1)
        rfm['frequency'] = customer_agg['total_orders'] - 1
        
        # Recency = days between first and last purchase
        rfm['recency'] = (
            customer_agg['last_purchase'] - customer_agg['first_purchase']
        ).dt.days.astype(float)
        
        # T = customer age = days from first purchase to observation end
        rfm['T'] = (
            obs_end - customer_agg['first_purchase']
        ).dt.days.astype(float)
        
        # Monetary value = average order value (for repeat customers only)
        # For customers with frequency=0 (single purchase), set monetary_value 
        # to their single order value for feature purposes
        repeat_customers = orders.groupby(COL_CUSTOMER).filter(
            lambda x: x[COL_INVOICE].nunique() > 1
        )
        
        if len(repeat_customers) > 0:
            avg_order_value = (
                repeat_customers.groupby(COL_CUSTOMER)[COL_TOTAL_PRICE]
                .mean()
            )
            rfm['monetary_value'] = avg_order_value
        
        # Fill NaN monetary_value for single-purchase customers with their single order value
        single_purchase_value = (
            orders.groupby(COL_CUSTOMER)[COL_TOTAL_PRICE].mean()
        )
        rfm['monetary_value'] = rfm['monetary_value'].fillna(single_purchase_value)
        
        # Also store total revenue for reference
        rfm['total_revenue'] = customer_agg['total_revenue']
        
        logger.info(f"RFM summary: {len(rfm):,} customers")
        logger.info(f"  Repeat customers (freq > 0): {(rfm['frequency'] > 0).sum():,}")
        logger.info(f"  Single-purchase customers: {(rfm['frequency'] == 0).sum():,}")
        logger.info(f"  Avg frequency: {rfm['frequency'].mean():.2f}")
        logger.info(f"  Avg recency: {rfm['recency'].mean():.1f} days")
        logger.info(f"  Avg monetary: ₹{rfm['monetary_value'].mean():,.0f}")
        
        return rfm
    
    def build_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build behavioral features from transaction data.
        
        Features:
        - category_diversity: Number of unique products purchased
        - avg_items_per_order: Average quantity per invoice
        - avg_basket_size: Average distinct products per invoice
        - weekend_ratio: Fraction of purchases on weekends
        - purchase_regularity: Coefficient of variation of inter-purchase intervals
        - days_between_purchases: Mean inter-purchase interval
        - country_is_uk: Binary flag for UK customers
        - avg_unit_price: Average price of items purchased
        - total_items: Total quantity of items purchased
        
        Parameters
        ----------
        df : pd.DataFrame
            Cleaned transaction data.
            
        Returns
        -------
        pd.DataFrame
            Behavioral features indexed by CustomerID.
        """
        logger.info("Building behavioral features...")
        
        features = pd.DataFrame(index=df[COL_CUSTOMER].unique())
        features.index.name = COL_CUSTOMER
        
        # ---- Category diversity ----
        cat_diversity = df.groupby(COL_CUSTOMER)[COL_STOCK_CODE].nunique()
        features['category_diversity'] = cat_diversity
        
        # ---- Items per order ----
        items_per_order = (
            df.groupby([COL_CUSTOMER, COL_INVOICE])[COL_QUANTITY].sum()
            .groupby(COL_CUSTOMER).mean()
        )
        features['avg_items_per_order'] = items_per_order
        
        # ---- Basket size (distinct products per order) ----
        basket_size = (
            df.groupby([COL_CUSTOMER, COL_INVOICE])[COL_STOCK_CODE].nunique()
            .groupby(COL_CUSTOMER).mean()
        )
        features['avg_basket_size'] = basket_size
        
        # ---- Weekend ratio ----
        df_temp = df.copy()
        df_temp['is_weekend'] = df_temp[COL_DATE].dt.dayofweek >= 5
        weekend_ratio = df_temp.groupby(COL_CUSTOMER)['is_weekend'].mean()
        features['weekend_ratio'] = weekend_ratio
        
        # ---- Purchase regularity (CV of inter-purchase intervals) ----
        order_dates = (
            df.groupby([COL_CUSTOMER, COL_INVOICE])[COL_DATE].first()
            .reset_index()
            .sort_values([COL_CUSTOMER, COL_DATE])
        )
        
        def compute_regularity(group):
            if len(group) < 2:
                return pd.Series({
                    'purchase_regularity': np.nan,
                    'days_between_purchases': np.nan
                })
            dates = group[COL_DATE].sort_values()
            intervals = dates.diff().dt.days.dropna()
            if len(intervals) == 0 or intervals.mean() == 0:
                return pd.Series({
                    'purchase_regularity': np.nan,
                    'days_between_purchases': np.nan
                })
            cv = intervals.std() / intervals.mean() if intervals.mean() > 0 else np.nan
            return pd.Series({
                'purchase_regularity': cv,
                'days_between_purchases': intervals.mean()
            })
        
        regularity = order_dates.groupby(COL_CUSTOMER).apply(
            compute_regularity, include_groups=False
        )
        features['purchase_regularity'] = regularity['purchase_regularity']
        features['days_between_purchases'] = regularity['days_between_purchases']
        
        # ---- Country ----
        country = df.groupby(COL_CUSTOMER)[COL_COUNTRY].first()
        features['country_is_uk'] = (country == 'United Kingdom').astype(int)
        
        # ---- Average unit price ----
        avg_price = df.groupby(COL_CUSTOMER)[COL_PRICE].mean()
        features['avg_unit_price'] = avg_price
        
        # ---- Total items purchased ----
        total_items = df.groupby(COL_CUSTOMER)[COL_QUANTITY].sum()
        features['total_items'] = total_items
        
        # Fill NaN values
        features['purchase_regularity'] = features['purchase_regularity'].fillna(0)
        features['days_between_purchases'] = features['days_between_purchases'].fillna(0)
        
        logger.info(f"Built {len(features.columns)} behavioral features for {len(features):,} customers")
        
        return features
    
    def build_cohort_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign cohort labels based on first purchase month.
        
        Parameters
        ----------
        df : pd.DataFrame
            Cleaned transaction data.
            
        Returns
        -------
        pd.DataFrame
            Cohort features indexed by CustomerID.
        """
        logger.info("Building cohort features...")
        
        first_purchase = df.groupby(COL_CUSTOMER)[COL_DATE].min()
        
        cohort = pd.DataFrame(index=first_purchase.index)
        cohort.index.name = COL_CUSTOMER
        
        # Cohort month (year-month of first purchase)
        cohort['cohort_month'] = first_purchase.dt.to_period('M')
        
        # Cohort index (months since first purchase relative to observation end)
        obs_period = pd.Timestamp(self.observation_end).to_period('M')
        cohort['cohort_index'] = (
            obs_period - cohort['cohort_month']
        ).apply(lambda x: x.n if hasattr(x, 'n') else 0)
        
        # Convert cohort_month to string for storage
        cohort['cohort_month'] = cohort['cohort_month'].astype(str)
        
        logger.info(f"Cohort range: {cohort['cohort_month'].min()} to {cohort['cohort_month'].max()}")
        logger.info(f"Cohort index range: {cohort['cohort_index'].min()} to {cohort['cohort_index'].max()}")
        
        return cohort
    
    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the complete customer feature matrix.
        
        Combines:
        - RFM summary (frequency, recency, T, monetary_value)
        - Behavioral features
        - Cohort features
        
        Parameters
        ----------
        df : pd.DataFrame
            Cleaned transaction data (observation window).
            
        Returns
        -------
        pd.DataFrame
            Complete customer feature matrix.
        """
        logger.info("Building complete customer feature matrix...")
        
        rfm = self.build_rfm_summary(df)
        behavioral = self.build_behavioral_features(df)
        cohort = self.build_cohort_features(df)
        
        # Merge all features
        features = rfm.join(behavioral, how='left').join(cohort, how='left')
        
        logger.info(f"\nComplete feature matrix: {features.shape}")
        logger.info(f"Features: {features.columns.tolist()}")
        
        return features
    
    def build_holdout_actuals(self, holdout_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute actual holdout period metrics for validation.
        
        Parameters
        ----------
        holdout_df : pd.DataFrame
            Transaction data from the holdout period.
            
        Returns
        -------
        pd.DataFrame
            Actual holdout metrics indexed by CustomerID:
            - holdout_transactions: number of orders in holdout
            - holdout_revenue: total revenue in holdout
        """
        logger.info("Computing holdout actuals...")
        
        orders = (
            holdout_df.groupby([COL_CUSTOMER, COL_INVOICE])
            [COL_TOTAL_PRICE].sum()
            .reset_index()
        )
        
        actuals = orders.groupby(COL_CUSTOMER).agg(
            holdout_transactions=(COL_INVOICE, 'nunique'),
            holdout_revenue=(COL_TOTAL_PRICE, 'sum')
        )
        
        logger.info(f"Holdout actuals for {len(actuals):,} customers")
        logger.info(f"  Avg transactions: {actuals['holdout_transactions'].mean():.2f}")
        logger.info(f"  Avg revenue: ₹{actuals['holdout_revenue'].mean():,.0f}")
        
        return actuals
    
    def save_features(
        self, 
        features: pd.DataFrame,
        rfm: Optional[pd.DataFrame] = None,
        features_path: Optional[Path] = None,
        rfm_path: Optional[Path] = None
    ):
        """Save feature files to parquet."""
        features_path = features_path or CUSTOMER_FEATURES_FILE
        features_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(features_path)
        logger.info(f"Saved customer features to {features_path}")
        
        if rfm is not None:
            rfm_path = rfm_path or RFM_SUMMARY_FILE
            rfm.to_parquet(rfm_path)
            logger.info(f"Saved RFM summary to {rfm_path}")


def build_cohort_retention_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a cohort retention matrix for visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction data.
        
    Returns
    -------
    pd.DataFrame
        Cohort retention matrix (rows=cohort, cols=period, values=retention %).
    """
    logger.info("Building cohort retention matrix...")
    
    # Assign cohort based on first purchase month
    df = df.copy()
    first_purchase = df.groupby(COL_CUSTOMER)[COL_DATE].min().dt.to_period('M')
    df['cohort'] = df[COL_CUSTOMER].map(first_purchase)
    
    # Transaction month
    df['order_period'] = df[COL_DATE].dt.to_period('M')
    
    # Cohort index (months since first purchase)
    df['cohort_index'] = (df['order_period'] - df['cohort']).apply(
        lambda x: x.n if hasattr(x, 'n') else 0
    )
    
    # Count unique customers per cohort per period
    cohort_data = (
        df.groupby(['cohort', 'cohort_index'])[COL_CUSTOMER]
        .nunique()
        .reset_index()
        .rename(columns={COL_CUSTOMER: 'customers'})
    )
    
    # Pivot to matrix
    retention_matrix = cohort_data.pivot(
        index='cohort', columns='cohort_index', values='customers'
    )
    
    # Convert to retention percentages (relative to cohort size at index 0)
    cohort_sizes = retention_matrix.iloc[:, 0]
    retention_pct = retention_matrix.divide(cohort_sizes, axis=0) * 100
    
    logger.info(f"Retention matrix shape: {retention_pct.shape}")
    
    return retention_pct


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    from src.data.loader import load_raw_data
    from src.data.preprocessor import DataPreprocessor
    
    # Load and clean
    raw_df = load_raw_data()
    preprocessor = DataPreprocessor(convert_to_inr=True)
    clean_df = preprocessor.clean(raw_df)
    
    # Split
    obs_df, holdout_df = preprocessor.temporal_split(clean_df)
    
    # Build features
    fe = FeatureEngineer()
    features = fe.build_all_features(obs_df)
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"\nFeature statistics:")
    print(features.describe())
    
    # Build holdout actuals
    actuals = fe.build_holdout_actuals(holdout_df)
    print(f"\nHoldout actuals shape: {actuals.shape}")
