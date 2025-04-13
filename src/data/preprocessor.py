"""
Data Preprocessor Module
========================
Cleans and transforms raw transaction data for CLV analysis.
Handles cancellations, missing values, outliers, and non-product entries.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    COL_INVOICE, COL_STOCK_CODE, COL_QUANTITY, COL_PRICE,
    COL_CUSTOMER, COL_DATE, COL_TOTAL_PRICE, COL_COUNTRY,
    NON_PRODUCT_CODES, OUTLIER_PERCENTILE,
    OBSERVATION_START, OBSERVATION_END, HOLDOUT_START, HOLDOUT_END,
    GBP_TO_INR, CLEAN_TRANSACTIONS_FILE
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Cleans raw transaction data for CLV analysis.
    
    Pipeline:
    1. Remove duplicates
    2. Drop missing CustomerIDs
    3. Remove cancellations (Invoice starts with 'C')
    4. Filter invalid quantities/prices
    5. Remove non-product stock codes
    6. Create TotalPrice column
    7. Convert GBP to INR
    8. Cap outliers
    """
    
    def __init__(self, convert_to_inr: bool = True):
        """
        Parameters
        ----------
        convert_to_inr : bool
            If True, convert GBP prices to INR.
        """
        self.convert_to_inr = convert_to_inr
        self.cleaning_report = {}
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full cleaning pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction data with standardized column names.
            
        Returns
        -------
        pd.DataFrame
            Cleaned transaction data.
        """
        initial_rows = len(df)
        self.cleaning_report["initial_rows"] = initial_rows
        logger.info(f"Starting cleaning pipeline with {initial_rows:,} rows...")
        
        # Step 1: Remove exact duplicates
        df[COL_STOCK_CODE] = df[COL_STOCK_CODE].astype(str)
        df[COL_INVOICE] = df[COL_INVOICE].astype(str)
        df = df.drop_duplicates()
        self._log_step("remove_duplicates", initial_rows, len(df))
        
        # Step 2: Drop rows with missing CustomerID
        rows_before = len(df)
        df = df.dropna(subset=[COL_CUSTOMER])
        df[COL_CUSTOMER] = df[COL_CUSTOMER].astype(int)
        self._log_step("drop_missing_customer", rows_before, len(df))
        
        # Step 3: Remove cancelled orders (Invoice starts with 'C')
        rows_before = len(df)
        df = df[~df[COL_INVOICE].astype(str).str.startswith('C')]
        self._log_step("remove_cancellations", rows_before, len(df))
        
        # Step 4: Filter invalid quantities and prices
        rows_before = len(df)
        df = df[(df[COL_QUANTITY] > 0) & (df[COL_PRICE] > 0)]
        self._log_step("filter_invalid_qty_price", rows_before, len(df))
        
        # Step 5: Remove non-product stock codes
        rows_before = len(df)
        # Also remove stock codes that don't look like product codes
        df = df[~df[COL_STOCK_CODE].astype(str).str.upper().isin(
            [code.upper() for code in NON_PRODUCT_CODES]
        )]
        self._log_step("remove_non_products", rows_before, len(df))
        
        # Step 6: Create TotalPrice column
        df[COL_TOTAL_PRICE] = df[COL_QUANTITY] * df[COL_PRICE]
        
        # Step 7: Convert to INR if configured
        if self.convert_to_inr:
            df[COL_PRICE] = df[COL_PRICE] * GBP_TO_INR
            df[COL_TOTAL_PRICE] = df[COL_TOTAL_PRICE] * GBP_TO_INR
            logger.info(f"Converted prices to INR (rate: 1 GBP = {GBP_TO_INR} INR)")
        
        # Step 8: Cap extreme outliers on TotalPrice
        rows_before = len(df)
        upper_cap = df[COL_TOTAL_PRICE].quantile(OUTLIER_PERCENTILE / 100)
        df = df[df[COL_TOTAL_PRICE] <= upper_cap]
        self._log_step("cap_outliers", rows_before, len(df))
        
        # Sort by date
        df = df.sort_values([COL_CUSTOMER, COL_DATE]).reset_index(drop=True)
        
        # Final stats
        self.cleaning_report["final_rows"] = len(df)
        self.cleaning_report["unique_customers"] = df[COL_CUSTOMER].nunique()
        self.cleaning_report["unique_invoices"] = df[COL_INVOICE].nunique()
        self.cleaning_report["date_range"] = (
            df[COL_DATE].min().strftime("%Y-%m-%d"),
            df[COL_DATE].max().strftime("%Y-%m-%d")
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Cleaning complete: {initial_rows:,} → {len(df):,} rows ({len(df)/initial_rows:.1%} retained)")
        logger.info(f"Unique customers: {self.cleaning_report['unique_customers']:,}")
        logger.info(f"Unique invoices: {self.cleaning_report['unique_invoices']:,}")
        logger.info(f"Date range: {self.cleaning_report['date_range'][0]} to {self.cleaning_report['date_range'][1]}")
        logger.info(f"{'='*60}")
        
        return df
    
    def temporal_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into observation and holdout windows.
        
        Parameters
        ----------
        df : pd.DataFrame
            Cleaned transaction data.
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (observation_df, holdout_df)
        """
        observation_df = df[
            (df[COL_DATE] >= OBSERVATION_START) & 
            (df[COL_DATE] <= OBSERVATION_END)
        ].copy()
        
        holdout_df = df[
            (df[COL_DATE] >= HOLDOUT_START) & 
            (df[COL_DATE] <= HOLDOUT_END)
        ].copy()
        
        logger.info(f"Observation window: {OBSERVATION_START.date()} to {OBSERVATION_END.date()}")
        logger.info(f"  → {len(observation_df):,} transactions, {observation_df[COL_CUSTOMER].nunique():,} customers")
        logger.info(f"Holdout window: {HOLDOUT_START.date()} to {HOLDOUT_END.date()}")
        logger.info(f"  → {len(holdout_df):,} transactions, {holdout_df[COL_CUSTOMER].nunique():,} customers")
        
        return observation_df, holdout_df
    
    def _log_step(self, step_name: str, before: int, after: int):
        """Log a cleaning step."""
        removed = before - after
        self.cleaning_report[step_name] = {
            "before": before, "after": after, "removed": removed
        }
        logger.info(f"  [{step_name}] {before:,} → {after:,} (removed {removed:,})")
    
    def get_cleaning_report(self) -> dict:
        """Get the cleaning report as a dictionary."""
        return self.cleaning_report
    
    def save_clean_data(self, df: pd.DataFrame, filepath: Optional[Path] = None):
        """Save cleaned data to parquet."""
        if filepath is None:
            filepath = CLEAN_TRANSACTIONS_FILE
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved clean data to {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    from src.data.loader import load_raw_data
    
    raw_df = load_raw_data()
    
    preprocessor = DataPreprocessor(convert_to_inr=True)
    clean_df = preprocessor.clean(raw_df)
    
    print(f"\nCleaning Report:")
    for k, v in preprocessor.get_cleaning_report().items():
        print(f"  {k}: {v}")
    
    preprocessor.save_clean_data(clean_df)
    
    print(f"\nSample data:")
    print(clean_df.head(10))
