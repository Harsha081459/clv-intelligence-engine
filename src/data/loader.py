"""
Data Loader Module
==================
Handles loading the UCI Online Retail II dataset from Excel/CSV.
Supports loading from both sheets and concatenating into a single DataFrame.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    RAW_DATA_FILE, COLUMN_RENAME_MAP, COL_DATE
)

logger = logging.getLogger(__name__)


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the UCI Online Retail II dataset from an Excel file.
    
    The dataset has two sheets:
    - 'Year 2009-2010' (~525K rows)
    - 'Year 2010-2011' (~541K rows)
    
    Both sheets are concatenated into a single DataFrame with
    standardized column names.
    
    Parameters
    ----------
    filepath : Path, optional
        Path to the Excel file. Defaults to config.RAW_DATA_FILE.
        
    Returns
    -------
    pd.DataFrame
        Raw transaction data with standardized column names.
    """
    if filepath is None:
        filepath = RAW_DATA_FILE
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            f"Please download the UCI Online Retail II dataset and place it at: {filepath}"
        )
    
    logger.info(f"Loading dataset from {filepath}...")
    
    # Check file extension
    if filepath.suffix in ['.xlsx', '.xls']:
        # Load both sheets
        logger.info("Loading Excel sheets...")
        sheets = pd.read_excel(filepath, sheet_name=None, engine='openpyxl')
        
        sheet_names = list(sheets.keys())
        logger.info(f"Found sheets: {sheet_names}")
        
        # Concatenate all sheets
        df = pd.concat(sheets.values(), ignore_index=True)
        logger.info(f"Combined {len(sheet_names)} sheets: {sum(len(s) for s in sheets.values())} total rows")
        
    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
        logger.info(f"Loaded CSV: {len(df)} rows")
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Rename columns to standardized names
    # The actual Excel uses 'Invoice', 'Price', 'Customer ID' (with space)
    # We standardize to 'InvoiceNo', 'UnitPrice', 'CustomerID'
    existing_cols = set(df.columns)
    rename_map = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in existing_cols}
    
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed columns: {rename_map}")
    
    # Ensure InvoiceDate is datetime
    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    
    # Sort by date
    df = df.sort_values(COL_DATE).reset_index(drop=True)
    
    logger.info(f"Loaded {len(df):,} rows, {df.columns.tolist()}")
    logger.info(f"Date range: {df[COL_DATE].min()} to {df[COL_DATE].max()}")
    
    return df


def load_processed_data(filepath: Path) -> pd.DataFrame:
    """
    Load a processed parquet file.
    
    Parameters
    ----------
    filepath : Path
        Path to the parquet file.
        
    Returns
    -------
    pd.DataFrame
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Processed data not found at {filepath}. Run the pipeline first.")
    
    return pd.read_parquet(filepath)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    df = load_raw_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
