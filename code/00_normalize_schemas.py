#!/usr/bin/env python3
"""
Schema normalization for raw data files.

Purpose: Standardize raw data schemas before ingestion
- Stocks & Index: normalize to date, ticker, adj_close
- Risk-free: normalize to date, rf (monthly decimal)
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def setup_logging():
    """Set up logging configuration."""
    log_dir = Path('output/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'normalize_schemas.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def normalize_stock_file(file_path: Path, ticker: str, logger: logging.Logger) -> pd.DataFrame:
    """Normalize a single stock file to standard schema."""
    logger.info(f"Normalizing {file_path.name}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Handle different date column names
        date_cols = ['date', 'timestamp', 'time']
        date_col = None
        for col in date_cols:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError(f"No date column found in {file_path.name}")
        
        # Handle different price column names
        price_cols = ['adj_close', 'adjclose', 'close', 'adjusted_close']
        price_col = None
        for col in price_cols:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None:
            raise ValueError(f"No price column found in {file_path.name}")
        
        # Create normalized dataframe
        normalized_df = pd.DataFrame({
            'date': pd.to_datetime(df[date_col]),
            'ticker': ticker,
            'adj_close': pd.to_numeric(df[price_col], errors='coerce')
        })
        
        # Remove rows with missing data
        normalized_df = normalized_df.dropna()
        
        # Sort by date
        normalized_df = normalized_df.sort_values('date')
        
        logger.info(f"Normalized {file_path.name}: {len(normalized_df)} rows")
        return normalized_df
        
    except Exception as e:
        logger.error(f"Failed to normalize {file_path.name}: {e}")
        return pd.DataFrame()

def normalize_market_file(file_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Normalize market index file to standard schema."""
    logger.info(f"Normalizing market file {file_path.name}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Handle different date column names
        date_cols = ['date', 'timestamp', 'time']
        date_col = None
        for col in date_cols:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError(f"No date column found in {file_path.name}")
        
        # Handle different price column names
        price_cols = ['adj_close', 'adjclose', 'close', 'adjusted_close']
        price_col = None
        for col in price_cols:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None:
            raise ValueError(f"No price column found in {file_path.name}")
        
        # Create normalized dataframe
        normalized_df = pd.DataFrame({
            'date': pd.to_datetime(df[date_col]),
            'ticker': 'DAX',
            'adj_close': pd.to_numeric(df[price_col], errors='coerce')
        })
        
        # Remove rows with missing data
        normalized_df = normalized_df.dropna()
        
        # Sort by date
        normalized_df = normalized_df.sort_values('date')
        
        logger.info(f"Normalized market file: {len(normalized_df)} rows")
        return normalized_df
        
    except Exception as e:
        logger.error(f"Failed to normalize market file {file_path.name}: {e}")
        return pd.DataFrame()

def normalize_risk_free_file(file_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Normalize risk-free rate file to standard schema."""
    logger.info(f"Normalizing risk-free file {file_path.name}")
    
    try:
        # The risk-free file has a complex column structure, so we need special handling
        df = pd.read_csv(file_path)
        
        # The column name contains the data, so we need to parse it
        col_name = df.columns[0]
        
        # Split the complex column into components
        # Format: '2015-09-30,"2015Sep","-0.1052"'
        parsed_data = []
        for value in df[col_name]:
            try:
                # Split by comma and extract date and rate
                parts = str(value).split(',')
                if len(parts) >= 3:
                    date_str = parts[0].strip('"')
                    rate_str = parts[2].strip('"')
                    
                    # Convert rate to decimal
                    rate = float(rate_str)
                    
                    # Convert annualized rate to monthly
                    # If rate is already in decimal form (e.g., -0.1052 = -10.52%)
                    if abs(rate) > 1:
                        rate = rate / 100  # Convert percentage to decimal
                    
                    # Convert to monthly rate
                    monthly_rate = (1 + rate) ** (1/12) - 1
                    
                    parsed_data.append({
                        'date': pd.to_datetime(date_str),
                        'rf': monthly_rate
                    })
            except Exception as e:
                logger.warning(f"Failed to parse risk-free data: {value}, error: {e}")
                continue
        
        if not parsed_data:
            raise ValueError("No valid risk-free data found")
        
        normalized_df = pd.DataFrame(parsed_data)
        normalized_df = normalized_df.dropna()
        normalized_df = normalized_df.sort_values('date')
        
        logger.info(f"Normalized risk-free file: {len(normalized_df)} rows")
        return normalized_df
        
    except Exception as e:
        logger.error(f"Failed to normalize risk-free file {file_path.name}: {e}")
        return pd.DataFrame()

def main():
    """Main normalization function."""
    logger = setup_logging()
    logger.info("Starting schema normalization")
    
    try:
        # Normalize stock files
        stocks_dir = Path("data/raw/stocks")
        stock_files = list(stocks_dir.glob("*.csv"))
        
        logger.info(f"Found {len(stock_files)} stock files")
        
        normalized_stocks = []
        for file_path in stock_files:
            # Extract ticker from filename (e.g., sap_de_m.csv -> SAP)
            ticker = file_path.stem.split('_')[0].upper()
            normalized_df = normalize_stock_file(file_path, ticker, logger)
            if not normalized_df.empty:
                normalized_stocks.append(normalized_df)
        
        # Normalize market file
        market_file = Path("data/raw/market/dax_tr.csv")
        if market_file.exists():
            normalized_market = normalize_market_file(market_file, logger)
        else:
            logger.error("Market file not found")
            normalized_market = pd.DataFrame()
        
        # Normalize risk-free file
        rf_file = Path("data/raw/risk_free/rf_eur.csv")
        if rf_file.exists():
            normalized_rf = normalize_risk_free_file(rf_file, logger)
        else:
            logger.error("Risk-free file not found")
            normalized_rf = pd.DataFrame()
        
        # Save normalized data
        output_dir = Path("data/staging")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if normalized_stocks:
            stocks_combined = pd.concat(normalized_stocks, ignore_index=True)
            stocks_combined.to_parquet(output_dir / "stocks_normalized.parquet", index=False)
            logger.info(f"Saved normalized stocks: {len(stocks_combined)} rows")
        
        if not normalized_market.empty:
            normalized_market.to_parquet(output_dir / "market_normalized.parquet", index=False)
            logger.info(f"Saved normalized market: {len(normalized_market)} rows")
        
        if not normalized_rf.empty:
            normalized_rf.to_parquet(output_dir / "risk_free_normalized.parquet", index=False)
            logger.info(f"Saved normalized risk-free: {len(normalized_rf)} rows")
        
        logger.info("Schema normalization completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Schema normalization failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
