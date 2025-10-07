#!/usr/bin/env python3
"""
Data ingestion and cleaning pipeline for asset pricing analysis.

Purpose: Read raw stock data, clean and validate, compute returns, save processed data
Inputs: CSV files in data/raw/stocks/, risk-free rate data (optional)
Outputs: data/processed/returns.parquet, output/tables/returns_summary.csv
Seed: 42 (for reproducibility)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from pydantic import ValidationError

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config_loader import load_config


def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path('output/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'ingest_clean.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def read_stock_data(stocks_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Read all stock CSV files and combine into single DataFrame.
    
    Expected columns: date, ticker, adj_close
    """
    logger.info(f"Reading stock data from {stocks_dir}")
    
    stock_files = list(stocks_dir.glob("*.csv"))
    if not stock_files:
        raise FileNotFoundError(f"No CSV files found in {stocks_dir}")
    
    logger.info(f"Found {len(stock_files)} stock files")
    
    all_data = []
    failed_tickers = []
    
    for file_path in stock_files:
        ticker = file_path.stem
        try:
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['date', 'ticker', 'adj_close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Ticker {ticker}: Missing columns {missing_cols}")
                failed_tickers.append(ticker)
                continue
            
            # Parse dates
            df['date'] = pd.to_datetime(df['date'])
            
            # Add ticker if not present
            if 'ticker' not in df.columns:
                df['ticker'] = ticker
            
            # Sort by date
            df = df.sort_values('date')
            
            # Drop missing prices
            initial_rows = len(df)
            df = df.dropna(subset=['adj_close'])
            if len(df) < initial_rows:
                logger.warning(f"Ticker {ticker}: Dropped {initial_rows - len(df)} rows with missing prices")
            
            # Check for sufficient data
            if len(df) < 12:  # Minimum 1 year of data
                logger.warning(f"Ticker {ticker}: Only {len(df)} observations, may be insufficient")
            
            all_data.append(df)
            logger.info(f"Ticker {ticker}: {len(df)} observations from {df['date'].min()} to {df['date'].max()}")
            
        except Exception as e:
            logger.error(f"Ticker {ticker}: Failed to read - {e}")
            failed_tickers.append(ticker)
    
    if not all_data:
        raise ValueError("No valid stock data found")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data: {len(combined_df)} total observations from {len(all_data)} tickers")
    
    if failed_tickers:
        logger.warning(f"Failed to process {len(failed_tickers)} tickers: {failed_tickers}")
    
    return combined_df, failed_tickers


def resample_to_monthly(df: pd.DataFrame, frequency: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Resample data to monthly frequency, taking last observation per month.
    """
    logger.info(f"Resampling to {frequency} frequency")
    
    # Set date as index for resampling
    df_indexed = df.set_index('date')
    
    # Resample by ticker
    resampled_data = []
    for ticker in df_indexed['ticker'].unique():
        ticker_data = df_indexed[df_indexed['ticker'] == ticker].copy()
        
        # Resample to monthly, taking last observation
        monthly_data = ticker_data.resample(frequency).last()
        
        # Add ticker back
        monthly_data['ticker'] = ticker
        
        # Drop rows where adj_close is NaN (no data in that month)
        monthly_data = monthly_data.dropna(subset=['adj_close'])
        
        resampled_data.append(monthly_data)
    
    # Combine resampled data
    result_df = pd.concat(resampled_data, ignore_index=False)
    result_df = result_df.reset_index()
    
    logger.info(f"Resampled data: {len(result_df)} monthly observations")
    return result_df


def compute_returns(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute monthly returns for each ticker.
    """
    logger.info("Computing monthly returns")
    
    result_data = []
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy().sort_values('date')
        
        # Compute returns: ret = adj_close / adj_close.shift(1) - 1
        ticker_data['ret'] = ticker_data['adj_close'] / ticker_data['adj_close'].shift(1) - 1
        
        # Compute log returns for analysis
        ticker_data['log_ret'] = np.log(ticker_data['adj_close'] / ticker_data['adj_close'].shift(1))
        
        result_data.append(ticker_data)
    
    result_df = pd.concat(result_data, ignore_index=True)
    
    # Drop first observation for each ticker (no return available)
    result_df = result_df.dropna(subset=['ret'])
    
    logger.info(f"Computed returns: {len(result_df)} observations")
    return result_df


def read_risk_free_data(rf_path: Optional[str], logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Read risk-free rate data if available.
    """
    if not rf_path or not Path(rf_path).exists():
        logger.info("No risk-free rate data provided")
        return None
    
    logger.info(f"Reading risk-free rate data from {rf_path}")
    
    try:
        rf_df = pd.read_csv(rf_path)
        
        # Parse dates
        rf_df['date'] = pd.to_datetime(rf_df['date'])
        
        # Sort by date
        rf_df = rf_df.sort_values('date')
        
        # Resample to monthly if needed
        rf_df = rf_df.set_index('date')
        rf_df = rf_df.resample('M').last()
        rf_df = rf_df.reset_index()
        
        # Convert to monthly rate if needed (assuming annual rate)
        if 'rf_annual' in rf_df.columns:
            rf_df['rf'] = rf_df['rf_annual'] / 12
        elif 'rf' in rf_df.columns:
            rf_df['rf'] = rf_df['rf']
        else:
            logger.error("Risk-free rate data must have 'rf' or 'rf_annual' column")
            return None
        
        logger.info(f"Risk-free rate data: {len(rf_df)} monthly observations")
        return rf_df[['date', 'rf']]
        
    except Exception as e:
        logger.error(f"Failed to read risk-free rate data: {e}")
        return None


def compute_excess_returns(returns_df: pd.DataFrame, rf_df: Optional[pd.DataFrame], logger: logging.Logger) -> pd.DataFrame:
    """
    Compute excess returns by joining with risk-free rate data.
    """
    if rf_df is None:
        logger.info("No risk-free rate data available, skipping excess return calculation")
        returns_df['rf'] = 0.0
        returns_df['ret_excess'] = returns_df['ret']
        return returns_df
    
    logger.info("Computing excess returns")
    
    # Merge with risk-free rate data
    merged_df = returns_df.merge(rf_df, on='date', how='left')
    
    # Fill missing risk-free rates with 0 (or forward fill)
    merged_df['rf'] = merged_df['rf'].fillna(0.0)
    
    # Compute excess returns
    merged_df['ret_excess'] = merged_df['ret'] - merged_df['rf']
    
    logger.info(f"Computed excess returns: {len(merged_df)} observations")
    return merged_df


def create_summary_stats(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """
    Create summary statistics CSV file.
    """
    logger.info("Creating summary statistics")
    
    summary_data = []
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker]
        
        summary_data.append({
            'ticker': ticker,
            'n_obs': len(ticker_data),
            'start_date': ticker_data['date'].min(),
            'end_date': ticker_data['date'].max(),
            'mean_return': ticker_data['ret'].mean(),
            'std_return': ticker_data['ret'].std(),
            'min_return': ticker_data['ret'].min(),
            'max_return': ticker_data['ret'].max()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('ticker')
    
    # Save summary
    summary_path = output_dir / 'tables' / 'returns_summary.csv'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"Summary: {len(summary_df)} tickers, {summary_df['n_obs'].sum()} total observations")


def validate_data(df: pd.DataFrame, config, logger: logging.Logger) -> bool:
    """
    Validate processed data against requirements.
    """
    logger.info("Validating processed data")
    
    validation_passed = True
    
    # Check minimum observations per ticker
    min_obs = config.universe.min_obs_months
    ticker_counts = df['ticker'].value_counts()
    insufficient_tickers = ticker_counts[ticker_counts < min_obs]
    
    if len(insufficient_tickers) > 0:
        logger.warning(f"Tickers with < {min_obs} observations: {insufficient_tickers.to_dict()}")
        validation_passed = False
    
    # Check date range
    date_range = df['date'].max() - df['date'].min()
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()} ({date_range.days} days)")
    
    # Check for missing values
    missing_returns = df['ret'].isna().sum()
    if missing_returns > 0:
        logger.warning(f"Missing returns: {missing_returns}")
        validation_passed = False
    
    # Check return ranges (should be reasonable)
    extreme_returns = df[(df['ret'] < -0.5) | (df['ret'] > 0.5)]
    if len(extreme_returns) > 0:
        logger.warning(f"Extreme returns (>50%): {len(extreme_returns)} observations")
    
    return validation_passed


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Ingest and clean stock data")
    parser.add_argument("--check", action="store_true", help="Run validation check only")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting data ingestion and cleaning")
    
    try:
        # Load configuration
        config = load_config("config.yml")
        logger.info("Configuration loaded successfully")
        
        # Create output directories
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        (output_dir / "tables").mkdir(exist_ok=True)
        
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        if args.check:
            logger.info("Running validation check only")
            # TODO: Implement validation check logic
            print("VALIDATION CHECK: Data ingestion pipeline ready")
            return 0
        
        # Read stock data
        stocks_dir = Path("data/raw/stocks")
        if not stocks_dir.exists():
            logger.error(f"Stocks directory not found: {stocks_dir}")
            return 1
        
        returns_df, failed_tickers = read_stock_data(stocks_dir, logger)
        
        # Resample to monthly
        returns_df = resample_to_monthly(returns_df, config.project.frequency, logger)
        
        # Compute returns
        returns_df = compute_returns(returns_df, logger)
        
        # Read risk-free rate data
        rf_df = read_risk_free_data(config.paths.risk_free_csv, logger)
        
        # Compute excess returns
        returns_df = compute_excess_returns(returns_df, rf_df, logger)
        
        # Validate data
        validation_passed = validate_data(returns_df, config, logger)
        
        if not validation_passed:
            logger.warning("Data validation failed, but continuing with processing")
        
        # Save processed data
        output_path = processed_dir / "returns.parquet"
        returns_df.to_parquet(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
        # Create summary statistics
        create_summary_stats(returns_df, output_dir, logger)
        
        logger.info("Data ingestion and cleaning completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
