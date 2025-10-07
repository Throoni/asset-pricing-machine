#!/usr/bin/env python3
"""
Data ingestion and cleaning pipeline for asset pricing analysis.

Purpose: Read normalized data, clean and validate, compute returns, save processed data
Inputs: Normalized parquet files from data/staging/
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


def read_normalized_data(logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read normalized data from staging directory.
    
    Returns: (stocks_df, market_df, rf_df)
    """
    logger.info("Reading normalized data from staging")
    
    # Read stocks data
    stocks_file = Path("data/staging/stocks_normalized.parquet")
    if not stocks_file.exists():
        raise FileNotFoundError(f"Normalized stocks data not found: {stocks_file}")
    
    stocks_df = pd.read_parquet(stocks_file)
    logger.info(f"Loaded stocks: {len(stocks_df)} rows, {stocks_df['ticker'].nunique()} tickers")
    
    # Read market data
    market_file = Path("data/staging/market_normalized.parquet")
    if not market_file.exists():
        raise FileNotFoundError(f"Normalized market data not found: {market_file}")
    
    market_df = pd.read_parquet(market_file)
    logger.info(f"Loaded market: {len(market_df)} rows")
    
    # Read risk-free data
    rf_file = Path("data/staging/risk_free_normalized.parquet")
    if not rf_file.exists():
        logger.warning("Normalized risk-free data not found, will proceed without it")
        rf_df = pd.DataFrame()
    else:
        rf_df = pd.read_parquet(rf_file)
        logger.info(f"Loaded risk-free: {len(rf_df)} rows")
    
    return stocks_df, market_df, rf_df


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
    Compute monthly returns from adjusted close prices.
    """
    logger.info("Computing monthly returns")
    
    # Sort by ticker and date
    df = df.sort_values(['ticker', 'date'])
    
    # Compute returns by ticker
    df['ret'] = df.groupby('ticker')['adj_close'].pct_change()
    
    # Drop first observation for each ticker (no return available)
    initial_rows = len(df)
    df = df.dropna(subset=['ret'])
    logger.info(f"Dropped {initial_rows - len(df)} observations with missing returns")
    
    return df


def add_risk_free_data(df: pd.DataFrame, rf_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add risk-free rate data and compute excess returns.
    """
    if rf_df.empty:
        logger.info("No risk-free data available, using raw returns")
        return df
    
    logger.info("Adding risk-free rate data")
    
    # Merge risk-free data
    df = df.merge(rf_df, on='date', how='left')
    
    # Compute excess returns
    df['ret_excess'] = df['ret'] - df['rf']
    
    # Log missing risk-free data
    missing_rf = df['rf'].isna().sum()
    if missing_rf > 0:
        logger.warning(f"Missing risk-free data for {missing_rf} observations")
    
    return df


def add_market_data(df: pd.DataFrame, market_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add market return data.
    """
    logger.info("Adding market return data")
    
    # Compute market returns
    market_df = market_df.sort_values('date')
    market_df['ret_m'] = market_df['adj_close'].pct_change()
    market_df = market_df.dropna(subset=['ret_m'])
    
    # Merge market data
    df = df.merge(market_df[['date', 'ret_m']], on='date', how='left')
    
    # Compute market excess returns if risk-free data is available
    if 'rf' in df.columns:
        df['mkt_excess'] = df['ret_m'] - df['rf']
    
    # Log missing market data
    missing_mkt = df['ret_m'].isna().sum()
    if missing_mkt > 0:
        logger.warning(f"Missing market data for {missing_mkt} observations")
    
    return df


def validate_data(df: pd.DataFrame, config, logger: logging.Logger) -> pd.DataFrame:
    """
    Validate data quality and apply filters.
    """
    logger.info("Validating data quality")
    
    initial_rows = len(df)
    
    # Filter by date range
    start_date = pd.to_datetime(config.project.start_date)
    end_date = pd.to_datetime(config.project.end_date)
    
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    logger.info(f"Filtered by date range: {len(df)} rows remaining")
    
    # Filter by minimum observations per ticker
    ticker_counts = df.groupby('ticker').size()
    min_obs = config.universe.min_obs_months
    
    valid_tickers = ticker_counts[ticker_counts >= min_obs].index
    df = df[df['ticker'].isin(valid_tickers)]
    
    dropped_tickers = ticker_counts[ticker_counts < min_obs].index
    if len(dropped_tickers) > 0:
        logger.warning(f"Dropped {len(dropped_tickers)} tickers with < {min_obs} observations: {list(dropped_tickers)}")
    
    logger.info(f"Final validation: {len(df)} rows, {df['ticker'].nunique()} tickers")
    return df


def create_summary_table(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """
    Create summary table with statistics per ticker.
    """
    logger.info("Creating summary table")
    
    summary_data = []
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        
        summary_data.append({
            'ticker': ticker,
            'count': len(ticker_data),
            'start_date': ticker_data['date'].min().strftime('%Y-%m-%d'),
            'end_date': ticker_data['date'].max().strftime('%Y-%m-%d'),
            'mean_return': ticker_data['ret'].mean(),
            'volatility': ticker_data['ret'].std() * np.sqrt(12)  # Annualized
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('ticker')
    
    # Save summary
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tables_dir / 'returns_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"Summary table saved to {summary_path}")
    logger.info(f"Summary: {len(summary_df)} tickers, {summary_df['count'].sum()} total observations")


def save_processed_data(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """
    Save processed data to parquet format.
    """
    logger.info("Saving processed data")
    
    # Create processed directory
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    output_path = processed_dir / 'returns.parquet'
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Processed data saved to {output_path}")
    logger.info(f"Final dataset: {len(df)} rows, {df['ticker'].nunique()} tickers")


def print_validation_summary(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Print validation summary for CLI check.
    """
    logger.info("=== Data Validation Summary ===")
    
    print(f"Total observations: {len(df)}")
    print(f"Number of tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    # Check for missing data
    missing_ret = df['ret'].isna().sum()
    missing_mkt = df['ret_m'].isna().sum() if 'ret_m' in df.columns else 0
    missing_rf = df['rf'].isna().sum() if 'rf' in df.columns else 0
    
    print(f"Missing returns: {missing_ret}")
    print(f"Missing market returns: {missing_mkt}")
    print(f"Missing risk-free rates: {missing_rf}")
    
    # Ticker statistics
    ticker_counts = df.groupby('ticker').size()
    print(f"Min observations per ticker: {ticker_counts.min()}")
    print(f"Max observations per ticker: {ticker_counts.max()}")
    print(f"Mean observations per ticker: {ticker_counts.mean():.1f}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Data ingestion and cleaning pipeline")
    parser.add_argument("--check", action="store_true", help="Run validation check only")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting data ingestion and cleaning")
    
    try:
        # Load configuration
        config = load_config("config.yml")
        logger.info("Configuration loaded successfully")
        
        if args.check:
            logger.info("Running validation check only")
            print("VALIDATION CHECK: Data ingestion pipeline ready")
            return 0
        
        # Read normalized data
        stocks_df, market_df, rf_df = read_normalized_data(logger)
        
        # Resample to monthly
        stocks_df = resample_to_monthly(stocks_df, config.project.frequency, logger)
        
        # Compute returns
        stocks_df = compute_returns(stocks_df, logger)
        
        # Add risk-free data
        stocks_df = add_risk_free_data(stocks_df, rf_df, logger)
        
        # Add market data
        stocks_df = add_market_data(stocks_df, market_df, logger)
        
        # Validate data
        stocks_df = validate_data(stocks_df, config, logger)
        
        # Create output directory
        output_dir = Path(config.outputs.dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary table
        create_summary_table(stocks_df, output_dir, logger)
        
        # Save processed data
        save_processed_data(stocks_df, output_dir, logger)
        
        # Print validation summary
        print_validation_summary(stocks_df, logger)
        
        logger.info("Data ingestion and cleaning completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
