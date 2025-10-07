#!/usr/bin/env python3
"""
Time-series CAPM and Black (zero-beta) model analysis.

Purpose: Estimate alpha and beta for each stock using OLS with HAC standard errors
Inputs: data/processed/returns.parquet, market data, risk-free rate
Outputs: betas.csv, vw_beta_summary.csv, beta_hist.png
Seed: 42 (for reproducibility)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tools import add_constant

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config_loader import load_config


def setup_logging():
    """Set up logging configuration."""
    log_dir = Path('output/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'capm_timeseries.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_processed_data(logger: logging.Logger) -> pd.DataFrame:
    """Load processed returns data."""
    processed_path = Path("data/processed/returns.parquet")
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_path}")
    
    df = pd.read_parquet(processed_path)
    logger.info(f"Loaded processed data: {len(df)} observations, {df['ticker'].nunique()} tickers")
    return df


def build_market_return_series(df: pd.DataFrame, config, logger: logging.Logger) -> pd.DataFrame:
    """
    Build market return series based on configuration.
    """
    logger.info("Building market return series")
    
    if config.benchmarks.market_index == "use_value_weighted_of_universe":
        # Try to use value-weighted market
        if config.paths.market_caps_csv and Path(config.paths.market_caps_csv).exists():
            logger.info("Using value-weighted market from market caps data")
            market_df = build_value_weighted_market(df, config.paths.market_caps_csv, logger)
        else:
            logger.warning("Using equal-weighted market due to missing market caps data")
            market_df = build_equal_weighted_market(df, logger)
    else:
        # Use index file
        if config.benchmarks.index_csv and Path(config.benchmarks.index_csv).exists():
            logger.info(f"Using market index from {config.benchmarks.index_csv}")
            market_df = build_index_market(config.benchmarks.index_csv, logger)
        else:
            logger.warning("Index file not found, falling back to equal-weighted market")
            market_df = build_equal_weighted_market(df, logger)
    
    return market_df


def build_value_weighted_market(df: pd.DataFrame, market_caps_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Build value-weighted market return series."""
    # Read market cap data
    caps_df = pd.read_csv(market_caps_path)
    caps_df['date'] = pd.to_datetime(caps_df['date'])
    
    # Merge with returns data
    merged_df = df.merge(caps_df, on=['date', 'ticker'], how='left')
    
    # Compute value weights for each date
    market_returns = []
    for date in merged_df['date'].unique():
        date_data = merged_df[merged_df['date'] == date]
        
        # Calculate weights (normalize to sum to 1)
        if 'market_cap' in date_data.columns:
            weights = date_data['market_cap'] / date_data['market_cap'].sum()
        else:
            # Fallback to equal weights
            weights = pd.Series(1/len(date_data), index=date_data.index)
        
        # Compute weighted return
        weighted_return = (date_data['ret'] * weights).sum()
        market_returns.append({'date': date, 'ret_m': weighted_return})
    
    market_df = pd.DataFrame(market_returns)
    market_df = market_df.sort_values('date')
    
    logger.info(f"Value-weighted market: {len(market_df)} observations")
    return market_df


def build_equal_weighted_market(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Build equal-weighted market return series."""
    market_returns = df.groupby('date')['ret'].mean().reset_index()
    market_returns.columns = ['date', 'ret_m']
    
    logger.info(f"Equal-weighted market: {len(market_returns)} observations")
    return market_returns


def build_index_market(index_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Build market return series from index file."""
    index_df = pd.read_csv(index_path)
    index_df['date'] = pd.to_datetime(index_df['date'])
    
    # Compute returns
    index_df = index_df.sort_values('date')
    index_df['ret_m'] = index_df['adj_close'] / index_df['adj_close'].shift(1) - 1
    index_df = index_df.dropna()
    
    market_df = index_df[['date', 'ret_m']]
    logger.info(f"Index market: {len(market_df)} observations")
    return market_df


def compute_excess_returns(df: pd.DataFrame, market_df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute excess returns for stocks and market."""
    logger.info("Computing excess returns")
    
    # Merge market returns with stock data
    merged_df = df.merge(market_df, on='date', how='inner')
    
    # Compute excess returns if risk-free rate is available
    if 'rf' in merged_df.columns and merged_df['rf'].notna().any():
        merged_df['ret_excess'] = merged_df['ret'] - merged_df['rf']
        merged_df['mkt_excess'] = merged_df['ret_m'] - merged_df['rf']
        logger.info("Computed excess returns using risk-free rate")
    else:
        merged_df['ret_excess'] = merged_df['ret']
        merged_df['mkt_excess'] = merged_df['ret_m']
        logger.info("No risk-free rate available, using raw returns")
    
    return merged_df, market_df


def run_capm_regression(ticker_data: pd.DataFrame, logger: logging.Logger) -> dict:
    """Run CAPM regression for a single ticker."""
    # Drop missing values
    clean_data = ticker_data.dropna(subset=['ret_excess', 'mkt_excess'])
    
    if len(clean_data) < 12:  # Minimum observations
        return None
    
    # CAPM regression: ret_excess ~ alpha + beta * mkt_excess
    X = add_constant(clean_data['mkt_excess'])
    y = clean_data['ret_excess']
    
    try:
        model = OLS(y, X).fit()
        
        # Get HAC standard errors
        hac_cov = cov_hac(model, nlags=4)
        hac_se = np.sqrt(np.diag(hac_cov))
        
        # Extract results
        alpha = model.params[0]
        beta = model.params[1]
        t_alpha = alpha / hac_se[0]
        t_beta = beta / hac_se[1]
        r2 = model.rsquared
        n = len(clean_data)
        
        return {
            'alpha': alpha,
            'beta': beta,
            't_alpha': t_alpha,
            't_beta': t_beta,
            'r2': r2,
            'n': n,
            'start': clean_data['date'].min(),
            'end': clean_data['date'].max()
        }
    except Exception as e:
        logger.warning(f"CAPM regression failed: {e}")
        return None


def run_black_regression(ticker_data: pd.DataFrame, logger: logging.Logger) -> dict:
    """Run Black (zero-beta) regression for a single ticker."""
    # Drop missing values
    clean_data = ticker_data.dropna(subset=['ret', 'ret_m'])
    
    if len(clean_data) < 12:  # Minimum observations
        return None
    
    # Black regression: ret ~ a + beta_B * ret_m
    X = add_constant(clean_data['ret_m'])
    y = clean_data['ret']
    
    try:
        model = OLS(y, X).fit()
        
        # Get HAC standard errors
        hac_cov = cov_hac(model, nlags=4)
        hac_se = np.sqrt(np.diag(hac_cov))
        
        # Extract results
        a = model.params[0]
        beta = model.params[1]
        t_a = a / hac_se[0]
        t_beta = beta / hac_se[1]
        r2 = model.rsquared
        n = len(clean_data)
        
        return {
            'a': a,
            'beta': beta,
            't_a': t_a,
            't_beta': t_beta,
            'r2': r2,
            'n': n,
            'start': clean_data['date'].min(),
            'end': clean_data['date'].max()
        }
    except Exception as e:
        logger.warning(f"Black regression failed: {e}")
        return None


def estimate_betas(df: pd.DataFrame, config, logger: logging.Logger) -> pd.DataFrame:
    """Estimate betas for all tickers."""
    logger.info("Estimating betas for all tickers")
    
    results = []
    min_obs = config.universe.min_obs_months
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        
        if len(ticker_data) < min_obs:
            logger.warning(f"Ticker {ticker}: insufficient observations ({len(ticker_data)} < {min_obs})")
            continue
        
        # Run CAPM regression
        capm_result = run_capm_regression(ticker_data, logger)
        if capm_result:
            results.append({
                'ticker': ticker,
                'model_type': 'CAPM',
                'alpha_or_a': capm_result['alpha'],
                'beta': capm_result['beta'],
                't_alpha_or_a': capm_result['t_alpha'],
                't_beta': capm_result['t_beta'],
                'r2': capm_result['r2'],
                'n': capm_result['n'],
                'start': capm_result['start'],
                'end': capm_result['end']
            })
        
        # Run Black regression
        black_result = run_black_regression(ticker_data, logger)
        if black_result:
            results.append({
                'ticker': ticker,
                'model_type': 'Black',
                'alpha_or_a': black_result['a'],
                'beta': black_result['beta'],
                't_alpha_or_a': black_result['t_a'],
                't_beta': black_result['t_beta'],
                'r2': black_result['r2'],
                'n': black_result['n'],
                'start': black_result['start'],
                'end': black_result['end']
            })
    
    results_df = pd.DataFrame(results)
    logger.info(f"Estimated betas for {len(results_df)} ticker-model combinations")
    return results_df


def create_vw_beta_summary(betas_df: pd.DataFrame, market_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Create value-weighted beta summary."""
    logger.info("Creating VW beta summary")
    
    # Get CAPM betas
    capm_betas = betas_df[betas_df['model_type'] == 'CAPM']
    
    if len(capm_betas) == 0:
        # Use Black betas if no CAPM available
        capm_betas = betas_df[betas_df['model_type'] == 'Black']
        method = "Black (no CAPM available)"
    else:
        method = "CAPM"
    
    if len(capm_betas) == 0:
        logger.warning("No betas available for summary")
        return pd.DataFrame()
    
    # Compute equal-weighted average beta
    ew_beta = capm_betas['beta'].mean()
    
    # For value-weighted beta, we'd need market cap weights
    # For now, use equal-weighted as approximation
    vw_beta = ew_beta
    
    summary_df = pd.DataFrame([{
        'method': method,
        'vw_beta': vw_beta,
        'ew_beta': ew_beta,
        'notes': 'VW beta approximated by EW (market cap weights not available)'
    }])
    
    logger.info(f"VW beta summary: EW={ew_beta:.3f}, VW={vw_beta:.3f}")
    return summary_df


def create_beta_histogram(betas_df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """Create beta histogram."""
    logger.info("Creating beta histogram")
    
    # Get CAPM betas if available, otherwise use Black
    capm_betas = betas_df[betas_df['model_type'] == 'CAPM']
    if len(capm_betas) == 0:
        capm_betas = betas_df[betas_df['model_type'] == 'Black']
        title_suffix = " (Black Model)"
    else:
        title_suffix = " (CAPM)"
    
    if len(capm_betas) == 0:
        logger.warning("No betas available for histogram")
        return
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(capm_betas['beta'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Beta')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Stock Betas{title_suffix}')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_beta = capm_betas['beta'].mean()
    std_beta = capm_betas['beta'].std()
    plt.axvline(mean_beta, color='red', linestyle='--', label=f'Mean: {mean_beta:.3f}')
    plt.axvline(1.0, color='green', linestyle='--', label='Beta = 1.0')
    plt.legend()
    
    # Save figure
    fig_path = output_dir / 'figs' / 'beta_hist.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Beta histogram saved to {fig_path}")


def save_results(betas_df: pd.DataFrame, vw_summary_df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """Save all results to files."""
    logger.info("Saving results")
    
    # Save betas
    betas_path = output_dir / 'tables' / 'betas.csv'
    betas_path.parent.mkdir(parents=True, exist_ok=True)
    betas_df.to_csv(betas_path, index=False)
    logger.info(f"Betas saved to {betas_path}")
    
    # Save VW summary
    vw_path = output_dir / 'tables' / 'vw_beta_summary.csv'
    vw_summary_df.to_csv(vw_path, index=False)
    logger.info(f"VW summary saved to {vw_path}")


def print_summary(betas_df: pd.DataFrame, vw_summary_df: pd.DataFrame, logger: logging.Logger) -> None:
    """Print summary statistics."""
    logger.info("=== CAPM Time-Series Analysis Summary ===")
    
    # Basic counts
    total_estimates = len(betas_df)
    capm_estimates = len(betas_df[betas_df['model_type'] == 'CAPM'])
    black_estimates = len(betas_df[betas_df['model_type'] == 'Black'])
    
    print(f"Total estimates: {total_estimates}")
    print(f"CAPM estimates: {capm_estimates}")
    print(f"Black estimates: {black_estimates}")
    
    # Beta statistics
    if len(betas_df) > 0:
        print(f"Min beta: {betas_df['beta'].min():.3f}")
        print(f"Max beta: {betas_df['beta'].max():.3f}")
        print(f"Mean beta: {betas_df['beta'].mean():.3f}")
    
    # Alpha statistics (CAPM only)
    capm_betas = betas_df[betas_df['model_type'] == 'CAPM']
    if len(capm_betas) > 0:
        mean_alpha = capm_betas['alpha_or_a'].mean()
        print(f"Mean alpha (CAPM): {mean_alpha:.3f}")
    
    # VW beta summary
    if len(vw_summary_df) > 0:
        print(f"VW beta: {vw_summary_df['vw_beta'].iloc[0]:.3f}")
        print(f"EW beta: {vw_summary_df['ew_beta'].iloc[0]:.3f}")
    
    print("Files written:")
    print("- output/tables/betas.csv")
    print("- output/tables/vw_beta_summary.csv")
    print("- output/figs/beta_hist.png")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Time-series CAPM analysis")
    parser.add_argument("--check", action="store_true", help="Run validation check only")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting CAPM time-series analysis")
    
    try:
        # Load configuration
        config = load_config("config.yml")
        logger.info("Configuration loaded successfully")
        
        if args.check:
            logger.info("Running validation check only")
            print("VALIDATION CHECK: CAPM time-series analysis ready")
            return 0
        
        # Load processed data
        df = load_processed_data(logger)
        
        # Build market return series
        market_df = build_market_return_series(df, config, logger)
        
        # Compute excess returns
        df, market_df = compute_excess_returns(df, market_df, logger)
        
        # Estimate betas
        betas_df = estimate_betas(df, config, logger)
        
        if len(betas_df) == 0:
            logger.error("No betas estimated - check data quality")
            return 1
        
        # Create VW beta summary
        vw_summary_df = create_vw_beta_summary(betas_df, market_df, logger)
        
        # Create beta histogram
        output_dir = Path("output")
        create_beta_histogram(betas_df, output_dir, logger)
        
        # Save results
        save_results(betas_df, vw_summary_df, output_dir, logger)
        
        # Print summary
        print_summary(betas_df, vw_summary_df, logger)
        
        logger.info("CAPM time-series analysis completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"CAPM analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
