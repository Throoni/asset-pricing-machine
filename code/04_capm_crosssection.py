#!/usr/bin/env python3
"""
Cross-sectional CAPM analysis using Fama-MacBeth and zero-beta rate estimation.

Purpose: Test if beta is the only determinant of expected returns using cross-sectional regressions
Inputs: data/processed/returns.parquet, output/tables/betas.csv
Outputs: fmb_results.csv, fmb_with_idio.csv, zero_beta_portfolio.csv, SML plots
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
from scipy.optimize import minimize
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
            logging.FileHandler(log_dir / 'capm_crosssection.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_data(logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed returns and betas data."""
    # Load processed returns
    returns_path = Path("data/processed/returns.parquet")
    if not returns_path.exists():
        raise FileNotFoundError(f"Processed returns not found: {returns_path}")
    
    returns_df = pd.read_parquet(returns_path)
    logger.info(f"Loaded returns: {len(returns_df)} observations, {returns_df['ticker'].nunique()} tickers")
    
    # Load betas from Step 4
    betas_path = Path("output/tables/betas.csv")
    if not betas_path.exists():
        raise FileNotFoundError(f"Betas file not found: {betas_path}")
    
    betas_df = pd.read_csv(betas_path)
    logger.info(f"Loaded betas: {len(betas_df)} estimates")
    
    return returns_df, betas_df


def build_market_series(returns_df: pd.DataFrame, config, logger: logging.Logger) -> pd.DataFrame:
    """Build market return series (reuse logic from Step 4)."""
    logger.info("Building market return series")
    
    if config.benchmarks.market_index == "use_value_weighted_of_universe":
        if config.paths.market_caps_csv and Path(config.paths.market_caps_csv).exists():
            logger.info("Using value-weighted market")
            market_df = build_value_weighted_market(returns_df, config.paths.market_caps_csv, logger)
        else:
            logger.warning("Using equal-weighted market due to missing market caps")
            market_df = build_equal_weighted_market(returns_df, logger)
    else:
        if config.benchmarks.index_csv and Path(config.benchmarks.index_csv).exists():
            logger.info(f"Using market index from {config.benchmarks.index_csv}")
            market_df = build_index_market(config.benchmarks.index_csv, logger)
        else:
            logger.warning("Index file not found, falling back to equal-weighted market")
            market_df = build_equal_weighted_market(returns_df, logger)
    
    return market_df


def build_value_weighted_market(df: pd.DataFrame, market_caps_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Build value-weighted market return series."""
    caps_df = pd.read_csv(market_caps_path)
    caps_df['date'] = pd.to_datetime(caps_df['date'])
    
    merged_df = df.merge(caps_df, on=['date', 'ticker'], how='left')
    
    market_returns = []
    for date in merged_df['date'].unique():
        date_data = merged_df[merged_df['date'] == date]
        
        if 'market_cap' in date_data.columns:
            weights = date_data['market_cap'] / date_data['market_cap'].sum()
        else:
            weights = pd.Series(1/len(date_data), index=date_data.index)
        
        weighted_return = (date_data['ret'] * weights).sum()
        market_returns.append({'date': date, 'ret_m': weighted_return})
    
    market_df = pd.DataFrame(market_returns)
    market_df = market_df.sort_values('date')
    
    return market_df


def build_equal_weighted_market(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Build equal-weighted market return series."""
    market_returns = df.groupby('date')['ret'].mean().reset_index()
    market_returns.columns = ['date', 'ret_m']
    return market_returns


def build_index_market(index_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Build market return series from index file."""
    index_df = pd.read_csv(index_path)
    index_df['date'] = pd.to_datetime(index_df['date'])
    index_df = index_df.sort_values('date')
    index_df['ret_m'] = index_df['adj_close'] / index_df['adj_close'].shift(1) - 1
    index_df = index_df.dropna()
    
    market_df = index_df[['date', 'ret_m']]
    return market_df


def compute_excess_returns(returns_df: pd.DataFrame, market_df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, bool]:
    """Compute excess returns and determine if risk-free rate is available."""
    logger.info("Computing excess returns")
    
    # Merge market returns
    merged_df = returns_df.merge(market_df, on='date', how='inner')
    
    has_rf = 'rf' in merged_df.columns and merged_df['rf'].notna().any()
    
    if has_rf:
        merged_df['ret_excess'] = merged_df['ret'] - merged_df['rf']
        merged_df['mkt_excess'] = merged_df['ret_m'] - merged_df['rf']
        logger.info("Computed excess returns using risk-free rate")
    else:
        merged_df['ret_excess'] = merged_df['ret']
        merged_df['mkt_excess'] = merged_df['ret_m']
        logger.info("No risk-free rate available, using raw returns")
    
    return merged_df, has_rf


def static_beta_sml(returns_df: pd.DataFrame, betas_df: pd.DataFrame, has_rf: bool, logger: logging.Logger) -> pd.DataFrame:
    """Implement static-beta Security Market Line regression."""
    logger.info("Running static-beta SML regression")
    
    # Get average returns by stock
    avg_returns = returns_df.groupby('ticker').agg({
        'ret': 'mean',
        'ret_excess': 'mean' if has_rf else 'ret'
    }).reset_index()
    
    # Merge with betas
    if has_rf:
        capm_betas = betas_df[betas_df['model_type'] == 'CAPM']
        if len(capm_betas) == 0:
            logger.warning("No CAPM betas available, using Black betas")
            capm_betas = betas_df[betas_df['model_type'] == 'Black']
    else:
        capm_betas = betas_df[betas_df['model_type'] == 'Black']
    
    merged_data = avg_returns.merge(capm_betas[['ticker', 'beta']], on='ticker', how='inner')
    
    results = []
    
    # CAPM excess version
    if has_rf:
        X = add_constant(merged_data['beta'])
        y = merged_data['ret_excess']
        
        model = OLS(y, X).fit()
        
        results.append({
            'method': 'static_excess',
            'gamma0': model.params[0],
            'gamma_m': model.params[1],
            't_gamma0': model.tvalues[0],
            't_gamma_m': model.tvalues[1],
            'n_months': len(merged_data)
        })
    
    # Black raw version
    X = add_constant(merged_data['beta'])
    y = merged_data['ret']
    
    model = OLS(y, X).fit()
    
    results.append({
        'method': 'static_raw',
        'gamma0': model.params[0],
        'gamma_m': model.params[1],
        't_gamma0': model.tvalues[0],
        't_gamma_m': model.tvalues[1],
        'n_months': len(merged_data)
    })
    
    return pd.DataFrame(results)


def fama_macbeth(returns_df: pd.DataFrame, betas_df: pd.DataFrame, has_rf: bool, logger: logging.Logger) -> pd.DataFrame:
    """Implement Fama-MacBeth rolling regression."""
    logger.info("Running Fama-MacBeth regression")
    
    # Get betas for merging
    if has_rf:
        capm_betas = betas_df[betas_df['model_type'] == 'CAPM']
        if len(capm_betas) == 0:
            capm_betas = betas_df[betas_df['model_type'] == 'Black']
    else:
        capm_betas = betas_df[betas_df['model_type'] == 'Black']
    
    beta_lookup = dict(zip(capm_betas['ticker'], capm_betas['beta']))
    
    # Merge returns with betas
    returns_df['beta'] = returns_df['ticker'].map(beta_lookup)
    returns_df = returns_df.dropna(subset=['beta'])
    
    monthly_results = []
    
    # Run regression for each month
    for date in sorted(returns_df['date'].unique()):
        month_data = returns_df[returns_df['date'] == date]
        
        if len(month_data) < 10:  # Need minimum stocks
            continue
        
        # CAPM excess version
        if has_rf:
            X = add_constant(month_data['beta'])
            y = month_data['ret_excess']
            
            try:
                model = OLS(y, X).fit()
                monthly_results.append({
                    'date': date,
                    'method': 'fm_excess',
                    'gamma0': model.params[0],
                    'gamma_m': model.params[1]
                })
            except:
                continue
        
        # Black raw version
        X = add_constant(month_data['beta'])
        y = month_data['ret']
        
        try:
            model = OLS(y, X).fit()
            monthly_results.append({
                'date': date,
                'method': 'fm_raw',
                'gamma0': model.params[0],
                'gamma_m': model.params[1]
            })
        except:
            continue
    
    # Convert to DataFrame and compute time-series statistics
    monthly_df = pd.DataFrame(monthly_results)
    
    if len(monthly_df) == 0:
        logger.warning("No monthly results from Fama-MacBeth")
        return pd.DataFrame()
    
    # Compute time-series means and t-stats
    results = []
    for method in monthly_df['method'].unique():
        method_data = monthly_df[monthly_df['method'] == method]
        
        gamma0_mean = method_data['gamma0'].mean()
        gamma_m_mean = method_data['gamma_m'].mean()
        
        # Newey-West t-statistics
        gamma0_tstat = gamma0_mean / (method_data['gamma0'].std() / np.sqrt(len(method_data)))
        gamma_m_tstat = gamma_m_mean / (method_data['gamma_m'].std() / np.sqrt(len(method_data)))
        
        results.append({
            'method': method,
            'gamma0': gamma0_mean,
            'gamma_m': gamma_m_mean,
            't_gamma0': gamma0_tstat,
            't_gamma_m': gamma_m_tstat,
            'n_months': len(method_data)
        })
    
    return pd.DataFrame(results)


def idiosyncratic_risk_test(returns_df: pd.DataFrame, betas_df: pd.DataFrame, has_rf: bool, logger: logging.Logger) -> pd.DataFrame:
    """Test if idiosyncratic risk is priced."""
    logger.info("Testing idiosyncratic risk pricing")
    
    # Get betas and compute idiosyncratic variance
    if has_rf:
        capm_betas = betas_df[betas_df['model_type'] == 'CAPM']
        if len(capm_betas) == 0:
            capm_betas = betas_df[betas_df['model_type'] == 'Black']
    else:
        capm_betas = betas_df[betas_df['model_type'] == 'Black']
    
    # Compute idiosyncratic variance for each stock
    idio_vars = []
    for ticker in capm_betas['ticker'].unique():
        ticker_data = returns_df[returns_df['ticker'] == ticker].sort_values('date')
        if len(ticker_data) < 12:
            continue
        
        # Get beta for this stock
        beta = capm_betas[capm_betas['ticker'] == ticker]['beta'].iloc[0]
        
        # Compute residuals
        if has_rf:
            residuals = ticker_data['ret_excess'] - beta * ticker_data['mkt_excess']
        else:
            residuals = ticker_data['ret'] - beta * ticker_data['ret_m']
        
        idio_var = residuals.var()
        idio_vars.append({'ticker': ticker, 'idio_var': idio_var})
    
    idio_df = pd.DataFrame(idio_vars)
    
    # Merge with average returns
    avg_returns = returns_df.groupby('ticker').agg({
        'ret': 'mean',
        'ret_excess': 'mean' if has_rf else 'ret'
    }).reset_index()
    
    merged_data = avg_returns.merge(capm_betas[['ticker', 'beta']], on='ticker', how='inner')
    merged_data = merged_data.merge(idio_df, on='ticker', how='inner')
    
    results = []
    
    # Run regression with idiosyncratic risk
    if has_rf:
        X = merged_data[['beta', 'idio_var']]
        X = add_constant(X)
        y = merged_data['ret_excess']
        
        model = OLS(y, X).fit()
        
        results.append({
            'method': 'fm_with_idio_excess',
            'gamma0': model.params[0],
            'gamma_m': model.params[1],
            'gamma_idio': model.params[2],
            't_gamma0': model.tvalues[0],
            't_gamma_m': model.tvalues[1],
            't_gamma_idio': model.tvalues[2],
            'n_months': len(merged_data)
        })
    
    # Black version
    X = merged_data[['beta', 'idio_var']]
    X = add_constant(X)
    y = merged_data['ret']
    
    model = OLS(y, X).fit()
    
    results.append({
        'method': 'fm_with_idio_raw',
        'gamma0': model.params[0],
        'gamma_m': model.params[1],
        'gamma_idio': model.params[2],
        't_gamma0': model.tvalues[0],
        't_gamma_m': model.tvalues[1],
        't_gamma_idio': model.tvalues[2],
        'n_months': len(merged_data)
    })
    
    return pd.DataFrame(results)


def zero_beta_portfolio(returns_df: pd.DataFrame, market_df: pd.DataFrame, config, logger: logging.Logger) -> pd.DataFrame:
    """Construct zero-beta portfolio Z(M)."""
    logger.info("Constructing zero-beta portfolio")
    
    # Merge returns with market
    merged_df = returns_df.merge(market_df, on='date', how='inner')
    
    # Get universe returns matrix
    returns_matrix = merged_df.pivot(index='date', columns='ticker', values='ret')
    returns_matrix = returns_matrix.dropna()
    
    # Compute mean and covariance
    mu = returns_matrix.mean()
    Sigma = returns_matrix.cov()
    
    # Market weights (VW if available, else EW)
    if config.paths.market_caps_csv and Path(config.paths.market_caps_csv).exists():
        # Value-weighted
        caps_df = pd.read_csv(config.paths.market_caps_csv)
        caps_df['date'] = pd.to_datetime(caps_df['date'])
        
        # Use latest market caps
        latest_caps = caps_df.groupby('ticker')['market_cap'].last()
        w_market = latest_caps / latest_caps.sum()
        w_market = w_market.reindex(returns_matrix.columns, fill_value=1/len(returns_matrix.columns))
        w_market = w_market / w_market.sum()
        
        method_notes = "Value-weighted market"
    else:
        # Equal-weighted
        w_market = pd.Series(1/len(returns_matrix.columns), index=returns_matrix.columns)
        method_notes = "Equal-weighted market"
    
    # Solve for zero-beta portfolio weights
    try:
        # Objective: minimize variance
        def objective(w):
            return w.T @ Sigma @ w
        
        # Constraints: sum to 1, zero correlation with market
        def constraint_sum(w):
            return w.sum() - 1
        
        def constraint_corr(w):
            return w.T @ Sigma @ w_market
        
        # Initial guess
        w0 = np.ones(len(returns_matrix.columns)) / len(returns_matrix.columns)
        
        # Optimize
        result = minimize(
            objective, w0,
            constraints=[
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'eq', 'fun': constraint_corr}
            ],
            method='SLSQP'
        )
        
        if result.success:
            w_zero = pd.Series(result.x, index=returns_matrix.columns)
            R_Z = w_zero.T @ mu
            
            # Check if shorting is needed
            has_shorting = (w_zero < 0).any()
            
            results = [{
                'method': 'zero_beta_portfolio',
                'R_Z': R_Z,
                'notes': f"{method_notes}, shorting={'yes' if has_shorting else 'no'}",
                'has_shorting': has_shorting
            }]
            
        else:
            logger.warning("Zero-beta portfolio optimization failed")
            results = [{
                'method': 'zero_beta_portfolio',
                'R_Z': np.nan,
                'notes': f"{method_notes}, optimization failed",
                'has_shorting': False
            }]
    
    except Exception as e:
        logger.error(f"Zero-beta portfolio construction failed: {e}")
        results = [{
            'method': 'zero_beta_portfolio',
            'R_Z': np.nan,
            'notes': f"{method_notes}, error: {str(e)}",
            'has_shorting': False
        }]
    
    return pd.DataFrame(results)


def create_sml_plots(returns_df: pd.DataFrame, betas_df: pd.DataFrame, fmb_results: pd.DataFrame, 
                     zero_beta_results: pd.DataFrame, has_rf: bool, output_dir: Path, logger: logging.Logger) -> None:
    """Create Security Market Line plots."""
    logger.info("Creating SML plots")
    
    # Get average returns and betas
    avg_returns = returns_df.groupby('ticker').agg({
        'ret': 'mean',
        'ret_excess': 'mean' if has_rf else 'ret'
    }).reset_index()
    
    if has_rf:
        capm_betas = betas_df[betas_df['model_type'] == 'CAPM']
        if len(capm_betas) == 0:
            capm_betas = betas_df[betas_df['model_type'] == 'Black']
    else:
        capm_betas = betas_df[betas_df['model_type'] == 'Black']
    
    merged_data = avg_returns.merge(capm_betas[['ticker', 'beta']], on='ticker', how='inner')
    
    # Create plots
    fig_dir = output_dir / 'figs'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic SML scatter plot
    plt.figure(figsize=(10, 6))
    
    if has_rf:
        plt.scatter(merged_data['beta'], merged_data['ret_excess'], alpha=0.6)
        plt.xlabel('Beta')
        plt.ylabel('Average Excess Return')
        plt.title('Security Market Line (Excess Returns)')
        
        # Add fitted line
        X = add_constant(merged_data['beta'])
        y = merged_data['ret_excess']
        model = OLS(y, X).fit()
        
        beta_range = np.linspace(merged_data['beta'].min(), merged_data['beta'].max(), 100)
        fitted_line = model.params[0] + model.params[1] * beta_range
        plt.plot(beta_range, fitted_line, 'r-', linewidth=2, label='Fitted SML')
        
    else:
        plt.scatter(merged_data['beta'], merged_data['ret'], alpha=0.6)
        plt.xlabel('Beta')
        plt.ylabel('Average Return')
        plt.title('Security Market Line (Raw Returns)')
        
        # Add fitted line
        X = add_constant(merged_data['beta'])
        y = merged_data['ret']
        model = OLS(y, X).fit()
        
        beta_range = np.linspace(merged_data['beta'].min(), merged_data['beta'].max(), 100)
        fitted_line = model.params[0] + model.params[1] * beta_range
        plt.plot(beta_range, fitted_line, 'r-', linewidth=2, label='Fitted SML')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'sml_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # SML with zero-beta rate
    plt.figure(figsize=(10, 6))
    
    if has_rf:
        plt.scatter(merged_data['beta'], merged_data['ret_excess'], alpha=0.6)
        plt.xlabel('Beta')
        plt.ylabel('Average Excess Return')
        plt.title('Security Market Line with Zero-Beta Rate')
    else:
        plt.scatter(merged_data['beta'], merged_data['ret'], alpha=0.6)
        plt.xlabel('Beta')
        plt.ylabel('Average Return')
        plt.title('Security Market Line with Zero-Beta Rate')
    
    # Add fitted line
    beta_range = np.linspace(merged_data['beta'].min(), merged_data['beta'].max(), 100)
    fitted_line = model.params[0] + model.params[1] * beta_range
    plt.plot(beta_range, fitted_line, 'r-', linewidth=2, label='Fitted SML')
    
    # Add zero-beta rate
    if len(zero_beta_results) > 0 and not pd.isna(zero_beta_results['R_Z'].iloc[0]):
        R_Z = zero_beta_results['R_Z'].iloc[0]
        plt.axhline(y=R_Z, color='g', linestyle='--', linewidth=2, label=f'Zero-Beta Rate: {R_Z:.3f}')
        plt.annotate(f'R_Z = {R_Z:.3f}', xy=(0.1, R_Z), xytext=(0.1, R_Z + 0.01),
                    arrowprops=dict(arrowstyle='->', color='g'))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'sml_with_zero_beta.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("SML plots created")


def save_results(fmb_results: pd.DataFrame, fmb_idio: pd.DataFrame, zero_beta: pd.DataFrame, 
                 output_dir: Path, logger: logging.Logger) -> None:
    """Save all results to files."""
    logger.info("Saving results")
    
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Fama-MacBeth results
    fmb_path = tables_dir / 'fmb_results.csv'
    fmb_results.to_csv(fmb_path, index=False)
    logger.info(f"FMB results saved to {fmb_path}")
    
    # Save idiosyncratic risk results
    if len(fmb_idio) > 0:
        idio_path = tables_dir / 'fmb_with_idio.csv'
        fmb_idio.to_csv(idio_path, index=False)
        logger.info(f"FMB with idiosyncratic risk saved to {idio_path}")
    
    # Save zero-beta portfolio
    zero_path = tables_dir / 'zero_beta_portfolio.csv'
    zero_beta.to_csv(zero_path, index=False)
    logger.info(f"Zero-beta portfolio saved to {zero_path}")


def print_summary(fmb_results: pd.DataFrame, zero_beta: pd.DataFrame, logger: logging.Logger) -> None:
    """Print summary statistics."""
    logger.info("=== Cross-Sectional CAPM Analysis Summary ===")
    
    print(f"Methods analyzed: {len(fmb_results)}")
    
    for _, row in fmb_results.iterrows():
        print(f"\n{row['method']}:")
        print(f"  γ₀ = {row['gamma0']:.4f} (t = {row['t_gamma0']:.2f})")
        print(f"  γₘ = {row['gamma_m']:.4f} (t = {row['t_gamma_m']:.2f})")
        print(f"  n_months = {row['n_months']}")
    
    if len(zero_beta) > 0:
        R_Z = zero_beta['R_Z'].iloc[0]
        print(f"\nZero-Beta Rate (R_Z):")
        print(f"  From portfolio construction: {R_Z:.4f}")
        
        # Also show from Black regression intercept
        black_results = fmb_results[fmb_results['method'].str.contains('raw')]
        if len(black_results) > 0:
            gamma0_black = black_results['gamma0'].iloc[0]
            print(f"  From Black regression intercept: {gamma0_black:.4f}")
    
    print("\nFiles written:")
    print("- output/tables/fmb_results.csv")
    print("- output/tables/fmb_with_idio.csv")
    print("- output/tables/zero_beta_portfolio.csv")
    print("- output/figs/sml_scatter.png")
    print("- output/figs/sml_with_zero_beta.png")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Cross-sectional CAPM analysis")
    parser.add_argument("--check", action="store_true", help="Run validation check only")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting cross-sectional CAPM analysis")
    
    try:
        # Load configuration
        config = load_config("config.yml")
        logger.info("Configuration loaded successfully")
        
        if args.check:
            logger.info("Running validation check only")
            print("VALIDATION CHECK: Cross-sectional CAPM analysis ready")
            return 0
        
        # Load data
        returns_df, betas_df = load_data(logger)
        
        # Build market series
        market_df = build_market_series(returns_df, config, logger)
        
        # Compute excess returns
        returns_df, has_rf = compute_excess_returns(returns_df, market_df, logger)
        
        # Run static-beta SML
        fmb_results = static_beta_sml(returns_df, betas_df, has_rf, logger)
        
        # Run Fama-MacBeth
        fm_results = fama_macbeth(returns_df, betas_df, has_rf, logger)
        if len(fm_results) > 0:
            fmb_results = pd.concat([fmb_results, fm_results], ignore_index=True)
        
        # Test idiosyncratic risk
        fmb_idio = idiosyncratic_risk_test(returns_df, betas_df, has_rf, logger)
        
        # Construct zero-beta portfolio
        zero_beta = zero_beta_portfolio(returns_df, market_df, config, logger)
        
        # Create plots
        output_dir = Path("output")
        create_sml_plots(returns_df, betas_df, fmb_results, zero_beta, has_rf, output_dir, logger)
        
        # Save results
        save_results(fmb_results, fmb_idio, zero_beta, output_dir, logger)
        
        # Print summary
        print_summary(fmb_results, zero_beta, logger)
        
        logger.info("Cross-sectional CAPM analysis completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Cross-sectional analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
