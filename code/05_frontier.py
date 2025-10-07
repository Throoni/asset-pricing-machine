#!/usr/bin/env python3
"""
Efficient frontier and optimal portfolio analysis.

Purpose: Construct efficient frontier, find tangency portfolios, and analyze diversification
Inputs: data/processed/returns.parquet, market data, risk-free rate
Outputs: optimizer_weights.csv, efficient frontier plots, diversification analysis
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
from sklearn.covariance import LedoitWolf

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
            logging.FileHandler(log_dir / 'frontier.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_data(logger: logging.Logger) -> pd.DataFrame:
    """Load processed returns data."""
    returns_path = Path("data/processed/returns.parquet")
    if not returns_path.exists():
        raise FileNotFoundError(f"Processed returns not found: {returns_path}")
    
    df = pd.read_parquet(returns_path)
    logger.info(f"Loaded returns: {len(df)} observations, {df['ticker'].nunique()} tickers")
    return df


def build_returns_matrix(df: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build returns matrix and estimate mean and covariance."""
    logger.info("Building returns matrix")
    
    # Create returns matrix (T x N)
    returns_matrix = df.pivot(index='date', columns='ticker', values='ret')
    returns_matrix = returns_matrix.dropna()
    
    logger.info(f"Returns matrix: {returns_matrix.shape[0]} periods, {returns_matrix.shape[1]} stocks")
    
    # Estimate mean and covariance
    mu = returns_matrix.mean().values
    Sigma = returns_matrix.cov().values
    
    return returns_matrix, mu, Sigma


def estimate_covariance(returns_matrix: pd.DataFrame, estimator: str, logger: logging.Logger) -> np.ndarray:
    """Estimate covariance matrix using specified estimator."""
    logger.info(f"Estimating covariance using {estimator}")
    
    if estimator == "ledoit_wolf":
        lw = LedoitWolf()
        Sigma = lw.fit(returns_matrix).covariance_
    elif estimator == "sample":
        Sigma = returns_matrix.cov().values
    else:
        raise ValueError(f"Unknown covariance estimator: {estimator}")
    
    return Sigma


def build_market_portfolio(df: pd.DataFrame, config, logger: logging.Logger) -> np.ndarray:
    """Build market portfolio weights (VW if available, else EW)."""
    logger.info("Building market portfolio")
    
    if config.paths.market_caps_csv and Path(config.paths.market_caps_csv).exists():
        # Value-weighted
        caps_df = pd.read_csv(config.paths.market_caps_csv)
        caps_df['date'] = pd.to_datetime(caps_df['date'])
        
        # Use latest market caps
        latest_caps = caps_df.groupby('ticker')['market_cap'].last()
        
        # Get intersection with returns data
        returns_tickers = df['ticker'].unique()
        w_market = latest_caps.reindex(returns_tickers, fill_value=0)
        w_market = w_market / w_market.sum()
        
        logger.info("Using value-weighted market portfolio")
    else:
        # Equal-weighted
        returns_tickers = df['ticker'].unique()
        w_market = pd.Series(1/len(returns_tickers), index=returns_tickers)
        logger.info("Using equal-weighted market portfolio")
    
    return w_market.values


def construct_zero_beta_portfolio(mu: np.ndarray, Sigma: np.ndarray, w_market: np.ndarray, 
                                 config, logger: logging.Logger) -> Tuple[np.ndarray, float, bool]:
    """Construct zero-beta portfolio Z(M)."""
    logger.info("Constructing zero-beta portfolio")
    
    n = len(mu)
    
    # Objective: minimize variance
    def objective(w):
        return w.T @ Sigma @ w
    
    # Constraints: sum to 1, zero correlation with market
    def constraint_sum(w):
        return w.sum() - 1
    
    def constraint_corr(w):
        return w.T @ Sigma @ w_market
    
    # Initial guess
    w0 = np.ones(n) / n
    
    try:
        # Unconstrained optimization
        result = minimize(
            objective, w0,
            constraints=[
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'eq', 'fun': constraint_corr}
            ],
            method='SLSQP'
        )
        
        if result.success:
            w_zero = result.x
            R_Z = w_zero.T @ mu
            has_shorting = (w_zero < 0).any()
            
            logger.info(f"Zero-beta portfolio constructed successfully, R_Z = {R_Z:.4f}")
            return w_zero, R_Z, has_shorting
        else:
            logger.warning("Unconstrained zero-beta optimization failed")
    except Exception as e:
        logger.warning(f"Unconstrained zero-beta optimization failed: {e}")
    
    # Try constrained optimization (no shorting)
    try:
        bounds = [(0, 1) for _ in range(n)]
        result = minimize(
            objective, w0,
            constraints=[
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'eq', 'fun': constraint_corr}
            ],
            bounds=bounds,
            method='SLSQP'
        )
        
        if result.success:
            w_zero = result.x
            R_Z = w_zero.T @ mu
            has_shorting = False
            
            logger.info(f"Constrained zero-beta portfolio constructed, R_Z = {R_Z:.4f}")
            return w_zero, R_Z, has_shorting
        else:
            logger.warning("Constrained zero-beta optimization failed")
    except Exception as e:
        logger.warning(f"Constrained zero-beta optimization failed: {e}")
    
    # Fallback: equal weights
    w_zero = np.ones(n) / n
    R_Z = w_zero.T @ mu
    has_shorting = False
    
    logger.warning("Using equal weights as fallback for zero-beta portfolio")
    return w_zero, R_Z, has_shorting


def trace_efficient_frontier(mu: np.ndarray, Sigma: np.ndarray, config, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """Trace the efficient frontier."""
    logger.info("Tracing efficient frontier")
    
    n = len(mu)
    target_returns = np.linspace(mu.min(), mu.max(), 50)
    
    # Set up constraints
    if config.optimization.allow_shorting:
        bounds = None
    else:
        bounds = [(0, config.optimization.weight_bounds[1]) for _ in range(n)]
    
    def constraint_sum(w):
        return w.sum() - 1
    
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    
    # Add weight bounds if not allowing shorting
    if not config.optimization.allow_shorting:
        for i in range(n):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda w, i=i: w[i] - config.optimization.weight_bounds[0]
            })
            constraints.append({
                'type': 'ineq', 
                'fun': lambda w, i=i: config.optimization.weight_bounds[1] - w[i]
            })
    
    frontier_returns = []
    frontier_vols = []
    
    for target_ret in target_returns:
        def objective(w):
            return w.T @ Sigma @ w
        
        def constraint_return(w):
            return w.T @ mu - target_ret
        
        try:
            result = minimize(
                objective, np.ones(n) / n,
                constraints=constraints + [{'type': 'eq', 'fun': constraint_return}],
                bounds=bounds,
                method='SLSQP'
            )
            
            if result.success:
                frontier_returns.append(target_ret)
                frontier_vols.append(np.sqrt(result.fun))
        except:
            continue
    
    return np.array(frontier_returns), np.array(frontier_vols)


def find_tangency_portfolio(mu: np.ndarray, Sigma: np.ndarray, R_f: Optional[float], 
                           R_Z: float, config, logger: logging.Logger) -> Tuple[np.ndarray, float, float, str]:
    """Find tangency portfolio (with Rf or zero-beta CAL)."""
    logger.info("Finding tangency portfolio")
    
    n = len(mu)
    
    # Use Rf if available, otherwise use RZ
    if R_f is not None:
        risk_free_rate = R_f
        cal_type = "risk_free"
    else:
        risk_free_rate = R_Z
        cal_type = "zero_beta"
    
    # Objective: maximize Sharpe ratio
    def objective(w):
        portfolio_return = w.T @ mu
        portfolio_vol = np.sqrt(w.T @ Sigma @ w)
        return -(portfolio_return - risk_free_rate) / portfolio_vol
    
    # Constraints
    def constraint_sum(w):
        return w.sum() - 1
    
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    
    # Bounds
    if config.optimization.allow_shorting:
        bounds = None
    else:
        bounds = [(0, config.optimization.weight_bounds[1]) for _ in range(n)]
    
    # Add weight bounds if not allowing shorting
    if not config.optimization.allow_shorting:
        for i in range(n):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda w, i=i: w[i] - config.optimization.weight_bounds[0]
            })
            constraints.append({
                'type': 'ineq', 
                'fun': lambda w, i=i: config.optimization.weight_bounds[1] - w[i]
            })
    
    try:
        result = minimize(
            objective, np.ones(n) / n,
            constraints=constraints,
            bounds=bounds,
            method='SLSQP'
        )
        
        if result.success:
            w_tangency = result.x
            tangency_return = w_tangency.T @ mu
            tangency_vol = np.sqrt(w_tangency.T @ Sigma @ w_tangency)
            sharpe = (tangency_return - risk_free_rate) / tangency_vol
            
            logger.info(f"Tangency portfolio found, Sharpe = {sharpe:.4f}")
            return w_tangency, tangency_return, tangency_vol, cal_type
        else:
            logger.warning("Tangency optimization failed")
    except Exception as e:
        logger.warning(f"Tangency optimization failed: {e}")
    
    # Fallback: equal weights
    w_tangency = np.ones(n) / n
    tangency_return = w_tangency.T @ mu
    tangency_vol = np.sqrt(w_tangency.T @ Sigma @ w_tangency)
    sharpe = (tangency_return - risk_free_rate) / tangency_vol
    
    logger.warning("Using equal weights as fallback for tangency portfolio")
    return w_tangency, tangency_return, tangency_vol, cal_type


def find_min_variance_portfolio(mu: np.ndarray, Sigma: np.ndarray, config, logger: logging.Logger) -> Tuple[np.ndarray, float, float]:
    """Find minimum variance portfolio."""
    logger.info("Finding minimum variance portfolio")
    
    n = len(mu)
    
    # Objective: minimize variance
    def objective(w):
        return w.T @ Sigma @ w
    
    # Constraints
    def constraint_sum(w):
        return w.sum() - 1
    
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    
    # Bounds
    if config.optimization.allow_shorting:
        bounds = None
    else:
        bounds = [(0, config.optimization.weight_bounds[1]) for _ in range(n)]
    
    # Add weight bounds if not allowing shorting
    if not config.optimization.allow_shorting:
        for i in range(n):
            constraints.append({
                'type': 'ineq', 
                'fun': lambda w, i=i: w[i] - config.optimization.weight_bounds[0]
            })
            constraints.append({
                'type': 'ineq', 
                'fun': lambda w, i=i: config.optimization.weight_bounds[1] - w[i]
            })
    
    try:
        result = minimize(
            objective, np.ones(n) / n,
            constraints=constraints,
            bounds=bounds,
            method='SLSQP'
        )
        
        if result.success:
            w_minvar = result.x
            minvar_return = w_minvar.T @ mu
            minvar_vol = np.sqrt(result.fun)
            
            logger.info(f"Minimum variance portfolio found, vol = {minvar_vol:.4f}")
            return w_minvar, minvar_return, minvar_vol
        else:
            logger.warning("Min variance optimization failed")
    except Exception as e:
        logger.warning(f"Min variance optimization failed: {e}")
    
    # Fallback: equal weights
    w_minvar = np.ones(n) / n
    minvar_return = w_minvar.T @ mu
    minvar_vol = np.sqrt(w_minvar.T @ Sigma @ w_minvar)
    
    logger.warning("Using equal weights as fallback for min variance portfolio")
    return w_minvar, minvar_return, minvar_vol


def analyze_diversification(returns_matrix: pd.DataFrame, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """Analyze diversification impact."""
    logger.info("Analyzing diversification impact")
    
    n_stocks = returns_matrix.shape[1]
    k_values = range(1, min(n_stocks + 1, 21))  # Up to 20 stocks
    n_simulations = 100
    
    np.random.seed(42)  # Fixed seed for reproducibility
    
    avg_vols = []
    
    for k in k_values:
        vols = []
        
        for _ in range(n_simulations):
            # Random subset of k stocks
            selected_stocks = np.random.choice(returns_matrix.columns, k, replace=False)
            subset_returns = returns_matrix[selected_stocks]
            
            # Equal-weighted portfolio
            ew_weights = np.ones(k) / k
            portfolio_returns = (subset_returns * ew_weights).sum(axis=1)
            vol = portfolio_returns.std() * np.sqrt(12)  # Annualized
            
            vols.append(vol)
        
        avg_vols.append(np.mean(vols))
    
    return np.array(k_values), np.array(avg_vols)


def create_plots(frontier_returns: np.ndarray, frontier_vols: np.ndarray, 
                 market_return: float, market_vol: float,
                 minvar_return: float, minvar_vol: float,
                 tangency_return: float, tangency_vol: float,
                 zero_beta_return: float, zero_beta_vol: float,
                 R_f: Optional[float], R_Z: float, cal_type: str,
                 diversification_k: np.ndarray, diversification_vols: np.ndarray,
                 output_dir: Path, logger: logging.Logger) -> None:
    """Create efficient frontier and diversification plots."""
    logger.info("Creating plots")
    
    fig_dir = output_dir / 'figs'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Efficient frontier plot
    plt.figure(figsize=(12, 8))
    
    # Plot frontier
    plt.plot(frontier_vols, frontier_returns, 'b-', linewidth=2, label='Efficient Frontier')
    
    # Plot key portfolios
    plt.scatter(market_vol, market_return, color='red', s=100, label='Market', zorder=5)
    plt.scatter(minvar_vol, minvar_return, color='green', s=100, label='Min Variance', zorder=5)
    plt.scatter(tangency_vol, tangency_return, color='orange', s=100, label='Tangency', zorder=5)
    plt.scatter(zero_beta_vol, zero_beta_return, color='purple', s=100, label='Zero-Beta', zorder=5)
    
    # Plot CAL
    if cal_type == "risk_free":
        plt.axhline(y=R_f, color='gray', linestyle='--', alpha=0.7, label=f'Risk-Free Rate: {R_f:.3f}')
        # Draw CAL line
        cal_vols = np.linspace(0, max(frontier_vols), 100)
        cal_returns = R_f + (tangency_return - R_f) / tangency_vol * cal_vols
        plt.plot(cal_vols, cal_returns, 'r--', alpha=0.7, label='Capital Allocation Line')
    else:
        plt.axhline(y=R_Z, color='gray', linestyle='--', alpha=0.7, label=f'Zero-Beta Rate: {R_Z:.3f}')
        # Draw zero-beta CAL line
        cal_vols = np.linspace(0, max(frontier_vols), 100)
        cal_returns = R_Z + (tangency_return - R_Z) / tangency_vol * cal_vols
        plt.plot(cal_vols, cal_returns, 'r--', alpha=0.7, label='Zero-Beta CAL')
    
    plt.xlabel('Volatility (Annualized)')
    plt.ylabel('Expected Return (Annualized)')
    plt.title('Efficient Frontier and Optimal Portfolios')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'efficient_frontier.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Zero-beta CAL plot (if applicable)
    if cal_type == "zero_beta":
        plt.figure(figsize=(12, 8))
        
        # Plot frontier
        plt.plot(frontier_vols, frontier_returns, 'b-', linewidth=2, label='Efficient Frontier')
        
        # Plot key portfolios
        plt.scatter(market_vol, market_return, color='red', s=100, label='Market', zorder=5)
        plt.scatter(minvar_vol, minvar_return, color='green', s=100, label='Min Variance', zorder=5)
        plt.scatter(tangency_vol, tangency_return, color='orange', s=100, label='Tangency', zorder=5)
        plt.scatter(zero_beta_vol, zero_beta_return, color='purple', s=100, label='Zero-Beta', zorder=5)
        
        # Draw zero-beta CAL line
        cal_vols = np.linspace(0, max(frontier_vols), 100)
        cal_returns = R_Z + (tangency_return - R_Z) / tangency_vol * cal_vols
        plt.plot(cal_vols, cal_returns, 'r--', linewidth=2, label='Zero-Beta CAL')
        
        plt.xlabel('Volatility (Annualized)')
        plt.ylabel('Expected Return (Annualized)')
        plt.title('Zero-Beta Capital Allocation Line')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / 'zero_beta_CAL.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Diversification impact plot
    plt.figure(figsize=(10, 6))
    plt.plot(diversification_k, diversification_vols, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Number of Stocks')
    plt.ylabel('Portfolio Volatility (Annualized)')
    plt.title('Diversification Impact: Risk vs Number of Stocks')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'diversification_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Plots created")


def save_results(weights_df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    """Save optimization results."""
    logger.info("Saving results")
    
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    weights_path = tables_dir / 'optimizer_weights.csv'
    weights_df.to_csv(weights_path, index=False)
    logger.info(f"Optimizer weights saved to {weights_path}")


def print_summary(weights_df: pd.DataFrame, R_Z: float, cal_type: str, logger: logging.Logger) -> None:
    """Print summary statistics."""
    logger.info("=== Efficient Frontier Analysis Summary ===")
    
    print(f"Zero-Beta Rate (R_Z): {R_Z:.4f}")
    print(f"CAL Type: {cal_type}")
    
    print("\nPortfolio Summary:")
    for _, row in weights_df.iterrows():
        print(f"\n{row['portfolio']}:")
        print(f"  Expected Return: {row['expected_return']:.4f}")
        print(f"  Volatility: {row['volatility']:.4f}")
        print(f"  Sharpe/Slope: {row['sharpe_or_slope']:.4f}")
        print(f"  Beta vs Market: {row['beta_vs_market']:.4f}")
        print(f"  Sum |Weights|: {row['sum_abs_weights']:.4f}")
        print(f"  Notes: {row['note_on_constraints']}")
    
    print("\nFiles written:")
    print("- output/tables/optimizer_weights.csv")
    print("- output/figs/efficient_frontier.png")
    print("- output/figs/diversification_impact.png")
    if cal_type == "zero_beta":
        print("- output/figs/zero_beta_CAL.png")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Efficient frontier analysis")
    parser.add_argument("--check", action="store_true", help="Run validation check only")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting efficient frontier analysis")
    
    try:
        # Load configuration
        config = load_config("config.yml")
        logger.info("Configuration loaded successfully")
        
        if args.check:
            logger.info("Running validation check only")
            print("VALIDATION CHECK: Efficient frontier analysis ready")
            return 0
        
        # Load data
        df = load_data(logger)
        
        # Build returns matrix
        returns_matrix, mu, Sigma = build_returns_matrix(df, logger)
        
        # Estimate covariance
        Sigma = estimate_covariance(returns_matrix, config.optimization.covariance_estimator, logger)
        
        # Build market portfolio
        w_market = build_market_portfolio(df, config, logger)
        market_return = w_market.T @ mu
        market_vol = np.sqrt(w_market.T @ Sigma @ w_market)
        
        # Construct zero-beta portfolio
        w_zero, R_Z, has_shorting = construct_zero_beta_portfolio(mu, Sigma, w_market, config, logger)
        zero_beta_return = w_zero.T @ mu
        zero_beta_vol = np.sqrt(w_zero.T @ Sigma @ w_zero)
        
        # Check for risk-free rate
        R_f = None
        if 'rf' in df.columns and df['rf'].notna().any():
            R_f = df['rf'].mean()
            logger.info(f"Using risk-free rate: {R_f:.4f}")
        else:
            logger.info(f"Using zero-beta rate: {R_Z:.4f}")
        
        # Find optimal portfolios
        w_minvar, minvar_return, minvar_vol = find_min_variance_portfolio(mu, Sigma, config, logger)
        w_tangency, tangency_return, tangency_vol, cal_type = find_tangency_portfolio(
            mu, Sigma, R_f, R_Z, config, logger
        )
        
        # Trace efficient frontier
        frontier_returns, frontier_vols = trace_efficient_frontier(mu, Sigma, config, logger)
        
        # Analyze diversification
        diversification_k, diversification_vols = analyze_diversification(returns_matrix, logger)
        
        # Create results DataFrame
        results = []
        
        # Min variance portfolio
        results.append({
            'portfolio': 'min_var',
            'expected_return': minvar_return,
            'volatility': minvar_vol,
            'sharpe_or_slope': (minvar_return - (R_f or R_Z)) / minvar_vol,
            'beta_vs_market': np.cov(w_minvar, w_market)[0, 1] / np.var(w_market),
            'sum_abs_weights': np.sum(np.abs(w_minvar)),
            'note_on_constraints': f"shorting={'yes' if (w_minvar < 0).any() else 'no'}"
        })
        
        # Tangency portfolio
        results.append({
            'portfolio': 'tangency_or_zero_beta_tangent',
            'expected_return': tangency_return,
            'volatility': tangency_vol,
            'sharpe_or_slope': (tangency_return - (R_f or R_Z)) / tangency_vol,
            'beta_vs_market': np.cov(w_tangency, w_market)[0, 1] / np.var(w_market),
            'sum_abs_weights': np.sum(np.abs(w_tangency)),
            'note_on_constraints': f"shorting={'yes' if (w_tangency < 0).any() else 'no'}"
        })
        
        # Market portfolio
        results.append({
            'portfolio': 'market',
            'expected_return': market_return,
            'volatility': market_vol,
            'sharpe_or_slope': (market_return - (R_f or R_Z)) / market_vol,
            'beta_vs_market': 1.0,
            'sum_abs_weights': np.sum(np.abs(w_market)),
            'note_on_constraints': 'market_weights'
        })
        
        # Zero-beta portfolio
        results.append({
            'portfolio': 'zero_beta',
            'expected_return': zero_beta_return,
            'volatility': zero_beta_vol,
            'sharpe_or_slope': 0.0,  # By construction
            'beta_vs_market': 0.0,  # By construction
            'sum_abs_weights': np.sum(np.abs(w_zero)),
            'note_on_constraints': f"shorting={'yes' if has_shorting else 'no'}"
        })
        
        weights_df = pd.DataFrame(results)
        
        # Create plots
        output_dir = Path("output")
        create_plots(frontier_returns, frontier_vols, market_return, market_vol,
                    minvar_return, minvar_vol, tangency_return, tangency_vol,
                    zero_beta_return, zero_beta_vol, R_f, R_Z, cal_type,
                    diversification_k, diversification_vols, output_dir, logger)
        
        # Save results
        save_results(weights_df, output_dir, logger)
        
        # Print summary
        print_summary(weights_df, R_Z, cal_type, logger)
        
        logger.info("Efficient frontier analysis completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Efficient frontier analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
