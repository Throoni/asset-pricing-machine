#!/usr/bin/env python3
"""
Validation and Intelligence Layer for Asset Pricing Machine

This module provides automated validation of model consistency, diagnostics,
and generates comprehensive reports for the asset pricing analysis.
"""

import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
def setup_logging():
    """Set up logging for validation module."""
    log_dir = Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "validation.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_data_files():
    """Load all required data files for validation."""
    logger = logging.getLogger(__name__)
    
    data = {}
    files = {
        'fmb_results': 'output/tables/fmb_results.csv',
        'zero_beta_portfolio': 'output/tables/zero_beta_portfolio.csv',
        'optimizer_weights': 'output/tables/optimizer_weights.csv',
        'betas': 'output/tables/betas.csv',
        'returns': 'data/processed/returns.parquet'
    }
    
    for key, filepath in files.items():
        path = Path(filepath)
        if path.exists():
            try:
                if filepath.endswith('.parquet'):
                    data[key] = pd.read_parquet(path)
                else:
                    data[key] = pd.read_csv(path)
                logger.info(f"Loaded {key}: {len(data[key])} rows")
            except Exception as e:
                logger.warning(f"Failed to load {key} from {filepath}: {e}")
                data[key] = None
        else:
            logger.warning(f"File not found: {filepath}")
            data[key] = None
    
    return data

def check_fmb_consistency(data):
    """Check consistency of Fama-MacBeth results across methods."""
    logger = logging.getLogger(__name__)
    logger.info("Checking Fama-MacBeth consistency")
    
    if data['fmb_results'] is None:
        logger.warning("FMB results not available for consistency check")
        return None
    
    df = data['fmb_results']
    
    # Extract gamma_m values for different methods
    gamma_m_values = {}
    for _, row in df.iterrows():
        method = row['method']
        gamma_m = row['gamma_m']
        t_gamma_m = row['t_gamma_m']
        gamma_m_values[method] = {
            'gamma_m': gamma_m,
            't_gamma_m': t_gamma_m,
            'significant': abs(t_gamma_m) > 1.96
        }
    
    # Check consistency
    consistency_checks = {
        'excess_vs_raw': None,
        'static_vs_fm': None,
        'sign_consistency': None,
        'magnitude_consistency': None
    }
    
    # Compare excess vs raw methods
    if 'static_excess' in gamma_m_values and 'static_raw' in gamma_m_values:
        excess_gamma = gamma_m_values['static_excess']['gamma_m']
        raw_gamma = gamma_m_values['static_raw']['gamma_m']
        consistency_checks['excess_vs_raw'] = {
            'excess_gamma': excess_gamma,
            'raw_gamma': raw_gamma,
            'difference': abs(excess_gamma - raw_gamma),
            'consistent': abs(excess_gamma - raw_gamma) < 0.01
        }
    
    # Compare static vs Fama-MacBeth
    if 'static_excess' in gamma_m_values and 'fm_excess' in gamma_m_values:
        static_gamma = gamma_m_values['static_excess']['gamma_m']
        fm_gamma = gamma_m_values['fm_excess']['gamma_m']
        consistency_checks['static_vs_fm'] = {
            'static_gamma': static_gamma,
            'fm_gamma': fm_gamma,
            'difference': abs(static_gamma - fm_gamma),
            'consistent': abs(static_gamma - fm_gamma) < 0.02
        }
    
    # Check sign consistency
    gamma_signs = [v['gamma_m'] for v in gamma_m_values.values()]
    all_positive = all(g > 0 for g in gamma_signs)
    all_negative = all(g < 0 for g in gamma_signs)
    consistency_checks['sign_consistency'] = {
        'all_positive': all_positive,
        'all_negative': all_negative,
        'consistent_signs': all_positive or all_negative,
        'signs': gamma_signs
    }
    
    # Check magnitude consistency (coefficient of variation)
    if len(gamma_signs) > 1:
        cv = np.std(gamma_signs) / abs(np.mean(gamma_signs)) if np.mean(gamma_signs) != 0 else np.inf
        consistency_checks['magnitude_consistency'] = {
            'coefficient_of_variation': cv,
            'consistent_magnitude': cv < 0.5
        }
    
    logger.info("FMB consistency check completed")
    return {
        'gamma_m_values': gamma_m_values,
        'consistency_checks': consistency_checks
    }

def check_zero_beta_consistency(data):
    """Check consistency of zero-beta rate estimates."""
    logger = logging.getLogger(__name__)
    logger.info("Checking zero-beta consistency")
    
    if data['zero_beta_portfolio'] is None:
        logger.warning("Zero-beta portfolio data not available")
        return None
    
    zb_df = data['zero_beta_portfolio']
    
    # Extract R_Z values
    rz_values = {}
    for _, row in zb_df.iterrows():
        method = row.get('method', 'unknown')
        rz = row.get('R_Z', np.nan)
        if not np.isnan(rz):
            rz_values[method] = rz
    
    # Check consistency between methods
    consistency_checks = {}
    if len(rz_values) > 1:
        rz_list = list(rz_values.values())
        mean_rz = np.mean(rz_list)
        std_rz = np.std(rz_list)
        cv = std_rz / abs(mean_rz) if mean_rz != 0 else np.inf
        
        consistency_checks = {
            'rz_values': rz_values,
            'mean_rz': mean_rz,
            'std_rz': std_rz,
            'coefficient_of_variation': cv,
            'consistent': cv < 0.2  # Within 20% variation
        }
    else:
        consistency_checks = {
            'rz_values': rz_values,
            'consistent': True,
            'note': 'Only one R_Z estimate available'
        }
    
    logger.info("Zero-beta consistency check completed")
    return consistency_checks

def compute_diagnostics(data):
    """Compute comprehensive diagnostics for the analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Computing diagnostics")
    
    diagnostics = {}
    
    # Portfolio diagnostics
    if data['optimizer_weights'] is not None:
        ow_df = data['optimizer_weights']
        
        # Sharpe ratios
        sharpe_ratios = {}
        for _, row in ow_df.iterrows():
            portfolio = row['portfolio']
            sharpe = row.get('sharpe_or_slope', np.nan)
            if not np.isnan(sharpe):
                sharpe_ratios[portfolio] = sharpe
        
        diagnostics['sharpe_ratios'] = sharpe_ratios
        
        # Portfolio concentration
        concentration = {}
        for _, row in ow_df.iterrows():
            portfolio = row['portfolio']
            sum_abs_weights = row.get('sum_abs_weights', np.nan)
            if not np.isnan(sum_abs_weights):
                concentration[portfolio] = {
                    'sum_abs_weights': sum_abs_weights,
                    'concentrated': sum_abs_weights > 2.0
                }
        
        diagnostics['concentration'] = concentration
    
    # Beta diagnostics
    if data['betas'] is not None:
        betas_df = data['betas']
        
        # Alpha analysis
        alpha_stats = {}
        for model_type in betas_df['model_type'].unique():
            model_data = betas_df[betas_df['model_type'] == model_type]
            alphas = model_data['alpha_or_a'].dropna()
            
            if len(alphas) > 0:
                alpha_stats[model_type] = {
                    'mean': alphas.mean(),
                    'std': alphas.std(),
                    'min': alphas.min(),
                    'max': alphas.max(),
                    'significant_positive': (alphas > 0).sum(),
                    'significant_negative': (alphas < 0).sum(),
                    'count': len(alphas)
                }
        
        diagnostics['alpha_stats'] = alpha_stats
        
        # Beta analysis
        beta_stats = {}
        for model_type in betas_df['model_type'].unique():
            model_data = betas_df[betas_df['model_type'] == model_type]
            betas = model_data['beta'].dropna()
            
            if len(betas) > 0:
                beta_stats[model_type] = {
                    'mean': betas.mean(),
                    'std': betas.std(),
                    'min': betas.min(),
                    'max': betas.max(),
                    'high_beta': (betas > 1.5).sum(),
                    'low_beta': (betas < 0.5).sum(),
                    'count': len(betas)
                }
        
        diagnostics['beta_stats'] = beta_stats
    
    # Returns diagnostics
    if data['returns'] is not None:
        returns_df = data['returns']
        
        # Market return analysis
        if 'ret_m' in returns_df.columns:
            market_returns = returns_df['ret_m'].dropna()
            diagnostics['market_returns'] = {
                'mean': market_returns.mean(),
                'std': market_returns.std(),
                'sharpe': market_returns.mean() / market_returns.std() if market_returns.std() > 0 else 0,
                'min': market_returns.min(),
                'max': market_returns.max(),
                'count': len(market_returns)
            }
        
        # Risk-free rate analysis
        if 'rf' in returns_df.columns:
            rf_rates = returns_df['rf'].dropna()
            diagnostics['risk_free'] = {
                'mean': rf_rates.mean(),
                'std': rf_rates.std(),
                'min': rf_rates.min(),
                'max': rf_rates.max(),
                'latest': rf_rates.iloc[-1] if len(rf_rates) > 0 else np.nan,
                'count': len(rf_rates)
            }
    
    logger.info("Diagnostics computation completed")
    return diagnostics

def generate_validation_report(fmb_consistency, zb_consistency, diagnostics):
    """Generate comprehensive validation report."""
    logger = logging.getLogger(__name__)
    logger.info("Generating validation report")
    
    # Create output directory
    output_dir = Path("output/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate HTML report
    html_content = generate_html_report(fmb_consistency, zb_consistency, diagnostics)
    
    # Save HTML report
    html_path = output_dir / "validation_report.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Validation report saved to {html_path}")
    
    # Print console summary
    print_console_summary(fmb_consistency, zb_consistency, diagnostics)
    
    return html_path

def generate_html_report(fmb_consistency, zb_consistency, diagnostics):
    """Generate HTML validation report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Asset Pricing Machine - Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .check {{ margin: 15px 0; padding: 10px; border-left: 4px solid #3498db; background-color: #f8f9fa; }}
            .pass {{ border-left-color: #27ae60; }}
            .fail {{ border-left-color: #e74c3c; }}
            .warning {{ border-left-color: #f39c12; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
            .metric {{ font-weight: bold; color: #2c3e50; }}
            .value {{ color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Asset Pricing Machine - Validation Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This report provides comprehensive validation of the asset pricing analysis, including model consistency checks, diagnostic metrics, and quality assessments.</p>
            </div>
    """
    
    # Fama-MacBeth Consistency Section
    if fmb_consistency:
        html += """
            <h2>Fama-MacBeth Consistency Analysis</h2>
        """
        
        # Gamma M values table
        html += "<h3>Market Risk Premium Estimates (γₘ)</h3>"
        html += "<table><tr><th>Method</th><th>γₘ</th><th>t-statistic</th><th>Significant</th></tr>"
        
        for method, values in fmb_consistency['gamma_m_values'].items():
            significance = "Yes" if values['significant'] else "No"
            html += f"<tr><td>{method}</td><td>{values['gamma_m']:.4f}</td><td>{values['t_gamma_m']:.2f}</td><td>{significance}</td></tr>"
        
        html += "</table>"
        
        # Consistency checks
        html += "<h3>Consistency Checks</h3>"
        checks = fmb_consistency['consistency_checks']
        
        if checks['excess_vs_raw']:
            check = checks['excess_vs_raw']
            status = "pass" if check['consistent'] else "fail"
            html += f"""
            <div class="check {status}">
                <strong>Excess vs Raw Methods:</strong> 
                Difference = {check['difference']:.4f} 
                ({'✓ Consistent' if check['consistent'] else '✗ Inconsistent'})
            </div>
            """
        
        if checks['static_vs_fm']:
            check = checks['static_vs_fm']
            status = "pass" if check['consistent'] else "fail"
            html += f"""
            <div class="check {status}">
                <strong>Static vs Fama-MacBeth:</strong> 
                Difference = {check['difference']:.4f} 
                ({'✓ Consistent' if check['consistent'] else '✗ Inconsistent'})
            </div>
            """
        
        if checks['sign_consistency']:
            check = checks['sign_consistency']
            status = "pass" if check['consistent_signs'] else "warning"
            html += f"""
            <div class="check {status}">
                <strong>Sign Consistency:</strong> 
                {'✓ All signs consistent' if check['consistent_signs'] else '⚠ Mixed signs'} 
                (Signs: {check['signs']})
            </div>
            """
    
    # Zero-Beta Consistency Section
    if zb_consistency:
        html += """
            <h2>Zero-Beta Rate Consistency Analysis</h2>
        """
        
        html += "<table><tr><th>Method</th><th>R_Z</th></tr>"
        for method, rz in zb_consistency['rz_values'].items():
            html += f"<tr><td>{method}</td><td>{rz:.4f}</td></tr>"
        html += "</table>"
        
        if 'coefficient_of_variation' in zb_consistency:
            cv = zb_consistency['coefficient_of_variation']
            status = "pass" if zb_consistency['consistent'] else "warning"
            html += f"""
            <div class="check {status}">
                <strong>R_Z Consistency:</strong> 
                CV = {cv:.3f} 
                ({'✓ Consistent' if zb_consistency['consistent'] else '⚠ High variation'})
            </div>
            """
    
    # Diagnostics Section
    if diagnostics:
        html += """
            <h2>Diagnostic Metrics</h2>
        """
        
        # Sharpe Ratios
        if 'sharpe_ratios' in diagnostics:
            html += "<h3>Portfolio Sharpe Ratios</h3>"
            html += "<table><tr><th>Portfolio</th><th>Sharpe Ratio</th></tr>"
            for portfolio, sharpe in diagnostics['sharpe_ratios'].items():
                html += f"<tr><td>{portfolio}</td><td>{sharpe:.3f}</td></tr>"
            html += "</table>"
        
        # Alpha Statistics
        if 'alpha_stats' in diagnostics:
            html += "<h3>Alpha Statistics by Model</h3>"
            html += "<table><tr><th>Model</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Count</th></tr>"
            for model, stats in diagnostics['alpha_stats'].items():
                html += f"<tr><td>{model}</td><td>{stats['mean']:.4f}</td><td>{stats['std']:.4f}</td><td>{stats['min']:.4f}</td><td>{stats['max']:.4f}</td><td>{stats['count']}</td></tr>"
            html += "</table>"
        
        # Beta Statistics
        if 'beta_stats' in diagnostics:
            html += "<h3>Beta Statistics by Model</h3>"
            html += "<table><tr><th>Model</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Count</th></tr>"
            for model, stats in diagnostics['beta_stats'].items():
                html += f"<tr><td>{model}</td><td>{stats['mean']:.3f}</td><td>{stats['std']:.3f}</td><td>{stats['min']:.3f}</td><td>{stats['max']:.3f}</td><td>{stats['count']}</td></tr>"
            html += "</table>"
        
        # Market Returns
        if 'market_returns' in diagnostics:
            mr = diagnostics['market_returns']
            html += f"""
            <h3>Market Return Analysis</h3>
            <div class="check pass">
                <strong>Mean Return:</strong> {mr['mean']:.4f} | 
                <strong>Volatility:</strong> {mr['std']:.4f} | 
                <strong>Sharpe:</strong> {mr['sharpe']:.3f}
            </div>
            """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

def print_console_summary(fmb_consistency, zb_consistency, diagnostics):
    """Print validation summary to console."""
    print("\n" + "="*60)
    print("ASSET PRICING MACHINE - VALIDATION SUMMARY")
    print("="*60)
    
    # Fama-MacBeth Summary
    if fmb_consistency:
        print("\n📊 FAMA-MACBETH CONSISTENCY:")
        for method, values in fmb_consistency['gamma_m_values'].items():
            sig = "✓" if values['significant'] else "○"
            print(f"  {method:15} γₘ = {values['gamma_m']:7.4f} (t={values['t_gamma_m']:5.2f}) {sig}")
        
        checks = fmb_consistency['consistency_checks']
        if checks['excess_vs_raw']:
            status = "✓" if checks['excess_vs_raw']['consistent'] else "✗"
            print(f"  Excess vs Raw:     {status} (diff = {checks['excess_vs_raw']['difference']:.4f})")
        
        if checks['static_vs_fm']:
            status = "✓" if checks['static_vs_fm']['consistent'] else "✗"
            print(f"  Static vs FM:      {status} (diff = {checks['static_vs_fm']['difference']:.4f})")
    
    # Zero-Beta Summary
    if zb_consistency:
        print("\n🎯 ZERO-BETA CONSISTENCY:")
        for method, rz in zb_consistency['rz_values'].items():
            print(f"  {method:15} R_Z = {rz:.4f}")
        
        if 'coefficient_of_variation' in zb_consistency:
            cv = zb_consistency['coefficient_of_variation']
            status = "✓" if zb_consistency['consistent'] else "⚠"
            print(f"  Consistency:       {status} (CV = {cv:.3f})")
    
    # Diagnostics Summary
    if diagnostics:
        print("\n📈 DIAGNOSTICS:")
        
        if 'sharpe_ratios' in diagnostics:
            print("  Portfolio Sharpe Ratios:")
            for portfolio, sharpe in diagnostics['sharpe_ratios'].items():
                print(f"    {portfolio:20} {sharpe:.3f}")
        
        if 'alpha_stats' in diagnostics:
            print("  Alpha Statistics:")
            for model, stats in diagnostics['alpha_stats'].items():
                print(f"    {model:15} mean={stats['mean']:7.4f}, std={stats['std']:7.4f}")
        
        if 'market_returns' in diagnostics:
            mr = diagnostics['market_returns']
            print(f"  Market: mean={mr['mean']:.4f}, vol={mr['std']:.4f}, Sharpe={mr['sharpe']:.3f}")
    
    print("\n" + "="*60)
    print("Validation complete! Check output/reports/validation_report.html for detailed report.")
    print("="*60)

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Asset Pricing Machine Validation")
    parser.add_argument("--check", action="store_true", help="Run validation checks only")
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting validation analysis")
    
    # Load data
    data = load_data_files()
    
    # Run consistency checks
    fmb_consistency = check_fmb_consistency(data)
    zb_consistency = check_zero_beta_consistency(data)
    
    # Compute diagnostics
    diagnostics = compute_diagnostics(data)
    
    if args.check:
        # Just print summary for --check
        print_console_summary(fmb_consistency, zb_consistency, diagnostics)
    else:
        # Generate full report
        report_path = generate_validation_report(fmb_consistency, zb_consistency, diagnostics)
        logger.info(f"Validation analysis completed. Report saved to {report_path}")

if __name__ == "__main__":
    main()
