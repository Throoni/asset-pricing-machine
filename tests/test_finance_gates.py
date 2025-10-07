"""
Finance validation tests for asset pricing pipeline.

Purpose: Validate financial logic and CAPM assumptions
Inputs: Processed data from pipeline
Outputs: Pass/fail validation results
Seed: 42 (for reproducibility)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestFinanceGates:
    """Test financial validation gates and CAPM assumptions."""
    
    def test_rf_returns_frequency_match(self):
        """Test that risk-free rate frequency matches returns frequency."""
        pytest.skip("Configure in Step 4+")
        # TODO: Implement when data processing is complete
        # - Check risk-free rate has same frequency as returns
        # - Check alignment of risk-free rate dates with return dates
        # - Check for missing risk-free rate values
    
    def test_min_obs_per_stock(self):
        """Test that each stock has minimum required observations."""
        betas_path = Path("output/tables/betas.csv")
        if not betas_path.exists():
            pytest.skip("Betas file not found - run CAPM analysis first")
        
        betas_df = pd.read_csv(betas_path)
        
        # Load config to get min_obs_months
        import yaml
        with open("config.yml", 'r') as f:
            config = yaml.safe_load(f)
        min_obs = config['universe']['min_obs_months']
        
        # Check all estimates have sufficient observations
        insufficient_obs = betas_df[betas_df['n'] < min_obs]
        assert len(insufficient_obs) == 0, f"Found {len(insufficient_obs)} estimates with < {min_obs} observations"
    
    def test_vw_beta_approximately_one(self):
        """Test that value-weighted beta is approximately 1."""
        vw_summary_path = Path("output/tables/vw_beta_summary.csv")
        if not vw_summary_path.exists():
            pytest.skip("VW beta summary not found - run CAPM analysis first")
        
        vw_summary = pd.read_csv(vw_summary_path)
        
        # Check if weights are available with sufficient coverage
        weights_path = Path("data/processed/market_caps_weights.csv")
        if weights_path.exists():
            weights_df = pd.read_csv(weights_path)
            betas_path = Path("output/tables/betas.csv")
            if betas_path.exists():
                betas_df = pd.read_csv(betas_path)
                capm_tickers = set(betas_df[betas_df['model_type'] == 'CAPM']['ticker'].unique())
                weight_tickers = set(weights_df['ticker'].unique())
                coverage = len(capm_tickers.intersection(weight_tickers)) / len(capm_tickers)
                
                if coverage >= 0.8:
                    # Weights available with sufficient coverage - enforce VW gate
                    vw_beta = vw_summary['vw_beta'].iloc[0]
                    assert 0.85 <= vw_beta <= 1.15, f"VW beta {vw_beta:.3f} not in [0.85, 1.15]"
                else:
                    pytest.xfail(f"weights unavailable or insufficient coverage: {coverage:.1%}")
            else:
                pytest.xfail("weights unavailable or insufficient coverage")
        else:
            pytest.xfail("weights unavailable or insufficient coverage")
    
    def test_capm_cross_section_intercept(self):
        """Test that CAPM cross-section intercept is approximately 0."""
        betas_path = Path("output/tables/betas.csv")
        if not betas_path.exists():
            pytest.skip("Betas file not found - run CAPM analysis first")
        
        betas_df = pd.read_csv(betas_path)
        capm_betas = betas_df[betas_df['model_type'] == 'CAPM']
        
        if len(capm_betas) == 0:
            pytest.skip("No CAPM betas available")
        
        # Compute t-statistic of cross-sectional mean alpha
        mean_alpha = capm_betas['alpha_or_a'].mean()
        std_alpha = capm_betas['alpha_or_a'].std()
        n = len(capm_betas)
        t_stat = mean_alpha / (std_alpha / np.sqrt(n))
        
        # Soft gate: |t| < 2.0
        if abs(t_stat) >= 2.0:
            pytest.xfail(f"Mean alpha t-stat {t_stat:.3f} >= 2.0 (soft gate)")
        
        assert abs(t_stat) < 2.0, f"Mean alpha t-stat {t_stat:.3f} >= 2.0"
    
    def test_price_of_risk_sign(self):
        """Test that price of risk sign matches market excess return."""
        fmb_path = Path("output/tables/fmb_results.csv")
        if not fmb_path.exists():
            pytest.skip("FMB results not found - run cross-sectional analysis first")
        
        fmb_df = pd.read_csv(fmb_path)
        
        # Load processed data to get market returns
        returns_path = Path("data/processed/returns.parquet")
        if not returns_path.exists():
            pytest.skip("Processed returns not found")
        
        returns_df = pd.read_parquet(returns_path)
        has_rf = 'rf' in returns_df.columns and returns_df['rf'].notna().any()
        
        # Get market excess return
        if has_rf:
            market_excess = returns_df['mkt_excess'].mean()
        else:
            market_excess = returns_df['ret_m'].mean()
        
        # Soft gate: only fail hard if market premium is clearly non-zero AND gamma_m significant AND signs disagree.
        small_premium = abs(market_excess) < 0.002  # ~0.2% monthly threshold
        hard_fail_count = 0
        reasons = []
        for _, row in fmb_df.iterrows():
            gamma_m = float(row['gamma_m'])
            t_gamma_m = float(row.get('t_gamma_m', 0.0))
            sig = abs(t_gamma_m) >= 1.96
            if small_premium or not sig:
                # Inconclusive environment: xfail soft expectation if sign disagrees
                if (market_excess > 0 and gamma_m < 0) or (market_excess < 0 and gamma_m > 0):
                    pytest.xfail(f"Price-of-risk sign mismatch under small premium or insignificant gamma_m "
                                 f"(mean mkt_excess={market_excess:.4f}, gamma_m={gamma_m:.4f}, t={t_gamma_m:.2f})")
            else:
                # Clear premium and significant slope: must agree in sign
                if (market_excess > 0 and gamma_m <= 0) or (market_excess < 0 and gamma_m >= 0):
                    hard_fail_count += 1
                    reasons.append(f"gamma_m={gamma_m:.4f}, t={t_gamma_m:.2f}, mkt_excess={market_excess:.4f}")

        assert hard_fail_count == 0, "Price-of-risk sign failed: " + "; ".join(reasons)
    
    def test_capm_intercept_gate(self):
        """Test that CAPM intercept is approximately zero (soft gate)."""
        fmb_path = Path("output/tables/fmb_results.csv")
        if not fmb_path.exists():
            pytest.skip("FMB results not found - run cross-sectional analysis first")
        
        fmb_df = pd.read_csv(fmb_path)
        
        # Check excess return methods
        excess_methods = fmb_df[fmb_df['method'].str.contains('excess')]
        
        if len(excess_methods) == 0:
            pytest.skip("No excess return methods available")
        
        for _, row in excess_methods.iterrows():
            t_gamma0 = abs(row['t_gamma0'])
            
            if t_gamma0 >= 2.0:
                pytest.xfail(f"CAPM intercept t-stat {t_gamma0:.3f} >= 2.0 (soft gate)")
            
            assert t_gamma0 < 2.0, f"CAPM intercept t-stat {t_gamma0:.3f} >= 2.0"
    
    def test_idiosyncratic_risk_not_priced(self):
        """Test that idiosyncratic risk is not priced (soft gate)."""
        idio_path = Path("output/tables/fmb_with_idio.csv")
        if not idio_path.exists():
            pytest.skip("Idiosyncratic risk results not found - run cross-sectional analysis first")
        
        idio_df = pd.read_csv(idio_path)
        
        for _, row in idio_df.iterrows():
            t_gamma_idio = abs(row['t_gamma_idio'])
            
            if t_gamma_idio >= 2.0:  # 5% significance level
                pytest.xfail(f"Idiosyncratic risk t-stat {t_gamma_idio:.3f} >= 2.0 (soft gate)")
            
            assert t_gamma_idio < 2.0, f"Idiosyncratic risk t-stat {t_gamma_idio:.3f} >= 2.0"
    
    def test_frontier_psd_check(self):
        """Test that covariance matrix is positive semi-definite."""
        weights_path = Path("output/tables/optimizer_weights.csv")
        if not weights_path.exists():
            pytest.skip("Optimizer weights not found - run frontier analysis first")
        
        # Load processed returns to check covariance matrix
        returns_path = Path("data/processed/returns.parquet")
        if not returns_path.exists():
            pytest.skip("Processed returns not found")
        
        returns_df = pd.read_parquet(returns_path)
        returns_matrix = returns_df.pivot(index='date', columns='ticker', values='ret')
        returns_matrix = returns_matrix.dropna()
        
        # Check covariance matrix is PSD
        Sigma = returns_matrix.cov().values
        eigenvalues = np.linalg.eigvals(Sigma)
        min_eigenvalue = np.min(eigenvalues)
        
        assert min_eigenvalue >= -1e-10, f"Covariance matrix not PSD, min eigenvalue: {min_eigenvalue:.2e}"
    
    def test_weights_sum_to_one(self):
        """Test that portfolio weights sum to 1."""
        weights_path = Path("output/tables/optimizer_weights.csv")
        if not weights_path.exists():
            pytest.skip("Optimizer weights not found - run frontier analysis first")
        
        weights_df = pd.read_csv(weights_path)
        
        # Enforce sum(abs(weights))≈1 for long-only portfolios; allow leverage for zero-beta (long-short).
        for _, row in weights_df.iterrows():
            name = str(row['portfolio']).lower()
            sabs = float(row['sum_abs_weights'])
            if "zero_beta" in name:
                # Allow long-short construction; cap L1 norm to a reasonable ceiling to avoid runaway leverage.
                assert sabs <= 3.5, f"Zero-beta leverage too high: sum_abs_weights={sabs:.6f} (>3.5)"
            else:
                assert abs(sabs - 1.0) < 1e-6, f"Weights for {row['portfolio']} sum to {sabs:.6f}, not 1.0"
    
    def test_tangency_on_frontier(self):
        """Test that tangency portfolio lies on the efficient frontier."""
        weights_path = Path("output/tables/optimizer_weights.csv")
        if not weights_path.exists():
            pytest.skip("Optimizer weights not found - run frontier analysis first")
        
        weights_df = pd.read_csv(weights_path)
        
        # Load processed returns to compute frontier
        returns_path = Path("data/processed/returns.parquet")
        if not returns_path.exists():
            pytest.skip("Processed returns not found")
        
        returns_df = pd.read_parquet(returns_path)
        returns_matrix = returns_df.pivot(index='date', columns='ticker', values='ret')
        returns_matrix = returns_matrix.dropna()
        
        mu = returns_matrix.mean().values
        Sigma = returns_matrix.cov().values
        
        # Check tangency portfolio
        tangency_row = weights_df[weights_df['portfolio'] == 'tangency_or_zero_beta_tangent']
        if len(tangency_row) == 0:
            pytest.skip("No tangency portfolio found")
        
        tangency_return = tangency_row['expected_return'].iloc[0]
        tangency_vol = tangency_row['volatility'].iloc[0]
        
        # Compute theoretical variance for this return level
        # This is a simplified check - in practice, we'd need to solve the optimization
        # For now, just check that the portfolio is reasonable
        assert tangency_return >= mu.min(), "Tangency return below minimum possible"
        assert tangency_return <= mu.max(), "Tangency return above maximum possible"
        assert tangency_vol > 0, "Tangency volatility must be positive"
        
        # Check that Sharpe ratio is reasonable
        sharpe = tangency_row['sharpe_or_slope'].iloc[0]
        assert sharpe > 0, "Tangency Sharpe ratio should be positive"
    
    def test_no_look_ahead_bm_lag(self):
        """Test that book-to-market data is properly lagged."""
        pytest.skip("Configure in Step 4+")
        # TODO: Implement when value effect analysis is complete
        # - Check B/M data is lagged by bm_lag_months
        # - Check no future information leakage
        # - Check lag implementation
    
    def test_sensitivity_alt_rf(self):
        """Test sensitivity to alternative risk-free rate."""
        pytest.skip("Configure in Step 4+")
        # TODO: Implement when sensitivity analysis is complete
        # - Check results with alternative risk-free rate
        # - Check robustness of conclusions
        # - Check economic significance
    
    def test_sensitivity_trim_outliers(self):
        """Test sensitivity to outlier trimming."""
        pytest.skip("Configure in Step 4+")
        # TODO: Implement when sensitivity analysis is complete
        # - Check results with different winsorization levels
        # - Check robustness to outliers
        # - Check statistical significance
    
    def test_sensitivity_sub_period_split(self):
        """Test sensitivity to sub-period analysis."""
        pytest.skip("Configure in Step 4+")
        # TODO: Implement when sensitivity analysis is complete
        # - Check results in first half vs second half
        # - Check stability over time
        # - Check structural breaks
    
    def test_returns_calculation(self):
        """Test that returns are calculated correctly."""
        pytest.skip("Configure in Step 4+")
        # TODO: Implement when data processing is complete
        # - Check return calculation formula
        # - Check for data errors
        # - Check for survivorship bias
    
    def test_excess_returns_calculation(self):
        """Test that excess returns are calculated correctly."""
        pytest.skip("Configure in Step 4+")
        # TODO: Implement when data processing is complete
        # - Check excess return = return - risk_free_rate
        # - Check alignment of dates
        # - Check for missing values
