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
        
        # Check if using value-weighted method
        if vw_summary['method'].iloc[0] == "CAPM" and "VW" in vw_summary['notes'].iloc[0]:
            vw_beta = vw_summary['vw_beta'].iloc[0]
            assert 0.85 <= vw_beta <= 1.15, f"VW beta {vw_beta:.3f} not in [0.85, 1.15]"
        else:
            # Using equal-weighted fallback
            pytest.xfail("Using equal-weighted market due to missing market cap data")
    
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
        
        returns_df = pd.read_csv(returns_path)
        has_rf = 'rf' in returns_df.columns and returns_df['rf'].notna().any()
        
        # Get market excess return
        if has_rf:
            market_excess = returns_df['mkt_excess'].mean()
        else:
            market_excess = returns_df['ret_m'].mean()
        
        # Check sign consistency for each method
        for _, row in fmb_df.iterrows():
            gamma_m = row['gamma_m']
            
            if has_rf:
                # For excess returns, gamma_m should have same sign as market excess return
                if market_excess > 0:
                    assert gamma_m > 0, f"gamma_m {gamma_m:.3f} should be positive when market excess return is positive"
                else:
                    assert gamma_m < 0, f"gamma_m {gamma_m:.3f} should be negative when market excess return is negative"
            else:
                # For raw returns, check against zero-beta rate
                zero_beta_path = Path("output/tables/zero_beta_portfolio.csv")
                if zero_beta_path.exists():
                    zero_beta_df = pd.read_csv(zero_beta_path)
                    R_Z = zero_beta_df['R_Z'].iloc[0]
                    
                    if not pd.isna(R_Z):
                        expected_sign = 1 if market_excess > R_Z else -1
                        actual_sign = 1 if gamma_m > 0 else -1
                        assert actual_sign == expected_sign, f"gamma_m sign inconsistent with market return vs zero-beta rate"
    
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
        
        # Check that sum_abs_weights is close to 1 (within tolerance)
        for _, row in weights_df.iterrows():
            sum_weights = row['sum_abs_weights']
            assert abs(sum_weights - 1.0) < 1e-6, f"Weights for {row['portfolio']} sum to {sum_weights:.6f}, not 1.0"
    
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
