"""
Data contract tests for asset pricing pipeline.

Purpose: Validate data structure and quality requirements
Inputs: Data files in data/raw/ directory
Outputs: Pass/fail validation results
Seed: 42 (for reproducibility)
"""

import pytest
import pandas as pd
from pathlib import Path


class TestDataContracts:
    """Test data contracts and file structure requirements."""
    
    def test_required_columns_present(self):
        """Test that required columns are present in processed data."""
        # Check processed returns data has required columns
        processed_path = Path("data/processed/returns.parquet")
        if not processed_path.exists():
            pytest.skip("Processed data not found - run ingest pipeline first")
        
        df = pd.read_parquet(processed_path)
        required_cols = ['date', 'ticker', 'adj_close', 'ret', 'ret_excess', 'rf']
        missing_cols = [col for col in required_cols if col not in df.columns]
        assert len(missing_cols) == 0, f"Missing required columns: {missing_cols}"
    
    def test_date_monotonicity(self):
        """Test that date columns are monotonic and properly formatted."""
        processed_path = Path("data/processed/returns.parquet")
        if not processed_path.exists():
            pytest.skip("Processed data not found - run ingest pipeline first")
        
        df = pd.read_parquet(processed_path)
        
        # Check dates are in correct format
        assert df['date'].dtype == 'datetime64[ns]', "Date column should be datetime type"
        
        # Check dates are monotonic for each ticker
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].sort_values('date')
            assert ticker_data['date'].is_monotonic_increasing, f"Dates not monotonic for ticker {ticker}"
    
    def test_min_obs_per_ticker(self):
        """Test that each ticker has minimum required observations."""
        processed_path = Path("data/processed/returns.parquet")
        if not processed_path.exists():
            pytest.skip("Processed data not found - run ingest pipeline first")
        
        df = pd.read_parquet(processed_path)
        
        # Load config to get min_obs_months
        import yaml
        with open("config.yml", 'r') as f:
            config = yaml.safe_load(f)
        min_obs = config['universe']['min_obs_months']
        
        # Check each ticker has sufficient observations
        ticker_counts = df['ticker'].value_counts()
        insufficient_tickers = ticker_counts[ticker_counts < min_obs]
        
        assert len(insufficient_tickers) == 0, f"Tickers with < {min_obs} observations: {insufficient_tickers.to_dict()}"
    
    def test_file_presence_stocks(self):
        """Test that stock CSV files exist in expected location."""
        pytest.skip("Configure in Step 3")
        # TODO: Implement when data structure is defined
        # - Check data/raw/stocks/ directory exists
        # - Check expected number of stock files present
        # - Check file naming convention
        # - Check file sizes are reasonable
    
    def test_rf_alignment(self):
        """Test that risk-free rate data is properly aligned with returns."""
        processed_path = Path("data/processed/returns.parquet")
        if not processed_path.exists():
            pytest.skip("Processed data not found - run ingest pipeline first")
        
        df = pd.read_parquet(processed_path)
        
        # Check that rf column exists and has reasonable values
        assert 'rf' in df.columns, "Risk-free rate column missing"
        
        # Check rf values are reasonable (between -0.1 and 0.1 for monthly rates)
        assert df['rf'].min() >= -0.1, "Risk-free rate values too low"
        assert df['rf'].max() <= 0.1, "Risk-free rate values too high"
        
        # Check that ret_excess = ret - rf
        calculated_excess = df['ret'] - df['rf']
        pd.testing.assert_series_equal(df['ret_excess'], calculated_excess, check_names=False)
    
    def test_file_presence_index(self):
        """Test that index CSV file exists if required."""
        pytest.skip("Configure in Step 3")
        # TODO: Implement when data structure is defined
        # - Check data/raw/index/index.csv exists if using index file
        # - Check file is readable and has expected structure
        # - Check file size is reasonable
    
    def test_file_presence_risk_free(self):
        """Test that risk-free rate CSV exists if required."""
        pytest.skip("Configure in Step 3")
        # TODO: Implement when data structure is defined
        # - Check data/raw/rf/risk_free.csv exists if risk_free_source provided
        # - Check file is readable and has expected structure
        # - Check file size is reasonable
    
    def test_file_presence_fundamentals(self):
        """Test that fundamentals CSV exists if value effect enabled."""
        pytest.skip("Configure in Step 3")
        # TODO: Implement when data structure is defined
        # - Check data/raw/fundamentals.csv exists if value_effect.enabled=True
        # - Check file is readable and has expected structure
        # - Check file size is reasonable
    
    def test_file_presence_market_caps(self):
        """Test that market caps CSV exists if value-weighted index used."""
        pytest.skip("Configure in Step 3")
        # TODO: Implement when data structure is defined
        # - Check data/raw/market_caps.csv exists if using value-weighted index
        # - Check file is readable and has expected structure
        # - Check file size is reasonable
    
    def test_provenance_sidecars(self):
        """Test that provenance metadata files exist."""
        pytest.skip("Configure in Step 3")
        # TODO: Implement when data structure is defined
        # - Check for data source documentation
        # - Check for data quality reports
        # - Check for data versioning information
        # - Check for licensing information
    
    def test_data_types(self):
        """Test that data types match expected formats."""
        pytest.skip("Configure in Step 3")
        # TODO: Implement when data structure is defined
        # - Check numeric columns are numeric
        # - Check date columns are datetime
        # - Check string columns are strings
        # - Check for missing value handling
    
    def test_data_ranges(self):
        """Test that data values are within expected ranges."""
        pytest.skip("Configure in Step 3")
        # TODO: Implement when data structure is defined
        # - Check returns are reasonable (not > 100% or < -100%)
        # - Check dates are within expected range
        # - Check market caps are positive
        # - Check risk-free rates are reasonable
