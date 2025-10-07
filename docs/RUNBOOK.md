# Runbook

This document provides step-by-step commands to run the asset pricing analysis pipeline.

## Prerequisites

1. **Python Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Data Preparation:**
   - Place 25 stock CSV files in `data/raw/stocks/`
   - Place index data in `data/raw/index/index.csv` (if using index file)
   - Place risk-free rate data in `data/raw/rf/risk_free.csv` (if available)
   - Place fundamentals data in `data/raw/fundamentals.csv` (if available)
   - Place market cap data in `data/raw/market_caps.csv` (if available)

## Configuration

1. **Edit Configuration:**
   ```bash
   # Edit config.yml to fill in TODO values
   nano config.yml
   ```

2. **Validate Configuration:**
   ```bash
   python code/01_config.py --check config.yml
   ```

## Pipeline Execution

### Step 1: Data Ingestion and Cleaning
```bash
# Validate data before processing
python code/02_ingest_clean.py --check

# Run full data ingestion and cleaning
python code/02_ingest_clean.py
```

### Step 2: Time Series CAPM Analysis
```bash
# Validate CAPM analysis before running
python code/03_capm_timeseries.py --check

# Run full CAPM time-series analysis
python code/03_capm_timeseries.py
```

### Step 3: Cross-Sectional CAPM Test
```bash
# Validate cross-sectional analysis before running
python code/04_capm_crosssection.py --check

# Run full cross-sectional CAPM analysis
python code/04_capm_crosssection.py
```

### Step 4: Efficient Frontier Analysis
```bash
# Validate frontier analysis before running
python code/05_frontier.py --check

# Run full efficient frontier analysis
python code/05_frontier.py
```

### Step 5: Value Effect Analysis
```bash
# TODO: Command will be added in Step 7
python code/06_value_alpha.py
```

### Step 6: Generate Report Assets
```bash
# TODO: Command will be added in Step 8
python code/07_build_report_assets.py
```

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Data contract tests
pytest tests/test_data_contracts.py -v

# Finance validation tests
pytest tests/test_finance_gates.py -v

# Pipeline integrity tests
pytest tests/test_pipeline_integrity.py -v
```

## Data File Formats

### Stock CSV Files
- **Location:** `data/raw/stocks/`
- **Format:** One CSV per stock
- **Required Columns:** `date`, `ticker`, `adj_close`
- **Naming:** `{TICKER}.csv` (e.g., `ABI.BR.csv`)
- **Date Format:** YYYY-MM-DD
- **Price Format:** Adjusted close prices (decimal)

### Index CSV File
- **Location:** `data/raw/index/index.csv`
- **Format:** Single CSV with index data
- **Required Columns:** TODO - Will be defined in Step 3

### Risk-Free Rate CSV
- **Location:** `data/raw/rf/risk_free.csv`
- **Format:** Single CSV with risk-free rate data
- **Required Columns:** TODO - Will be defined in Step 3

### Fundamentals CSV
- **Location:** `data/raw/fundamentals.csv`
- **Format:** Single CSV with book-to-market data
- **Required Columns:** TODO - Will be defined in Step 3

### Market Cap CSV
- **Location:** `data/raw/market_caps.csv`
- **Format:** Single CSV with market cap data
- **Required Columns:** TODO - Will be defined in Step 3

## Output Files

### Tables
- **Location:** `output/tables/`
- **Format:** CSV files with analysis results
- **Files:** 
  - `returns_summary.csv` - Summary statistics for each ticker
  - `betas.csv` - Alpha and beta estimates for each stock
  - `vw_beta_summary.csv` - Value-weighted and equal-weighted beta summary
  - `fmb_results.csv` - Fama-MacBeth cross-sectional regression results
  - `fmb_with_idio.csv` - Fama-MacBeth with idiosyncratic risk test
  - `zero_beta_portfolio.csv` - Zero-beta portfolio construction results
  - `optimizer_weights.csv` - Portfolio optimization results and weights
  - TODO - Additional tables will be defined as pipeline develops

### Processed Data
- **Location:** `data/processed/`
- **Format:** Parquet files with processed data
- **Files:**
  - `returns.parquet` - Cleaned stock returns with excess returns

### Figures
- **Location:** `output/figs/`
- **Format:** PNG/PDF files with charts and plots
- **Files:** 
  - `beta_hist.png` - Histogram of stock betas
  - `sml_scatter.png` - Security Market Line scatter plot
  - `sml_with_zero_beta.png` - SML with zero-beta rate annotation
  - `efficient_frontier.png` - Efficient frontier with optimal portfolios
  - `zero_beta_CAL.png` - Zero-beta Capital Allocation Line (if applicable)
  - `diversification_impact.png` - Diversification impact analysis
  - TODO - Additional figures will be defined as pipeline develops

### Summary
- **Location:** `output/summary.json`
- **Format:** JSON with key metrics and validation flags
- **Purpose:** Quick status check and CI validation

## Troubleshooting

### Common Issues
1. **Configuration Errors:**
   - Run `python code/01_config.py --check config.yml`
   - Check all TODO values are filled

2. **Data Issues:**
   - Verify data files are in correct locations
   - Check file formats match requirements
   - Run data contract tests

3. **Dependency Issues:**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

### Log Files
- **Location:** `output/logs/`
- **Format:** Text files with detailed execution logs
- **Purpose:** Debugging and audit trail

## Interpreting Results

### Cross-Sectional Analysis (Fama-MacBeth)

**γ₀ (gamma-zero) - Intercept:**
- **CAPM excess returns:** Should be close to zero (no abnormal returns)
- **Black raw returns:** Should be close to zero-beta rate (RZ)
- **Significance:** |t-stat| < 2.0 indicates CAPM holds

**γₘ (gamma-m) - Market Risk Premium:**
- **CAPM excess returns:** Market risk premium (expected excess return for beta=1)
- **Black raw returns:** Market risk premium above zero-beta rate
- **Significance:** Should be positive and significant

**Zero-Beta Rate (RZ):**
- **From regression:** Intercept of Black raw returns regression
- **From portfolio:** Expected return of zero-beta portfolio
- **Interpretation:** Risk-free rate in Black model (replaces risk-free rate)

### Idiosyncratic Risk Test
- **Theory:** Idiosyncratic risk should not be priced (γ_idio ≈ 0)
- **Test:** |t-stat| < 2.0 indicates idiosyncratic risk is not significant
- **Implication:** Only systematic risk (beta) matters for expected returns

### Efficient Frontier Analysis

**Efficient Frontier Plot:**
- **Blue curve:** Efficient frontier showing best risk-return combinations
- **Red dot:** Market portfolio (VW or EW)
- **Green dot:** Minimum variance portfolio (lowest risk)
- **Orange dot:** Tangency portfolio (highest Sharpe ratio)
- **Purple dot:** Zero-beta portfolio (uncorrelated with market)
- **Dashed line:** Capital Allocation Line (CAL) from risk-free rate or zero-beta rate

**Zero-Beta CAL Plot (if no risk-free rate):**
- **Blue curve:** Efficient frontier
- **Red dashed line:** Zero-beta CAL through RZ tangent to frontier
- **Interpretation:** Shows optimal portfolio when risk-free rate is not available

**Diversification Impact Plot:**
- **X-axis:** Number of stocks in portfolio
- **Y-axis:** Portfolio volatility (risk)
- **Interpretation:** Shows diminishing returns to diversification
- **Key insight:** Most diversification benefits come from first 20-30 stocks

**Portfolio Weights Table:**
- **Min Var:** Lowest risk portfolio
- **Tangency:** Highest risk-adjusted return portfolio
- **Market:** Market portfolio (benchmark)
- **Zero-Beta:** Uncorrelated portfolio (replaces risk-free rate)
