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
# TODO: Command will be added in Step 5
python code/04_capm_crosssection.py
```

### Step 4: Efficient Frontier Analysis
```bash
# TODO: Command will be added in Step 6
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
