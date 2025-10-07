# Asset Pricing Machine
Reproducible pipeline for CAPM / Zero-Beta CAPM and mean-variance analysis.
See docs/PROMPT.md for the system prompt and /report for the paper.

## Quick Usage

```bash
# one-off self-check
python main.py health

# build processed data
python main.py ingest

# run CAPM (time-series) and cross-section
python main.py ts
python main.py cs

# run efficient frontier
python main.py frontier

# run validation and intelligence layer
python main.py validate

# launch interactive dashboard
python main.py dashboard

# run everything end-to-end (with a health check at start and end)
python main.py all

# or via Makefile
make health
make ingest
make ts
make cs
make frontier
make validate
make dashboard
make all
make test
```

## Dashboard

The interactive dashboard provides real-time visualization of pipeline outputs and validation results.

### Features
- **Summary Cards**: Key metrics (Sharpe ratio, Zero-beta rate, number of tickers, sample period)
- **SML Tab**: Security Market Line scatter plot with fitted regression line
- **Frontier Tab**: Efficient frontier with highlighted portfolios
- **Betas Tab**: Beta distribution histograms by model type
- **Validation Tab**: Fama-MacBeth consistency comparisons
- **Live Reload**: "Reload Data" button to refresh all visualizations

### Usage
```bash
# Start dashboard
python main.py dashboard
# or
make dashboard

# Access at: http://127.0.0.1:8050
# Stop with: Ctrl+C
```

### Requirements
The dashboard requires additional packages:
```bash
pip install dash plotly
```
```
