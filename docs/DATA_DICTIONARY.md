# Data Dictionary

This document defines all columns and variables that will be created during the asset pricing analysis pipeline.

## Core Time Series Data

### date
- **Type:** datetime
- **Description:** Date index for all time series data
- **Format:** YYYY-MM-DD
- **Frequency:** Monthly (or as configured)

### ticker
- **Type:** string
- **Description:** Stock ticker symbol
- **Example:** "ABI.BR", "SOLB.BR"

## Return Data

### ret_m
- **Type:** float
- **Description:** Monthly stock return
- **Calculation:** (Price_t / Price_t-1) - 1
- **Units:** Decimal (0.05 = 5%)

### ret_excess
- **Type:** float
- **Description:** Excess return over risk-free rate
- **Calculation:** ret_m - rf_m
- **Units:** Decimal

### rf_m
- **Type:** float
- **Description:** Monthly risk-free rate
- **Source:** TODO - Risk-free rate source
- **Units:** Decimal

## Market Data

### mkt_ret
- **Type:** float
- **Description:** Market portfolio return
- **Calculation:** Value-weighted or cap-weighted index return
- **Units:** Decimal

### mkt_ret_excess
- **Type:** float
- **Description:** Market excess return
- **Calculation:** mkt_ret - rf_m
- **Units:** Decimal

## CAPM Analysis

### beta
- **Type:** float
- **Description:** Stock beta (market sensitivity)
- **Calculation:** Cov(ret_excess, mkt_ret_excess) / Var(mkt_ret_excess)
- **Units:** Dimensionless

### alpha
- **Type:** float
- **Description:** CAPM alpha (abnormal return)
- **Calculation:** ret_excess - beta * mkt_ret_excess
- **Units:** Decimal

### r_squared
- **Type:** float
- **Description:** R-squared of CAPM regression
- **Range:** [0, 1]
- **Units:** Dimensionless

### t_stat_alpha
- **Type:** float
- **Description:** T-statistic for alpha significance
- **Calculation:** alpha / std_error_alpha
- **Units:** Dimensionless

## Portfolio Data

### weight
- **Type:** float
- **Description:** Portfolio weight for each stock
- **Constraints:** Sum to 1, within bounds [0, 0.25]
- **Units:** Decimal

### portfolio_ret
- **Type:** float
- **Description:** Portfolio return
- **Calculation:** Sum(weight * ret_m)
- **Units:** Decimal

### portfolio_ret_excess
- **Type:** float
- **Description:** Portfolio excess return
- **Calculation:** portfolio_ret - rf_m
- **Units:** Decimal

## Value Effect Data

### bm_ratio
- **Type:** float
- **Description:** Book-to-market ratio
- **Calculation:** Book_value / Market_value
- **Lag:** 6 months (configurable)
- **Units:** Dimensionless

### value_tertile
- **Type:** int
- **Description:** Value tertile (1=low, 2=mid, 3=high)
- **Calculation:** Based on bm_ratio breakpoints
- **Range:** [1, 3]

## Risk Metrics

### volatility
- **Type:** float
- **Description:** Stock return volatility
- **Calculation:** Std(ret_m) * sqrt(12) for annualized
- **Units:** Decimal

### sharpe_ratio
- **Type:** float
- **Description:** Risk-adjusted return
- **Calculation:** Mean(ret_excess) / Std(ret_excess)
- **Units:** Dimensionless

### max_drawdown
- **Type:** float
- **Description:** Maximum peak-to-trough decline
- **Calculation:** Max(cumulative_ret - running_max(cumulative_ret))
- **Units:** Decimal

## Market Cap Data

### market_cap
- **Type:** float
- **Description:** Market capitalization
- **Units:** Currency units (EUR, USD, etc.)

### weight_mkt_cap
- **Type:** float
- **Description:** Market cap weight in index
- **Calculation:** market_cap / sum(market_cap)
- **Units:** Decimal

## Quality Checks

### n_obs
- **Type:** int
- **Description:** Number of observations per stock
- **Minimum:** 24 months (configurable)

### data_quality_flag
- **Type:** bool
- **Description:** Data quality indicator
- **True:** Sufficient data, no major gaps
- **False:** Insufficient data or quality issues
