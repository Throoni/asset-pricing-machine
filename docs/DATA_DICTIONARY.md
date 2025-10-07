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

### log_ret
- **Type:** float
- **Description:** Log return (continuously compounded)
- **Calculation:** log(adj_close / adj_close.shift(1))
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

### mkt_excess
- **Type:** float
- **Description:** Market excess return (alias for mkt_ret_excess)
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

## CAPM Analysis Results

### ticker
- **Type:** string
- **Description:** Stock ticker symbol
- **Example:** "ABI.BR", "SOLB.BR"

### model_type
- **Type:** string
- **Description:** Regression model type
- **Values:** "CAPM", "Black"
- **CAPM:** Uses excess returns (ret_excess ~ alpha + beta * mkt_excess)
- **Black:** Uses raw returns (ret ~ a + beta * ret_m)

### alpha_or_a
- **Type:** float
- **Description:** Intercept coefficient (alpha for CAPM, a for Black)
- **CAPM:** Alpha (abnormal return after accounting for market risk)
- **Black:** Intercept (expected return when market return is zero)
- **Units:** Decimal

### beta
- **Type:** float
- **Description:** Market sensitivity coefficient
- **CAPM:** Beta (sensitivity to market excess returns)
- **Black:** Beta (sensitivity to market returns)
- **Units:** Dimensionless

### t_alpha_or_a
- **Type:** float
- **Description:** T-statistic for intercept coefficient
- **Calculation:** alpha_or_a / standard_error
- **Units:** Dimensionless

### t_beta
- **Type:** float
- **Description:** T-statistic for beta coefficient
- **Calculation:** beta / standard_error
- **Units:** Dimensionless

### r2
- **Type:** float
- **Description:** R-squared of regression
- **Range:** [0, 1]
- **Units:** Dimensionless

### n
- **Type:** int
- **Description:** Number of observations used in regression
- **Minimum:** config.universe.min_obs_months
- **Units:** Count

### start
- **Type:** datetime
- **Description:** Start date of regression period
- **Format:** YYYY-MM-DD
- **Units:** Date

### end
- **Type:** datetime
- **Description:** End date of regression period
- **Format:** YYYY-MM-DD
- **Units:** Date

## VW Beta Summary

### method
- **Type:** string
- **Description:** Method used for beta estimation
- **Values:** "CAPM", "Black (no CAPM available)"
- **CAPM:** When risk-free rate is available
- **Black:** When risk-free rate is not available

### vw_beta
- **Type:** float
- **Description:** Value-weighted average beta
- **Calculation:** Weighted average of individual stock betas
- **Units:** Dimensionless

### ew_beta
- **Type:** float
- **Description:** Equal-weighted average beta
- **Calculation:** Simple average of individual stock betas
- **Units:** Dimensionless

### notes
- **Type:** string
- **Description:** Additional information about the calculation
- **Example:** "VW beta approximated by EW (market cap weights not available)"

## Cross-Sectional Analysis Results

### method
- **Type:** string
- **Description:** Cross-sectional regression method
- **Values:** "static_excess", "static_raw", "fm_excess", "fm_raw", "fm_with_idio_excess", "fm_with_idio_raw"
- **static_excess:** Static beta SML using excess returns
- **static_raw:** Static beta SML using raw returns
- **fm_excess:** Fama-MacBeth using excess returns
- **fm_raw:** Fama-MacBeth using raw returns
- **fm_with_idio_excess:** Fama-MacBeth with idiosyncratic risk (excess returns)
- **fm_with_idio_raw:** Fama-MacBeth with idiosyncratic risk (raw returns)

### gamma0
- **Type:** float
- **Description:** Intercept coefficient (γ₀)
- **CAPM excess:** Should be close to zero
- **Black raw:** Should be close to zero-beta rate (RZ)
- **Units:** Decimal

### gamma_m
- **Type:** float
- **Description:** Market risk premium coefficient (γₘ)
- **CAPM excess:** Market risk premium
- **Black raw:** Market risk premium above zero-beta rate
- **Units:** Decimal

### t_gamma0
- **Type:** float
- **Description:** T-statistic for intercept coefficient
- **Calculation:** gamma0 / standard_error
- **Units:** Dimensionless

### t_gamma_m
- **Type:** float
- **Description:** T-statistic for market risk premium coefficient
- **Calculation:** gamma_m / standard_error
- **Units:** Dimensionless

### n_months
- **Type:** int
- **Description:** Number of observations used in regression
- **Static methods:** Number of stocks
- **Fama-MacBeth:** Number of months
- **Units:** Count

### gamma_idio
- **Type:** float
- **Description:** Idiosyncratic risk coefficient (only in fm_with_idio methods)
- **CAPM theory:** Should be zero (idiosyncratic risk not priced)
- **Units:** Decimal

### t_gamma_idio
- **Type:** float
- **Description:** T-statistic for idiosyncratic risk coefficient
- **Calculation:** gamma_idio / standard_error
- **Units:** Dimensionless

## Zero-Beta Portfolio

### method
- **Type:** string
- **Description:** Zero-beta portfolio construction method
- **Values:** "zero_beta_portfolio"

### R_Z
- **Type:** float
- **Description:** Zero-beta rate (expected return of zero-beta portfolio)
- **Calculation:** w_Z^T * μ (portfolio weights times expected returns)
- **Units:** Decimal

### notes
- **Type:** string
- **Description:** Additional information about portfolio construction
- **Example:** "Value-weighted market, shorting=no"

### has_shorting
- **Type:** bool
- **Description:** Whether portfolio construction required shorting
- **True:** Some weights are negative
- **False:** All weights are non-negative
