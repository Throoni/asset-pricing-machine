#!/usr/bin/env python3
"""
Configuration loader with strict validation for asset pricing analysis.

Purpose: Load and validate configuration from config.yml with no hidden assumptions.
Inputs: config.yml file path
Outputs: Validated Config object or clear error messages
Seed: 42 (for reproducibility)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, validator


class ProjectConfig(BaseModel):
    """Project-level configuration."""
    country: str = Field(..., description="Country for analysis (e.g., Belgium, Germany)")
    base_currency: str = Field(..., description="Base currency (e.g., EUR, USD)")
    timezone: str = Field(default="Europe/Brussels", description="Timezone for data")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    frequency: Literal["D", "W", "M"] = Field(default="M", description="Data frequency")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        """Validate date format is YYYY-MM-DD."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")

    @validator('end_date')
    def validate_date_order(cls, v, values):
        """Validate start_date < end_date."""
        if 'start_date' in values:
            start = datetime.strptime(values['start_date'], '%Y-%m-%d')
            end = datetime.strptime(v, '%Y-%m-%d')
            if start >= end:
                raise ValueError(f"start_date ({values['start_date']}) must be before end_date ({v})")
        return v


class UniverseConfig(BaseModel):
    """Universe selection configuration."""
    method: Literal["static_list", "top_n_mcap"] = Field(..., description="Universe selection method")
    min_obs_months: int = Field(..., description="Minimum observations per stock")
    allow_unbalanced: bool = Field(..., description="Allow unbalanced panels")
    path_static_list: Optional[str] = Field(None, description="Path to static stock list folder")


class BenchmarksConfig(BaseModel):
    """Benchmark and risk model configuration."""
    market_index: str = Field(..., description="Market index specification")
    risk_model: Literal["rf_capm", "zero_beta", "both"] = Field(..., description="Risk model type")
    risk_free_source: Optional[str] = Field(None, description="Risk-free rate source")
    index_csv: Optional[str] = Field(None, description="Path to index CSV file")

    @validator('risk_free_source')
    def validate_risk_free_source(cls, v, values):
        """Require risk_free_source if risk_model includes rf_capm."""
        if values.get('risk_model') in ['rf_capm', 'both'] and not v:
            raise ValueError("risk_free_source is required when risk_model includes 'rf_capm'")
        return v


class PathsConfig(BaseModel):
    """Data file paths configuration."""
    risk_free_csv: Optional[str] = Field(None, description="Risk-free rate CSV path")
    fundamentals_csv: Optional[str] = Field(None, description="Fundamentals CSV path")
    market_caps_csv: Optional[str] = Field(None, description="Market caps CSV path")


class RegressionConfig(BaseModel):
    """Regression analysis configuration."""
    nw_lags: int = Field(..., description="Newey-West lags")
    winsorize_pct: float = Field(..., description="Winsorization percentage")


class OptimizationConfig(BaseModel):
    """Portfolio optimization configuration."""
    allow_shorting: bool = Field(..., description="Allow short positions")
    weight_bounds: list = Field(..., description="Weight bounds [min, max]")
    covariance_estimator: str = Field(..., description="Covariance estimator method")

    @validator('weight_bounds')
    def validate_weight_bounds(cls, v):
        """Validate weight bounds format and values."""
        if len(v) != 2:
            raise ValueError("weight_bounds must have exactly 2 elements [min, max]")
        if v[0] >= v[1]:
            raise ValueError("weight_bounds[0] must be less than weight_bounds[1]")
        if v[0] < 0 or v[1] > 1:
            raise ValueError("weight_bounds must be between 0 and 1")
        return v


class ValueEffectConfig(BaseModel):
    """Value effect analysis configuration."""
    enabled: bool = Field(..., description="Enable value effect analysis")
    bm_lag_months: int = Field(..., description="Book-to-market lag in months")
    sort_breakpoints: str = Field(..., description="Sort breakpoints method")


class OutputsConfig(BaseModel):
    """Output configuration."""
    dir: str = Field(..., description="Output directory")


class Config(BaseModel):
    """Main configuration model."""
    project: ProjectConfig
    universe: UniverseConfig
    benchmarks: BenchmarksConfig
    paths: PathsConfig
    regression: RegressionConfig
    optimization: OptimizationConfig
    value_effect: ValueEffectConfig
    outputs: OutputsConfig


def load_config(config_path: str) -> Config:
    """Load and validate configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    try:
        return Config(**config_data)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")


def main():
    """CLI for configuration validation."""
    parser = argparse.ArgumentParser(description="Validate configuration file")
    parser.add_argument("--check", required=True, help="Path to config.yml file")
    args = parser.parse_args()
    
    try:
        config = load_config(args.check)
        print("CONFIG OK")
        print(f"Project: {config.project.country} ({config.project.base_currency})")
        print(f"Period: {config.project.start_date} to {config.project.end_date}")
        print(f"Frequency: {config.project.frequency}")
        print(f"Risk model: {config.benchmarks.risk_model}")
        return 0
    except Exception as e:
        print(f"CONFIG ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
