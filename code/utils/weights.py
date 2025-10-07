"""
Robust value-weighted (VW) weights loader for CAPM analysis.

Handles multiple CSV schemas and normalizes weights for the universe.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Set, Optional


def load_vw_weights(config, universe_tickers: Set[str]) -> Optional[pd.DataFrame]:
    """
    Load and normalize value-weighted weights for the given universe.
    
    Args:
        config: Configuration object with paths.market_caps_csv
        universe_tickers: Set of ticker symbols in the universe
        
    Returns:
        DataFrame with columns [ticker, weight, source, asof] or None if insufficient coverage
    """
    logger = logging.getLogger(__name__)
    
    # Check if market caps CSV exists
    market_caps_path = config.paths.market_caps_csv
    if not market_caps_path or not Path(market_caps_path).exists():
        logger.warning(f"Market caps file not found: {market_caps_path}")
        return None
    
    try:
        # Load the CSV
        df = pd.read_csv(market_caps_path)
        logger.info(f"Loaded market caps from {market_caps_path}: {len(df)} rows")
        
        # Normalize column names (lowercase, strip spaces)
        df.columns = df.columns.str.lower().str.strip()
        
        # Normalize ticker column
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].str.strip().str.upper()
        else:
            logger.error("No 'ticker' column found in market caps CSV")
            return None
        
        # Handle different schemas
        if 'weight' in df.columns and 'market_cap' in df.columns:
            # Schema D: date, ticker, market_cap, weight (prefer weight)
            logger.info("Using existing weight column")
            weights_df = df[['ticker', 'weight']].copy()
            source = "provided_weights"
            
        elif 'weight' in df.columns:
            # Schema A: ticker, weight
            logger.info("Using weight column only")
            weights_df = df[['ticker', 'weight']].copy()
            source = "provided_weights"
            
        elif 'market_cap' in df.columns:
            # Schema B or C: ticker, market_cap OR date, ticker, market_cap
            logger.info("Computing weights from market_cap")
            if 'date' in df.columns:
                # Schema C: date, ticker, market_cap (single as-of date)
                asof_date = df['date'].iloc[0] if len(df['date'].unique()) == 1 else None
                if asof_date:
                    logger.info(f"Using as-of date: {asof_date}")
                weights_df = df[['ticker', 'market_cap']].copy()
            else:
                # Schema B: ticker, market_cap
                weights_df = df[['ticker', 'market_cap']].copy()
                asof_date = None
            
            # Compute weights
            total_market_cap = weights_df['market_cap'].sum()
            weights_df['weight'] = weights_df['market_cap'] / total_market_cap
            weights_df = weights_df[['ticker', 'weight']]
            source = "computed_from_market_cap"
            
        else:
            logger.error("No 'weight' or 'market_cap' column found in market caps CSV")
            return None
        
        # Filter to universe tickers
        universe_set = {ticker.upper().strip() for ticker in universe_tickers}
        weights_df = weights_df[weights_df['ticker'].isin(universe_set)]
        
        # Check coverage
        coverage = len(weights_df) / len(universe_set)
        logger.info(f"Weight coverage: {len(weights_df)}/{len(universe_set)} = {coverage:.1%}")
        
        if coverage < 0.8:
            logger.warning(f"Insufficient weight coverage: {coverage:.1%} < 80%")
            return None
        
        # Re-normalize weights to sum to 1 within the covered set
        total_weight = weights_df['weight'].sum()
        weights_df['weight'] = weights_df['weight'] / total_weight
        
        # Add metadata columns
        weights_df['source'] = source
        if 'asof_date' in locals() and asof_date:
            weights_df['asof'] = asof_date
        else:
            weights_df['asof'] = None
        
        # Validate weights
        weight_sum = weights_df['weight'].sum()
        if abs(weight_sum - 1.0) > 1e-6:
            logger.error(f"Weights do not sum to 1: {weight_sum}")
            return None
        
        if not weights_df['weight'].ge(0).all():
            logger.error("Negative weights found")
            return None
        
        if weights_df['weight'].isna().any():
            logger.error("NaN weights found")
            return None
        
        logger.info(f"Validated weights: sum={weight_sum:.6f}, min={weights_df['weight'].min():.6f}, max={weights_df['weight'].max():.6f}")
        
        # Save normalized weights
        output_path = Path("data/processed/market_caps_weights.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weights_df.to_csv(output_path, index=False)
        logger.info(f"Saved normalized weights to {output_path}")
        
        return weights_df
        
    except Exception as e:
        logger.error(f"Error loading weights: {e}")
        return None
