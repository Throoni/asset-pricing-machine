#!/usr/bin/env python3
"""
Utility to check and fix Euribor (risk-free rate) scaling issues.

This script:
1. Loads raw Euribor data from data/raw/risk_free/rf_eur.csv
2. Converts from percent to decimal monthly returns
3. Compares with current pipeline values
4. Fixes scaling if needed
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

def setup_logging():
    """Set up basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def load_raw_euribor():
    """Load and parse raw Euribor data."""
    logger = logging.getLogger(__name__)
    
    rf_path = Path("data/raw/risk_free/rf_eur.csv")
    if not rf_path.exists():
        logger.error(f"Raw Euribor file not found: {rf_path}")
        return None
    
    try:
        # Read the CSV - the entire row is quoted, so we need to parse manually
        with open(rf_path, 'r') as f:
            lines = f.readlines()
        
        logger.info(f"Loaded raw Euribor data: {len(lines)} lines")
        
        # Parse the data manually
        data = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Remove outer quotes and split by comma
            line = line.strip('"')
            # Handle doubled quotes in the data
            line = line.replace('""', '"')
            
            # Split by comma, but be careful with quoted fields
            parts = []
            current_part = ""
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    parts.append(current_part)
                    current_part = ""
                else:
                    current_part += char
            
            if current_part:
                parts.append(current_part)
            
            if len(parts) >= 3:
                date_str = parts[0]
                time_period = parts[1]
                rate_str = parts[2].strip('"')
                
                try:
                    # Parse date
                    date = pd.to_datetime(date_str)
                    
                    # Parse rate (remove quotes if present)
                    rate = float(rate_str.strip('"'))
                    
                    data.append({
                        'date': date,
                        'time_period': time_period,
                        'rate_percent': rate
                    })
                except Exception as e:
                    logger.warning(f"Could not parse line {i+1}: {line} - {e}")
                    continue
        
        if not data:
            logger.error("No valid data found in Euribor file")
            return None
        
        df = pd.DataFrame(data)
        df = df.sort_values('date')
        
        logger.info(f"Cleaned Euribor data: {len(df)} rows")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Rate range: {df['rate_percent'].min():.4f}% to {df['rate_percent'].max():.4f}%")
        return df
        
    except Exception as e:
        logger.error(f"Error loading raw Euribor data: {e}")
        return None

def convert_to_monthly_decimal(rf_df):
    """Convert Euribor from percent to monthly decimal returns."""
    logger = logging.getLogger(__name__)
    
    # Convert percent to annual decimal
    rf_df['rate_annual'] = rf_df['rate_percent'] / 100.0
    
    # Convert to monthly decimal via compounding
    rf_df['rate_monthly'] = (1 + rf_df['rate_annual']) ** (1/12) - 1
    
    logger.info("Converted Euribor to monthly decimal returns")
    return rf_df

def load_pipeline_rf():
    """Load current pipeline risk-free rate."""
    logger = logging.getLogger(__name__)
    
    # Try to load from processed data
    processed_path = Path("data/processed/returns.parquet")
    if processed_path.exists():
        try:
            df = pd.read_parquet(processed_path)
            if 'rf' in df.columns:
                logger.info("Loaded pipeline RF from processed data")
                return df[['date', 'rf']].drop_duplicates().sort_values('date')
        except Exception as e:
            logger.warning(f"Could not load from processed data: {e}")
    
    # Try to load from staging
    staging_path = Path("data/staging/risk_free_normalized.parquet")
    if staging_path.exists():
        try:
            df = pd.read_parquet(staging_path)
            logger.info("Loaded pipeline RF from staging data")
            return df
        except Exception as e:
            logger.warning(f"Could not load from staging data: {e}")
    
    logger.error("Could not load pipeline RF data")
    return None

def compare_rf_series(published_df, pipeline_df):
    """Compare published and pipeline risk-free rates."""
    logger = logging.getLogger(__name__)
    
    # Find overlapping dates
    published_dates = set(published_df['date'].dt.date)
    pipeline_dates = set(pipeline_df['date'].dt.date)
    overlap_dates = published_dates.intersection(pipeline_dates)
    
    if len(overlap_dates) == 0:
        logger.warning("No overlapping dates found, using latest values")
        # Use latest values
        published_latest = published_df['rate_monthly'].iloc[-1]
        pipeline_latest = pipeline_df['rf'].iloc[-1]
        ratio = pipeline_latest / published_latest
        return published_latest, pipeline_latest, ratio, None
    
    # Find the most recent overlapping date
    latest_overlap = max(overlap_dates)
    
    published_val = published_df[published_df['date'].dt.date == latest_overlap]['rate_monthly'].iloc[0]
    pipeline_val = pipeline_df[pipeline_df['date'].dt.date == latest_overlap]['rf'].iloc[0]
    
    ratio = pipeline_val / published_val
    
    logger.info(f"Comparing RF at {latest_overlap}:")
    logger.info(f"  Published monthly: {published_val:.6f} ({published_val*100:.4f}%)")
    logger.info(f"  Pipeline monthly:  {pipeline_val:.6f} ({pipeline_val*100:.4f}%)")
    logger.info(f"  Ratio: {ratio:.4f}")
    
    return published_val, pipeline_val, ratio, latest_overlap

def fix_rf_scaling(published_df):
    """Fix the risk-free rate scaling in the pipeline."""
    logger = logging.getLogger(__name__)
    
    # Update staging data
    staging_path = Path("data/staging/risk_free_normalized.parquet")
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create corrected staging data
    corrected_df = published_df[['date', 'rate_monthly']].copy()
    corrected_df = corrected_df.rename(columns={'rate_monthly': 'rf'})
    corrected_df.to_parquet(staging_path, index=False)
    
    logger.info(f"Updated staging RF data: {staging_path}")
    
    # Also update processed data if it exists
    processed_path = Path("data/processed/returns.parquet")
    if processed_path.exists():
        try:
            df = pd.read_parquet(processed_path)
            if 'rf' in df.columns:
                # Update RF column with corrected values
                df = df.merge(corrected_df, on='date', how='left', suffixes=('_old', ''))
                df['rf'] = df['rf'].fillna(df['rf_old'])
                df = df.drop(columns=['rf_old'])
                df.to_parquet(processed_path, index=False)
                logger.info(f"Updated processed RF data: {processed_path}")
        except Exception as e:
            logger.warning(f"Could not update processed data: {e}")
    
    return True

def main():
    """Main function."""
    logger = setup_logging()
    logger.info("Starting Euribor scaling check and fix")
    
    # Load raw Euribor data
    raw_rf = load_raw_euribor()
    if raw_rf is None:
        logger.error("Failed to load raw Euribor data")
        sys.exit(1)
    
    # Convert to monthly decimal
    published_rf = convert_to_monthly_decimal(raw_rf)
    
    # Load pipeline RF
    pipeline_rf = load_pipeline_rf()
    if pipeline_rf is None:
        logger.error("Failed to load pipeline RF data")
        sys.exit(1)
    
    # Compare series
    published_val, pipeline_val, ratio, overlap_date = compare_rf_series(published_rf, pipeline_rf)
    
    # Check if fix is needed
    if 0.5 <= ratio <= 1.5:
        logger.info("✅ RF scaling looks correct (ratio in [0.5, 1.5])")
        print("✅ RF scaling looks correct - no changes needed")
        sys.exit(0)
    else:
        logger.warning(f"⚠️ RF scaling issue detected (ratio: {ratio:.4f})")
        print(f"⚠️ RF scaling issue detected (ratio: {ratio:.4f})")
        print("Applying fix...")
        
        # Apply fix
        if fix_rf_scaling(published_rf):
            logger.info("✅ RF scaling fix applied successfully")
            print("✅ RF scaling fix applied successfully")
            sys.exit(0)
        else:
            logger.error("❌ Failed to apply RF scaling fix")
            print("❌ Failed to apply RF scaling fix")
            sys.exit(1)

if __name__ == "__main__":
    main()
