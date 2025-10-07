from __future__ import annotations
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

def load_rf_monthly_decimal(rf_path: str) -> pd.DataFrame:
    """
    Load risk-free CSV that may contain quoted fields:
    Expected raw rows like:
      DATE,"TIME PERIOD","Euribor 1-month - Historical close ..."
      2025-08-31,"2025Aug","1.8900"
    Values are in PERCENT per ANNUM. We convert to monthly decimal via compounding:
      rf_month = (1 + percent/100)**(1/12) - 1
    Returns DataFrame with columns: date (datetime64[ns]), rf (float monthly decimal)
    """
    p = Path(rf_path)
    text = p.read_text(encoding="utf-8").splitlines()
    # skip header, parse rows; tolerate quotes and commas inside quotes
    rows = []
    for i, line in enumerate(text):
        if i == 0:
            continue
        
        # Remove outer quotes and handle doubled quotes
        line = line.strip().strip('"')
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
            date_str = parts[0].strip().strip('"')
            val_str = parts[2].strip().strip('"')
        else:
            continue

        if not date_str or not val_str:
            continue
        try:
            dt = pd.to_datetime(date_str)
            percent = float(val_str)
        except Exception:
            continue

        rf_month = (1.0 + percent/100.0)**(1.0/12.0) - 1.0
        rows.append((dt, rf_month))

    df = pd.DataFrame(rows, columns=["date", "rf"]).sort_values("date").reset_index(drop=True)
    return df

def is_recent(date: pd.Timestamp, max_age_days: int = 120) -> bool:
    now = pd.Timestamp.now(tz=timezone.utc).normalize()
    if date.tzinfo is None:
        date = date.tz_localize("UTC")
    delta = (now - date).days
    return delta <= max_age_days
