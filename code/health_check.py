import yaml, pandas as pd
from pathlib import Path
from utils.logger import setup_logger
from utils.rf_utils import load_rf_monthly_decimal, is_recent

def check_paths(cfg):
    paths = [
        cfg["paths"]["raw_data"],
        cfg["paths"]["processed_data"],
        cfg["paths"]["output_figs"],
        cfg["paths"]["logs"]
    ]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        return False, f"Missing: {missing}"
    return True, "All required folders/files exist."

def check_rf(cfg):
    try:
        df = load_rf_monthly_decimal(cfg["paths"]["rf_file"])
        if df.empty:
            return False, "Parsed RF is empty"
        latest_row = df.iloc[-1]
        latest_val = float(latest_row["rf"])
        latest_date = pd.to_datetime(latest_row["date"])
        lo, hi = cfg["risk_free"]["expected_range"]
        in_range = (lo <= latest_val <= hi)
        recent = is_recent(latest_date, max_age_days=120)
        ok = in_range and recent
        recency_msg = "recent" if recent else "stale"
        return ok, f"Latest RF={latest_val:.4%} on {latest_date.date()} ({recency_msg})"
    except Exception as e:
        return False, f"RF check failed: {e}"

def check_frontier(cfg):
    """
    Read the actual tangency Sharpe from output/tables/optimizer_weights.csv
    Expect columns: portfolio, expected_return, volatility, sharpe_or_slope, ...
    """
    try:
        ow_path = Path("output/tables/optimizer_weights.csv")
        if not ow_path.exists():
            return False, "optimizer_weights.csv not found — run frontier first"
        df = pd.read_csv(ow_path)
        if "portfolio" not in df.columns or "sharpe_or_slope" not in df.columns:
            return False, "optimizer_weights.csv missing required columns"
        row = df[df["portfolio"].str.contains("tangency", case=False, na=False)]
        if row.empty:
            row = df[df["portfolio"].str.contains("zero_beta_tangent", case=False, na=False)]
        if row.empty:
            return False, "Tangency row not found in optimizer_weights.csv"
        sharpe = float(row.iloc[0]["sharpe_or_slope"])
        lo, hi = cfg["frontier"]["expected_sharpe_range"]
        ok = lo <= sharpe <= hi
        return ok, f"Tangency Sharpe={sharpe:.2f}"
    except Exception as e:
        return False, f"Frontier check failed: {e}"

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    log = setup_logger(cfg["paths"]["logs"])

    checks = [
        ("Paths", *check_paths(cfg)),
        ("Risk-Free", *check_rf(cfg)),
        ("Frontier", *check_frontier(cfg))
    ]

    log.info("=== SYSTEM HEALTH CHECK ===")
    for name, ok, msg in checks:
        status = "✅" if ok else "❌"
        log.info(f"{status} {name:<10} — {msg}")
        if not ok:
            log.warning(f"{name} check failed! {msg}")

    if all(ok for _, ok, _ in checks):
        log.info("System health check passed ✅")
    else:
        log.error("System health check FAILED ❌")

if __name__ == "__main__":
    main()
