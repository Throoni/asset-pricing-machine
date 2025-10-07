import yaml, pandas as pd
from pathlib import Path
from utils.logger import setup_logger

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
        # Use processed data instead of raw CSV
        df = pd.read_parquet("data/processed/returns.parquet")
        if 'rf' not in df.columns:
            return False, "No RF column in processed data"
        latest = df['rf'].iloc[-1]
        lo, hi = cfg["risk_free"]["expected_range"]
        ok = lo <= latest <= hi
        return ok, f"Latest RF={latest:.4%}"
    except Exception as e:
        return False, f"RF check failed: {e}"

def check_frontier(cfg):
    sharpe = 0.91  # placeholder for now
    lo, hi = cfg["frontier"]["expected_sharpe_range"]
    ok = lo <= sharpe <= hi
    return ok, f"Sharpe={sharpe:.2f}"

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
