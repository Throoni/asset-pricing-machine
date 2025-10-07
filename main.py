import argparse
import sys
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent

def run(cmd, cwd=HERE):
    print(f"▶ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=cwd)

def cmd_health(args):
    sys.exit(run([sys.executable, "code/health_check.py"]))

def cmd_ingest(args):
    sys.exit(run([sys.executable, "code/02_ingest_clean.py"]))

def cmd_ts(args):
    # time-series CAPM
    rc = run([sys.executable, "code/03_capm_timeseries.py", "--check"])
    if rc == 0:
        rc = run([sys.executable, "code/03_capm_timeseries.py"])
    sys.exit(rc)

def cmd_cs(args):
    # cross-sectional CAPM
    rc = run([sys.executable, "code/04_capm_crosssection.py", "--check"])
    if rc == 0:
        rc = run([sys.executable, "code/04_capm_crosssection.py"])
    sys.exit(rc)

def cmd_frontier(args):
    # portfolio frontier
    rc = run([sys.executable, "code/05_frontier.py", "--check"])
    if rc == 0:
        rc = run([sys.executable, "code/05_frontier.py"])
    sys.exit(rc)

def cmd_validate(args):
    # validation and intelligence
    rc = run([sys.executable, "code/06_validation.py", "--check"])
    if rc == 0:
        rc = run([sys.executable, "code/06_validation.py"])
    sys.exit(rc)

def cmd_all(args):
    steps = [
        [sys.executable, "code/health_check.py"],
        [sys.executable, "code/02_ingest_clean.py"],
        [sys.executable, "code/03_capm_timeseries.py"],
        [sys.executable, "code/04_capm_crosssection.py"],
        [sys.executable, "code/05_frontier.py"],
        [sys.executable, "code/health_check.py"],
    ]
    rc = 0
    for step in steps:
        rc = run(step)
        if rc != 0:
            break
    sys.exit(rc)

def main():
    p = argparse.ArgumentParser(description="Asset Pricing Machine CLI")
    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("health").set_defaults(fn=cmd_health)
    sp.add_parser("ingest").set_defaults(fn=cmd_ingest)
    sp.add_parser("ts").set_defaults(fn=cmd_ts)
    sp.add_parser("cs").set_defaults(fn=cmd_cs)
    sp.add_parser("frontier").set_defaults(fn=cmd_frontier)
    sp.add_parser("validate").set_defaults(fn=cmd_validate)
    sp.add_parser("all").set_defaults(fn=cmd_all)

    args = p.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()
