import subprocess
import sys
import pytest
from pathlib import Path

def run_health():
    """Runs the health check and returns stdout as text."""
    result = subprocess.run(
        [sys.executable, "code/health_check.py"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True
    )
    return result.stdout + result.stderr, result.returncode

def test_health_passes():
    """Health check must pass end-to-end."""
    output, rc = run_health()
    assert rc == 0, f"Health check exited with {rc}\n{output}"
    assert "System health check passed" in output, f"Unexpected output:\n{output}"
