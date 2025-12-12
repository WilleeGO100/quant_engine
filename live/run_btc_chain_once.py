from __future__ import annotations

import os
import sys
import subprocess
import logging
from typing import Optional, Tuple

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ORDERS_CSV = os.path.join(DATA_DIR, "paper_orders_btc.csv")

LIVE_SIGNAL_SCRIPT = os.path.join(PROJECT_ROOT, "live", "main_live_btc_csv.py")
EXECUTOR_SCRIPT = os.path.join(PROJECT_ROOT, "live", "execute_latest_btc_intent_coinbase.py")


def _python_exe() -> str:
    # Use the currently running interpreter (your venv python)
    return sys.executable


def _latest_intent_sig() -> Optional[Tuple[str, str, str, str]]:
    """
    Returns (timestamp, action, side, qty_btc_str) for latest ENTER/EXIT row.
    """
    if not os.path.exists(ORDERS_CSV):
        return None

    df = pd.read_csv(ORDERS_CSV)
    if df.empty:
        return None

    df.columns = [c.strip().lower() for c in df.columns]
    if "timestamp" not in df.columns or "action" not in df.columns:
        return None

    df = df[df["action"].astype(str).str.upper().str.startswith(("ENTER", "EXIT"))]
    if df.empty:
        return None

    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return None

    r = df.iloc[-1]
    ts = str(r.get("timestamp"))
    action = str(r.get("action")).upper().strip()
    side = str(r.get("side", "")).upper().strip()
    qty = float(r.get("qty_btc", 0.0) or 0.0)
    qty_s = f"{qty:.8f}"
    return (ts, action, side, qty_s)


def _run_script(path: str) -> int:
    cmd = [_python_exe(), path]
    logging.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return int(proc.returncode)


def main():
    # Snapshot latest intent BEFORE
    before = _latest_intent_sig()

    # 1) Run live signal generator (one-shot)
    rc1 = _run_script(LIVE_SIGNAL_SCRIPT)
    if rc1 != 0:
        logging.error("Live signal script failed with code %d", rc1)
        sys.exit(rc1)

    # Snapshot latest intent AFTER
    after = _latest_intent_sig()

    if after is None:
        logging.info("No intent file found or no ENTER/EXIT rows. Nothing to execute.")
        return

    if before == after:
        logging.info("No NEW intent produced. Skipping executor.")
        return

    logging.info("New intent detected: %s", after)

    # 2) Run executor (DEDUP will still protect you if needed)
    rc2 = _run_script(EXECUTOR_SCRIPT)
    if rc2 != 0:
        logging.error("Executor script failed with code %d", rc2)
        sys.exit(rc2)

    logging.info("Chain completed successfully.")


if __name__ == "__main__":
    main()
