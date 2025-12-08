# live/test_traderspost_paper_btc.py
#
# Quick sanity check for TradersPostExecutor using your Config.
# - Adds project ROOT_DIR to sys.path so `config` resolves
# - Loads settings.yaml via load_config()
# - Builds TradersPostExecutor(config=cfg)
# - Sends a SMALL BTCUSD LONG test signal
#
# With:
#   live_trading = False  -> prints payload only
#   live_trading = True   -> POSTs to TradersPost with test=true

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# --------------------------------------------------
# Path setup (project root)
# --------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --------------------------------------------------
# Local imports (now ROOT_DIR is on sys.path)
# --------------------------------------------------
from config.config import load_config
from engine.execution.traderspost_executor import TradersPostExecutor


def main() -> None:
    cfg = load_config()

    webhook_url = getattr(cfg, "traderspost_webhook_url", "")
    trade_symbol = getattr(cfg, "trade_symbol", "BTCUSD")
    acct_id = getattr(cfg, "traderspost_account_id", "(none)")
    live_trading = bool(getattr(cfg, "live_trading", False))

    print("=== TradersPost BTC Paper Test ===")
    print(f"webhook_url         : {webhook_url}")
    print(f"traderspost_ticker  : {trade_symbol}")
    print(f"traderspost_acct_id : {acct_id}")
    print(f"live_trading flag   : {live_trading}")
    print("==================================\n")

    executor = TradersPostExecutor(config=cfg)

    # Use a tiny test size
    size = float(getattr(cfg, "default_position_size", 0.01))
    test_flag = True  # keep True until you're 100% sure

    meta = {
        "source": "test_traderspost_paper_btc",
        "note": "This is a test signal from Python, equivalent to a curl webhook.",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    print("[TEST] Sending LONG BTCUSD position as test.")
    print("      If live_trading=False, this will ONLY print the payload.")
    print("      If live_trading=True, it will POST to TradersPost with test=true.\n")

    executor.send_order("LONG", size, test_flag, meta)

    print("\n[TEST] Done. Check the console output and, if live_trading=True, your TradersPost paper account.")


if __name__ == "__main__":
    main()
