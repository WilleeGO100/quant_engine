# test_traderspost_paper_btc.py
#
# Minimal sanity check that our Python POST to TradersPost
# matches what the curl webhook example would do.
#
# Flow:
#   1) Load settings.yaml via load_config()
#   2) Instantiate TradersPostExecutor with that config
#   3) Build a fake LONG BTCUSD decision (0.01 size)
#   4) Call executor.send_order(...) with test=True
#
# If live_trading == False in settings.yaml:
#   - It will ONLY print the payload (no HTTP request sent).
#
# If live_trading == True:
#   - It will actually POST to TradersPost, but still with "test": true
#     in the payload, so it should register as a test/paper order.

from __future__ import annotations

from typing import Any, Dict

from config.config import load_config
from engine.execution.traderspost_executor import TradersPostExecutor


def main() -> None:
    # 1) Load YAML settings (settings.yaml)
    cfg = load_config()

    # cfg might be a dataclass / object or a plain dict.
    # For the executor we want a dict-like interface.
    if hasattr(cfg, "__dict__"):
        cfg_dict: Dict[str, Any] = cfg.__dict__
    else:
        cfg_dict = dict(cfg)

    print("=== TradersPost BTC Paper Test ===")
    print(f"webhook_url         : {cfg_dict.get('traderspost_webhook_url')}")
    print(f"traderspost_ticker  : {cfg_dict.get('traderspost_ticker')}")
    print(f"traderspost_acct_id : {cfg_dict.get('traderspost_account_id')}")
    print(f"live_trading flag   : {cfg_dict.get('live_trading')}")
    print("==================================")

    # 2) Instantiate executor from config
    executor = TradersPostExecutor(config=cfg_dict)

    # 3) Build a sample decision as if from your BTC PO3 engine
    signal = "LONG"       # -> action = "buy"
    size = cfg_dict.get("default_position_size", 0.01)  # default to 0.01 BTC

    meta: Dict[str, Any] = {
        "source": "test_traderspost_paper_btc",
        "note": "This is a test signal from Python, equivalent to a curl webhook.",
    }

    # IMPORTANT:
    # - test=True ensures the payload contains "test": true.
    # - If live_trading == False in settings.yaml, the executor will ONLY print the payload
    #   and will NOT send any HTTP request (full dry run).
    print(
        "\n[TEST] Sending LONG BTCUSD position as test.\n"
        "      If live_trading=False, this will ONLY print the payload.\n"
        "      If live_trading=True, it will POST to TradersPost with test=true."
    )

    executor.send_order(signal=signal, size=size, test=True, meta=meta)

    print("\n[TEST] Done. Check the console output and, if live_trading=True, your TradersPost paper account.")


if __name__ == "__main__":
    main()
