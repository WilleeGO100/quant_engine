# test_live_es_rest.py
#
# Smoke-test for TwelveData REST time_series feed.
# It pulls the last N 5-minute candles and prints them.
#
# Usage (from project root):
#   cd C:\Python312\quant_engine
#   python test_live_es_rest.py
#
# No trades, no strategy, just prints data.

from __future__ import annotations

from config.config import load_config
from engine.data.twelvedata_client import TwelveDataClient


def main() -> None:
    cfg = load_config()

    api_key = getattr(cfg, "twelvedata_api_key", "")
    symbol = getattr(cfg, "twelvedata_symbol", "ES=F")

    if not api_key:
        print("[TEST_LIVE_ES_REST] ERROR: twelvedata_api_key is empty in settings.yaml")
        return

    print(f"[TEST_LIVE_ES_REST] Using symbol={symbol}, api_key length={len(api_key)}")

    client = TwelveDataClient(api_key=api_key)

    print("[TEST_LIVE_ES_REST] Fetching last 10 bars (5min)...")
    df = client.fetch_recent_5m(symbol=symbol, bars=10)

    if df is None or df.empty:
        print("[TEST_LIVE_ES_REST] ERROR: No data returned.")
        return

    print("\n[TEST_LIVE_ES_REST] DataFrame tail (10 bars):")
    print(df.tail(10))

    last = df.iloc[-1]
    print("\n[TEST_LIVE_ES_REST] Latest bar:")
    print(last)


if __name__ == "__main__":
    main()
