# test_live_es_feed.py
#
# Simple smoke test for the live ES/MNQ data feed using TwelveData.
#
# Usage (from project root):
#   cd C:\Python312\quant_engine
#   python test_live_es_feed.py
#
# This does NOT send any trades, it only prints recent candles.

from __future__ import annotations

from config.config import load_config
from engine.data.live_es_twelvedata import TwelveDataClient


def main() -> None:
    cfg = load_config()

    api_key = getattr(cfg, "twelvedata_api_key", "")
    symbol = getattr(cfg, "twelvedata_symbol", "ES=F")

    if not api_key:
        print("[TEST_LIVE_ES] ERROR: twelvedata_api_key is empty in settings.yaml")
        return

    print(f"[TEST_LIVE_ES] Using symbol={symbol}, api_key length={len(api_key)}")

    client = TwelveDataClient(api_key=api_key)

    print("[TEST_LIVE_ES] Fetching last 10 bars (5m)...")
    df = client.fetch_recent_5m(symbol=symbol, bars=10)

    if df is None or df.empty:
        print("[TEST_LIVE_ES] ERROR: No data returned.")
        return

    print("[TEST_LIVE_ES] Got DataFrame:")
    print(df.tail(10))

    last = df.iloc[-1]
    print("\n[TEST_LIVE_ES] Latest bar:")
    print(last)


if __name__ == "__main__":
    main()
