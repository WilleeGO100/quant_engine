# engine/data/live_es_twelvedata.py
#
# Fetches recent 5-minute OHLCV data for ES/MNQ from TwelveData.
# This is the FIRST building block for live ES data.
#
# We are NOT touching any of your CSV / backtests code here.
# This module will be used by a test script first, then by main_live.py.

from __future__ import annotations

from typing import Optional

import requests
import pandas as pd


class TwelveDataClient:
    """
    Minimal TwelveData client for 5-minute futures data.
    """

    BASE_URL = "https://api.twelvedata.com/time_series"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_recent_5m(
        self,
        symbol: str,
        bars: int = 50,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch recent 5-minute candles for the given symbol.

        Returns:
            DataFrame with columns:
            ['datetime', 'open', 'high', 'low', 'close', 'volume']
            sorted oldest -> newest

            or None on error.
        """
        params = {
            "symbol": symbol,
            "interval": "5min",
            "outputsize": bars,
            "apikey": self.api_key,
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
        except Exception as e:
            print(f"[TwelveData] Request error: {e}")
            return None

        if resp.status_code != 200:
            print(f"[TwelveData] HTTP {resp.status_code}: {resp.text}")
            return None

        data = resp.json()

        if "values" not in data:
            print(f"[TwelveData] Unexpected response: {data}")
            return None

        values = data["values"]

        df = pd.DataFrame(values)

        # Ensure correct column types
        # TwelveData returns strings; we convert prices/volume to float.
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype(float)

        # Rename datetime column to 'timestamp' to match your engine style
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "timestamp"}, inplace=True)

        # Sort from oldest -> newest
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def fetch_last_bar(
        self,
        symbol: str,
    ) -> Optional[dict]:
        """
        Convenience helper: returns ONLY the latest 5m bar as a dict.

        Keys:
          'timestamp', 'open', 'high', 'low', 'close', 'volume'
        """
        df = self.fetch_recent_5m(symbol=symbol, bars=1)
        if df is None or df.empty:
            return None

        row = df.iloc[-1]
        return {
            "timestamp": row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0)),
        }
