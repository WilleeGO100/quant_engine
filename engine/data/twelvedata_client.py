# engine/data/twelvedata_client.py
#
# Minimal TwelveData REST client for 5-minute OHLCV data.
# Uses the "time_series" endpoint (the one you see in the API playground).
#
# This does NOT change any of your existing CSV / backtests code.
# It is just a reusable piece we will later plug into main_live.py.

from __future__ import annotations

from typing import Optional

import requests
import pandas as pd


class TwelveDataClient:
    """
    Minimal TwelveData REST client.

    Example:
        client = TwelveDataClient(api_key="...")
        df = client.fetch_recent_5m(symbol="ES=F", bars=50)
        last_bar = client.fetch_last_bar(symbol="ES=F")
    """

    BASE_URL = "https://api.twelvedata.com/time_series"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, params: dict) -> Optional[dict]:
        """Internal helper to call TwelveData time_series endpoint."""
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
        except Exception as e:
            print(f"[TwelveData] Request error: {e}")
            return None

        if resp.status_code != 200:
            print(f"[TwelveData] HTTP {resp.status_code}: {resp.text}")
            return None

        try:
            data = resp.json()
        except Exception as e:
            print(f"[TwelveData] JSON decode error: {e}")
            return None

        if "values" not in data:
            # TwelveData errors come back as {"code": ..., "message": ...}
            print(f"[TwelveData] Unexpected response: {data}")
            return None

        return data

    def fetch_recent_5m(self, symbol: str, bars: int = 50) -> Optional[pd.DataFrame]:
        """
        Fetch recent 5-minute candles for the given symbol.

        Returns:
            pandas.DataFrame with columns:
                ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            sorted oldest -> newest

            or None on error.
        """
        params = {
            "symbol": symbol,
            "interval": "5min",
            "outputsize": bars,
            "apikey": self.api_key,
        }

        data = self._request(params)
        if data is None:
            return None

        values = data["values"]
        df = pd.DataFrame(values)

        # Convert columns to correct types
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "timestamp"}, inplace=True)

        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype(float)

        # Sort from oldest -> newest
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def fetch_last_bar(self, symbol: str) -> Optional[dict]:
        """
        Convenience helper: return ONLY the most recent 5m bar as a dict.

        Keys:
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        """
        df = self.fetch_recent_5m(symbol=symbol, bars=1)
        if df is None or df.empty:
            print("[TwelveData] No data returned for last bar.")
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
