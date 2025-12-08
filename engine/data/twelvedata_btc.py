# engine/data/twelvedata_btc.py
#
# TwelveData BTC/USD 5m data access layer (REST polling).
# - Uses TWELVEDATA_API_KEY from environment or .env
# - Provides:
#     fetch_btcusd_5m_history(outputsize=500)
#     fetch_latest_btcusd_5m_bar()
#
# All timestamps returned as timezone-aware UTC pandas DateTimeIndex.

from __future__ import annotations

import os
import time
import logging
from typing import Optional

import requests
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# 1) Basic config
# ---------------------------------------------------------------------

# Load .env if present
load_dotenv()

TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

if not TWELVEDATA_API_KEY:
    raise RuntimeError(
        "TWELVEDATA_API_KEY not set. "
        "Add it to a .env file or your environment."
    )

BASE_URL = "https://api.twelvedata.com/time_series"

# IMPORTANT: TwelveData supports BTC/USD (fiat), not BTC/USDT on basic plans
SYMBOL = "BTC/USD"
INTERVAL = "5min"
TIMEZONE = "UTC"

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# 2) Low-level request helper
# ---------------------------------------------------------------------

def _call_twelvedata(
    params: dict,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> dict:
    """
    Internal helper to call TwelveData with basic retry logic.
    Raises RuntimeError on final failure.
    """
    params = dict(params)  # shallow copy
    params.setdefault("apikey", TWELVEDATA_API_KEY)

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.get(BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if "status" in data and data["status"] == "error":
                # TwelveData-style error payload
                message = data.get("message", "Unknown TwelveData error")
                raise RuntimeError(f"TwelveData error: {message}")

            return data

        except Exception as exc:
            logger.warning(
                "TwelveData request failed on attempt %s/%s: %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt >= max_retries:
                raise RuntimeError(f"TwelveData request failed: {exc}") from exc
            time.sleep(backoff_seconds)


# ---------------------------------------------------------------------
# 3) Public functions: history + latest bar
# ---------------------------------------------------------------------

def _values_to_dataframe(values: list[dict]) -> pd.DataFrame:
    """
    Convert TwelveData 'values' list into a pandas DataFrame
    indexed by UTC datetime, sorted oldest → newest.
    """
    if not values:
        raise ValueError("No 'values' returned from TwelveData.")

    # TwelveData returns newest first by default; reverse to oldest→newest
    values = list(reversed(values))

    records = []
    index = []

    for row in values:
        # Example row:
        # {"datetime":"2025-12-03 13:25:00","open":"...", "high":"...", ...}
        ts = pd.to_datetime(row["datetime"], utc=True)
        index.append(ts)

        records.append(
            {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
            }
        )

    df = pd.DataFrame(records, index=index)
    df.index.name = "datetime"
    return df


def fetch_btcusd_5m_history(outputsize: int = 500) -> pd.DataFrame:
    """
    Fetch historical BTC/USD 5m candles.

    Parameters
    ----------
    outputsize : int
        Number of candles to request (max allowed by your plan; 500 is typical).

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume
        Index: UTC datetime (oldest → newest)
    """
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": outputsize,
        "timezone": TIMEZONE,
        "order": "desc",  # newest first; we reverse in helper
        "format": "JSON",
    }

    data = _call_twelvedata(params)

    values = data.get("values")
    if not values:
        raise RuntimeError(f"No 'values' in TwelveData response: {data}")

    df = _values_to_dataframe(values)
    logger.info(
        "Fetched %s historical %s %s candles (from %s to %s).",
        len(df),
        SYMBOL,
        INTERVAL,
        df.index[0],
        df.index[-1],
    )
    return df


def fetch_latest_btcusd_5m_bar() -> pd.DataFrame:
    """
    Fetch the latest *closed* 5m BTC/USD candle as a 1-row DataFrame.

    Returns
    -------
    pd.DataFrame
        Single row with columns: open, high, low, close, volume
        Index: UTC datetime of that candle.
    """
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": 1,
        "timezone": TIMEZONE,
        "order": "desc",  # newest first
        "format": "JSON",
    }

    data = _call_twelvedata(params)

    values = data.get("values")
    if not values:
        raise RuntimeError(f"No 'values' in TwelveData latest-bar response: {data}")

    df = _values_to_dataframe(values)
    latest_ts = df.index[-1]
    logger.info("Fetched latest %s %s bar at %s.", SYMBOL, INTERVAL, latest_ts)
    return df.tail(1)


# ---------------------------------------------------------------------
# 4) Simple CLI test
# ---------------------------------------------------------------------

def _demo():
    """
    Run this file directly to sanity-check your TwelveData connectivity.
    """
    print("=== TwelveData BTC/USD 5m demo ===")
    hist = fetch_make_btc_5m_csv.py(outputsize=20)
    print("\nLast 5 historical candles:")
    print(hist.tail(5))

    latest = fetch_latest_btcusd_5m_bar()
    print("\nLatest closed 5m bar:")
    print(latest)


if __name__ == "__main__":
    _demo()
