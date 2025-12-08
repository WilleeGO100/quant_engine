# live/main_live_btc.py
#
# LIVE / PAPER VERSION OF BTC PO3 ENGINE
# -------------------------------------
# This script:
#   1) Pulls recent BTC/USD 5m candles from TwelveData
#   2) Builds the institutional multi-timeframe feature frame
#   3) Instantiates BTCPO3PowerStrategy (same logic as backtest)
#   4) Evaluates the latest bar and prints LONG / SHORT / FLAT decision
#
# No real trading happens here. This is a "live backtest" / signal monitor.

from __future__ import annotations

import os
import sys
import logging
from typing import Dict, Any

import requests
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 0. Ensure project root is on sys.path
# ---------------------------------------------------------------------------
# When you run:
#   .venv\Scripts\python.exe live\main_live_btc.py
# Python's sys.path[0] is C:\Python312\quant_engine\live
# We need C:\Python312\quant_engine on the path so that `import engine...` works.

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("DEBUG: PROJECT_ROOT on sys.path ->", PROJECT_ROOT)

# Now imports like `from engine.features...` should work
from engine.features.btc_multiframe_features import build_btc_multiframe_features
from engine.strategies.smc_po3_power_btc import (
    BTCPO3PowerStrategy,
    BTCPO3PowerConfig,
)


# ---------------------------------------------------------------------------
# 1. Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# 2. TwelveData 5m BTC fetcher
# ---------------------------------------------------------------------------
def fetch_btc_5m_from_twelvedata(
    api_key: str,
    symbol: str = "BTC/USD",
    interval: str = "5min",
    outputsize: int = 500,
) -> pd.DataFrame:
    """
    Fetch latest BTC/USD 5m candles from TwelveData.

    Returns a DataFrame indexed by timestamp (ascending), with columns:
        ["open", "high", "low", "close", "volume"]

    If volume is missing from the feed, we synthesize a zero-volume column
    so that the rest of the pipeline still runs.
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "order": "ASC",  # earliest -> latest
    }

    logging.info(
        "Requesting %s %s data from TwelveData (outputsize=%d)...",
        symbol,
        interval,
        outputsize,
    )
    resp = requests.get(url, params=params, timeout=10)

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to decode TwelveData JSON: {e}")

    if "status" in data and data["status"] == "error":
        raise RuntimeError(f"TwelveData error: {data.get('message')}")

    if "values" not in data:
        raise RuntimeError(f"Unexpected TwelveData payload: {data}")

    values = data["values"]
    if not values:
        raise RuntimeError("TwelveData returned no candles.")

    # Convert to DataFrame
    df = pd.DataFrame(values)

    # Expected keys from TwelveData: datetime, open, high, low, close, [volume?]
    base_required = {"datetime", "open", "high", "low", "close"}
    missing_base = base_required - set(df.columns)
    if missing_base:
        raise RuntimeError(f"Missing core OHLC columns from TwelveData: {missing_base}")

    # If volume is missing, synthesize it as zeros (so pipeline still works)
    if "volume" not in df.columns:
        logging.warning(
            "TwelveData BTC feed has no 'volume' column. "
            "Synthesizing volume=0.0 for all rows."
        )
        df["volume"] = 0.0

    # Convert dtypes
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df = df.sort_values("datetime").set_index("datetime")

    logging.info(
        "Fetched %d BTC 5m candles (from %s to %s).",
        len(df),
        df.index[0],
        df.index[-1],
    )

    return df


# ---------------------------------------------------------------------------
# 3. Feature builder wrapper
# ---------------------------------------------------------------------------
def build_live_btc_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Wraps your institutional feature builder so the live engine
    can call it exactly like the backtest.
    """
    logging.info(
        "Building BTC multiframe features for live frame (rows=%d)...",
        len(prices),
    )

    # This function name matches what your backtest already uses:
    #   build_btc_multiframe_features(prices)
    features = build_btc_multiframe_features(prices)

    logging.info(
        "Live BTC feature frame built. Final rows: %d (index from %s to %s).",
        len(features),
        features.index[0] if len(features) > 0 else "n/a",
        features.index[-1] if len(features) > 0 else "n/a",
    )
    return features


# ---------------------------------------------------------------------------
# 4. Live evaluation helper
# ---------------------------------------------------------------------------
def evaluate_latest_bar(
    features: pd.DataFrame,
    strategy: BTCPO3PowerStrategy,
) -> Dict[str, Any]:
    """
    Take the *latest* row of the feature frame and ask the strategy
    for a decision. Returns a dict summarizing the decision and
    context for logging/printing.
    """
    if features.empty:
        raise RuntimeError("Feature frame is empty – nothing to evaluate.")

    latest_idx = features.index[-1]
    latest_row = features.iloc[-1]

    # Your strategy's on_bar takes a single row: on_bar(self, row)
    decision = strategy.on_bar(latest_row)

    result: Dict[str, Any] = {
        "timestamp": latest_idx,
        "raw_decision": decision,
    }

    # If decision is a simple string like "LONG", "SHORT", "FLAT":
    if isinstance(decision, str):
        result["side"] = decision
    # If decision is a dict-like object with more detail:
    elif isinstance(decision, dict):
        result["side"] = decision.get("side") or decision.get("action")
        result["context"] = {
            k: v for k, v in decision.items() if k not in ("side", "action")
        }
    else:
        # Fallback – just store repr
        result["side"] = str(decision)

    return result


# ---------------------------------------------------------------------------
# 5. Main entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    # 5A. Load environment and API key
    load_dotenv()
    api_key = os.getenv("TWELVEDATA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "TWELVEDATA_API_KEY not found in environment. "
            "Make sure it's set in your .env file."
        )

    # 5B. Fetch recent BTC 5m candles
    prices = fetch_btc_5m_from_twelvedata(api_key=api_key, outputsize=500)

    # 5C. Build institutional feature frame (same as backtest)
    features = build_live_btc_features(prices)

    # Drop any rows that are incomplete (NaNs from rolling windows, etc.)
    features = features.dropna()
    if features.empty:
        raise RuntimeError("All feature rows are NaN after dropna – check feature builder.")

    # 5D. Instantiate the BTC PO3 strategy with SAME CONFIG STYLE as backtest
    config = BTCPO3PowerConfig()
    strategy = BTCPO3PowerStrategy(config=config)

    logging.info("BTCPO3PowerStrategy instantiated for live evaluation.")

    # 5E. Evaluate the latest bar
    result = evaluate_latest_bar(features, strategy)

    ts = result.get("timestamp")
    side = result.get("side")
    raw_decision = result.get("raw_decision")
    context = result.get("context", {})

    logging.info("-------------------------------------------------------")
    logging.info("[LIVE] Latest bar timestamp: %s", ts)
    logging.info("[LIVE] Strategy side       : %s", side)
    logging.info("[LIVE] Raw decision object : %r", raw_decision)
    if context:
        logging.info("[LIVE] Extra context      : %r", context)
    logging.info("-------------------------------------------------------")

    # Also print a clean, one-line human summary:
    print(
        f"[LIVE] {ts} | Decision: {side} | "
        f"Context: {context if context else 'n/a'}"
    )


if __name__ == "__main__":
    main()
