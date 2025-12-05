# live/btc_po3_paper.py
#
# ALL-IN-ONE BTC PO3 LIVE PAPER ENGINE (MATCHED TO YOUR PROJECT)
#
# - Pulls BTC/USD 5m from TwelveData (REST)
# - Builds features via engine/features/btc_multiframe_features.py
# - Uses BTCPO3PowerStrategy from engine/strategies/smc_po3_power_btc.py
# - Runs a live paper engine with ATR-based SL/TP & time stop
#
# REQUIREMENTS:
#   pip install requests pandas python-dotenv
#   .env must contain: TWELVEDATA_API_KEY=YOUR_KEY

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

import requests
import pandas as pd
from dotenv import load_dotenv

# ============================================================
# 1) TwelveData configuration
# ============================================================

load_dotenv()

TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
if not TWELVEDATA_API_KEY:
    raise RuntimeError("TWELVEDATA_API_KEY missing in .env")

BASE_URL = "https://api.twelvedata.com/time_series"
SYMBOL = "BTC/USD"
INTERVAL = "5min"
TIMEZONE = "UTC"
API_SAVER_MODE = True
poll = 180 if API_SAVER_MODE else 60


# ============================================================
# 2) Logging
# ============================================================

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    f = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    h.setFormatter(f)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ============================================================
# 3) Import your real feature builder + BTC PO3 strategy
# ============================================================

# Feature pipeline
try:
    from engine.features.btc_multiframe_features import build_btc_multiframe_features
except ImportError as e:
    raise ImportError(
        "Cannot import build_btc_multiframe_features from "
        "engine/features/btc_multiframe_features.py"
    ) from e

# BTC PO3 strategy + config
try:
    from engine.strategies.smc_po3_power_btc import (
        BTCPO3PowerStrategy,
        BTCPO3PowerConfig,
    )
except ImportError as e:
    raise ImportError(
        "Cannot import BTCPO3PowerStrategy and BTCPO3PowerConfig from "
        "engine/strategies/smc_po3_power_btc.py"
    ) from e

# Name of the ATR column in your BTC feature frame
# If your features use a different name (e.g. "atr_14"), change this.
ATR_COLUMN_NAME = "atr"

# ============================================================
# 4) TwelveData REST helpers
# ============================================================

def _call_twelvedata(params: dict) -> dict:
    params = {**params, "apikey": TWELVEDATA_API_KEY}
    resp = requests.get(BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "status" in data and data["status"] == "error":
        raise RuntimeError(f"TwelveData error: {data.get('message')}")
    return data


def _df_from_values(values: list[dict]) -> pd.DataFrame:
    # TwelveData returns newest first → reverse to oldest→newest
    values = list(reversed(values))
    rows = []
    idx = []
    for v in values:
        ts = pd.to_datetime(v["datetime"], utc=True)
        idx.append(ts)
        rows.append(
            {
                "open": float(v["open"]),
                "high": float(v["high"]),
                "low": float(v["low"]),
                "close": float(v["close"]),
                "volume": float(v.get("volume", 0.0)),
            }
        )
    df = pd.DataFrame(rows, index=idx)
    df.index.name = "datetime"
    return df


def fetch_history(n: int = 500) -> pd.DataFrame:
    data = _call_twelvedata(
        {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "outputsize": n,
            "timezone": TIMEZONE,
            "order": "desc",
        }
    )
    values = data.get("values")
    if not values:
        raise RuntimeError(f"No 'values' in TwelveData history response: {data}")
    df = _df_from_values(values)
    logger.info(
        "Fetched %s historical %s %s candles (from %s to %s).",
        len(df),
        SYMBOL,
        INTERVAL,
        df.index[0],
        df.index[-1],
    )
    return df


def fetch_latest() -> pd.DataFrame:
    data = _call_twelvedata(
        {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "outputsize": 1,
            "timezone": TIMEZONE,
            "order": "desc",
        }
    )
    values = data.get("values")
    if not values:
        raise RuntimeError(f"No 'values' in TwelveData latest-bar response: {data}")
    df = _df_from_values(values)
    latest_ts = df.index[-1]
    logger.info("Fetched latest %s %s bar at %s.", SYMBOL, INTERVAL, latest_ts)
    return df.tail(1)

# ============================================================
# 5) Paper position model
# ============================================================

@dataclass
class Position:
    side: str       # "LONG" or "SHORT"
    entry: float
    size: float
    stop: float
    tp: float
    bars: int = 0

    def pnl(self, price: float) -> float:
        if self.side == "LONG":
            return (price - self.entry) * self.size
        else:
            return (self.entry - price) * self.size

# ============================================================
# 6) Live BTC PO3 paper engine
# ============================================================

class LiveBTCPO3Engine:
    """
    Live paper engine:
      - Keeps raw BTC/USD 5m OHLCV
      - Builds institutional BTC features each step
      - Feeds latest row into BTCPO3PowerStrategy.on_bar(row_dict)
      - Manages a single paper position with ATR-based SL/TP
    """

    def __init__(
        self,
        equity: float = 100_000.0,
        stop_mult: float = 1.2,
        tp_mult: float = 2.5,
        max_bars: int = 96,
        poll_secs: int = 180,
    ):
        self.equity = equity
        self.stop_mult = stop_mult
        self.tp_mult = tp_mult
        self.max_bars = max_bars
        self.poll_secs = poll_secs

        self.prices: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.strategy = BTCPO3PowerStrategy(BTCPO3PowerConfig())

        self.position: Optional[Position] = None
        self.last_ts: Optional[pd.Timestamp] = None

    # -----------------------------
    # Initialization
    # -----------------------------
    def initialize(self, history_bars: int = 500) -> None:
        logger.info("Initializing BTC PO3 live paper engine with %s history bars...", history_bars)

        # 1) Fetch initial OHLCV
        self.prices = fetch_history(history_bars)

        # 2) Build multiframe BTC feature frame
        self.features = build_btc_multiframe_features(self.prices.copy())

        if not isinstance(self.features, pd.DataFrame):
            raise RuntimeError("build_btc_multiframe_features did not return a DataFrame.")

        if ATR_COLUMN_NAME not in self.features.columns:
            raise RuntimeError(
                f"ATR column '{ATR_COLUMN_NAME}' not found in features. "
                "Update ATR_COLUMN_NAME to match your feature pipeline."
            )

        self.last_ts = self.prices.index[-1]
        logger.info(
            "Engine initialized. Last bar time: %s | Feature rows: %s",
            self.last_ts,
            len(self.features),
        )

    # -----------------------------
    # Main loop
    # -----------------------------
    def run(self) -> None:
        if self.prices is None or self.features is None:
            raise RuntimeError("Call initialize() before run().")

        logger.info("Starting live BTC PO3 paper loop (poll every %s seconds)...", self.poll_secs)

        while True:
            try:
                self._step()
            except Exception as exc:
                logger.exception("Error in live loop: %s", exc)
            time.sleep(self.poll_secs)

    # -----------------------------
    # One poll step
    # -----------------------------
    def _step(self) -> None:
        latest_df = fetch_latest()
        latest_ts = latest_df.index[-1]

        if self.last_ts is not None and latest_ts <= self.last_ts:
            # No new closed 5m bar yet
            return

        logger.info("New 5m bar detected at %s", latest_ts)

        # 1) Append new bar to price history
        self.prices = pd.concat([self.prices, latest_df])
        self.prices = self.prices[~self.prices.index.duplicated(keep="last")]

        # 2) Rebuild features from scratch (simple, safe version)
        self.features = build_btc_multiframe_features(self.prices.copy())

        latest_features = self.features.iloc[-1]
        row_dict = latest_features.to_dict()

        # 3) Feed to BTCPO3PowerStrategy (expects a dict)
        signal = self.strategy.on_bar(row_dict)
        logger.info("Strategy signal at %s: %s", latest_ts, signal)

        # 4) Extract close & ATR for risk
        close_price = float(row_dict.get("close", self.prices["close"].iloc[-1]))
        atr_value = float(latest_features[ATR_COLUMN_NAME])

        # 5) Update any open position and maybe open a new one
        self._update_position(close_price, signal, atr_value)

        self.last_ts = latest_ts

    # -----------------------------
    # Position & risk management
    # -----------------------------
    def _update_position(self, price: float, signal: str, atr: float) -> None:
        # 1) Manage existing position
        if self.position is not None:
            self.position.bars += 1

            hit_tp = False
            hit_sl = False

            if self.position.side == "LONG":
                if price >= self.position.tp:
                    hit_tp = True
                elif price <= self.position.stop:
                    hit_sl = True
            else:  # SHORT
                if price <= self.position.tp:
                    hit_tp = True
                elif price >= self.position.stop:
                    hit_sl = True

            time_exit = self.position.bars >= self.max_bars

            if hit_tp or hit_sl or time_exit:
                reason = "TP" if hit_tp else "SL" if hit_sl else "TIME"
                pnl = self.position.pnl(price)
                self.equity += pnl

                logger.info(
                    "EXIT %s | entry=%.2f | exit=%.2f | pnl=%.2f | reason=%s | equity=%.2f",
                    self.position.side,
                    self.position.entry,
                    price,
                    pnl,
                    reason,
                    self.equity,
                )

                self.position = None

        # 2) If flat, maybe open a new position
        if self.position is None and signal in ("LONG", "SHORT"):
            if signal == "LONG":
                stop = price - self.stop_mult * atr
                tp = price + self.tp_mult * atr
            else:  # SHORT
                stop = price + self.stop_mult * atr
                tp = price - self.tp_mult * atr

            self.position = Position(
                side=signal,
                entry=price,
                size=1.0,   # later we'll tie this to real risk sizing
                stop=stop,
                tp=tp,
            )

            logger.info(
                "ENTER %s | entry=%.2f | SL=%.2f | TP=%.2f | equity=%.2f",
                signal,
                price,
                stop,
                tp,
                self.equity,
            )

# ============================================================
# 7) Entry point
# ============================================================

def main() -> None:
    engine = LiveBTCPO3Engine(
        equity=100_000.0,
        stop_mult=1.2,
        tp_mult=2.5,
        max_bars=96,
        poll_secs=180,
    )
    engine.initialize(history_bars=500)
    engine.run()


if __name__ == "__main__":
    main()
