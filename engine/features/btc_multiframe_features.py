# engine/features/btc_multiframe_features.py
#
# BTC 5m multi-timeframe "institutional" feature builder.
#
# Input:
#   data/5m_btc.csv with columns:
#       timestamp, open, high, low, close, volume
#
# Output:
#   DataFrame indexed by timestamp with at least:
#       open, high, low, close, volume
#       open_1h, high_1h, low_1h, close_1h
#       open_4h, high_4h, low_4h, close_4h
#       open_1d, high_1d, low_1d, close_1d
#       atr_5m, rvol_5m
#       week_high, week_low, week_pos
#       regime_trend_up (0/1), regime_vol_high (0/1)
#       session
#       above_prev_high, below_prev_low
#       vwap_dist_atr
#
# Main public entry:
#   build_btc_5m_multiframe_features_institutional() -> (df, n_rows)

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------
# Resolve project root and CSV path
# ------------------------------------------------------

# This file lives at: quant_engine/engine/features/btc_multiframe_features.py
# Project root is two levels up.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BTC_5M_CSV_PATH = os.path.join(BASE_DIR, "data", "5m_btc.csv")


# ------------------------------------------------------
# Core loader
# ------------------------------------------------------


def _load_btc_5m_csv(csv_path: str = BTC_5M_CSV_PATH) -> pd.DataFrame:
    """
    Load BTC 5m CSV, normalize columns, set timestamp index.
    Expected columns: timestamp, open, high, low, close, volume
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"BTC 5m CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    if "timestamp" not in df.columns:
        raise ValueError("BTC CSV missing required 'timestamp' column.")

    # Parse timestamps, UTC, and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Set index
    df = df.set_index("timestamp")

    # Ensure numeric OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])

    return df


# ------------------------------------------------------
# Feature helpers
# ------------------------------------------------------


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Classic Wilder ATR on 5m bars.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return atr


def _compute_rvol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Relative volume: volume / rolling mean(volume, window)
    """
    vol = df["volume"]
    vol_ma = vol.rolling(window, min_periods=window).mean()
    rvol = vol / vol_ma
    return rvol


def _resample_ohlc(
    df: pd.DataFrame,
    rule: str,
    suffix: str,
) -> pd.DataFrame:
    """
    Resample 5m OHLC to higher timeframe and align back to 5m via merge_asof.
    rule: "1h", "4h", "1d" etc.
    suffix: "_1h", "_4h", "_1d"
    """
    ohlc = df[["open", "high", "low", "close"]].resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    ohlc = ohlc.dropna(how="any")
    ohlc = ohlc.add_suffix(suffix)

    # Align back to original 5m index (backward: use last completed HTF bar)
    merged = pd.merge_asof(
        df.sort_index(),
        ohlc.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    return merged


def _compute_week_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute week_high, week_low (based on close) and position in weekly range.
    """
    # Use ISO weeks
    week_index = df.index.to_period("W-MON")
    close = df["close"]

    week_high = close.groupby(week_index).transform("max")
    week_low = close.groupby(week_index).transform("min")

    span = (week_high - week_low).replace(0, np.nan)

    week_pos = (close - week_low) / span
    week_pos = week_pos.clip(0.0, 1.0).fillna(0.5)

    df["week_high"] = week_high
    df["week_low"] = week_low
    df["week_pos"] = week_pos

    return df


def _compute_trend_vol_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple trend + volatility regime flags.
    - regime_trend_up: 1 if fast EMA > slow EMA, else 0
    - regime_vol_high: 1 if atr_pct above 75th percentile, else 0
    """
    close = df["close"]

    ema_fast = close.ewm(span=50, adjust=False).mean()
    ema_slow = close.ewm(span=200, adjust=False).mean()

    df["regime_trend_up"] = (ema_fast > ema_slow).astype(int)

    # Vol regime based on ATR as % of price
    atr = df["atr_5m"]
    atr_pct = atr / close.replace(0, np.nan)
    df["atr_pct"] = atr_pct

    # Define "high vol" as above 75th percentile of atr_pct
    thresh = atr_pct.quantile(0.75)
    df["regime_vol_high"] = (atr_pct > thresh).astype(int)

    return df


def _compute_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each bar with a rough session:
      ASIA   : 00:00–07:00 UTC
      LONDON : 07:00–13:00 UTC
      NY     : 13:00–21:00 UTC
      OFF    : 21:00–24:00 UTC
    """
    hours = df.index.hour

    session = np.where(
        (hours >= 0) & (hours < 7),
        "ASIA",
        np.where(
            (hours >= 7) & (hours < 13),
            "LONDON",
            np.where((hours >= 13) & (hours < 21), "NY", "OFF"),
        ),
    )

    df["session"] = session.astype(str)
    return df


def _compute_prev_day_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute previous-day high/low and mark sweeps:
        above_prev_high: high > prev_day_high
        below_prev_low : low < prev_day_low
    """
    # Daily highs/lows
    daily_high = df["high"].resample("1d").max()
    daily_low = df["low"].resample("1d").min()

    # Shift by 1 day to get previous day's extremes
    prev_high = daily_high.shift(1)
    prev_low = daily_low.shift(1)

    prev_daily = pd.concat([prev_high, prev_low], axis=1)
    prev_daily.columns = ["prev_day_high", "prev_day_low"]
    prev_daily = prev_daily.dropna(how="any")

    # Align back to 5m
    df = pd.merge_asof(
        df.sort_index(),
        prev_daily.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    df["above_prev_high"] = (df["high"] > df["prev_day_high"]).fillna(False)
    df["below_prev_low"] = (df["low"] < df["prev_day_low"]).fillna(False)

    return df


def _compute_vwap_dist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute intraday VWAP and distance from VWAP in ATR units.

    VWAP approximation uses typical price (H+L+C)/3.
    """
    price = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].fillna(0.0)

    # Group by date for intraday VWAP
    date_index = df.index.normalize()

    cum_pv = (price * vol).groupby(date_index).cumsum()
    cum_v = vol.groupby(date_index).cumsum().replace(0, np.nan)

    vwap = cum_pv / cum_v
    df["vwap"] = vwap

    # Distance from VWAP in ATR units
    atr = df["atr_5m"].replace(0, np.nan)
    df["vwap_dist_atr"] = (df["close"] - df["vwap"]) / atr

    return df


# ------------------------------------------------------
# Main public builder
# ------------------------------------------------------


def build_btc_5m_multiframe_features_institutional() -> Tuple[pd.DataFrame, int]:
    """
    Full institutional-style feature builder for BTC 5m.

    Returns:
        df (pd.DataFrame): Indexed by timestamp with engineered features.
        n_rows (int): Number of rows after feature construction.
    """
    # 1) Load base 5m data
    df = _load_btc_5m_csv(BTC_5M_CSV_PATH)

    # 2) Compute ATR and RVOL on 5m
    df["atr_5m"] = _compute_atr(df, period=14)
    df["rvol_5m"] = _compute_rvol(df, window=20)

    # 3) Resample to 1H, 4H, 1D OHLC and align back
    df = _resample_ohlc(df, rule="1h", suffix="_1h")
    df = _resample_ohlc(df, rule="4h", suffix="_4h")
    df = _resample_ohlc(df, rule="1d", suffix="_1d")

    # 4) Week range features
    df = _compute_week_range_features(df)

    # 5) Trend & volatility regimes
    df = _compute_trend_vol_regimes(df)

    # 6) Session tags
    df = _compute_sessions(df)

    # 7) Previous-day sweep flags
    df = _compute_prev_day_sweeps(df)

    # 8) VWAP distance
    df = _compute_vwap_dist(df)

    # 9) Final cleanup: drop rows where we don't have core features yet
    required_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr_5m",
        "rvol_5m",
        "open_1h",
        "high_1h",
        "low_1h",
        "close_1h",
        "open_4h",
        "high_4h",
        "low_4h",
        "close_4h",
        "open_1d",
        "high_1d",
        "low_1d",
        "close_1d",
        "week_high",
        "week_low",
        "week_pos",
        "regime_trend_up",
        "regime_vol_high",
    ]

    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    n_rows = len(df)
    return df, n_rows
