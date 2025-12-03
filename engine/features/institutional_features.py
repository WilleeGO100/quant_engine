# engine/features/institutional_features.py
#
# Institutional-style feature engineering on top of your existing
# 5m multi-timeframe feature sets (ES or BTC).
#
# It expects a DataFrame with at least:
#   - 'timestamp' (datetime-like, UTC or tz-aware)
#   - 'open', 'high', 'low', 'close', 'volume'
#
# It adds:
#   Core:
#     atr_14, rvol_20, ret_z_50,
#     regime_vol, regime_trend,
#     session,
#     vwap_day, vwap_dist_atr,
#     rolling_imbalance_20, cum_delta
#
#   Context:
#     day_open, day_high, day_low, day_close,
#     day_range, day_pos_in_range,
#     prev_day_high, prev_day_low, prev_day_close,
#     in_prev_day_range, above_prev_high, below_prev_low,
#     week_high_5d, week_low_5d, pos_in_week_range
#
# Public:
#   add_institutional_features(df, tz="US/Eastern")  # generic
#   build_es_5m_multiframe_features_institutional(tz="US/Eastern")
#   build_btc_5m_multiframe_features_institutional(tz="US/Eastern")
#
# The ES/BTC-specific wrappers call their respective multi-timeframe builders
# and then decorate the result with institutional features.

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from engine.features.es_multiframe_features import build_es_5m_multiframe_features
from engine.features.btc_multiframe_features import build_btc_5m_multiframe_features


# -----------------------------------------------------------------------------
# Generic institutional feature builder
# -----------------------------------------------------------------------------

def add_institutional_features(
    df: pd.DataFrame,
    tz: str = "US/Eastern",
) -> pd.DataFrame:
    """
    Add institutional-style features to an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'open', 'high', 'low', 'close', 'volume'
        and ideally 'timestamp' as datetime-like.
    tz : str
        Timezone used for session labelling. Defaults to US/Eastern.

    Returns
    -------
    pd.DataFrame
        Same df with extra columns added.
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # Ensure timestamp is datetime and tz-aware
    # -------------------------------------------------------------------------
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        # Convert to target timezone for sessions / daily grouping
        df["timestamp"] = ts.dt.tz_convert(tz)
    else:
        # Create a dummy index-based timestamp if none exists
        df["timestamp"] = pd.date_range(
            start="2000-01-01", periods=len(df), freq="5min", tz=tz
        )

    # Shortcuts
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    ts_local = df["timestamp"]

    # -------------------------------------------------------------------------
    # 1) ATR (14)
    # -------------------------------------------------------------------------
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    df["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

    # -------------------------------------------------------------------------
    # 2) Relative Volume (20)
    # -------------------------------------------------------------------------
    vol_mean_20 = volume.rolling(window=20, min_periods=1).mean()
    df["rvol_20"] = volume / vol_mean_20.replace(0, np.nan)

    # -------------------------------------------------------------------------
    # 3) Z-score of returns (50)
    # -------------------------------------------------------------------------
    rets = close.pct_change().fillna(0.0)
    ret_mean_50 = rets.rolling(window=50, min_periods=10).mean()
    ret_std_50 = rets.rolling(window=50, min_periods=10).std()
    df["ret_z_50"] = (rets - ret_mean_50) / ret_std_50.replace(0, np.nan)

    # -------------------------------------------------------------------------
    # 4) Regime detection (volatility + trend)
    # -------------------------------------------------------------------------
    # Volatility regime based on ATR% of price
    atr_pct = df["atr_14"] / close.replace(0, np.nan)
    # Simple cutoffs (tune later):
    # <0.5% -> LOW, >1.5% -> HIGH, else NORMAL
    vol_regime = np.where(
        atr_pct < 0.005,
        "LOW",
        np.where(atr_pct > 0.015, "HIGH", "NORMAL"),
    )
    df["regime_vol"] = vol_regime

    # Trend regime using EMA(50) vs EMA(200)
    ema_fast = close.ewm(span=50, adjust=False).mean()
    ema_slow = close.ewm(span=200, adjust=False).mean()
    diff = ema_fast - ema_slow
    # Threshold so we don't flip on noise
    thresh = close * 0.001  # 0.1% of price
    trend_regime = np.where(
        diff > thresh, "UP", np.where(diff < -thresh, "DOWN", "FLAT")
    )
    df["regime_trend"] = trend_regime

    # -------------------------------------------------------------------------
    # 5) Session labels (Asia / London / NY / Off)
    # -------------------------------------------------------------------------
    hours = ts_local.dt.hour

    def _session_for_hour(h: int) -> str:
        # All in US/Eastern
        # Approximate windows:
        #   ASIA   20:00 - 03:00
        #   LONDON 03:00 - 09:00
        #   NY     09:00 - 16:00
        #   OFF    remaining
        if h >= 20 or h < 3:
            return "ASIA"
        if 3 <= h < 9:
            return "LONDON"
        if 9 <= h < 16:
            return "NY"
        return "OFF"

    df["session"] = hours.apply(_session_for_hour)

    # -------------------------------------------------------------------------
    # 6) VWAP (per day) + distance in ATR
    # -------------------------------------------------------------------------
    typical_price = (high + low + close) / 3.0
    date_key = ts_local.dt.date

    grouped_day = df.groupby(date_key, group_keys=False)

    cum_pv = grouped_day.apply(
        lambda g: (g["volume"] * typical_price.loc[g.index]).cumsum()
    )
    cum_vol = grouped_day["volume"].cumsum()

    df["vwap_day"] = cum_pv / cum_vol.replace(0, np.nan)
    df["vwap_dist_atr"] = (close - df["vwap_day"]) / df["atr_14"].replace(0, np.nan)

    # -------------------------------------------------------------------------
    # 7) Rolling imbalance (20) + cumulative delta
    # -------------------------------------------------------------------------
    sign_ret = np.sign(rets).replace(0, 0.0)
    delta = sign_ret * volume

    df["rolling_imbalance_20"] = delta.rolling(window=20, min_periods=1).sum()
    df["cum_delta"] = delta.cumsum()

    # -------------------------------------------------------------------------
    # 8) Daily context: today's range & position
    # -------------------------------------------------------------------------
    # Daily OHLC by date, then broadcast back to 5m bars
    day_ohlc = grouped_day.agg(
        day_open=("open", "first"),
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_close=("close", "last"),
    )

    # Map back to df
    df["day_open"] = date_key.map(day_ohlc["day_open"])
    df["day_high"] = date_key.map(day_ohlc["day_high"])
    df["day_low"] = date_key.map(day_ohlc["day_low"])
    df["day_close"] = date_key.map(day_ohlc["day_close"])

    df["day_range"] = df["day_high"] - df["day_low"]
    # Position in today's range: 0 at low, 1 at high
    day_range_safe = df["day_range"].replace(0, np.nan)
    df["day_pos_in_range"] = (close - df["day_low"]) / day_range_safe

    # -------------------------------------------------------------------------
    # 9) Yesterday context: previous day's range & close
    # -------------------------------------------------------------------------
    # Build a daily-level frame with previous-day values, then map back.

    day_ohlc = day_ohlc.copy()
    day_ohlc.index = pd.to_datetime(day_ohlc.index)

    # Shift by 1 day to get previous day's stats
    prev_day = day_ohlc.shift(1)

    date_key_ts = pd.to_datetime(date_key)

    df["prev_day_high"] = date_key_ts.map(
        prev_day["day_high"].reindex(prev_day.index)
    )
    df["prev_day_low"] = date_key_ts.map(
        prev_day["day_low"].reindex(prev_day.index)
    )
    df["prev_day_close"] = date_key_ts.map(
        prev_day["day_close"].reindex(prev_day.index)
    )

    # Flags relative to yesterday's range
    df["in_prev_day_range"] = (
        (close >= df["prev_day_low"]) & (close <= df["prev_day_high"])
    )
    df["above_prev_high"] = close > df["prev_day_high"]
    df["below_prev_low"] = close < df["prev_day_low"]

    # -------------------------------------------------------------------------
    # 10) 5-day context: rolling week range & position
    # -------------------------------------------------------------------------
    daily_for_week = day_ohlc.copy()
    daily_for_week["week_high_5d"] = (
        daily_for_week["day_high"].rolling(window=5, min_periods=1).max()
    )
    daily_for_week["week_low_5d"] = (
        daily_for_week["day_low"].rolling(window=5, min_periods=1).min()
    )

    week_high_map = daily_for_week["week_high_5d"]
    week_low_map = daily_for_week["week_low_5d"]

    df["week_high_5d"] = date_key_ts.map(
        week_high_map.reindex(week_high_map.index)
    )
    df["week_low_5d"] = date_key_ts.map(
        week_low_map.reindex(week_low_map.index)
    )

    week_range = df["week_high_5d"] - df["week_low_5d"]
    week_range_safe = week_range.replace(0, np.nan)
    df["pos_in_week_range"] = (close - df["week_low_5d"]) / week_range_safe

    return df


# -----------------------------------------------------------------------------
# ES + institutional
# -----------------------------------------------------------------------------

def build_es_5m_multiframe_features_institutional(
    tz: str = "US/Eastern",
) -> Tuple[pd.DataFrame, int]:
    """
    Call ES 5m multi-timeframe feature builder, then decorate with
    institutional features.
    """
    base_df = build_es_5m_multiframe_features()
    if base_df is None or base_df.empty:
        raise RuntimeError("build_es_5m_multiframe_features() returned no data.")

    full_df = add_institutional_features(base_df, tz=tz)
    return full_df, len(full_df)


# -----------------------------------------------------------------------------
# BTC + institutional
# -----------------------------------------------------------------------------

def build_btc_5m_multiframe_features_institutional(
    tz: str = "US/Eastern",
) -> Tuple[pd.DataFrame, int]:
    """
    Call BTC 5m multi-timeframe feature builder, then decorate with
    institutional features.
    """
    base_df = build_btc_5m_multiframe_features()
    if base_df is None or base_df.empty:
        raise RuntimeError("build_btc_5m_multiframe_features() returned no data.")

    full_df = add_institutional_features(base_df, tz=tz)
    return full_df, len(full_df)
