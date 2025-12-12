# engine/features/btc_features.py
#
# BTC 5m institutional-style feature builder, modeled after ES features.
#
# - Loads btc_5m_live.csv (or another filename) from ROOT/data by default
# - Normalizes time to UTC
# - Computes:
#     * atr_14, atr_pct_5m
#     * rvol_5m
#     * candle ranges (body / wicks / total)
#     * simple session tag (asia / london / ny / off)
#     * weekly position (0..1 in weekly low→high)
#     * vwap + vwap_dist_atr
#     * ema_fast / ema_slow
#     * regime_trend_up
#

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BTC_5M_FILENAME = "btc_5m_live.csv"


# =====================================================
# CORE HELPERS
# =====================================================

def _load_btc_raw(path: Path) -> pd.DataFrame:
    """
    Load a BTC 5m OHLCV CSV.

    Expected columns:
        time, open, high, low, close, [volume]
    """
    if not path.exists():
        raise FileNotFoundError(f"BTC CSV not found: {path}")

    df = pd.read_csv(path)

    if "time" not in df.columns:
        # Try first column as datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], utc=True)
        df = df.rename(columns={df.columns[0]: "time"})
    else:
        df["time"] = pd.to_datetime(df["time"], utc=True)

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required price column: '{col}'")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df.sort_values("time").reset_index(drop=True)
    return df


def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame is indexed by UTC time.
    """
    if "time" not in df.columns:
        raise ValueError("Expected 'time' column in BTC DataFrame.")

    time = pd.to_datetime(df["time"], utc=True)
    df = df.copy()
    df["time"] = time
    df = df.set_index("time").sort_index()
    return df


# =====================================================
# TECHNICAL INDICATORS / FEATURES
# =====================================================

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Simple ATR using rolling mean of True Range.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def _compute_rvol(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Relative volume = current volume / rolling mean volume.
    """
    vol_mean = volume.rolling(window=window, min_periods=5).mean()
    rvol = volume / vol_mean.replace(0.0, np.nan)
    rvol = rvol.replace([np.inf, -np.inf], np.nan)
    rvol = rvol.fillna(1.0)
    return rvol


def _compute_candle_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute body / wick / total ranges.
    """
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    body = (close - open_).abs()
    upper_wick = (high - np.maximum(open_, close)).clip(lower=0.0)
    lower_wick = (np.minimum(open_, close) - low).clip(lower=0.0)
    total_range = high - low

    return pd.DataFrame(
        {
            "body_range": body,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "total_range": total_range,
        },
        index=df.index,
    )


def _classify_session_utc(time_series: pd.Series) -> pd.Series:
    """
    Simple UTC-based session tag:
      - 00:00–07:59 : asia
      - 08:00–15:59 : london
      - 16:00–23:59 : ny
    """
    ts = pd.to_datetime(time_series, utc=True)
    hours = ts.dt.hour

    sess = pd.Series(index=ts.index, dtype="object")

    sess[(hours >= 0) & (hours < 8)] = "asia"
    sess[(hours >= 8) & (hours < 16)] = "london"
    sess[(hours >= 16) & (hours < 24)] = "ny"

    return sess.fillna("off")


def _compute_week_position(df: pd.DataFrame) -> pd.Series:
    """
    Weekly position of price: 0 at weekly low, 1 at weekly high.
    If range is zero, returns 0.5.
    """
    close = df["close"].astype(float)
    time = pd.to_datetime(df["time"], utc=True)
    # FIX: use .dt.to_period so we operate on the datetime *values*,
    # not the RangeIndex, which caused "unsupported Type RangeIndex".
    week_index = time.dt.to_period("W-MON")

    weekly_low = close.groupby(week_index).transform("min")
    weekly_high = close.groupby(week_index).transform("max")

    rng = (weekly_high - weekly_low).replace(0, np.nan)
    week_pos = (close - weekly_low) / rng
    week_pos = week_pos.clip(0, 1)
    week_pos = week_pos.fillna(0.5)

    return week_pos


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Session-agnostic running VWAP.
    """
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    vol_safe = volume.replace(0.0, np.nan)
    cum_pv = (close * vol_safe).cumsum()
    cum_vol = vol_safe.cumsum()

    vwap = cum_pv / cum_vol.replace(0.0, np.nan)
    vwap = vwap.fillna(method="ffill").fillna(close)
    return vwap


def _compute_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _compute_regime_trend_up(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    ema_fast = _compute_ema(close, fast)
    ema_slow = _compute_ema(close, slow)
    return ema_fast > ema_slow


def _compute_vwap_dist_atr(
    close: pd.Series, vwap: pd.Series, atr: pd.Series
) -> pd.Series:
    dist = close - vwap
    dist_atr = dist / atr.replace(0.0, np.nan)
    dist_atr = dist_atr.replace([np.inf, -np.inf], np.nan)
    dist_atr = dist_atr.fillna(0.0)
    return dist_atr


# =====================================================
# MAIN FEATURE BUILDER
# =====================================================

def build_btc_5m_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build BTC 5m institutional-style features.

    Input:
        df_raw:
            columns: time, open, high, low, close, [volume]
    """
    df = df_raw.copy()

    # Ensure basic schema
    if "time" not in df.columns:
        raise ValueError("Expected 'time' column in BTC raw DataFrame.")

    df["time"] = pd.to_datetime(df["time"], utc=True)

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required price column: '{col}'")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df.sort_values("time").reset_index(drop=True)

    # Attach time index for rolling/groupby
    df_indexed = _ensure_time_index(df)

    # ATR + ATR%
    atr_14 = _compute_atr(df_indexed, period=14)
    atr_pct_5m = atr_14 / df_indexed["close"].replace(0.0, np.nan)

    # RVOL
    rvol_5m = _compute_rvol(df_indexed["volume"], window=20)

    # Candle structure
    ranges = _compute_candle_ranges(df_indexed)

    # Session tag
    session_type = _classify_session_utc(df_indexed.index.to_series())

    # Weekly position
    # Use the *original* df with 'time' column for weekly grouping
    df_for_week = df.copy()
    week_pos = _compute_week_position(df_for_week)

    # VWAP + VWAP distance in ATR units
    vwap = _compute_vwap(df_indexed)
    vwap_dist_atr = _compute_vwap_dist_atr(
        df_indexed["close"], vwap, atr_14
    )

    # EMA + regime
    close_series = df_indexed["close"]
    ema_fast = _compute_ema(close_series, span=20)
    ema_slow = _compute_ema(close_series, span=50)
    regime_trend_up = _compute_regime_trend_up(close_series, fast=20, slow=50)

    # Combine everything back into a single frame (time as a column)
    feat = pd.DataFrame(
        {
            "time": df_indexed.index,
            "open": df_indexed["open"],
            "high": df_indexed["high"],
            "low": df_indexed["low"],
            "close": df_indexed["close"],
            "volume": df_indexed["volume"],
            "atr_14": atr_14,
            "atr_pct_5m": atr_pct_5m,
            "rvol_5m": rvol_5m,
            "body_range": ranges["body_range"],
            "upper_wick": ranges["upper_wick"],
            "lower_wick": ranges["lower_wick"],
            "total_range": ranges["total_range"],
            "session_type": session_type,
            "week_pos": week_pos.values,  # align by row
            "vwap": vwap,
            "vwap_dist_atr": vwap_dist_atr,
            "ema_fast_20": ema_fast,
            "ema_slow_50": ema_slow,
            "regime_trend_up": regime_trend_up,
        }
    )

    return feat


# =====================================================
# CONVENIENCE WRAPPER
# =====================================================

def load_and_build_btc_5m_features(
    filename: str | Path = BTC_5M_FILENAME,
) -> pd.DataFrame:
    """
    Convenience function:

        df_feat = load_and_build_btc_5m_features()

    which:
      - loads ROOT/data/<filename> (default: btc_5m_live.csv)
      - runs build_btc_5m_features(df_raw)
      - returns the feature DataFrame.
    """
    path = Path(filename)
    if not path.is_absolute():
        path = ROOT / "data" / filename

    df_raw = _load_btc_raw(path)
    df_feat = build_btc_5m_features(df_raw)
    return df_feat
