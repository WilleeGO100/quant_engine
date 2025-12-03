# engine/features/es_features.py

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


SessionLabel = Literal["ASIA", "LONDON", "NY", "OFF"]


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Classic Wilder ATR."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def _compute_session(time_index: pd.DatetimeIndex) -> pd.Series:
    """
    Very simple ES session tags based on UTC hour.

    ASIA   ~ 00:00–06:59 UTC
    LONDON ~ 07:00–12:59 UTC
    NY     ~ 13:00–20:59 UTC
    OFF    ~ everything else
    """
    hours = time_index.hour

    session = np.full(len(time_index), "OFF", dtype=object)

    session[(hours >= 0) & (hours < 7)] = "ASIA"
    session[(hours >= 7) & (hours < 13)] = "LONDON"
    session[(hours >= 13) & (hours < 21)] = "NY"

    return pd.Series(session, index=time_index, name="session")


def _compute_trend_flag(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> pd.Series:
    """
    Simple trend classification:
      +1 = bullish (fast EMA > slow EMA)
      -1 = bearish (fast EMA < slow EMA)
       0 = flat / undefined
    """
    close = df["close"]
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()

    trend = np.where(ema_f > ema_s, 1, np.where(ema_f < ema_s, -1, 0))
    return pd.Series(trend, index=df.index, name="trend_flag")


def _compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns + rolling realized volatility.
    """
    close = df["close"]

    log_ret = np.log(close / close.shift(1))
    log_ret.name = "ret_log_1"

    # 1-hour realized vol on 5m bars -> 12 bars ≈ 1h
    rv_1h = (
        log_ret.rolling(12)
        .std()
        .rename("rv_1h")
    )

    # 1-day realized vol on 5m bars -> 12 * 24 = 288 bars
    rv_1d = (
        log_ret.rolling(288)
        .std()
        .rename("rv_1d")
    )

    return pd.concat([log_ret, rv_1h, rv_1d], axis=1)


def _compute_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candle body / wick / total range features + simple regime label.
    """
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    close = df["close"]

    total_range = (high - low).rename("range_total")
    body = (close - open_).abs().rename("range_body")
    upper_wick = (high - close.where(close > open_, open_)).rename("range_upper_wick")
    lower_wick = (open_.where(close > open_, close) - low).rename("range_lower_wick")

    # Normalize by ATR later; for now, just raw + percentile ranks
    range_pct_rank = total_range.rank(pct=True).rename("range_total_pct")

    return pd.concat(
        [total_range, body, upper_wick, lower_wick, range_pct_rank],
        axis=1,
    )


def build_es_5m_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point.

    Takes a raw ES 5m DataFrame with columns:
        time, open, high, low, close, volume

    Returns a NEW DataFrame with:
        - All original columns
        - ATR(14)
        - log returns + 1h / 1d realized volatility
        - range features (body/wicks/total)
        - session label (ASIA/LONDON/NY/OFF)
        - trend_flag (+1/-1/0)
    """

    df = df_raw.copy()

    # Ensure 'time' is datetime and set as index for features that need it
    # (works for naive or timezone-aware dtypes)
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], utc=True)

    df = df.sort_values("time").reset_index(drop=True)
    df = df.set_index("time")

    # ATR
    df["atr_14"] = _compute_atr(df, period=14)

    # Returns + realized vol
    ret_block = _compute_returns(df)
    df = pd.concat([df, ret_block], axis=1)

    # Range features
    range_block = _compute_range_features(df)
    df = pd.concat([df, range_block], axis=1)

    # Session labels
    df["session"] = _compute_session(df.index)

    # Trend classification
    df["trend_flag"] = _compute_trend_flag(df)

    # Drop initial NaNs from rolling windows
    df = df.dropna().reset_index()  # bring 'time' back as a column

    return df
