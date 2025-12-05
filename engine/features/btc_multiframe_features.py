# engine/features/btc_multiframe_features.py
#
# Clean institutional-style BTC 5m feature builder.
#
# Input:
#   prices: DataFrame with columns [open, high, low, close, volume]
#           indexed by datetime (ideally UTC, 5m bars)
#
# Output:
#   features: DataFrame with:
#       - open, high, low, close, volume
#       - atr               (14-period ATR)
#       - atr_pct_5m        (atr / close)
#       - rvol_5m           (volume / 20-bar rolling volume mean)
#       - session_type      ("ASIA", "LONDON", "NY")
#       - week_pos          (0–1, position within weekly low→high range)
#       - vwap              (running VWAP)
#       - vwap_dist_atr     ((close - vwap) / atr)
#       - regime_trend_up   (True if fast EMA > slow EMA)
#       - has_sweep         (liquidity sweep flag)
#       - sweep_bull        (bullish sweep)
#       - sweep_bear        (bearish sweep)

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from engine.modules.sweeps import compute_sweep_flags

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    f = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    h.setFormatter(f)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


# ------------------------------------------------------------
# Helper: ensure index & columns are sane
# ------------------------------------------------------------
def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)

    # Enforce UTC (drop tz info if present)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_convert("UTC")
    else:
        prices.index = prices.index.tz_localize("UTC")

    prices = prices.sort_index()

    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in prices.columns:
            raise ValueError(f"Missing required column in prices: '{col}'")

    # Volume is optional; if missing, create zeros
    if "volume" not in prices.columns:
        prices["volume"] = 0.0

    return prices


# ------------------------------------------------------------
# ATR
# ------------------------------------------------------------
def _compute_atr(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    high = prices["high"]
    low = prices["low"]
    close = prices["close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()

    return atr


# ------------------------------------------------------------
# RVOL (relative volume)
# ------------------------------------------------------------
def _compute_rvol(volume: pd.Series, window: int = 20) -> pd.Series:
    vol_mean = volume.rolling(window=window, min_periods=5).mean()
    rvol = volume / vol_mean
    rvol = rvol.replace([np.inf, -np.inf], np.nan)
    rvol = rvol.fillna(1.0)
    return rvol


# ------------------------------------------------------------
# Session classification
# ------------------------------------------------------------
def _classify_session(index: pd.DatetimeIndex) -> pd.Series:
    """
    Simple UTC-based session classification:
      - ASIA  : 00:00–07:59
      - LONDON: 08:00–15:59
      - NY    : 16:00–23:59
    """
    hours = index.hour
    session = pd.Series(index=index, dtype="object")

    session[(hours >= 0) & (hours < 8)] = "ASIA"
    session[(hours >= 8) & (hours < 16)] = "LONDON"
    session[(hours >= 16) & (hours < 24)] = "NY"

    session = session.fillna("NY")
    return session


# ------------------------------------------------------------
# Weekly position (0–1 within weekly low→high)
# ------------------------------------------------------------
def _compute_week_position(prices: pd.DataFrame) -> pd.Series:
    """
    For each bar, compute its position within the current week's
    low→high range, where week starts Monday.
    """
    close = prices["close"]

    # Weekly index (Monday-based)
    week_index = prices.index.to_period("W-MON")

    weekly_low = close.groupby(week_index).transform("min")
    weekly_high = close.groupby(week_index).transform("max")

    rng = weekly_high - weekly_low
    rng = rng.replace(0, np.nan)

    week_pos = (close - weekly_low) / rng
    week_pos = week_pos.clip(0.0, 1.0)
    week_pos = week_pos.fillna(0.5)  # mid-week if degenerate

    return week_pos


# ------------------------------------------------------------
# VWAP + VWAP distance in ATR units
# ------------------------------------------------------------
def _compute_vwap(prices: pd.DataFrame) -> pd.Series:
    close = prices["close"]
    volume = prices["volume"]

    # Avoid 0 volume degeneracy: treat 0-volume as 1 for vwap math
    vol_safe = volume.replace(0, 1.0)

    cum_pv = (close * vol_safe).cumsum()
    cum_vol = vol_safe.cumsum()

    vwap = cum_pv / cum_vol
    return vwap


def _compute_vwap_dist_atr(close: pd.Series, vwap: pd.Series, atr: pd.Series) -> pd.Series:
    dist = (close - vwap)
    dist_atr = dist / atr.replace(0, np.nan)
    dist_atr = dist_atr.replace([np.inf, -np.inf], np.nan)
    dist_atr = dist_atr.fillna(0.0)
    return dist_atr


# ------------------------------------------------------------
# Regime: simple EMA-based "trend up"
# ------------------------------------------------------------
def _compute_regime_trend_up(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    regime_up = ema_fast > ema_slow
    return regime_up


# ------------------------------------------------------------
# MAIN ENTRY: build_btc_multiframe_features
# ------------------------------------------------------------
def build_btc_multiframe_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature builder for BTC 5m institutional frame.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns: at least [open, high, low, close] and optionally [volume]
        Index: DatetimeIndex (5m bars, ideally UTC)

    Returns
    -------
    pd.DataFrame
        Feature frame aligned with input index.
    """
    logger.info("Building BTC multiframe features from %s rows...", len(prices))

    df = _normalize_prices(prices.copy())

    # --- Core features ---
    df["atr"] = _compute_atr(df, period=14)
    df["atr_5m"] = df["atr"]
    df["atr_pct_5m"] = df["atr"] / df["close"]

    df["rvol_5m"] = _compute_rvol(df["volume"], window=20)
    df["session_type"] = _classify_session(df.index)
    df["week_pos"] = _compute_week_position(df)

    df["vwap"] = _compute_vwap(df)
    df["vwap_dist_atr"] = _compute_vwap_dist_atr(df["close"], df["vwap"], df["atr"])

    df["regime_trend_up"] = _compute_regime_trend_up(df["close"], fast=20, slow=50)

    # --- Liquidity sweep features ---
    sweep_df = compute_sweep_flags(df, lookback=5)
    df["sweep_bull"] = sweep_df["sweep_bull"]
    df["sweep_bear"] = sweep_df["sweep_bear"]
    df["has_sweep"] = sweep_df["has_sweep"]

    # Final forward-fill in case of early NaNs from rolling windows
    df = df.ffill()

    logger.info("BTC feature frame built. Final rows: %s", len(df))
    return df


# ------------------------------------------------------------
# Backwards-compatible alias for older code (no-arg version)
# ------------------------------------------------------------
def build_btc_5m_multiframe_features_institutional(
    prices: pd.DataFrame | None = None,
):
    """
    Backwards-compatible alias so older code that calls
        build_btc_5m_multiframe_features_institutional()
    with NO arguments still works.

    Behavior:
    - If `prices` is provided: use it directly, return (features_df, n_rows).
    - If `prices` is None: try to load raw BTC OHLC from a few common paths
      under the project `data/` folder, then build features.

    Returns
    -------
    (features_df, n_rows)
    """
    if prices is None:
        # Try several likely BTC 5m CSV names under ./data
        candidate_paths = [
            "data/btc_5m.csv",
            "data/btc_5m_raw.csv",
            "data/btc_usdt_5m.csv",
            "data/btc_5m_prices.csv",
        ]

        found_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                found_path = path
                break

        if found_path is None:
            raise RuntimeError(
                "Could not find any BTC 5m CSV file. "
                "Looked for: "
                + ", ".join(candidate_paths)
                + ". Please either place your raw BTC OHLC CSV in one of "
                  "these locations, or pass a DataFrame directly to "
                  "build_btc_multiframe_features(prices)."
            )

        try:
            raw = pd.read_csv(
                found_path,
                parse_dates=True,
                index_col=0,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load raw BTC data from {found_path}. "
                f"Make sure the CSV has a datetime-like index column."
            ) from exc

        prices = raw

    features_df = build_btc_multiframe_features(prices)
    n_rows = len(features_df)
    return features_df, n_rows
