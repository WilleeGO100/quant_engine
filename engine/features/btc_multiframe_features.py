# engine/features/btc_multiframe_features.py
#
# Clean BTC 5m institutional feature builder, plus
# backwards-compatible alias:
#   build_btc_5m_multiframe_features_institutional()

from __future__ import annotations

import logging
import os
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

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
    """
    Ensure we have:
      - DatetimeIndex (UTC)
      - columns: open, high, low, close, volume (volume default 0.0 if missing)
    """
    prices = prices.copy()

    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)

    # Force UTC (drop or convert tz if present)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_convert("UTC")
    else:
        prices.index = prices.index.tz_localize("UTC")

    prices = prices.sort_index()

    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in prices.columns:
            raise ValueError(f"Missing required column in prices: '{col}'")

    if "volume" not in prices.columns:
        prices["volume"] = 0.0

    return prices


# ------------------------------------------------------------
# ATR
# ------------------------------------------------------------

def _compute_atr(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    high = prices["high"].astype(float)
    low = prices["low"].astype(float)
    close = prices["close"].astype(float)

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
    rvol = volume / vol_mean.replace(0.0, np.nan)
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
    close = prices["close"].astype(float)
    week_index = prices.index.to_period("W-MON")

    weekly_low = close.groupby(week_index).transform("min")
    weekly_high = close.groupby(week_index).transform("max")

    rng = weekly_high - weekly_low
    rng = rng.replace(0, np.nan)

    week_pos = (close - weekly_low) / rng
    week_pos = week_pos.clip(0, 1)
    week_pos = week_pos.fillna(0.5)

    return week_pos


# ------------------------------------------------------------
# VWAP + VWAP distance in ATR units
# ------------------------------------------------------------

def _compute_vwap(prices: pd.DataFrame) -> pd.Series:
    close = prices["close"].astype(float)
    volume = prices["volume"].astype(float)

    vol_safe = volume.replace(0.0, np.nan)

    cum_pv = (close * vol_safe).cumsum()
    cum_vol = vol_safe.cumsum()

    vwap = cum_pv / cum_vol.replace(0.0, np.nan)
    vwap = vwap.fillna(method="ffill").fillna(close)
    return vwap


def _compute_vwap_dist_atr(
    close: pd.Series, vwap: pd.Series, atr: pd.Series
) -> pd.Series:
    dist = close - vwap
    dist_atr = dist / atr.replace(0.0, np.nan)
    dist_atr = dist_atr.replace([np.inf, -np.inf], np.nan)
    dist_atr = dist_atr.fillna(0.0)
    return dist_atr


# ------------------------------------------------------------
# Regime: simple EMA-based "trend up"
# ------------------------------------------------------------

def _compute_regime_trend_up(
    close: pd.Series, fast: int = 20, slow: int = 50
) -> pd.Series:
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

    prices:
        columns: open, high, low, close, [volume]
        index  : DatetimeIndex (5m, UTC)
    """
    logger.info("Building BTC multiframe features from %s rows...", len(prices))

    df = _normalize_prices(prices)

    # Core features
    df["atr"] = _compute_atr(df, period=14)
    df["atr_pct_5m"] = df["atr"] / df["close"].replace(0.0, np.nan)

    df["rvol_5m"] = _compute_rvol(df["volume"], window=20)
    df["session_type"] = _classify_session(df.index)
    df["week_pos"] = _compute_week_position(df)

    df["vwap"] = _compute_vwap(df)
    df["vwap_dist_atr"] = _compute_vwap_dist_atr(df["close"], df["vwap"], df["atr"])

    df["regime_trend_up"] = _compute_regime_trend_up(
        df["close"], fast=20, slow=50
    )

    # Placeholder for sweep – but we don't require it anywhere
    df["has_sweep"] = False

    df = df.ffill()

    logger.info("BTC feature frame built. Final rows: %s", len(df))
    return df


# ------------------------------------------------------------
# CSV AUTO-LOADER + BACKWARDS-COMPATIBLE ALIAS
# ------------------------------------------------------------

def _resolve_btc_csv_paths() -> List[str]:
    """
    Candidate paths (in order) for BTC 5m OHLCV CSV.

    We support your current layout:

        data/btc_5m.csv
        data/btc_5m_live.csv

    plus some legacy fallbacks.
    """
    candidates: List[str] = [
        # Primary project-layout paths
        os.path.join("data", "btc_5m.csv"),
        os.path.join("data", "btc_5m_live.csv"),
        # Root-level fallbacks (if someone runs from data/ root)
        "btc_5m_live.csv",
        "btc_5m.csv",
        # Older names you had in earlier versions
        os.path.join("data", "btc_5m_raw.csv"),
        os.path.join("data", "btc_usdt_5m.csv"),
        os.path.join("data", "btc_5m_prices.csv"),
    ]
    return candidates


def build_btc_5m_multiframe_features_institutional(
    prices: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Backwards-compatible wrapper:

    - If `prices` is provided: build features on that DataFrame.
    - If `prices` is None: auto-load a BTC 5m CSV from disk using
      a set of sensible candidate paths.
    """
    if prices is None:
        candidate_paths = _resolve_btc_csv_paths()

        found_path: Optional[str] = None
        for path in candidate_paths:
            if os.path.exists(path):
                found_path = path
                break

        if found_path is None:
            raise RuntimeError(
                "Could not find any BTC 5m CSV file. Looked for: "
                + ", ".join(candidate_paths)
            )

        logger.info("Loading raw BTC data from: %s", found_path)

        try:
            raw = pd.read_csv(
                found_path,
                parse_dates=True,
                index_col=0,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load raw BTC data from {found_path}. "
                "Make sure the first column is a datetime index."
            ) from exc

        prices = raw

    features_df = build_btc_multiframe_features(prices)
    n_rows = len(features_df)
    return features_df, n_rows
