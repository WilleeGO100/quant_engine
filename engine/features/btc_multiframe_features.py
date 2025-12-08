# engine/features/btc_multiframe_features.py
#
# BTC 5m multi-timeframe "institutional" feature builder.
#
# Input:
#   data/5m_btc.csv with columns:
#       timestamp, open, high, low, close, volume
#
# Output:
<<<<<<< HEAD
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
=======
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
>>>>>>> origin/phase5-features

import numpy as np
import pandas as pd

<<<<<<< HEAD
# ------------------------------------------------------
# Resolve project root and CSV path
# ------------------------------------------------------

# This file lives at: quant_engine/engine/features/btc_multiframe_features.py
# Project root is two levels up.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BTC_5M_CSV_PATH = os.path.join(BASE_DIR, "data", "5m_btc.csv")
=======
from engine.modules.sweeps import compute_sweep_flags
from engine.modules.fvg import compute_fvg_features

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    f = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    h.setFormatter(f)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
>>>>>>> origin/phase5-features


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
<<<<<<< HEAD
    Resample 5m OHLC to higher timeframe and align back to 5m via merge_asof.
    rule: "1h", "4h", "1d" etc.
    suffix: "_1h", "_4h", "_1d"
=======
    Simple UTC-based session classification:
      - ASIA  : 00:00–07:59
      - LONDON: 08:00–15:59
      - NY    : 16:00–23:59
>>>>>>> origin/phase5-features
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

<<<<<<< HEAD
    return merged
=======
    session = session.fillna("NY")
    return session
>>>>>>> origin/phase5-features


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

<<<<<<< HEAD
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
=======
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
    df["sweep_strength"] = sweep_df["sweep_strength"]
    # --- FVG features (placeholder for now) ---

    fvg_df = compute_fvg_features(df)
    df["bull_fvg_origin"] = fvg_df["bull_fvg_origin"]
    df["bear_fvg_origin"] = fvg_df["bear_fvg_origin"]
    df["in_bull_fvg"] = fvg_df["in_bull_fvg"]
    df["in_bear_fvg"] = fvg_df["in_bear_fvg"]

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
>>>>>>> origin/phase5-features
