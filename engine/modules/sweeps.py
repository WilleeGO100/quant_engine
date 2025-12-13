# engine/modules/sweeps.py
#
# Hybrid (ICT + crypto wick) liquidity sweep detector.
#
# Idea:
#   - Look at prior N bars to define liquidity pools (prev highs/lows).
#   - A "sweep" happens when:
#       * price pokes beyond those pools (stop run)
#       * the wick is large relative to the body/range
#       * the close returns back inside the prior liquidity zone
#
# Output columns:
#   sweep_bull      : True if this bar swept downside liquidity (bullish)
#   sweep_bear      : True if this bar swept upside liquidity (bearish)
#   has_sweep       : True if either bull or bear
#   sweep_strength  : 0 (none) to 3 (strong)
#
# This is backward-compatible with existing code that expects
# `compute_sweep_flags(df, lookback=5)`.

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sweep_flags(
    df: pd.DataFrame,
    lookback: int = 5,
    wick_ratio: float = 1.2,
    min_wick_frac: float = 0.35,
) -> pd.DataFrame:
    """
    Hybrid liquidity sweep detector.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ["open", "high", "low", "close"].
    lookback : int
        Number of previous bars to define liquidity highs/lows.
    wick_ratio : float
        Minimum wick/body ratio for a strong sweep.
    min_wick_frac : float
        Minimum wick/total-range fraction for a sweep.

    Returns
    -------
    pd.DataFrame
        Index aligned with df, with:
            sweep_bull, sweep_bear, has_sweep, sweep_strength
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("compute_sweep_flags expects a DatetimeIndex index.")

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"compute_sweep_flags: missing '{col}' column in input.")

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    # --------------------------------------------------------
    # 1) Liquidity pools: prior highs/lows (no current bar)
    # --------------------------------------------------------
    prev_high = h.shift(1).rolling(window=lookback, min_periods=2).max()
    prev_low = l.shift(1).rolling(window=lookback, min_periods=2).min()

    # --------------------------------------------------------
    # 2) Candle geometry: body, range, wicks
    # --------------------------------------------------------
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)

    # upper wick: high - max(open, close)
    upper_anchor = pd.concat([o, c], axis=1).max(axis=1)
    up_wick = (h - upper_anchor).clip(lower=0.0)

    # lower wick: min(open, close) - low
    lower_anchor = pd.concat([o, c], axis=1).min(axis=1)
    down_wick = (lower_anchor - l).clip(lower=0.0)

    up_wick_frac = (up_wick / rng).replace([np.inf, -np.inf], np.nan)
    down_wick_frac = (down_wick / rng).replace([np.inf, -np.inf], np.nan)

    # --------------------------------------------------------
    # 3) Basic sweep conditions (stop-run beyond prior pools)
    # --------------------------------------------------------
    bull_basic = (l < prev_low)  # ran stops below prior lows
    bear_basic = (h > prev_high)  # ran stops above prior highs

    # --------------------------------------------------------
    # 4) Wick dominance (crypto-style long wick sweeps)
    # --------------------------------------------------------
    bull_wick_ok = (
        (down_wick > wick_ratio * body) |
        (down_wick_frac >= min_wick_frac)
    )

    bear_wick_ok = (
        (up_wick > wick_ratio * body) |
        (up_wick_frac >= min_wick_frac)
    )

    # --------------------------------------------------------
    # 5) Close back into prior liquidity zone (ICT-style)
    # --------------------------------------------------------
    # Bullish sweep: close back ABOVE prior low region
    bull_close_ok = c > prev_low

    # Bearish sweep: close back BELOW prior high region
    bear_close_ok = c < prev_high

    sweep_bull = bull_basic & bull_wick_ok & bull_close_ok
    sweep_bear = bear_basic & bear_wick_ok & bear_close_ok
    has_sweep = sweep_bull | sweep_bear

    # --------------------------------------------------------
    # 6) Strength scoring (0–3)
    # --------------------------------------------------------
    strength = np.zeros(len(df), dtype=int)

    # Base level: any sweep
    strength[has_sweep.fillna(False).values] = 1

    # Level 2: sweep + strong wick dominance
    strong_bull = bull_basic & bull_wick_ok & bull_close_ok & (down_wick_frac >= (min_wick_frac + 0.1))
    strong_bear = bear_basic & bear_wick_ok & bear_close_ok & (up_wick_frac >= (min_wick_frac + 0.1))
    strength[(strong_bull | strong_bear).fillna(False).values] = 2

    # Level 3: strong wick + small body (classic stop-run candle)
    small_body = (body / rng) < 0.3
    strongest = (strong_bull | strong_bear) & small_body
    strength[strongest.fillna(False).values] = 3

    out = pd.DataFrame(index=df.index)
    out["sweep_bull"] = sweep_bull.fillna(False)
    out["sweep_bear"] = sweep_bear.fillna(False)
    out["has_sweep"] = has_sweep.fillna(False)
    out["sweep_strength"] = strength

    return out
