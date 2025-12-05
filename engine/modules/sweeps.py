# engine/modules/sweeps.py
#
# Simple liquidity sweep detection for BTC (or any OHLC series).
#
# Definition (per bar):
#   - Bearish sweep:
#       * current HIGH > highest HIGH of the previous N bars
#       * AND current CLOSE < that previous high region
#   - Bullish sweep:
#       * current LOW < lowest LOW of the previous N bars
#       * AND current CLOSE > that previous low region
#
# Output columns:
#   sweep_bear : True if this bar is a bearish sweep
#   sweep_bull : True if this bar is a bullish sweep
#   has_sweep  : True if either bull or bear sweep
#
# This is intentionally simple and fast. We can refine it later.

from __future__ import annotations

import pandas as pd


def compute_sweep_flags(
    df: pd.DataFrame,
    lookback: int = 5,
) -> pd.DataFrame:
    """
    Compute simple liquidity sweep flags over the last `lookback` bars.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ["high", "low", "close"] and a DatetimeIndex.
    lookback : int
        Number of previous bars to consider for prior highs/lows.

    Returns
    -------
    pd.DataFrame
        Index aligned with `df`, with boolean columns:
            ["has_sweep", "sweep_bull", "sweep_bear"]
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("compute_sweep_flags expects a DatetimeIndex index.")

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Highest/lowest of the *previous* N bars (shift(1) so we don't include current bar)
    prev_high = high.shift(1).rolling(window=lookback, min_periods=1).max()
    prev_low = low.shift(1).rolling(window=lookback, min_periods=1).min()

    # Bearish sweep: take out previous highs, close back below that prior high area
    sweep_bear = (high > prev_high) & (close < prev_high)

    # Bullish sweep: take out previous lows, close back above that prior low area
    sweep_bull = (low < prev_low) & (close > prev_low)

    has_sweep = sweep_bear | sweep_bull

    out = pd.DataFrame(index=df.index)
    out["sweep_bear"] = sweep_bear.fillna(False)
    out["sweep_bull"] = sweep_bull.fillna(False)
    out["has_sweep"] = has_sweep.fillna(False)

    return out
