# engine/modules/fvg.py

from __future__ import annotations

from typing import Tuple, List


def detect_fvg(
    highs: List[float],
    lows: List[float],
    lookback: int = 3,
    min_size: float = 2.0,
) -> Tuple[bool, bool]:
    """
    Detect bullish / bearish Fair Value Gaps (FVG) using a simple 3-candle model.

    We look at the last 3 candles: A, B, C
      - Bullish FVG:  low(C) > high(A)  AND  (low(C) - high(A)) >= min_size
      - Bearish FVG:  high(C) < low(A) AND  (low(A) - high(C)) >= min_size

    We also require a small "displacement" by making sure candle B range
    is not tiny compared to the gap.

    Parameters
    ----------
    highs, lows : lists of floats
        High/low series up to current bar.
    lookback : int
        Not used heavily here, but kept for future multi-gap logic.
    min_size : float
        Minimum size of gap, in price units (ES points).

    Returns
    -------
    (bull_fvg, bear_fvg) : tuple of bool
    """

    n = len(highs)
    if n < 3:
        return False, False

    # last 3 candles
    h_a, h_b, h_c = highs[-3], highs[-2], highs[-1]
    l_a, l_b, l_c = lows[-3], lows[-2], lows[-1]

    bull_fvg = False
    bear_fvg = False

    # Bullish: C's low is above A's high -> upside imbalance
    gap_up = l_c - h_a
    if gap_up >= min_size:
        # require some displacement: B range not tiny
        disp = abs(h_b - l_b)
        if disp >= min_size * 0.5:
            bull_fvg = True

    # Bearish: C's high is below A's low -> downside imbalance
    gap_down = l_a - h_c
    if gap_down >= min_size:
        disp = abs(h_b - l_b)
        if disp >= min_size * 0.5:
            bear_fvg = True

    return bull_fvg, bear_fvg
