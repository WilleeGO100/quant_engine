# engine/modules/structure.py

from __future__ import annotations

from typing import List, Tuple


def _is_pivot_high(highs: List[float], idx: int, left: int, right: int) -> bool:
    h = highs[idx]
    for i in range(idx - left, idx + right + 1):
        if i == idx:
            continue
        if i < 0 or i >= len(highs):
            return False
        if highs[i] >= h:
            return False
    return True


def _is_pivot_low(lows: List[float], idx: int, left: int, right: int) -> bool:
    l = lows[idx]
    for i in range(idx - left, idx + right + 1):
        if i == idx:
            continue
        if i < 0 or i >= len(lows):
            return False
        if lows[i] <= l:
            return False
    return True


def detect_structure(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    pivot_len: int = 5,
) -> Tuple[bool, bool, bool, bool, int]:
    """
    Very simple structural model:

    - Identify recent swing high / swing low using pivot_len.
    - If price breaks above last swing high -> BOS_UP
    - If price breaks below last swing low -> BOS_DOWN
    - If trend was down and we get BOS_UP -> CHOCH_UP
    - If trend was up and we get BOS_DOWN -> CHOCH_DOWN

    Returns:
      (bos_up, bos_down, choch_up, choch_down, trend_dir)

    where trend_dir is:
      +1 = bullish structure
      -1 = bearish structure
       0 = undefined / flat
    """

    n = len(highs)
    if n < pivot_len * 2 + 2:
        return False, False, False, False, 0

    left = right = pivot_len

    swing_high_idx = None
    swing_low_idx = None

    # find the latest swing high / low before the last bar
    for i in range(n - right - 2, pivot_len, -1):
        if swing_high_idx is None and _is_pivot_high(highs, i, left, right):
            swing_high_idx = i
        if swing_low_idx is None and _is_pivot_low(lows, i, left, right):
            swing_low_idx = i
        if swing_high_idx is not None and swing_low_idx is not None:
            break

    if swing_high_idx is None or swing_low_idx is None:
        return False, False, False, False, 0

    swing_high = highs[swing_high_idx]
    swing_low = lows[swing_low_idx]

    last_close = closes[-1]

    bos_up = last_close > swing_high
    bos_down = last_close < swing_low

    # crude trend: which swing is more recent
    trend_dir = 0
    if swing_high_idx > swing_low_idx:
        # last important event was a swing high -> likely downtrend
        trend_dir = -1
    elif swing_low_idx > swing_high_idx:
        trend_dir = 1

    choch_up = bos_up and trend_dir == -1
    choch_down = bos_down and trend_dir == 1

    return bos_up, bos_down, choch_up, choch_down, trend_dir
