# engine/modules/liquidity.py

from __future__ import annotations

from typing import List, Tuple


def detect_sweep(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    lookback: int = 10,
    wick_frac: float = 0.5,
) -> Tuple[bool, bool]:
    """
    Detect simple liquidity sweeps above / below recent highs/lows.

    A sweep UP is defined as:
      - current high > max(highs[-lookback-1:-1])
      - but current close is back inside the prior range
        (i.e., close is not at the extreme)
      - upper wick is at least `wick_frac` of the total range

    A sweep DOWN is the symmetric logic.

    Returns
    -------
    (sweep_up, sweep_down)
    """

    n = len(highs)
    if n < lookback + 2:
        return False, False

    h_cur = highs[-1]
    l_cur = lows[-1]
    c_cur = closes[-1]

    h_prev = highs[-(lookback + 1) : -1]
    l_prev = lows[-(lookback + 1) : -1]

    prior_high = max(h_prev)
    prior_low = min(l_prev)

    rng = h_cur - l_cur
    if rng <= 0:
        return False, False

    upper_wick = h_cur - max(c_cur, l_cur)
    lower_wick = min(c_cur, h_cur) - l_cur

    sweep_up = (
        h_cur > prior_high  # took out liquidity above
        and c_cur < h_cur   # closed off the high
        and upper_wick >= rng * wick_frac
    )

    sweep_down = (
        l_cur < prior_low   # took out liquidity below
        and c_cur > l_cur   # closed off the low
        and lower_wick >= rng * wick_frac
    )

    return sweep_up, sweep_down
