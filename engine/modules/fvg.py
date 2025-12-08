# engine/modules/fvg.py
#
# Fair Value Gap (FVG) detection + feature generator.
#
# We implement a classic 3-candle FVG model:
#
#   Bullish FVG origin at bar i:
#       low[i] > high[i-2]
#       -> gap zone: (high[i-2], low[i])
#
#   Bearish FVG origin at bar i:
#       high[i] < low[i-2]
#       -> gap zone: (high[i], low[i-2])
#
# We then track whether later bars are trading "inside" any active FVG.
#
# Output columns:
#   bull_fvg_origin : bool       # this bar *creates* a bullish FVG
#   bear_fvg_origin : bool       # this bar *creates* a bearish FVG
#   in_bull_fvg     : bool       # this bar is trading inside an unfilled bull FVG
#   in_bear_fvg     : bool       # this bar is trading inside an unfilled bear FVG
#
# This is backwards-compatible with older code that imports `detect_fvg`
# and newer code that calls `compute_fvg_features`.

from __future__ import annotations

import pandas as pd


def detect_fvg(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    Detect bullish and bearish 3-candle Fair Value Gaps.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ["high", "low", "close"].
    lookback : int
        Currently unused but kept for backwards compatibility.

    Returns
    -------
    pd.DataFrame
        Columns:
            bull_fvg_origin
            bear_fvg_origin
            in_bull_fvg
            in_bear_fvg
    """
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"detect_fvg: missing '{col}' column in input DataFrame.")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    index = df.index

    bull_fvg_origin = pd.Series(False, index=index)
    bear_fvg_origin = pd.Series(False, index=index)
    in_bull_fvg = pd.Series(False, index=index)
    in_bear_fvg = pd.Series(False, index=index)

    # Active FVG zones we track over time
    bull_zones = []  # each: {"top": float, "bottom": float}
    bear_zones = []  # each: {"top": float, "bottom": float}

    # We start from i=2 because we look back 2 bars (i-2)
    for i in range(2, len(df)):
        h_i = float(high.iloc[i])
        l_i = float(low.iloc[i])
        c_i = float(close.iloc[i])

        # --- 1) Check for new FVG origins at bar i ---

        # Bullish FVG: low[i] > high[i-2]
        h_prev2 = float(high.iloc[i - 2])
        l_prev2 = float(low.iloc[i - 2])

        if l_i > h_prev2:
            # Gap from high[i-2] up to low[i]
            bull_fvg_origin.iloc[i] = True
            bull_zones.append({"top": h_prev2, "bottom": l_i})

        # Bearish FVG: high[i] < low[i-2]
        if h_i < l_prev2:
            # Gap from high[i] up to low[i-2]
            bear_fvg_origin.iloc[i] = True
            bear_zones.append({"top": h_i, "bottom": l_prev2})

        # --- 2) Check if current bar is *inside* any existing FVGs ---
        # We require that the close is inside the zone bounds.

        # Bullish zones
        still_bull_zones = []
        for zone in bull_zones:
            top = zone["top"]
            bottom = zone["bottom"]

            # If price has completely traded through the zone, consider it filled
            # (rough heuristic: bar range covers the zone entirely).
            if (l_i <= top) and (h_i >= bottom):
                # Zone filled -> do not keep it
                continue

            # Otherwise, keep it active
            still_bull_zones.append(zone)

            # Check if close is between zone top and bottom
            if (c_i >= top) and (c_i <= bottom):
                in_bull_fvg.iloc[i] = True

        bull_zones = still_bull_zones

        # Bearish zones
        still_bear_zones = []
        for zone in bear_zones:
            top = zone["top"]
            bottom = zone["bottom"]

            if (l_i <= top) and (h_i >= bottom):
                # Zone filled
                continue

            still_bear_zones.append(zone)

            if (c_i >= top) and (c_i <= bottom):
                in_bear_fvg.iloc[i] = True

        bear_zones = still_bear_zones

    out = pd.DataFrame(index=index)
    out["bull_fvg_origin"] = bull_fvg_origin
    out["bear_fvg_origin"] = bear_fvg_origin
    out["in_bull_fvg"] = in_bull_fvg
    out["in_bear_fvg"] = in_bear_fvg

    return out


def compute_fvg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    New-style FVG feature generator used by the BTC feature builder.

    For now, this simply calls `detect_fvg` so both old and new code paths
    share the same output schema.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns:
            bull_fvg_origin
            bear_fvg_origin
            in_bull_fvg
            in_bear_fvg
    """
    return detect_fvg(df)

