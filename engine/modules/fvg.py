# engine/modules/fvg.py
#
# Fair Value Gap (FVG) feature scaffolding + backwards-compatible
# `detect_fvg` function for older strategies.
#
# Current behavior:
#   - All FVG flags are False (no FVG logic applied yet).
#   - This keeps the engine stable while allowing any code that
#     imports `detect_fvg` or `compute_fvg_features` to run.
#
# Later, we can upgrade `detect_fvg` to implement real 3-candle
# FVG detection and reuse it inside `compute_fvg_features`.

from __future__ import annotations

import pandas as pd


def detect_fvg(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    Backwards-compatible FVG detector stub.

    Parameters
    ----------
    df : pd.DataFrame
        Expected to contain ["high", "low", "close"].
    lookback : int
        Reserved for future use.

    Returns
    -------
    pd.DataFrame
        Index aligned with df, boolean columns:
            - bull_fvg_origin
            - bear_fvg_origin
            - in_bull_fvg
            - in_bear_fvg

        For now, all values are False (no FVGs detected).
    """
    out = pd.DataFrame(index=df.index)
    out["bull_fvg_origin"] = False
    out["bear_fvg_origin"] = False
    out["in_bull_fvg"] = False
    out["in_bear_fvg"] = False
    return out


def compute_fvg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    New-style FVG feature generator used by the BTC feature builder.

    For now, this simply delegates to `detect_fvg` so that both
    old and new code paths share the same output schema.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Same output as `detect_fvg`.
    """
    return detect_fvg(df)
