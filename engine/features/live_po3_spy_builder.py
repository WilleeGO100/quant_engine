# engine/features/live_po3_spy_builder.py
#
# Build a single "live" POWER SMC PO3 feature bar by:
#   1) Loading ES multiframe PO3 features (NOW WITH institutional features)
#   2) Taking the latest ES feature row
#   3) Overlaying the latest SPY 5m bar from TwelveData as extra columns
#
# This is called from main_live.py via build_live_po3_bar(cfg).
# The SMCPo3PowerStrategy still runs off the ES-based features, but the row
# now also contains:
#   - atr_14, rvol_20, ret_z_50
#   - regime_vol, regime_trend
#   - session
#   - vwap_day, vwap_dist_atr
#   - rolling_imbalance_20, cum_delta
# PLUS the live SPY fields (spy_timestamp, spy_open, ..., spy_volume).

from __future__ import annotations

from typing import Tuple, Optional

import pandas as pd

from engine.features.institutional_features import (
    build_es_5m_multiframe_features_institutional,
)
from engine.data.twelvedata_client import TwelveDataClient


def _select_latest_bar(df: pd.DataFrame) -> pd.Series:
    """Return the most recent row in the feature DataFrame."""
    if df.empty:
        raise ValueError("Features DataFrame is empty - cannot build live bar.")

    time_col = None
    for candidate in ("timestamp", "time", "datetime"):
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is not None:
        df_sorted = df.sort_values(time_col)
    else:
        df_sorted = df

    return df_sorted.iloc[-1]


def build_live_po3_bar(cfg) -> Tuple[pd.Series, int]:
    """
    Build a single enriched PO3 feature bar with live SPY overlay.

    Steps:
      1) Call build_es_5m_multiframe_features_institutional() to get ES 5m
         multi-timeframe PO3 features + institutional features.
      2) Take the latest ES feature row.
      3) Fetch the latest SPY 5m bar from TwelveData and attach as:
           spy_symbol, spy_timestamp, spy_open, spy_high,
           spy_low, spy_close, spy_volume

    Returns:
        bar   : pd.Series for the latest ES bar with all features + SPY overlay
        n_rows: number of rows in the ES features DataFrame
    """
    # 1) ES multi-timeframe + institutional features
    df_features, n_rows = build_es_5m_multiframe_features_institutional()
    if df_features is None or df_features.empty:
        raise RuntimeError("build_es_5m_multiframe_features_institutional() returned no data.")

    # 2) Latest ES feature row
    es_bar = _select_latest_bar(df_features)
    bar = es_bar.copy()

    # 3) Overlay latest SPY 5m bar from TwelveData
    api_key = getattr(cfg, "twelvedata_api_key", "")
    feed_symbol = getattr(cfg, "twelvedata_symbol", "SPY")

    if not api_key:
        print("[LIVE_PO3_SPY] WARNING: twelvedata_api_key is empty -> no SPY overlay.")
        return bar, n_rows

    client = TwelveDataClient(api_key=api_key)
    spy_bar = client.fetch_last_bar(symbol=feed_symbol)

    if spy_bar is None:
        print("[LIVE_PO3_SPY] WARNING: Could not fetch live SPY bar -> using ES-only features.")
        return bar, n_rows

    # Attach SPY info as extra columns
    bar["spy_symbol"] = feed_symbol
    bar["spy_timestamp"] = spy_bar.get("timestamp")
    bar["spy_open"] = spy_bar.get("open")
    bar["spy_high"] = spy_bar.get("high")
    bar["spy_low"] = spy_bar.get("low")
    bar["spy_close"] = spy_bar.get("close")
    bar["spy_volume"] = spy_bar.get("volume")

    print("[LIVE_PO3_SPY] Attached live SPY bar to ES PO3 institutional features:")
    print(
        f"  spy_timestamp={bar['spy_timestamp']}, "
        f"spy_open={bar['spy_open']}, "
        f"spy_high={bar['spy_high']}, "
        f"spy_low={bar['spy_low']}, "
        f"spy_close={bar['spy_close']}, "
        f"spy_volume={bar['spy_volume']}"
    )

    return bar, n_rows
