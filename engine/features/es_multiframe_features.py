# engine/features/es_multiframe_features.py

from pathlib import Path

import pandas as pd

from engine.data_loader import ROOT
from engine.features.es_features import build_es_5m_features


def _load_raw(path: Path) -> pd.DataFrame:
    """
    Load a raw ES CSV and ensure:
      - 'time' exists
      - 'time' is converted to UTC datetime
      - rows are sorted by time
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {path}")

    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError(f"'time' column missing in {path}")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _with_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Prefix all non-time columns with e.g. 'h1_' or 'h2_' so we can
    keep OHLCV from multiple timeframes in one DataFrame.
    """
    df = df.copy()
    cols = [c for c in df.columns if c != "time"]
    df = df[["time"] + cols]
    df.columns = ["time"] + [f"{prefix}_{c}" for c in cols]
    return df


def build_es_5m_multiframe_features(
    f_5m: str = "es_5m_clean.csv",
    f_1h: str = "es_1h_clean.csv",
    f_2h: str = "es_2h_clean.csv",
) -> pd.DataFrame:
    """
    Build a 5m ES feature set with aligned 1H and 2H OHLCV columns.

    Output columns include:
        - time, open, high, low, close, volume
        - atr_14, ret_log_1, rv_1h, rv_1d
        - range_total, range_body, range_upper_wick, range_lower_wick, range_total_pct
        - session, trend_flag
        - h1_open, h1_high, h1_low, h1_close, h1_volume
        - h2_open, h2_high, h2_low, h2_close, h2_volume
    """

    # ---------- 1) Load raw CSVs ----------
    path_5m = ROOT / "data" / "raw" / f_5m
    path_1h = ROOT / "data" / "raw" / f_1h
    path_2h = ROOT / "data" / "raw" / f_2h

    df_5m_raw = _load_raw(path_5m)
    df_1h = _load_raw(path_1h)
    df_2h = _load_raw(path_2h)

    # ---------- 2) Build 5m features ----------
    # This uses the same feature pipeline you already ran successfully
    df_5m_feat = build_es_5m_features(df_5m_raw)

    # ---------- 3) Prefix 1H / 2H ----------
    df_1h_p = _with_prefix(df_1h, "h1")
    df_2h_p = _with_prefix(df_2h, "h2")

    # ---------- 4) Merge-asof HTF onto 5m ----------
    df_5m_feat["time"] = pd.to_datetime(df_5m_feat["time"], utc=True)
    df_5m_feat = df_5m_feat.sort_values("time").reset_index(drop=True)

    df = pd.merge_asof(
        df_5m_feat,
        df_1h_p.sort_values("time"),
        on="time",
        direction="backward",
    )
    df = pd.merge_asof(
        df.sort_values("time"),
        df_2h_p.sort_values("time"),
        on="time",
        direction="backward",
    )

    return df
