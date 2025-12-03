# engine/data_loader.py

from pathlib import Path
import pandas as pd

# ROOT points to quant_engine/ project root
ROOT = Path(__file__).resolve().parent.parent


def _read_csv(path: Path) -> pd.DataFrame:
    """Internal helper to read a CSV with a 'time' column."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {path}")

    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError(f"CSV at {path} is missing 'time' column.")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def load_csv(filename: str, subdir: str = "raw") -> pd.DataFrame:
    """
    Generic single-timeframe CSV loader.

    Expected columns:
        time, open, high, low, close, volume
    """
    path = ROOT / "data" / subdir / filename
    df = _read_csv(path)

    # Ensure OHLCV columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col} in {path}")

    return df


def load_es_multi_tf(
    f_5m: str = "es_5m_clean.csv",
    f_1h: str = "es_1h_clean.csv",
    f_2h: str = "es_2h_clean.csv",
    subdir: str = "raw",
) -> pd.DataFrame:
    """
    Load ES 5m + 1h + 2h, and merge them into a single 5m-aligned DataFrame.

    Result columns:
        time, open, high, low, close, volume
        + h1_* columns (1H OHLCV)
        + h2_* columns (2H OHLCV)
    """

    # Base 5m data
    path_5m = ROOT / "data" / subdir / f_5m
    df_5m = _read_csv(path_5m)

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df_5m.columns:
            raise ValueError(f"5m data missing column: {col} in {path_5m}")

    # 1H and 2H
    path_1h = ROOT / "data" / subdir / f_1h
    path_2h = ROOT / "data" / subdir / f_2h

    df_1h = _read_csv(path_1h)
    df_2h = _read_csv(path_2h)

    def _with_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        df = df.copy()
        cols = [c for c in df.columns if c != "time"]
        df = df[["time"] + cols]
        df.columns = ["time"] + [f"{prefix}_{c}" for c in cols]
        return df

    df_1h_p = _with_prefix(df_1h, "h1")
    df_2h_p = _with_prefix(df_2h, "h2")

    # Merge-asof aligns each 5m bar with the latest <= 1h/2h bar
    df = pd.merge_asof(
        df_5m.sort_values("time"),
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
