import os
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Change this if your filename differs
SRC_FILE = os.path.join(DATA_DIR, "btcusd_1m.csv")

OUT_5M  = os.path.join(DATA_DIR, "btc_5m.csv")
OUT_30M = os.path.join(DATA_DIR, "btc_30m.csv")
OUT_1H  = os.path.join(DATA_DIR, "btc_1h.csv")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]

    # timestamp column normalization
    if "timestamp" not in df.columns:
        for alt in ["date", "time", "datetime"]:
            if alt in df.columns:
                df.rename(columns={alt: "timestamp"}, inplace=True)
                break

    if "timestamp" not in df.columns:
        raise ValueError(f"No timestamp column found. Columns: {list(df.columns)[:30]}")

    # volume column normalization (common variants)
    if "volume" not in df.columns:
        for alt in ["vol", "volume_(btc)", "volume btc", "volume_btc", "volumebtc", "qty", "quantity"]:
            if alt in df.columns:
                df.rename(columns={alt: "volume"}, inplace=True)
                break

    return df


def _parse_timestamp(series: pd.Series) -> pd.DatetimeIndex:
    s = series.copy()

    # If numeric epoch timestamps (very common in crypto datasets)
    if pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce")
        s = s.dropna()
        if s.empty:
            raise ValueError("Timestamp column is numeric but all values are NaN after coercion.")

        median = float(s.median())

        # Heuristic:
        # - seconds epoch ~ 1e9 (e.g., 1700000000)
        # - milliseconds epoch ~ 1e12 (e.g., 1700000000000)
        if median >= 1e12:
            dt = pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
        elif median >= 1e9:
            dt = pd.to_datetime(series, unit="s", utc=True, errors="coerce")
        else:
            # fallback: try plain parse
            dt = pd.to_datetime(series, utc=True, errors="coerce")
        return pd.DatetimeIndex(dt)

    # Otherwise parse as string datetime
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return pd.DatetimeIndex(dt)


def load_1m():
    df = pd.read_csv(SRC_FILE)
    df = _normalize_columns(df)

    # Parse timestamps correctly
    dt_index = _parse_timestamp(df["timestamp"])
    df["timestamp"] = dt_index

    # Drop rows with bad timestamps
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # Enforce numeric OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])

    # Quick sanity prints
    print(f"[OK] Loaded {len(df)} 1m bars")
    print(f"[OK] Time range: {df.index.min()}  ->  {df.index.max()}")
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlc = df[["open", "high", "low", "close"]].resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    })

    if "volume" in df.columns:
        vol = df["volume"].resample(rule).sum()
        out = ohlc.join(vol)
    else:
        out = ohlc

    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def main():
    df_1m = load_1m()

    df_5m  = resample(df_1m, "5min")
    df_30m = resample(df_1m, "30min")
    df_1h  = resample(df_1m, "1h")

    df_5m.to_csv(OUT_5M)
    df_30m.to_csv(OUT_30M)
    df_1h.to_csv(OUT_1H)

    print("[DONE]")
    print(f"  5m  bars : {len(df_5m)}")
    print(f"  30m bars : {len(df_30m)}")
    print(f"  1h  bars : {len(df_1h)}")


if __name__ == "__main__":
    main()
