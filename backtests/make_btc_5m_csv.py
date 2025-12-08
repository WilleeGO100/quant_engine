# backtests/make_btc_5m_csv.py
#
# Utility script to rebuild data/btc_5m.csv from TwelveData.
#
# Uses the existing engine.data.twelvedata_btc module, which expects
# your TWELVEDATA_API_KEY to be set in the .env file at project root.
#
# It will:
#   - fetch recent BTC/USD 5m candles
#   - sort by datetime
#   - ensure columns: ["datetime","open","high","low","close","volume"]
#   - save to data/btc_5m.csv

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# -------------------------------------------------------------------
# Ensure project root (quant_engine/) is on sys.path so `import engine`
# works even when this file is run as a script.
# -------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent  # .../quant_engine
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# NOTE: function name is fetch_btcusd_5m_history (no 't')
from engine.data.twelvedata_btc import fetch_btcusd_5m_history  # noqa: E402


def main() -> None:
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    print("Project root:", ROOT)
    print("Data directory:", data_dir)

    # Fetch history – tweak outputsize if you want more bars
    print("Fetching BTC/USD 5m history from TwelveData...")
    df = fetch_btcusd_5m_history(outputsize=2000)  # ~2000 bars of 5m data

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError("fetch_btcusd_5m_history returned no data or non-DataFrame.")

    # Ensure datetime index and column
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.set_index("datetime")
        else:
            raise RuntimeError("No DatetimeIndex or 'datetime' column found in fetched data.")

    df = df.sort_index()

    # Standardize columns – keep only what we need
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open", "o"):
            rename_map[col] = "open"
        elif lc in ("high", "h"):
            rename_map[col] = "high"
        elif lc in ("low", "l"):
            rename_map[col] = "low"
        elif lc in ("close", "c"):
            rename_map[col] = "close"
        elif "volume" in lc or lc == "v":
            rename_map[col] = "volume"

    df = df.rename(columns=rename_map)

    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required OHLC columns in fetched data: {missing}")

    # If volume is missing, create a dummy zero volume column
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # Re-order columns
    df = df[["open", "high", "low", "close", "volume"]]

    # Materialize datetime as a column for CSV
    out = df.copy()

    # Drop timezone: convert tz-aware index to naive (UTC)
    idx = out.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx_naive = idx.tz_convert("UTC").tz_localize(None)
    else:
        idx_naive = idx

    out.insert(0, "datetime", idx_naive)

    out_path = data_dir / "btc_5m.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
