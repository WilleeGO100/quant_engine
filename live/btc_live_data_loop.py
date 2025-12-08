# live/btc_live_data_loop.py
#
# Pure live BTC 5m data loop using TwelveData:
#   - Bootstraps last N 5m BTC bars from TwelveData
#   - Keeps them in memory
#   - Writes them to:
#       * data/btc_5m_live.csv        (rolling latest snapshot)
#       * data/live_btc_5m_history.csv (append-only permanent log)
#   - Every new closed 5m bar:
#       * fetch_latest_btcusd_5m_bar()
#       * if timestamp is new -> append + save
#
# This script does NOT run the PO3 strategy or TradersPost.
# It is a clean data-feed loop you can reuse for backtests and diagnostics.

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

# --------------------------------------------------
# Path setup (project root)
# --------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --------------------------------------------------
# Local imports
# --------------------------------------------------
from engine.data.twelvedata_btc import (
    fetch_btcusd_5m_history,
    fetch_latest_btcusd_5m_bar,
)

# --------------------------------------------------
# File paths
# --------------------------------------------------
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Rolling latest snapshot (good for feature builder, etc.)
RAW_5M_CSV = os.path.join(DATA_DIR, "btc_5m_live.csv")

# Permanent append-only log of all live 5m bars
LIVE_HISTORY_CSV = os.path.join(DATA_DIR, "live_btc_5m_history.csv")


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _seconds_until_next_5m(now: Optional[datetime] = None) -> int:
    """
    How many seconds until the next 5-minute boundary (UTC)?

    Bars close at hh:00, 05, 10, ... 55.
    We want to wake up just *after* the close, so we add +1s.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    epoch = int(now.timestamp())
    rem = epoch % 300  # 300 seconds = 5 minutes
    wait = 300 - rem + 1
    return max(wait, 1)


def _write_raw_5m_csv(df: pd.DataFrame) -> None:
    """
    Save the current in-memory 5m BTC history into data/btc_5m_live.csv.

    - Index is the candle datetime
    - We reset_index and rename it to 'timestamp' for convenience
      (this mirrors how other parts of the engine expect it).
    """
    out = df.copy().reset_index()

    if "datetime" in out.columns:
        out = out.rename(columns={"datetime": "timestamp"})

    os.makedirs(os.path.dirname(RAW_5M_CSV), exist_ok=True)
    out.to_csv(RAW_5M_CSV, index=False)


def _append_to_live_history(new_bars: pd.DataFrame) -> None:
    """
    Append one or more new bars to data/live_btc_5m_history.csv.

    - If the file does not exist, write header.
    - If it exists, append without header.

    Assumes new_bars index is datetime.
    """
    if new_bars.empty:
        return

    out = new_bars.copy().reset_index()
    if "datetime" in out.columns:
        out = out.rename(columns={"datetime": "timestamp"})

    os.makedirs(os.path.dirname(LIVE_HISTORY_CSV), exist_ok=True)

    file_exists = os.path.exists(LIVE_HISTORY_CSV)
    header = not file_exists

    out.to_csv(LIVE_HISTORY_CSV, mode="a", header=header, index=False)


# --------------------------------------------------
# Main live loop
# --------------------------------------------------
def run_btc_live_data_loop() -> None:
    print(
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} "
        "[INFO] BTC_LIVE_DATA_LOOP: Starting BTC 5m live data loop (TwelveData)..."
    )

    # 1) Bootstrap history from TwelveData
    print(
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} "
        "[INFO] BTC_LIVE_DATA_LOOP: Bootstrapping BTCUSD 5m history from TwelveData..."
    )

    # IMPORTANT: use outputsize=500 (matches your twelvedata_btc.py signature)
    hist_df = fetch_btcusd_5m_history(outputsize=500)
    if hist_df.empty:
        print(
            f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} "
            "[ERROR] BTC_LIVE_DATA_LOOP: No history returned from TwelveData. Exiting."
        )
        return

    # Ensure sorted by time (index = datetime)
    hist_df = hist_df.sort_index()
    last_ts = hist_df.index[-1]

    print(
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} "
        f"[INFO] BTC_LIVE_DATA_LOOP: Bootstrapped {len(hist_df)} bars "
        f"up to {last_ts}."
    )

    # Save initial snapshot + history
    _write_raw_5m_csv(hist_df)
    _append_to_live_history(hist_df)

    # 2) Main loop
    while True:
        sleep_s = _seconds_until_next_5m()
        print(
            f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} "
            f"[INFO] BTC_LIVE_DATA_LOOP: Sleeping {sleep_s} seconds until next 5m bar close..."
        )
        time.sleep(sleep_s)

        # Fetch latest (closed) 5m bar
        try:
            latest_df = fetch_latest_btcusd_5m_bar()
        except Exception as exc:
            print(
                f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} "
                f"[WARN] BTC_LIVE_DATA_LOOP: Error fetching latest bar: {exc}"
            )
            continue

        if latest_df is None or latest_df.empty:
            print(
                f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} "
                "[WARN] BTC_LIVE_DATA_LOOP: No latest bar returned from TwelveData."
            )
            continue

        latest_df = latest_df.sort_index()
        latest_ts = latest_df.index[-1]

        if latest_ts <= last_ts:
            # No new closed bar yet (API sometimes repeats last one)
            print(
                f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} "
                f"[INFO] BTC_LIVE_DATA_LOOP: No new 5m bar yet "
                f"(latest={latest_ts}, last_seen={last_ts})."
            )
            continue

        # We have at least one new bar; keep only truly new ones
        new_bars = latest_df[latest_df.index > last_ts]
        if new_bars.empty:
            continue

        # Update in-memory history
        hist_df = pd.concat([hist_df, new_bars]).sort_index()
        last_ts = hist_df.index[-1]

        # Save updated snapshot + append to permanent log
        _write_raw_5m_csv(hist_df)
        _append_to_live_history(new_bars)

        # Log a concise summary of the last bar
        last_bar = hist_df.iloc[-1]
        print(
            f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} "
            f"[INFO] BTC_LIVE_DATA_LOOP: New 5m bar @ {last_ts} | "
            f"o={last_bar['open']:.2f} h={last_bar['high']:.2f} "
            f"l={last_bar['low']:.2f} c={last_bar['close']:.2f} "
            f"vol={last_bar['volume']:.4f}"
        )


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    run_btc_live_data_loop()
