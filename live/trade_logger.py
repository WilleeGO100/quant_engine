# live/trade_logger.py
#
# Append live BTC trades to a CSV in the same format as btc_po3_paper_trades.csv:
# entry_time, exit_time, direction, entry_price, exit_price, pnl, bars_held, exit_reason

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd


@dataclass
class LiveTrade:
    entry_time: Union[pd.Timestamp, str]
    exit_time: Union[pd.Timestamp, str]
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    pnl: float
    bars_held: int
    exit_reason: str        # e.g. "TP", "SL", "FLIP", "TIME", "LIVE_MANUAL"


def _to_iso(ts: Union[pd.Timestamp, str]) -> str:
    """
    Normalize timestamps to ISO-8601 strings so they look like your paper CSV.
    """
    if isinstance(ts, str):
        return ts
    if isinstance(ts, pd.Timestamp):
        # ensure it's not NaT
        if pd.isna(ts):
            return ""
        # keep timezone info if present
        return ts.isoformat()
    # fallback
    return str(ts)


def append_live_trade(
    trade: LiveTrade,
    csv_path: str = "live/btc_po3_paper_trades.csv",
) -> None:
    """
    Append a single trade to the CSV at `csv_path`.
    If the file does not exist, it is created with a header.
    If it exists, the row is appended with no header.

    Columns:
        entry_time, exit_time, direction, entry_price,
        exit_price, pnl, bars_held, exit_reason
    """
    row = {
        "entry_time": _to_iso(trade.entry_time),
        "exit_time": _to_iso(trade.exit_time),
        "direction": trade.direction,
        "entry_price": float(trade.entry_price),
        "exit_price": float(trade.exit_price),
        "pnl": float(trade.pnl),
        "bars_held": int(trade.bars_held),
        "exit_reason": str(trade.exit_reason),
    }

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_row = pd.DataFrame([row])

    if path.exists():
        # append without header
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        # create with header
        df_row.to_csv(path, mode="w", header=True, index=False)
