# backtests/po3_power_backtest.py

from __future__ import annotations

from typing import List, Dict, Any

import pandas as pd

from engine.strategies.smc_po3_power import SMCPo3PowerStrategy
from engine.features.es_multiframe_features import build_es_5m_multiframe_features


def _load_data() -> pd.DataFrame:
    """
    Build the multiframe ES 5m feature set using your existing pipeline.

    This calls build_es_5m_multiframe_features(), which:
      - loads es_5m_clean.csv, es_1h_clean.csv, es_2h_clean.csv from data/raw
      - builds 5m features (ATR, ranges, sessions, etc.)
      - merges 1H and 2H OHLCV as h1_* and h2_* columns
    """

    df = build_es_5m_multiframe_features(
        f_5m="es_5m_clean.csv",
        f_1h="es_1h_clean.csv",
        f_2h="es_2h_clean.csv",
    )

    # Safety: make sure time is datetime and sorted
    if "time" not in df.columns:
        raise ValueError("Expected a 'time' column in multiframe features.")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def run_power_backtest() -> Dict[str, Any]:
    print("============================================")
    print("Running POWER SMC PO3 backtests")
    print("Using multiframe ES data with features")
    print("============================================")

    df = _load_data()
    strategy = SMCPo3PowerStrategy()

    trades: List[Dict[str, Any]] = []

    # Current open position
    position_side: str | None = None  # "LONG" or "SHORT"
    entry_price: float | None = None
    entry_time: pd.Timestamp | None = None
    bars_in_trade: int = 0

    # -------- main loop over bars --------
    for _, row in df.iterrows():
        price = float(row["close"])
        ts = row["time"]

        # Ask the strategy for a signal on this bar
        signal = strategy.on_bar(row)  # "LONG" / "SHORT" / "HOLD"
        strategy_reason = strategy.get_last_reason()  # explanation for this bar

        # No open position yet
        if position_side is None:
            if signal == "LONG":
                position_side = "LONG"
                entry_price = price
                entry_time = ts
                bars_in_trade = 0
            elif signal == "SHORT":
                position_side = "SHORT"
                entry_price = price
                entry_time = ts
                bars_in_trade = 0
            continue

        # We are in a trade
        bars_in_trade += 1

        # Check for reversal conditions:
        reverse_long_to_short = position_side == "LONG" and signal == "SHORT"
        reverse_short_to_long = position_side == "SHORT" and signal == "LONG"

        if reverse_long_to_short or reverse_short_to_long:
            exit_price = price
            exit_time = ts

            # Compute PnL in points
            if position_side == "LONG":
                pnl = exit_price - entry_price
            else:  # SHORT
                pnl = entry_price - exit_price

            trades.append(
                {
                    "side": position_side,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "bars": bars_in_trade,
                    "pnl": pnl,
                    "strategy_reason": strategy_reason,
                }
            )

            # Open a new trade in the direction of the new signal
            position_side = signal  # "LONG" or "SHORT"
            entry_price = price
            entry_time = ts
            bars_in_trade = 0

    # -------- summary stats --------
    num_trades = len(trades)
    if num_trades == 0:
        print("No trades taken.")
        return {"num_trades": 0}

    equity = 0.0
    max_equity = 0.0
    max_drawdown = 0.0
    wins = 0
    losses = 0

    for t in trades:
        equity += t["pnl"]
        max_equity = max(max_equity, equity)
        drawdown = max_equity - equity
        max_drawdown = max(max_drawdown, drawdown)

        if t["pnl"] > 0:
            wins += 1
        else:
            losses += 1

    win_rate = 100.0 * wins / num_trades if num_trades > 0 else 0.0

    print()
    # noinspection PyCompatibility
    print(f"Total Trades: {num_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"PnL: {equity:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}")
    print()

    # -------- sample trades --------
    print("Sample trades:")
    for t in trades[:10]:
        side = t["side"]
        entry_time = t["entry_time"]
        exit_time = t["exit_time"]
        entry_price = t["entry_price"]
        exit_price = t["exit_price"]
        bars = t["bars"]
        pnl = t["pnl"]
        reason = t["strategy_reason"]

        print(
            f"{side:<5} | {entry_time} -> {exit_time} | "
            f"{entry_price:.2f} -> {exit_price:.2f} | "
            f"bars: {bars:3d} | pnl: {pnl:7.2f} | reason: {reason}"
        )

    return {
        "num_trades": num_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "pnl": equity,
        "max_drawdown": max_drawdown,
    }


if __name__ == "__main__":
    run_power_backtest()

