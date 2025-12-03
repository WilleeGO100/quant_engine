# backtests/btc_po3_backtest.py
#
# Backtest harness for BTCPO3PowerStrategy on BTC 5m features.
#
# - Loads BTC 5m institutional features
# - Runs BTCPO3PowerStrategy bar-by-bar
# - Simulates simple entry/exit logic:
#     * FLAT -> LONG when signal == LONG
#     * FLAT -> SHORT when signal == SHORT
#     * LONG -> SHORT when signal flips to SHORT (close long, open short)
#     * SHORT -> LONG when signal flips to LONG (close short, open long)
#   (we will refine this later with proper stops/targets)
#
# - Computes:
#     * total PnL (1-unit per trade)
#     * win rate
#     * profit factor
#     * max drawdown
# - Saves all trades to btc_po3_trades.csv for inspection.

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from engine.features.btc_multiframe_features import (
    build_btc_5m_multiframe_features_institutional,
)
from engine.strategies.btc_po3_power import BTCPO3PowerStrategy


@dataclass
class Trade:
    side: str               # "LONG" or "SHORT"
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float              # absolute PnL in price units (1-unit per trade)
    pnl_pct: float          # PnL as % of entry price


def run_backtest() -> None:
    print("====================================================")
    print("[BACKTEST] BTC PO3 backtest starting...")
    print("====================================================")

    # 1) Load features
    df, _ = build_btc_5m_multiframe_features_institutional()
    print(f"[BACKTEST] Loaded {len(df)} feature rows.")

    # 2) Instantiate strategy
    strat = BTCPO3PowerStrategy()
    print("[BACKTEST] BTCPO3PowerStrategy instantiated.")

    trades: List[Trade] = []

    position = "FLAT"  # "FLAT", "LONG", "SHORT"
    entry_price: float = 0.0
    entry_time: pd.Timestamp | None = None

    # 3) Main backtest loop
    for ts, row in df.iterrows():
        price = float(row["close"])

        raw_signal = strat.on_bar(row)
        signal = str(raw_signal).upper().strip()

        # You’ll see these if you want to debug behavior per-bar later
        # print(ts, price, signal)

        if position == "FLAT":
            if signal == "LONG":
                position = "LONG"
                entry_price = price
                entry_time = ts
            elif signal == "SHORT":
                position = "SHORT"
                entry_price = price
                entry_time = ts

        elif position == "LONG":
            if signal == "SHORT":
                # CLOSE LONG
                exit_price = price
                exit_time = ts
                pnl = exit_price - entry_price
                pnl_pct = pnl / entry_price
                trades.append(
                    Trade(
                        side="LONG",
                        entry_time=entry_time,
                        exit_time=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    )
                )
                # OPEN SHORT
                position = "SHORT"
                entry_price = price
                entry_time = ts

            # If signal is HOLD or LONG, we simply stay in the position for now.

        elif position == "SHORT":
            if signal == "LONG":
                # CLOSE SHORT
                exit_price = price
                exit_time = ts
                pnl = entry_price - exit_price
                pnl_pct = pnl / entry_price
                trades.append(
                    Trade(
                        side="SHORT",
                        entry_time=entry_time,
                        exit_time=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    )
                )
                # OPEN LONG
                position = "LONG"
                entry_price = price
                entry_time = ts

            # If signal is HOLD or SHORT, we stay SHORT.

    # 4) Close any open position at the last bar (optional but common)
    if position in ("LONG", "SHORT") and entry_time is not None:
        ts_last = df.index[-1]
        price_last = float(df.iloc[-1]["close"])

        if position == "LONG":
            pnl = price_last - entry_price
            pnl_pct = pnl / entry_price
            trades.append(
                Trade(
                    side="LONG",
                    entry_time=entry_time,
                    exit_time=ts_last,
                    entry_price=entry_price,
                    exit_price=price_last,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
            )
        else:
            pnl = entry_price - price_last
            pnl_pct = pnl / entry_price
            trades.append(
                Trade(
                    side="SHORT",
                    entry_time=entry_time,
                    exit_time=ts_last,
                    entry_price=entry_price,
                    exit_price=price_last,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                )
            )

    # 5) Compute metrics
    if not trades:
        print("[BACKTEST] No trades generated by strategy.")
        return

    total_pnl = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    win_rate = (len(wins) / len(trades)) * 100.0

    gross_win = sum(t.pnl for t in wins) if wins else 0.0
    gross_loss = sum(t.pnl for t in losses) if losses else 0.0

    avg_win = gross_win / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0

    if gross_loss != 0:
        profit_factor = gross_win / abs(gross_loss)
    else:
        profit_factor = float("inf")

    # Equity curve + max drawdown (1-unit per trade, PnL in price units)
    equity = 0.0
    equity_curve = []
    for t in trades:
        equity += t.pnl
        equity_curve.append(equity)

    if equity_curve:
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
    else:
        max_dd = 0.0

    # 6) Print summary
    print("====================================================")
    print("==== BTC PO3 BACKTEST (1-unit per trade) ==========")
    print("====================================================")
    print(f"Bars tested     : {len(df)}")
    print(f"Trades taken    : {len(trades)}")
    print(f"Win rate        : {win_rate:.1f}%")
    print(f"Total PnL       : {total_pnl:.2f} (price units, 1-unit per trade)")
    print(f"Avg win         : {avg_win:.2f}")
    print(f"Avg loss        : {avg_loss:.2f}")
    print(f"Profit factor   : {profit_factor:.2f}")
    print(f"Max drawdown    : {max_dd:.2f}")

    # 7) Save trade log to CSV
    trades_df = pd.DataFrame(
        {
            "side": [t.side for t in trades],
            "entry_time": [t.entry_time for t in trades],
            "exit_time": [t.exit_time for t in trades],
            "entry_price": [t.entry_price for t in trades],
            "exit_price": [t.exit_price for t in trades],
            "pnl": [t.pnl for t in trades],
            "pnl_pct": [t.pnl_pct for t in trades],
        }
    )
    trades_df.to_csv("btc_po3_trades.csv", index=False)
    print("[BACKTEST] Trade log saved to btc_po3_trades.csv")
    print("====================================================")


if __name__ == "__main__":
    run_backtest()
