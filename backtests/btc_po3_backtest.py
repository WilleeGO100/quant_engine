"""
BTC PO3 BACKTEST ENGINE
=======================

This script runs your BTC PO3 Power Strategy against historical
BTC/USD 5-minute OHLCV data, computes institutional features,
applies signals, performs ATR-based SL/TP exits, and prints a full
performance report.

DEPENDENCIES:
    - engine/features/btc_multiframe_features.py
    - engine/strategies/smc_po3_power_btc.py

OUTPUT:
    - backtests/btc_po3_trades.csv
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import os
import sys
import math
from datetime import datetime
from collections import Counter

# ----------------------------
# FIX PYTHON PATH
# ----------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ----------------------------
# IMPORT STRATEGY + FEATURES
# ----------------------------
from engine.strategies.smc_po3_power_btc import (
    BTCPO3PowerStrategy,
    BTCPO3PowerConfig,
)

from engine.features.btc_multiframe_features import (
    build_btc_5m_multiframe_features_institutional,
)


# =====================================================
# Backtest Parameters
# =====================================================
ATR_PERIOD = 14
STOP_ATR_MULT = 1.2
TP_ATR_MULT = 2.5
MAX_HOLD_BARS = 96


# =====================================================
# Utility Functions
# =====================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(period).mean()


# =====================================================
# TRADE EXECUTION ENGINE
# =====================================================
def run_backtest():
    print("=" * 53)
    print("[BACKTEST] BTC PO3 backtest starting...")
    print("=" * 53)

    # -------------------------------------------------
    # LOAD AND BUILD FEATURES
    # -------------------------------------------------
    df, n_raw = build_btc_5m_multiframe_features_institutional()
    print(f"[BACKTEST] Loaded {len(df)} BTC 5m feature rows.")

    # ATR FOR SL/TP
    df["atr"] = compute_atr(df, ATR_PERIOD)
    df["atr"] = df["atr"].bfill()

    # ------------------------------------------------
    # 4) Instantiate strategy config
    # ------------------------------------------------
    config = BTCPO3PowerConfig(
        min_atr_pct=0.0004,
        max_atr_pct=0.08,
        min_rvol=0.5,
        allow_asia=True,
        allow_london=True,
        allow_ny=True,
        require_sweep=False,
        min_sweep_strength=0,
        require_fvg=False,
        verbose=False,
    )

    strategy = BTCPO3PowerStrategy(config)
    print("[BACKTEST] Strategy: BTCPO3PowerStrategy instantiated.")

    # ------------------------------------------------
    # 5) Main simulation loop
    # ------------------------------------------------
    position = None
    entry_price = None
    entry_ts = None
    stop_price = None
    tp_price = None
    bars_in_trade = 0

    trades = []
    exit_reasons = Counter()
    signal_counts = Counter()

    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------
    for i in range(len(df)):
        row = df.iloc[i]
        ts = row.name
        close = row["close"]
        atr = row["atr"]

        # UPDATE OPEN POSITIONS
        if position is not None:
            # Stop Loss
            if position == "LONG" and row["low"] <= stop_price:
                pnl = stop_price - entry_price
                trades.append((entry_ts, ts, "SL", position, entry_price, stop_price, pnl))
                position = None

            elif position == "SHORT" and row["high"] >= stop_price:
                pnl = entry_price - stop_price
                trades.append((entry_ts, ts, "SL", position, entry_price, stop_price, pnl))
                position = None

            # Take Profit
            elif position == "LONG" and row["high"] >= tp_price:
                pnl = tp_price - entry_price
                trades.append((entry_ts, ts, "TP", position, entry_price, tp_price, pnl))
                position = None

            elif position == "SHORT" and row["low"] <= tp_price:
                pnl = entry_price - tp_price
                trades.append((entry_ts, ts, "TP", position, entry_price, tp_price, pnl))
                position = None

            # Max Hold Time
            elif (ts - entry_ts).total_seconds() / 300 >= MAX_HOLD_BARS:
                exit_price = close
                pnl = exit_price - entry_price if position == "LONG" else entry_price - exit_price
                trades.append((entry_ts, ts, "TIME", position, entry_price, exit_price, pnl))
                position = None

        # =====================================================
        # DECISION MAKING (NO POSITION)
        # =====================================================
        if position is None:
            signal = strategy.on_bar(row)

            if signal in ("LONG", "SHORT"):
                position = signal
                entry_price = close
                entry_ts = ts

                # Set stop & TP
                if signal == "LONG":
                    stop_price = close - atr * STOP_ATR_MULT
                    tp_price = close + atr * TP_ATR_MULT
                else:
                    stop_price = close + atr * STOP_ATR_MULT
                    tp_price = close - atr * TP_ATR_MULT

    # =====================================================
    # REPORT
    # =====================================================
    print("=" * 53)
    print("==== BTC BACKTEST (1-unit per trade) ==============")
    print("=" * 53)

    if len(trades) == 0:
        print("No trades generated.")
        return

    results = pd.DataFrame(trades, columns=[
        "entry_ts", "exit_ts", "exit_reason", "side",
        "entry_price", "exit_price", "pnl"
    ])

    total_pnl = results["pnl"].sum()
    wins = results[results["pnl"] > 0]
    losses = results[results["pnl"] < 0]

    win_rate = len(wins) / len(results) * 100
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else np.inf
    max_dd = results["pnl"].cumsum().cummax() - results["pnl"].cumsum()
    max_dd = max_dd.max()

    print(f"Trades taken    : {len(results)}")
    print(f"Win rate        : {win_rate:.1f}%")
    print(f"Total PnL       : {total_pnl:.2f}")
    print(f"Profit factor   : {pf:.2f}")
    print(f"Max drawdown    : {max_dd:.2f}")
    print()

    # Save CSV
    out_path = os.path.join(ROOT, "backtests", "btc_po3_trades.csv")
    results.to_csv(out_path, index=False)
    print(f"[BACKTEST] Trade log saved to {out_path}")
    print("=" * 53)


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    run_backtest()
