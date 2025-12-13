# backtests/btc_po3_backtest.py
#
# Clean BTC PO3 backtest using the new btc_features.py pipeline.
#
# Expects:
#   - engine/features/btc_features.py
#   - engine/strategies/smc_po3_power_btc.py
#   - ROOT/data/btc_5m.csv or ROOT/data/btc_5m_live.csv
#
# Run from project root:
#   .\.venv\Scripts\python.exe -m backtests.btc_po3_backtest

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# ======================================================
# PATH / IMPORTS
# ======================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.features.btc_features import load_and_build_btc_5m_features
from engine.strategies.smc_po3_power_btc import (
    BTCPO3PowerStrategy,
    BTCPO3PowerConfig,
)

# ======================================================
# BACKTEST KNOBS
# ======================================================

# Preferred historical file name under ROOT/data.
# We will *resolve* between btc_5m.csv and btc_5m_live.csv at runtime.
PREFERRED_BTC_FILE_NAME = "btc_5m.csv"

# Risk parameters
STOP_ATR_MULT: float = 1.2
TP_ATR_MULT: float = 3.0
MAX_HOLD_BARS: int = 96
UNIT_SIZE: float = 1.0

# Use a separate ATR for SL/TP? Here we just reuse atr_14 from features
USE_SEPARATE_ATR: bool = False
ATR_PERIOD: int = 14  # only used if USE_SEPARATE_ATR = True

# Toggle a simple EMA trend baseline instead of PO3
USE_DUMMY_STRATEGY: bool = False

DUMMY_EMA_FAST: int = 20
DUMMY_EMA_SLOW: int = 50


# ======================================================
# DATA STRUCTURES
# ======================================================

@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    bars_held: int
    exit_reason: str


# ======================================================
# FILE RESOLUTION
# ======================================================

def _resolve_btc_filename() -> str:
    """
    Look in ROOT/data for a BTC 5m CSV and return the filename to pass
    into load_and_build_btc_5m_features().

    Priority:
      1) btc_5m.csv        (static backtest dataset)
      2) btc_5m_live.csv   (recent live export)
    """
    data_dir = os.path.join(ROOT_DIR, "data")

    candidates = [
        "btc_5m.csv",
        "btc_5m_live.csv",
    ]

    for fname in candidates:
        full_path = os.path.join(data_dir, fname)
        if os.path.exists(full_path):
            print(f"[BACKTEST] Using BTC data file: {full_path}")
            return fname

    # Fallback – keep old behavior if nothing was found
    print(
        "[BACKTEST] WARNING: No btc_5m*.csv found in ./data. "
        f"Falling back to preferred name: {PREFERRED_BTC_FILE_NAME}"
    )
    return PREFERRED_BTC_FILE_NAME


# ======================================================
# DUMMY TREND STRATEGY
# ======================================================

class DummyTrendStrategy:
    """
    Simple EMA cross trend strategy:
      LONG  when ema_fast > ema_slow
      SHORT when ema_fast < ema_slow
      HOLD  otherwise
    """

    def __init__(self, df: pd.DataFrame, fast_len: int = 20, slow_len: int = 50):
        self.fast = df["close"].ewm(span=fast_len, adjust=False).mean()
        self.slow = df["close"].ewm(span=slow_len, adjust=False).mean()

    def on_bar(self, row: Dict[str, Any]) -> str:
        idx = row["index"]
        f = float(self.fast.loc[idx])
        s = float(self.slow.loc[idx])

        if not np.isfinite(f) or not np.isfinite(s):
            return "HOLD"
        if f > s:
            return "LONG"
        if f < s:
            return "SHORT"
        return "HOLD"


# ======================================================
# ATR HELPERS (ONLY IF YOU WANT SEPARATE BACKTEST ATR)
# ======================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return atr


def compute_max_drawdown(trades: List[Trade]) -> float:
    if not trades:
        return 0.0

    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)

    return max_dd


# ======================================================
# CORE BACKTEST
# ======================================================

def run_backtest() -> None:
    print("====================================================")
    print("[BACKTEST] BTC PO3 backtest starting...")
    print("====================================================")

    # 1) Decide which CSV to use, then build BTC 5m feature frame
    btc_filename = _resolve_btc_filename()

    # load_and_build_btc_5m_features will:
    #   - resolve ROOT/data/<filename>
    #   - normalize timestamps
    #   - compute institutional features (atr_14, rvol, VWAP, etc.)
    feat = load_and_build_btc_5m_features(btc_filename)

    # Normalize to DatetimeIndex
    df = feat.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()

    n_rows = len(df)
    print(f"[BACKTEST] Loaded {n_rows} BTC 5m feature rows.")

    # Keep index in a column for DummyTrendStrategy
    df["index"] = df.index

    # 2) ATR for SL/TP
    if USE_SEPARATE_ATR:
        df["atr_bt"] = compute_atr(df, period=ATR_PERIOD)
    else:
        # reuse the feature ATR (atr_14)
        if "atr_14" not in df.columns:
            raise RuntimeError(
                "Feature column 'atr_14' not found in BTC features. "
                "Check engine/features/btc_features.py."
            )
        df["atr_bt"] = df["atr_14"].astype(float)

    # 3) Strategy
    if USE_DUMMY_STRATEGY:
        strategy = DummyTrendStrategy(
            df,
            fast_len=DUMMY_EMA_FAST,
            slow_len=DUMMY_EMA_SLOW,
        )
        print("[BACKTEST] Using DummyTrendStrategy (EMA cross baseline).")
    else:
        config = BTCPO3PowerConfig(
            # Volatility (ATR% of price)
            min_atr_pct=0.0008,
            max_atr_pct=0.06,
            # Relative volume
            min_rvol=1.0,
            max_rvol=None,
            # Session filters
            allow_asia=True,
            allow_london=True,
            allow_ny=True,
            # Trend regime
            use_trend_regime=True,
            # Weekly range
            min_week_pos_for_longs=0.06,
            max_week_pos_for_shorts=0.80,
            # VWAP distance (in ATR units)
            max_vwap_dist_atr_entry=1.1,
            # Sweep (kept for compatibility, unused)
            require_sweep=False,
            lookback_sweep_bars=5,
            verbose=False,
        )
        strategy = BTCPO3PowerStrategy(config=config)
        print("[BACKTEST] Using BTCPO3PowerStrategy (EMA/VWAP PO3).")

    trades: List[Trade] = []

    current_side: Optional[str] = None
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    bars_in_trade: int = 0
    stop_price: float = 0.0
    tp_price: float = 0.0

    bars_with_atr = 0
    long_signals = 0
    short_signals = 0
    hold_signals = 0

    # 4) Simulation loop
    for ts, row in df.iterrows():
        row_dict: Dict[str, Any] = row.to_dict()

        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row.get("atr_bt", np.nan))

        if not np.isfinite(atr) or atr <= 0:
            continue

        bars_with_atr += 1

        signal = strategy.on_bar(row_dict)  # "LONG", "SHORT", "HOLD"

        if signal == "LONG":
            long_signals += 1
        elif signal == "SHORT":
            short_signals += 1
        else:
            hold_signals += 1

        if current_side is None:
            # No open position: consider entry
            if signal in ("LONG", "SHORT"):
                current_side = signal
                entry_price = close
                entry_time = ts
                bars_in_trade = 0

                if current_side == "LONG":
                    stop_price = entry_price - STOP_ATR_MULT * atr
                    tp_price = entry_price + TP_ATR_MULT * atr
                else:
                    stop_price = entry_price + STOP_ATR_MULT * atr
                    tp_price = entry_price - TP_ATR_MULT * atr
        else:
            # Manage open position
            bars_in_trade += 1
            exit_now = False
            exit_reason = "none"
            exit_price = close

            if current_side == "LONG":
                if low <= stop_price:
                    exit_price = stop_price
                    exit_now = True
                    exit_reason = "SL"
                elif high >= tp_price:
                    exit_price = tp_price
                    exit_now = True
                    exit_reason = "TP"
            else:
                if high >= stop_price:
                    exit_price = stop_price
                    exit_now = True
                    exit_reason = "SL"
                elif low <= tp_price:
                    exit_price = tp_price
                    exit_now = True
                    exit_reason = "TP"

            # Time stop
            if not exit_now and bars_in_trade >= MAX_HOLD_BARS:
                exit_now = True
                exit_reason = "TIME"

            # Flip on opposite signal
            if (
                not exit_now
                and signal in ("LONG", "SHORT")
                and signal != current_side
            ):
                exit_now = True
                exit_reason = "FLIP"

            if exit_now:
                if current_side == "LONG":
                    pnl = (exit_price - entry_price) * UNIT_SIZE
                else:
                    pnl = (entry_price - exit_price) * UNIT_SIZE

                pnl_pct = pnl / entry_price if entry_price != 0 else 0.0

                trades.append(
                    Trade(
                        side=current_side,
                        entry_time=entry_time,
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        bars_held=bars_in_trade,
                        exit_reason=exit_reason,
                    )
                )

                current_side = None
                entry_price = 0.0
                entry_time = None
                bars_in_trade = 0
                stop_price = 0.0
                tp_price = 0.0

    # 5) Close any open trade at the final bar
    if current_side is not None and entry_time is not None:
        last_close = float(df.iloc[-1]["close"])
        ts = df.index[-1]
        if current_side == "LONG":
            pnl = (last_close - entry_price) * UNIT_SIZE
        else:
            pnl = (entry_price - last_close) * UNIT_SIZE
        pnl_pct = pnl / entry_price if entry_price != 0 else 0.0

        trades.append(
            Trade(
                side=current_side,
                entry_time=entry_time,
                exit_time=ts,
                entry_price=entry_price,
                exit_price=last_close,
                pnl=pnl,
                pnl_pct=pnl_pct,
                bars_held=bars_in_trade,
                exit_reason="END",
            )
        )

    # 6) Stats
    print("====================================================")
    print("==== BTC BACKTEST (1-unit per trade) ==============")
    print("====================================================")

    bars_tested = len(df)
    n_trades = len(trades)
    total_pnl = sum(t.pnl for t in trades)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]

    win_rate = (len(wins) / n_trades * 100.0) if n_trades > 0 else 0.0
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = sum(t.pnl for t in losses)
    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    else:
        profit_factor = np.nan

    max_dd = compute_max_drawdown(trades)

    print(f"Bars tested     : {bars_tested}")
    print(f"Bars w/ ATR     : {bars_with_atr}")
    print(f"Trades taken    : {n_trades}")
    print(f"Win rate        : {win_rate:.1f}%")
    print(f"Total PnL       : {total_pnl:.2f} (price units, 1-unit per trade)")
    print(f"Avg win         : {avg_win:.2f}")
    print(f"Avg loss        : {avg_loss:.2f}")
    print(f"Profit factor   : {profit_factor:.2f}")
    print(f"Max drawdown    : {max_dd:.2f}")
    print()
    print(f"Stop ATR mult   : {STOP_ATR_MULT}")
    print(f"TP ATR mult     : {TP_ATR_MULT}")
    print(f"Max hold (bars) : {MAX_HOLD_BARS}")
    print(f"ATR period      : {ATR_PERIOD}")
    print()
    print(f"LONG signals    : {long_signals}")
    print(f"SHORT signals   : {short_signals}")
    print(f"HOLD signals    : {hold_signals}")

    # 7) Save trade log
    trades_df = pd.DataFrame(
        {
            "side": [t.side for t in trades],
            "entry_time": [t.entry_time for t in trades],
            "exit_time": [t.exit_time for t in trades],
            "entry_price": [t.entry_price for t in trades],
            "exit_price": [t.exit_price for t in trades],
            "pnl": [t.pnl for t in trades],
            "pnl_pct": [t.pnl_pct for t in trades],
            "bars_held": [t.bars_held for t in trades],
            "exit_reason": [t.exit_reason for t in trades],
        }
    )
    trades_df.to_csv("btc_po3_trades.csv", index=False)
    print("[BACKTEST] Trade log saved to btc_po3_trades.csv")
    print("====================================================")


if __name__ == "__main__":
    run_backtest()
