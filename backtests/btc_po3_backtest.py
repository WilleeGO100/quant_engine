from __future__ import annotations

# ==========================================
# IMPORTS
# ==========================================
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# ==========================================
# PATH SETUP (PROJECT ROOT)
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Now we can import from engine.*
from engine.features.btc_multiframe_features import (
    build_btc_5m_multiframe_features_institutional,
)
from engine.strategies.smc_po3_power_btc import (
    BTCPO3PowerStrategy,
    BTCPO3PowerConfig,
)

# ==========================================
# TOGGLE: USE DUMMY STRATEGY OR REAL PO3
# ==========================================
USE_DUMMY_STRATEGY = False  # <-- set to False later to go back to BTCPO3PowerStrategy

# ==========================================
# RISK / BACKTEST PARAMETERS
# ==========================================
STOP_ATR_MULT = 1.2      # how many ATR below/above entry for stop
TP_ATR_MULT = 3.0        # how many ATR above/below entry for take profit
MAX_HOLD_BARS = 96       # max bars in trade (~8 hours on 5m)
UNIT_SIZE = 1.0          # 1-unit per trade for now
ATR_PERIOD = 14          # ATR lookback used for SL/TP

# ==========================================
# DATA STRUCTURES
# ==========================================
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


# ==========================================
# DUMMY STRATEGY (for pipeline sanity check)
# ==========================================
class DummyTrendStrategy:
    """
    Extremely simple trend strategy:
    - LONG when fast_ema > slow_ema
    - SHORT when fast_ema < slow_ema
    - HOLD otherwise
    """

    def __init__(self, df: pd.DataFrame, fast_len: int = 20, slow_len: int = 50):
        self.fast = df["close"].ewm(span=fast_len, adjust=False).mean()
        self.slow = df["close"].ewm(span=slow_len, adjust=False).mean()

    def on_bar(self, row_dict: Dict[str, Any]) -> str:
        idx = row_dict["index"]  # we will inject this before looping
        f = float(self.fast.loc[idx])
        s = float(self.slow.loc[idx])

        if not np.isfinite(f) or not np.isfinite(s):
            return "HOLD"

        if f > s:
            return "LONG"
        elif f < s:
            return "SHORT"
        else:
            return "HOLD"


# ==========================================
# HELPER: COMPUTE ATR FOR BACKTEST
# ==========================================
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute classic ATR from high/low/close.
    This is ONLY for SL/TP in the backtest, and does not change strategy logic.
    """
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


# ==========================================
# HELPER: EQUITY CURVE & MAX DRAWDOWN
# ==========================================
def compute_max_drawdown(trades: List[Trade]) -> float:
    """
    Compute max drawdown from sequence of trades (pnl in price units).
    """
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


# ==========================================
# CORE BACKTEST LOOP
# ==========================================
def run_backtest() -> None:
    print("====================================================")
    print("[BACKTEST] BTC PO3 backtest starting...")
    print("====================================================")

    # 1) Build BTC 5m institutional feature frame
    df, n_rows = build_btc_5m_multiframe_features_institutional()
    print(f"[BACKTEST] Loaded {n_rows} BTC 5m feature rows.")

    # keep the index in a column so DummyTrendStrategy can reference it
    df = df.copy()
    df["index"] = df.index

    # 1B) Compute internal ATR for SL/TP (separate from feature ATR)
    df["atr_bt"] = compute_atr(df, period=ATR_PERIOD)

    # 2) Instantiate strategy
    if USE_DUMMY_STRATEGY:
        strategy = DummyTrendStrategy(df)
        print("[BACKTEST] Using DummyTrendStrategy (EMA trend) for pipeline check.")
    else:
        config = BTCPO3PowerConfig(
            # ATR% filter: ignore ultra-dead + insane-wick zones
            # (ATR% = atr_5m / close, roughly)
            min_atr_pct=0.0008,      # 0.10% – avoid totally dead chop
            max_atr_pct=0.06,       # 5% – avoid insane liquidation spikes

            # RVOL filter: require at least "normal-ish" volume
            min_rvol=1.0,           # was 0.0 → now only trade when volume is decent

            # Session filters: BTC trades 24/7 but "real" flow is usually London+NY
            allow_asia=True,        # keep Asia for now (we can turn off later)
            allow_london=True,
            allow_ny=True,

            # Trend regime: keep OFF until we confirm the feature is clean
            use_trend_regime=True,

            # Weekly position: bias longs away from absolute highs, shorts away from lows
            min_week_pos_for_longs=0.20,   # only long above 10% of weekly range
            max_week_pos_for_shorts=0.80,  # only short below 90% of weekly range

            # VWAP distance: force trades reasonably near "value"
            max_vwap_dist_atr_entry=2.0,   # within 2 ATRs of VWAP

            # Sweep requirement: still off for now
            require_sweep=False,
            lookback_sweep_bars=5,

            verbose=False,
        )

        strategy = BTCPO3PowerStrategy(config=config)
        print("[BACKTEST] Using BTCPO3PowerStrategy with fully open filters.")

    # 3) Iterate over bars, simulate trades
    trades: List[Trade] = []

    current_side: str | None = None  # "LONG" or "SHORT"
    entry_price: float = 0.0
    entry_time: pd.Timestamp | None = None
    bars_in_trade: int = 0

    stop_price: float = 0.0
    tp_price: float = 0.0

    # Debug counters
    bars_with_atr = 0
    long_signals = 0
    short_signals = 0
    hold_signals = 0

    # We iterate over rows in time order
    for ts, row in df.iterrows():
        row_dict: Dict[str, Any] = row.to_dict()

        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row.get("atr_bt", np.nan))

        # Skip if ATR not ready yet
        if not np.isfinite(atr) or atr <= 0:
            continue

        bars_with_atr += 1

        # Strategy signal is based on the full feature row
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
                else:  # SHORT
                    stop_price = entry_price + STOP_ATR_MULT * atr
                    tp_price = entry_price - TP_ATR_MULT * atr

        else:
            # There is an open position; manage risk and potential exit
            bars_in_trade += 1
            exit_now = False
            exit_reason = "none"
            exit_price = close

            if current_side == "LONG":
                # 1) Check STOP first
                if low <= stop_price:
                    exit_price = stop_price
                    exit_now = True
                    exit_reason = "SL"
                # 2) Check TP
                elif high >= tp_price:
                    exit_price = tp_price
                    exit_now = True
                    exit_reason = "TP"
            else:  # SHORT
                if high >= stop_price:
                    exit_price = stop_price
                    exit_now = True
                    exit_reason = "SL"
                elif low <= tp_price:
                    exit_price = tp_price
                    exit_now = True
                    exit_reason = "TP"

            # 3) Max bars in trade (time stop)
            if not exit_now and bars_in_trade >= MAX_HOLD_BARS:
                exit_now = True
                exit_reason = "TIME"

            # 4) Optional: flip on hard opposite signal
            if not exit_now and signal in ("LONG", "SHORT") and signal != current_side:
                exit_now = True
                exit_reason = "FLIP"

            if exit_now:
                # Realize PnL
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

                # Reset position
                current_side = None
                entry_price = 0.0
                entry_time = None
                bars_in_trade = 0
                stop_price = 0.0
                tp_price = 0.0

    # 3B) If a trade is still open at the very end, close it at last close
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

    # 4) Print stats
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
    gross_loss = sum(t.pnl for t in losses)  # negative
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

    # 5) Save trades to CSV
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


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    run_backtest()
