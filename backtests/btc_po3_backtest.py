from __future__ import annotations

# ==========================================
# IMPORTS
# ==========================================
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

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
# RISK / BACKTEST PARAMETERS
# ==========================================
STOP_ATR_MULT = 1.2       # tuned: tighter stop
TP_ATR_MULT = 2.5         # tuned: slightly closer TP
MAX_HOLD_BARS = 96        # max bars in trade (~8 hours on 5m)
UNIT_SIZE = 1.0           # 1-unit per trade for now
ATR_PERIOD = 14           # just for display (used in features)


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


def compute_streaks(trades: List[Trade]) -> Dict[str, int]:
    """
    Compute longest win streak and loss streak based on trade PnL.
    """
    best_win_streak = 0
    best_loss_streak = 0
    cur_win_streak = 0
    cur_loss_streak = 0

    for t in trades:
        if t.pnl > 0:
            cur_win_streak += 1
            best_win_streak = max(best_win_streak, cur_win_streak)
            cur_loss_streak = 0
        elif t.pnl < 0:
            cur_loss_streak += 1
            best_loss_streak = max(best_loss_streak, cur_loss_streak)
            cur_win_streak = 0
        else:
            # flat pnl does not extend streaks, but also does not break them
            pass

    return {
        "best_win_streak": best_win_streak,
        "best_loss_streak": best_loss_streak,
    }


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

    # 2) Instantiate strategy with tuned config
    config = BTCPO3PowerConfig(
        # Volatility band (ATR%)
        min_atr_pct=0.0008,      # ~0.08%
        max_atr_pct=0.06,        # 6%

        # Relative volume filter
        min_rvol=1.0,            # only normal+ volume

        # Session filters (BTC: keep Asia too)
        allow_asia=True,
        allow_london=True,
        allow_ny=True,

        # Trend regime currently off (PO3 structure already encoded)
        use_trend_regime=False,

        # Weekly position: only trade away from absolute extremes
        min_week_pos_for_longs=0.20,
        max_week_pos_for_shorts=0.80,

        # VWAP distance: within 2 ATRs of VWAP
        max_vwap_dist_atr_entry=2.0,

        # Sweep logic currently optional
        require_sweep=False,
        lookback_sweep_bars=5,

        verbose=False,
    )
    strategy = BTCPO3PowerStrategy(config=config)
    print("[BACKTEST] Strategy: BTCPO3PowerStrategy instantiated.")

    # 3) Iterate over bars, simulate trades
    trades: List[Trade] = []

    current_side: Optional[str] = None  # "LONG" or "SHORT"
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    bars_in_trade: int = 0

    stop_price: float = 0.0
    tp_price: float = 0.0

    bars_with_atr = 0
    long_signals = 0
    short_signals = 0
    hold_signals = 0

    last_ts: Optional[pd.Timestamp] = None
    last_close: Optional[float] = None

    for ts, row in df.iterrows():
        last_ts = ts
        last_close = float(row["close"])

        row_dict: Dict[str, Any] = row.to_dict()

        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        atr = float(row.get("atr_5m", np.nan))

        # Skip if ATR not ready yet
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

                # If exit_reason == "FLIP" and we got a new signal, we could
                # immediately open the opposite side here. For now, we wait
                # for next bar to avoid over-complication.

    # If we still have an open position at the very end, close it at last close
    if current_side is not None and last_ts is not None and last_close is not None and entry_time is not None:
        if current_side == "LONG":
            pnl = (last_close - entry_price) * UNIT_SIZE
        else:
            pnl = (entry_price - last_close) * UNIT_SIZE
        pnl_pct = pnl / entry_price if entry_price != 0 else 0.0

        trades.append(
            Trade(
                side=current_side,
                entry_time=entry_time,
                exit_time=last_ts,
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
    streaks = compute_streaks(trades)

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
    print()
    print(f"Best win streak : {streaks['best_win_streak']}")
    print(f"Worst loss streak: {streaks['best_loss_streak']}")

    # 5) Exit reason breakdown
    if trades:
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        print()
        print("Exit reason counts:")
        for reason, count in sorted(exit_reasons.items()):
            print(f"  {reason:5s}: {count}")

    # 6) Daily PnL summary
    if trades:
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

        trades_df["exit_date"] = trades_df["exit_time"].dt.date
        daily_pnl = trades_df.groupby("exit_date")["pnl"].sum().sort_index()

        best_day = daily_pnl.max()
        worst_day = daily_pnl.min()
        best_day_date = daily_pnl.idxmax()
        worst_day_date = daily_pnl.idxmin()

        print()
        print("Daily PnL:")
        for d, v in daily_pnl.items():
            print(f"  {d}: {v:.2f}")

        print()
        print(f"Best day        : {best_day_date} ({best_day:.2f})")
        print(f"Worst day       : {worst_day_date} ({worst_day:.2f})")

        # 7) Save trades to CSV
        trades_df.drop(columns=["exit_date"], inplace=True)
        trades_df.to_csv("btc_po3_trades.csv", index=False)
        print()
        print("[BACKTEST] Trade log saved to btc_po3_trades.csv")
    else:
        print()
        print("[BACKTEST] No trades generated, not writing CSV.")

    print("====================================================")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    run_backtest()
