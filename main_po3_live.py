# main_po3_live.py
#
# Dry-run "live" loop for the POWER SMC PO3 strategy.
# Uses the same multiframe ES features as the backtests,
# but routes trades through an LLMApprovalAgent + executor.

from __future__ import annotations

from typing import Dict, Any, Optional

import pandas as pd

from engine.features.es_multiframe_features import build_es_5m_multiframe_features
from engine.strategies.smc_po3_power import SMCPo3PowerStrategy
from engine.agent.llm_approval import LLMApprovalAgent


# ----------------------------------------------------------------------
# 1. Simple "executor" stub
# ----------------------------------------------------------------------
def send_order(order: Dict[str, Any]) -> None:
    """
    Execution stub.

    For now this *only prints* the order. Later you can:
      - POST to TradersPost webhook
      - Call a broker REST API
      - Write to a message queue, etc.
    """
    print("\n[EXECUTOR] Would send order:")
    for k, v in order.items():
        print(f"  {k}: {v}")


# ----------------------------------------------------------------------
# 2. Load multiframe ES data
# ----------------------------------------------------------------------
def load_multiframe_es() -> pd.DataFrame:
    """
    Use your existing multiframe builder to get ES 5m + 1H + 2H features.
    """
    df = build_es_5m_multiframe_features(
        f_5m="es_5m_clean.csv",
        f_1h="es_1h_clean.csv",
        f_2h="es_2h_clean.csv",
    )

    if "time" not in df.columns:
        raise ValueError("Expected 'time' column in multiframe features")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 3. Live-style loop (over historical data for now)
# ----------------------------------------------------------------------
def run_live_sim() -> None:
    """
    Simulate a live auto-trading loop driven by the POWER SMC PO3 strategy.

    - Uses historical data (multiframe ES) as if it's live
    - On each bar:
        * strategy.on_bar(row) -> signal
        * strategy.get_last_reason() -> explanation
        * detect reversals (close + reverse)
        * ask LLMApprovalAgent to approve
        * if approved -> send_order(order_dict)
    """

    # Config you might later load from YAML
    config: Dict[str, Any] = {
        "symbol": "ES",          # logical symbol name
        "contract": "ESZ5",      # example; adjust as needed
        "size": 1,               # contracts per trade
        "llm": {
            "max_intraday_drawdown": None,  # e.g. -50.0 to turn on guardrail
        },
    }

    print("============================================")
    print("Running LIVE-SIM POWER SMC PO3 engine")
    print("Data source: multiframe ES features")
    print("Mode: dry-run (prints orders, does not trade)")
    print("============================================")

    df = load_multiframe_es()
    strategy = SMCPo3PowerStrategy()
    llm_agent = LLMApprovalAgent(config=config.get("llm", {}))

    # Live position state
    position_side: Optional[str] = None  # "LONG" or "SHORT"
    entry_price: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None
    bars_in_trade: int = 0
    equity: float = 0.0
    max_equity: float = 0.0

    for _, row in df.iterrows():
        price = float(row["close"])
        ts = row["time"]

        # --- step 1: strategy decision ---
        signal = strategy.on_bar(row)  # "LONG" / "SHORT" / "HOLD"
        reason = strategy.get_last_reason()

        # --- step 2: if no open position, maybe open one ---
        if position_side is None:
            if signal in ("LONG", "SHORT"):
                position_side = signal
                entry_price = price
                entry_time = ts
                bars_in_trade = 0

                # Build trade context for LLM (preview only)
                trade_context = {
                    "side": position_side,
                    "reason": reason,
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "current_drawdown": max_equity - equity,
                    "pnl_preview": 0.0,
                }

                approved, commentary = llm_agent.approve(trade_context)
                print(f"\n[LLM] {commentary}")

                if approved:
                    order = {
                        "action": "OPEN",
                        "side": position_side,
                        "symbol": config["symbol"],
                        "contract": config["contract"],
                        "size": config["size"],
                        "entry_price": entry_price,
                        "time": entry_time,
                        "strategy_reason": reason,
                    }
                    send_order(order)
                else:
                    print("[ENGINE] LLM vetoed opening trade.")
                    position_side = None
                    entry_price = None
                    entry_time = None
                    bars_in_trade = 0

            continue  # done with this bar

        # --- we have an open position ---
        bars_in_trade += 1

        reverse_long_to_short = position_side == "LONG" and signal == "SHORT"
        reverse_short_to_long = position_side == "SHORT" and signal == "LONG"

        if reverse_long_to_short or reverse_short_to_long:
            # Close existing position
            exit_price = price
            exit_time = ts

            if position_side == "LONG":
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price

            equity += pnl
            max_equity = max(max_equity, equity)

            # Build context for LLM approval
            trade_context = {
                "side": f"CLOSE_{position_side}_AND_OPEN_{signal}",
                "reason": reason,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "bars": bars_in_trade,
                "pnl_preview": pnl,
                "current_drawdown": max_equity - equity,
            }

            approved, commentary = llm_agent.approve(trade_context)
            print(f"\n[LLM] {commentary}")

            if approved:
                # First: close current
                close_order = {
                    "action": "CLOSE",
                    "side": position_side,
                    "symbol": config["symbol"],
                    "contract": config["contract"],
                    "size": config["size"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "bars": bars_in_trade,
                    "pnl": pnl,
                    "strategy_reason": reason,
                }
                send_order(close_order)

                # Second: open new in the direction of the new signal
                open_order = {
                    "action": "OPEN",
                    "side": signal,
                    "symbol": config["symbol"],
                    "contract": config["contract"],
                    "size": config["size"],
                    "entry_price": exit_price,
                    "time": exit_time,
                    "strategy_reason": reason,
                }
                send_order(open_order)

                # Update live position state to the new side
                position_side = signal
                entry_price = exit_price
                entry_time = exit_time
                bars_in_trade = 0
            else:
                print("[ENGINE] LLM vetoed reversal; staying flat.")
                position_side = None
                entry_price = None
                entry_time = None
                bars_in_trade = 0

    print("\n============================================")
    print("LIVE-SIM finished.")
    print(f"Final equity (points): {equity:.2f}")
    print("============================================")


if __name__ == "__main__":
    run_live_sim()
