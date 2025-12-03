# main_live.py
#
# LIVE-style runner for the POWER SMC PO3 strategy with live SPY overlay.
#
# Pipeline:
#   ES multi-timeframe features from CSV
#   + live SPY 5m bar from TwelveData (overlay)
#   -> SMCPo3PowerStrategy.on_bar()
#   -> LLMApprovalAgent
#   -> RiskManager
#   -> TradersPostExecutor
#
# ES PO3 features still drive the strategy logic.
# SPY fields are attached for context (LLM, risk, logging).
#
# Run one-shot:
#   cd C:\Python312\quant_engine
#   python main_live.py
#
# Continuous loop (recommended for forward test):
#   python main_live_loop.py

from __future__ import annotations

import sys
import traceback
from datetime import datetime

import pandas as pd

from config.config import load_config
from engine.agent.llm_approval import LLMApprovalAgent
from engine.risk.risk_manager import RiskManager
from engine.execution.traderspost_executor import TradersPostExecutor
from engine.strategies.smc_po3_power import SMCPo3PowerStrategy
from engine.features.live_po3_spy_builder import build_live_po3_bar


# ------------------------------------------------------------
# Live performance tracker
# ------------------------------------------------------------

class LivePerformanceTracker:
    """
    Simple forward-test tracker.

    - Tracks current side (LONG/SHORT/FLAT)
    - Tracks closed trades and realized PnL
    - Prints live stats whenever a trade closes (reverse) or opens.
    """

    def __init__(self) -> None:
        self.current_side: str = "FLAT"  # "LONG" | "SHORT" | "FLAT"
        self.entry_price: float | None = None
        self.entry_time: str | None = None
        self.trades: list[dict] = []
        self.realized_pnl: float = 0.0
        self.equity: float = 0.0
        self.max_drawdown: float = 0.0

    def _close_current_trade(self, exit_side: str, price: float, bar_time: str | None):
        if self.current_side not in ("LONG", "SHORT") or self.entry_price is None:
            return

        if self.current_side == "LONG":
            pnl = price - self.entry_price
        else:
            pnl = self.entry_price - price

        self.realized_pnl += pnl
        self.equity += pnl
        self.max_drawdown = min(self.max_drawdown, self.equity)

        trade = {
            "entry_side": self.current_side,
            "exit_side": exit_side,
            "entry_price": self.entry_price,
            "exit_price": price,
            "entry_time": self.entry_time,
            "exit_time": bar_time,
            "pnl": pnl,
        }
        self.trades.append(trade)

        print("[LIVE_STATS] Closed trade:",
              f"{trade['entry_side']} -> {trade['exit_side']} | "
              f"{trade['entry_price']} -> {trade['exit_price']} | "
              f"PnL={round(trade['pnl'], 2)}")

    def on_decision(self, bar: pd.Series, decision: dict) -> None:
        """
        Update tracker based on an approved (post-risk) decision.

        - Only reacts to LONG/SHORT.
        - If side changes, closes current trade and opens new one.
        """
        side = decision.get("signal", "").upper()
        if side not in ("LONG", "SHORT"):
            return

        price = float(bar.get("close", bar.get("last", 0.0)))
        bar_time = None
        for candidate in ("timestamp", "time", "datetime"):
            if candidate in bar.index:
                bar_time = str(bar[candidate])
                break

        if self.current_side == "FLAT":
            # Opening first trade
            self.current_side = side
            self.entry_price = price
            self.entry_time = bar_time
            print(f"[LIVE_STATS] Opened first trade: {side} @ {price}")
            return

        if side == self.current_side:
            # Same direction, ignore for now (no scale-in/out yet)
            return

        # Reversal: close previous, open new
        self._close_current_trade(exit_side=side, price=price, bar_time=bar_time)
        self.current_side = side
        self.entry_price = price
        self.entry_time = bar_time
        print(f"[LIVE_STATS] Reversed into: {side} @ {price}")

    def print_summary(self) -> None:
        n = len(self.trades)
        if n == 0:
            print("[LIVE_STATS] No closed trades yet.")
            return

        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        win_rate = 100.0 * wins / n
        print("--------------------------------------------------------")
        print("[LIVE_STATS] Forward-test summary so far:")
        print(f"  Trades: {n}")
        print(f"  Wins:   {wins} ({win_rate:.2f}%)")
        print(f"  PnL:    {round(self.realized_pnl, 2)}")
        print(f"  Max DD: {round(self.max_drawdown, 2)}")
        print("  Last trade:", self.trades[-1])
        print("--------------------------------------------------------")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def build_state_stub() -> dict:
    """Temporary placeholder for account/position state."""
    return {
        "open_position": None,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "daily_realized_pnl": 0.0,
        "trades_today": 0,
    }


def build_raw_decision(signal: str, cfg, bar: pd.Series) -> dict | None:
    """
    Wrap the strategy signal into the standard decision dict expected
    by LLMApprovalAgent and RiskManager.

    If signal is HOLD or invalid, returns None.
    """
    signal = (signal or "").upper().strip()
    if signal not in ("LONG", "SHORT"):
        print(f"[PO3_LIVE] Strategy returned non-trade signal: {signal!r} -> no trade.")
        return None

    # ES bar time
    bar_time = None
    for candidate in ("timestamp", "time", "datetime"):
        if candidate in bar.index:
            bar_time = str(bar[candidate])
            break

    # Optional SPY overlay info
    spy_timestamp = bar.get("spy_timestamp", None)
    spy_close = bar.get("spy_close", None)

    meta = {
        "source": "main_live_po3",
        "trade_symbol": getattr(cfg, "symbol", "MNQ"),
        "bar_time": bar_time,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    if spy_timestamp is not None:
        meta["spy_timestamp"] = spy_timestamp
    if spy_close is not None:
        meta["spy_close"] = spy_close
        meta["spy_symbol"] = bar.get("spy_symbol", getattr(cfg, "twelvedata_symbol", "SPY"))

    raw_decision = {
        "signal": signal,
        "size": float(getattr(cfg, "default_position_size", 1.0)),
        "stop_loss": None,
        "take_profit": None,
        "meta": meta,
    }
    print(f"[PO3_LIVE] Raw decision: {raw_decision}")
    return raw_decision


def create_llm_agent(cfg) -> LLMApprovalAgent:
    """Build the LLMApprovalAgent from config."""
    mode = getattr(cfg, "llm_mode", "off")
    api_key = getattr(cfg, "openai_api_key", "")
    model = getattr(cfg, "llm_model", "gpt-4.1-mini")
    temperature = float(getattr(cfg, "llm_temperature", 0.1))

    agent = LLMApprovalAgent(
        mode=mode,
        api_key=api_key,
        model=model,
        temperature=temperature,
    )
    return agent


# ------------------------------------------------------------
# Main pipeline (single run)
# ------------------------------------------------------------

def run_pipeline_once(tracker: LivePerformanceTracker | None = None) -> None:
    """
    Single pass of the PO3 live-style pipeline:
      1) Load config
      2) Build ES+SPY live PO3 bar
      3) Run SMCPo3PowerStrategy.on_bar()
      4) LLM approval
      5) Risk manager
      6) TradersPostExecutor
      7) (Optional) Update live performance tracker
    """
    print("============================================")
    print("[PO3_LIVE] Starting main_live PO3 pipeline run (ES+SPY)...")
    print("============================================")

    # 1) Load config
    cfg = load_config()
    print(
        f"[PO3_LIVE] Loaded config: trade_symbol={cfg.symbol}, "
        f"account_size={cfg.account_size}, "
        f"llm_mode={cfg.llm_mode}, "
        f"live_trading={cfg.live_trading}, "
        f"feed_symbol={getattr(cfg, 'twelvedata_symbol', 'SPY')}"
    )

    # 2) Build ES+SPY live PO3 bar
    try:
        bar, n_rows = build_live_po3_bar(cfg)
    except Exception as e:
        print("[PO3_LIVE] ERROR while building live PO3 bar:", e)
        traceback.print_exc()
        return

    print(f"[PO3_LIVE] ES feature DataFrame size: {n_rows} rows")
    print("[PO3_LIVE] Latest ES bar snapshot (OHLC):")
    try:
        print(bar[["open", "high", "low", "close"]])
    except Exception:
        print(bar)

    if "spy_close" in bar.index:
        print("[PO3_LIVE] Attached SPY overlay:")
        print(
            f"  spy_timestamp={bar.get('spy_timestamp')}, "
            f"spy_close={bar.get('spy_close')}, "
            f"spy_open={bar.get('spy_open')}, "
            f"spy_high={bar.get('spy_high')}, "
            f"spy_low={bar.get('spy_low')}, "
            f"spy_volume={bar.get('spy_volume')}"
        )

    # 3) Strategy decision via SMCPo3PowerStrategy
    strategy = SMCPo3PowerStrategy()
    signal = strategy.on_bar(bar)
    print(f"[PO3_LIVE] SMCPo3PowerStrategy.on_bar() signal: {signal}")

    raw_decision = build_raw_decision(signal, cfg, bar)
    if raw_decision is None:
        print("[PO3_LIVE] No actionable signal -> exit.")
        print("============================================")
        return

    # 4) LLM approval
    llm_agent = create_llm_agent(cfg)
    approved = llm_agent.approve(raw_decision)
    print(f"[PO3_LIVE] After LLM approval: {approved}")

    if not approved or (approved.get("signal", "HOLD").upper() == "HOLD"):
        print("[PO3_LIVE] LLM returned HOLD / no trade -> exit.")
        print("============================================")
        return

    # 5) Risk manager
    risk = RiskManager(config=cfg)
    current_state = build_state_stub()
    post_risk = risk.apply(approved, current_state=current_state)
    print(f"[PO3_LIVE] After RiskManager.apply(): {post_risk}")

    if not post_risk or (post_risk.get("signal", "HOLD").upper() == "HOLD"):
        print("[PO3_LIVE] RiskManager blocked or neutralized trade -> exit.")
        print("============================================")
        return

    # 6) TradersPost execution
    executor = TradersPostExecutor(config=cfg)
    is_live = bool(getattr(cfg, "live_trading", False))
    test_flag = not is_live  # test=True unless live_trading=True

    print(
        f"[PO3_LIVE] Sending order to TradersPost "
        f"(test={test_flag}, live_trading={is_live})..."
    )

    order_payload = {
        "signal": post_risk["signal"],
        "size": post_risk.get("size", cfg.default_position_size),
        "test": test_flag,
        "meta": post_risk.get("meta", {}),
    }

    executor.send_order(order_payload)

    # 7) Update performance tracker
    if tracker is not None:
        tracker.on_decision(bar, post_risk)
        tracker.print_summary()

    print("[PO3_LIVE] Order payload handed to TradersPostExecutor.")
    print("============================================")
    print("[PO3_LIVE] main_live PO3 pipeline run finished.")
    print("============================================")


if __name__ == "__main__":
    try:
        # One-shot run without external tracker
        run_pipeline_once()
    except Exception as e:
        print("[PO3_LIVE] FATAL ERROR in main_live.py")
        print(e)
        traceback.print_exc()
        sys.exit(1)
