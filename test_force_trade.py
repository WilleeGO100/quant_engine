# test_force_trade.py
"""
Quick harness to force a LONG trade through the full pipeline:

    synthetic decision -> LLMApprovalAgent -> RiskManager -> TradersPostExecutor

This is for **testing only**. It does NOT care about actual market data.
It just proves that the whole chain is wired correctly.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from config.config import load_config
from engine.agent.llm_approval import LLMApprovalAgent
from engine.execution.traderspost_executor import TradersPostExecutor
from engine.risk.risk_manager import RiskManager


def main() -> None:
    print("=== FORCING LONG SIGNAL THROUGH PIPELINE ===")

    # ------------------------------------------------------------------
    # 1) Load config
    # ------------------------------------------------------------------
    cfg = load_config()
    print(f"[TEST] Loaded config: symbol={cfg.symbol}, account_size={cfg.account_size}")

    # ------------------------------------------------------------------
    # 2) Build a synthetic "raw" strategy decision
    #    (this is what your strategy would normally output)
    # ------------------------------------------------------------------
    raw_decision: Dict[str, Any] = {
        "signal": "LONG",
        "size": 1.0,
        "stop_loss": None,
        "take_profit": None,
        "meta": {"source": "test_force_trade"},
    }
    print(f"[TEST] Raw decision: {json.dumps(raw_decision)}")

    # ------------------------------------------------------------------
    # 3) LLM approval layer
    #    mode comes from cfg.llm_mode ("echo" | "live" | "off")
    # ------------------------------------------------------------------
    llm_agent = LLMApprovalAgent(
        mode=cfg.llm_mode,
        api_key=getattr(cfg, "openai_api_key", None),
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
    )

    approved_by_llm = llm_agent.approve(raw_decision)
    print(f"[TEST] After LLM approval: {approved_by_llm}")

    if approved_by_llm is None or approved_by_llm.get("signal", "").upper() not in (
        "LONG",
        "SHORT",
    ):
        print("[TEST] LLM did not approve a trade (or returned HOLD) -> stopping.")
        return

    # ------------------------------------------------------------------
    # 4) Risk layer
    # ------------------------------------------------------------------
    risk = RiskManager(config=cfg)

    # In a real system this would come from your PnL tracker.
    # For now we keep it simple: no losses so far.
    current_state = {"daily_pnl": 0.0}

    approved_by_risk = risk.apply(approved_by_llm, current_state=current_state)
    if approved_by_risk is None:
        print("[TEST] RiskManager blocked the trade -> stopping.")
        return

    print(f"[TEST] After RiskManager: {approved_by_risk}")

    # ------------------------------------------------------------------
    # 5) TradersPost executor (still **test mode** = no real live order)
    # ------------------------------------------------------------------
    executor = TradersPostExecutor(config=cfg)

    print("[TEST] Sending TEST order to TradersPost (check their dashboard/logs).")
    response = executor.send_order(
        signal=approved_by_risk["signal"],
        size=approved_by_risk["size"],
        stop_loss=approved_by_risk.get("stop_loss"),
        take_profit=approved_by_risk.get("take_profit"),
        metadata=approved_by_risk.get("meta", {}),
        test=True,  # IMPORTANT: keep this True until you deliberately go live
    )

    print("[TEST] TradersPost response:")
    print(response)


if __name__ == "__main__":
    main()
