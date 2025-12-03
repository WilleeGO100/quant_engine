from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from config.config import load_config
from engine.agent.llm_approval import LLMApprovalAgent
from engine.risk.risk_manager import RiskManager
from engine.execution.traderspost_executor import TradersPostExecutor


def main():
    print("=== FORCING BTC LONG SIGNAL THROUGH PIPELINE ===")

    cfg = load_config()
    trade_symbol = getattr(cfg, "trade_symbol", getattr(cfg, "symbol", "BTCUSD"))
    feed_symbol = getattr(cfg, "feed_symbol", "BTCUSD")

    print(
        f"[BTC_TEST] Loaded config: trade_symbol={trade_symbol}, "
        f"account_size={cfg.account_size}"
    )

    # ------------------------------------------------------------------
    # 1) Build a fake raw_decision for BTC
    # ------------------------------------------------------------------
    now_utc = datetime.now(timezone.utc).isoformat()

    raw_decision = {
        "signal": "LONG",
        "size": float(cfg.default_position_size),
        "stop_loss": None,
        "take_profit": None,
        "meta": {
            "source": "test_btc_force_trade",
            "trade_symbol": trade_symbol,
            "feed_symbol": feed_symbol,
            "generated_at": now_utc,
        },
    }

    print(f"[BTC_TEST] Raw decision: {raw_decision}")

    # ------------------------------------------------------------------
    # 2) LLM approval
    # ------------------------------------------------------------------
    llm = LLMApprovalAgent(
        mode=cfg.llm_mode,
        api_key=cfg.openai_api_key,
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
    )

    print(
        f"[BTC_TEST] LLMApprovalAgent initialised with "
        f"mode={cfg.llm_mode}, model={cfg.llm_model}, "
        f"temperature={cfg.llm_temperature}"
    )

    approved = llm.approve(raw_decision)
    print(f"[BTC_TEST] After LLM approval: {approved}")

    if not approved or approved.get("signal", "HOLD") == "HOLD":
        print(
            "[BTC_TEST] LLM did not approve a trade (or returned HOLD) "
            "-> stopping before risk/execution."
        )
        return

    # ------------------------------------------------------------------
    # 3) Risk manager
    # ------------------------------------------------------------------
    risk = RiskManager(config=cfg)
    current_state = {
        "timestamp": now_utc,
        "equity": cfg.account_size,
    }

    post_risk = risk.apply(approved, current_state=current_state)
    print(f"[BTC_TEST] After RiskManager.apply(): {post_risk}")

    if not post_risk or post_risk.get("signal", "HOLD") == "HOLD":
        print("[BTC_TEST] Risk manager returned HOLD/None -> stopping.")
        return

    # ------------------------------------------------------------------
    # 4) TradersPost execution (test mode)
    # ------------------------------------------------------------------
    executor = TradersPostExecutor(config=cfg)
    test_flag = not bool(cfg.live_trading)

    print(
        f"[BTC_TEST] Sending BTC order to TradersPost "
        f"(test={test_flag}, live_trading={cfg.live_trading})..."
    )

    executor.send_order(
        post_risk["signal"],
        post_risk["size"],
        test_flag,
        post_risk.get("meta", {}),
    )

    print("[BTC_TEST] Done.")



if __name__ == "__main__":
    main()
