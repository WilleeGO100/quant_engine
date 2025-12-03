# main_live_spy.py
#
# Live runner using SPY 5m candles from TwelveData REST.
# Pipeline:
#   SPY live bar -> simple rule-based signal -> LLMApprovalAgent
#   -> RiskManager -> TradersPostExecutor
#
# This does NOT use the SMC PO3 strategy yet.
# It is meant to prove that live data can drive the
# LLM + risk + TradersPost chain end-to-end.
#
# Run from project root:
#   cd C:\Python312\quant_engine
#   python main_live_spy.py

from __future__ import annotations

import sys
import traceback
from datetime import datetime

from config.config import load_config
from engine.data.twelvedata_client import TwelveDataClient
from engine.agent.llm_approval import LLMApprovalAgent
from engine.risk.risk_manager import RiskManager
from engine.execution.traderspost_executor import TradersPostExecutor


def build_state_stub() -> dict:
    """
    Temporary placeholder for account/position state.

    Later we can replace this with a real persistence layer.
    For now, assume flat and zero PnL.
    """
    return {
        "open_position": None,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "daily_realized_pnl": 0.0,
        "trades_today": 0,
    }


def decide_from_spy_bar(bar: dict) -> str:
    """
    Simple rule-based signal from a single SPY 5m bar.

    - close > open  -> LONG
    - close < open  -> SHORT
    - close == open -> HOLD
    """
    c = bar["close"]
    o = bar["open"]

    if c > o:
        return "LONG"
    elif c < o:
        return "SHORT"
    else:
        return "HOLD"


def build_raw_decision(signal: str, cfg, spy_bar: dict) -> dict | None:
    """
    Wrap the raw signal into the decision dict expected by
    LLMApprovalAgent and RiskManager.

    If the signal is HOLD, return None (do nothing).
    """
    signal = (signal or "").upper().strip()
    if signal not in ("LONG", "SHORT"):
        print(f"[LIVE_SPY] Non-trade signal: {signal!r} -> no trade.")
        return None

    raw_decision = {
        "signal": signal,
        "size": float(getattr(cfg, "default_position_size", 1.0)),
        "stop_loss": None,
        "take_profit": None,
        "meta": {
            "source": "main_live_spy",
            "trade_symbol": getattr(cfg, "symbol", "MNQ"),  # what we tell TradersPost
            "feed_symbol": spy_bar.get("symbol", "SPY"),    # live data source
            "spy_timestamp": spy_bar.get("timestamp"),
            "spy_open": spy_bar.get("open"),
            "spy_high": spy_bar.get("high"),
            "spy_low": spy_bar.get("low"),
            "spy_close": spy_bar.get("close"),
            "spy_volume": spy_bar.get("volume"),
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
    }
    print(f"[LIVE_SPY] Raw decision: {raw_decision}")
    return raw_decision


def create_llm_agent(cfg) -> LLMApprovalAgent:
    """
    Build the LLMApprovalAgent from config.
    """
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


def run_pipeline_once() -> None:
    """
    Single pass of the live SPY pipeline:
      1) Load config
      2) Fetch latest SPY 5m bar from TwelveData
      3) Generate simple LONG/SHORT/HOLD signal
      4) LLM approval
      5) Risk manager
      6) TradersPostExecutor
    """
    print("============================================")
    print("[LIVE_SPY] Starting live SPY pipeline run...")
    print("============================================")

    # 1) Load config
    cfg = load_config()
    print(f"[LIVE_SPY] Config loaded: "
          f"trade_symbol={cfg.symbol}, "
          f"feed_symbol={getattr(cfg, 'twelvedata_symbol', 'SPY')}, "
          f"account_size={cfg.account_size}, "
          f"llm_mode={cfg.llm_mode}, "
          f"live_trading={cfg.live_trading}")

    api_key = getattr(cfg, "twelvedata_api_key", "")
    feed_symbol = getattr(cfg, "twelvedata_symbol", "SPY")

    if not api_key:
        print("[LIVE_SPY] ERROR: twelvedata_api_key is empty in settings.yaml")
        return

    # 2) Fetch latest SPY 5m bar
    client = TwelveDataClient(api_key=api_key)
    spy_bar = client.fetch_last_bar(symbol=feed_symbol)

    if spy_bar is None:
        print("[LIVE_SPY] ERROR: Could not fetch live SPY bar.")
        return

    spy_bar["symbol"] = feed_symbol

    print("[LIVE_SPY] Latest SPY bar:")
    for k, v in spy_bar.items():
        print(f"  {k}: {v}")

    # 3) Strategy decision from SPY bar
    signal = decide_from_spy_bar(spy_bar)
    print(f"[LIVE_SPY] Simple rule signal: {signal}")

    raw_decision = build_raw_decision(signal, cfg, spy_bar)
    if raw_decision is None:
        print("[LIVE_SPY] No actionable signal -> exit.")
        return

    # 4) LLM approval
    llm_agent = create_llm_agent(cfg)
    approved = llm_agent.approve(raw_decision)
    print(f"[LIVE_SPY] After LLM approval: {approved}")

    if not approved or (approved.get("signal", "HOLD").upper() == "HOLD"):
        print("[LIVE_SPY] LLM returned HOLD / no trade -> exit.")
        return

    # 5) Risk manager
    risk = RiskManager(config=cfg)
    current_state = build_state_stub()
    post_risk = risk.apply(approved, current_state=current_state)
    print(f"[LIVE_SPY] After RiskManager.apply(): {post_risk}")

    if not post_risk or (post_risk.get("signal", "HOLD").upper() == "HOLD"):
        print("[LIVE_SPY] RiskManager blocked or neutralized trade -> exit.")
        return

    # 6) TradersPost execution
    executor = TradersPostExecutor(config=cfg)
    is_live = bool(getattr(cfg, "live_trading", False))
    test_flag = not is_live  # test=True unless live_trading=True

    print(f"[LIVE_SPY] Sending order to TradersPost "
          f"(test={test_flag}, live_trading={is_live})...")

    # Build a generic order payload for TradersPostExecutor.
    order_payload = {
        "signal": post_risk["signal"],
        "size": post_risk.get("size", cfg.default_position_size),
        "test": test_flag,
        "meta": post_risk.get("meta", {}),
    }

    executor.send_order(order_payload)

    print("[LIVE_SPY] Order sent (check TradersPost dashboard / logs).")
    print("============================================")
    print("[LIVE_SPY] Pipeline run finished.")
    print("============================================")


if __name__ == "__main__":
    try:
        run_pipeline_once()
    except Exception as e:
        print("[LIVE_SPY] FATAL ERROR in main_live_spy.py")
        print(e)
        traceback.print_exc()
        sys.exit(1)
