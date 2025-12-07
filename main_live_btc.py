# main_live_btc.py
#
# LIVE BTC PO3 pipeline:
#   - Build 5m BTC institutional features from local CSV
#   - Run BTCPO3PowerStrategy on the latest bar
#   - If signal is LONG/SHORT:
#         raw_decision -> LLMApprovalAgent -> RiskManager -> TradersPostExecutor
#   - With live_trading=False this is a DRY RUN (no real orders sent)

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from config.config import load_config
from engine.features.btc_multiframe_features import (
    build_btc_5m_multiframe_features_institutional,
)
from engine.strategies.smc_po3_power_btc import (
    BTCPO3PowerStrategy,
    BTCPO3PowerConfig,
)
from engine.agent.llm_approval import LLMApprovalAgent
from engine.risk.risk_manager import RiskManager
from engine.execution.traderspost_executor import TradersPostExecutor


def _print_btc_snapshot(row):
    """
    Pretty-print the latest BTC bar.

    Note: timestamp is the index (row.name), not a column.
    """
    ts = row.name  # this is the pandas Timestamp for the row

    print("[BTC_LIVE] Latest BTC bar snapshot (OHLC):")
    print(f"timestamp : {ts}")
    print(f"open      : {row['open']}")
    print(f"high      : {row['high']}")
    print(f"low       : {row['low']}")
    print(f"close     : {row['close']}")
    if "volume" in row.index:
        print(f"volume    : {row['volume']}")


def main() -> None:
    print("====================================================")
    print("[BTC_LIVE] Starting main_live_btc PO3 pipeline run...")
    print("====================================================")

    # --------------------------------------------------
    # 1) Load config
    # --------------------------------------------------
    cfg = load_config()
    print(
        f"[BTC_LIVE] Loaded config: trade_symbol={getattr(cfg, 'trade_symbol', 'BTCUSD')}, "
        f"account_size={getattr(cfg, 'account_size', 0)}, "
        f"llm_mode={getattr(cfg, 'llm_mode', 'off')}, "
        f"live_trading={getattr(cfg, 'live_trading', False)}, "
        f"feed_symbol={getattr(cfg, 'feed_symbol', 'BTCUSD')}"
    )

    # --------------------------------------------------
    # 2) Build BTC 5m multi-timeframe institutional features
    # --------------------------------------------------
    print("[BTC_LIVE] Building BTC 5m multi-timeframe institutional features...")
    df, n_rows = build_btc_5m_multiframe_features_institutional()
    print(f"[BTC_LIVE] Feature DataFrame size: {n_rows} rows")

    if n_rows == 0 or df is None:
        print("[BTC_LIVE] No BTC feature rows returned -> aborting.")
        print("====================================================")
        return

    # Grab latest row
    last = df.iloc[-1]
    _print_btc_snapshot(last)

    # --------------------------------------------------
    # 3) Prepare strategy + latest bar dict
    # --------------------------------------------------
    print("[BTC_LIVE] Running BTCPO3PowerStrategy.on_bar()...")

    # Use the same config that worked in backtests/paper engine
    strat_cfg = BTCPO3PowerConfig(
        # ATR% filter: ignore ultra-dead + insane-wick zones
        min_atr_pct=0.0008,
        max_atr_pct=0.06,

        # RVOL filter
        min_rvol=1.0,

        # Session filters
        allow_asia=True,
        allow_london=True,
        allow_ny=True,

        # Trend regime
        use_trend_regime=True,

        # Weekly location constraints
        min_week_pos_for_longs=0.20,
        max_week_pos_for_shorts=0.80,

        # Distance from VWAP in ATR units
        max_vwap_dist_atr_entry=2.0,

        # Sweep requirement currently off
        require_sweep=False,
        lookback_sweep_bars=5,

        # TURN ON VERBOSE so we see which filter fails on HOLD
        verbose=True,
    )
    strategy = BTCPO3PowerStrategy(config=strat_cfg)

    # Strategy expects a dict-like bar with some specific keys.
    last_dict: Dict[str, Any] = last.to_dict()

    # 1) ATR% of price: atr_pct_5m = atr_5m / close
    close_price = float(last["close"])
    atr_5m = float(last.get("atr_5m", 0.0))
    if close_price != 0.0 and atr_5m > 0.0:
        last_dict["atr_pct_5m"] = atr_5m / close_price
    else:
        # fallback; strategy will treat this as "average" vol
        last_dict["atr_pct_5m"] = 0.01

    # 2) Relative volume: rvol_5m (already produced by feature builder)
    #    If missing, default to 1.0 (normal volume)
    if "rvol_5m" not in last_dict:
        last_dict["rvol_5m"] = 1.0

    # 3) Session tag: strategy looks for "session_type"
    #    Our feature frame uses "session"
    if "session" in last.index:
        last_dict["session_type"] = last["session"]
    else:
        last_dict["session_type"] = "NY"

    # 4) Weekly position, VWAP distance, regime_trend_up should already exist
    #    but we make sure there is at least a safe fallback:
    last_dict.setdefault("week_pos", 0.5)          # mid-week
    last_dict.setdefault("vwap_dist_atr", 0.0)     # on VWAP
    last_dict.setdefault("regime_trend_up", True)  # trending up

    # --------------------------------------------------
    # 4) Strategy raw signal on latest bar
    # --------------------------------------------------
    signal = strategy.on_bar(last_dict)
    print(f"[BTC_LIVE] Strategy raw signal: {signal}")

    if signal not in ("LONG", "SHORT"):
        # HOLD diagnostics: show exactly what the PO3 filters are seeing
        print("[BTC_LIVE] Strategy returned 'HOLD' -> no trade.")
        print("[BTC_LIVE] HOLD diagnostics (key PO3 inputs):")
        print(f"  atr_pct_5m     : {last_dict.get('atr_pct_5m')}")
        print(f"  rvol_5m        : {last_dict.get('rvol_5m')}")
        print(f"  week_pos       : {last_dict.get('week_pos')}")
        print(f"  vwap_dist_atr  : {last_dict.get('vwap_dist_atr')}")
        print(f"  session_type   : {last_dict.get('session_type')}")
        print(f"  regime_trend_up: {last_dict.get('regime_trend_up')}")
        print("====================================================")
        return

    # --------------------------------------------------
    # 5) Build raw_decision object
    # --------------------------------------------------
    now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw_decision: Dict[str, Any] = {
        "signal": signal,
        "size": 0.01,  # default BTC position size for now
        "stop_loss": None,
        "take_profit": None,
        "meta": {
            "source": "main_live_btc",
            "trade_symbol": getattr(cfg, "trade_symbol", "BTCUSD"),
            "feed_symbol": getattr(cfg, "feed_symbol", "BTCUSD"),
            "generated_at": now_utc,
            "last_price": float(last["close"]),
            "session": last_dict.get("session_type", "NY"),
            "week_pos": float(last_dict.get("week_pos", 0.5)),
            "vwap_dist_atr": float(last_dict.get("vwap_dist_atr", 0.0)),
        },
    }

    print(f"[BTC_LIVE] Raw decision: {raw_decision}")

    # --------------------------------------------------
    # 6) LLM approval layer
    # --------------------------------------------------
    llm_mode = getattr(cfg, "llm_mode", "off")
    llm_api_key = getattr(cfg, "openai_api_key", "")
    llm_model = getattr(cfg, "llm_model", "gpt-4.1-mini")
    llm_temp = getattr(cfg, "llm_temperature", 0.1)

    llm_agent = LLMApprovalAgent(
        mode=llm_mode,
        api_key=llm_api_key,
        model=llm_model,
        temperature=llm_temp,
    )

    approved = llm_agent.approve(raw_decision)
    print(f"[BTC_LIVE] After LLM approval: {approved}")

    if not approved or approved.get("signal") not in ("LONG", "SHORT"):
        print(
            "[BTC_LIVE] LLM did not approve a trade (or returned HOLD) -> stopping."
        )
        print("====================================================")
        return

    # --------------------------------------------------
    # 7) RiskManager layer
    # --------------------------------------------------
    risk = RiskManager(config=cfg)
    current_state: Dict[str, Any] = {}  # extend later with PnL/position state

    post_risk = risk.apply(approved, current_state=current_state)
    print(f"[BTC_LIVE] After RiskManager.apply(): {post_risk}")

    if not post_risk or post_risk.get("signal") not in ("LONG", "SHORT"):
        print("[BTC_LIVE] RiskManager returned HOLD/None -> no trade sent.")
        print("====================================================")
        return

    # --------------------------------------------------
    # 8) TradersPostExecutor – send test order (DRY RUN while live_trading=False)
    # --------------------------------------------------
    executor = TradersPostExecutor(config=cfg)

    # For safety: always send as test orders from this script.
    # live_trading flag inside the executor still controls whether the webhook is sent.
    test_flag = True

    print(
        f"[BTC_LIVE] Sending BTC order to TradersPost "
        f"(test={test_flag}, live_trading={getattr(cfg, 'live_trading', False)})..."
    )

    executor.send_order(
        post_risk["signal"],
        post_risk["size"],
        test_flag,
        post_risk.get("meta", {}),
    )

    print("[BTC_LIVE] Done.")
    print("====================================================")


if __name__ == "__main__":
    main()
