from __future__ import annotations

"""live/btc_po3_traderspost_df_loop.py

LIVE BTC PO3 ENGINE (DataFrame pipeline + LLM + simple risk)
------------------------------------------------------------
- Fetches BTC/USD 5m candles from TwelveData
- Maintains an in-memory rolling 5m OHLCV DataFrame
- Builds institutional multi-timeframe features directly from that DataFrame
- Runs BTCPO3PowerStrategy on the latest bar
- Wraps the signal into a decision dict
- Sends decision through:
    1) LLMApprovalAgent (off / echo / live)
    2) SimpleRiskManager (max trades per day, etc.)
- If approved, sends LONG / SHORT to TradersPostExecutor (test mode by default)
- Prints live session statistics so you can see what's happening.
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta, date
from typing import Optional, Dict, Any

import pandas as pd

# ---------------------------------------------------------------------------
# 0. Ensure project root is on sys.path
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# 1. Local imports
# ---------------------------------------------------------------------------
from config.config import load_config
from engine.data.twelvedata_btc import (
    fetch_btcusd_5m_history,
    fetch_latest_btcusd_5m_bar,
)
from engine.features.btc_multiframe_features import (
    build_btc_5m_multiframe_features_institutional_from_df,
)
from engine.strategies.smc_po3_power_btc import (
    BTCPO3PowerStrategy,
    BTCPO3PowerConfig,
)
from engine.execution.traderspost_executor import TradersPostExecutor
from engine.agent.llm_approval import LLMApprovalAgent

# ---------------------------------------------------------------------------
# 2. Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# 3. Helper: seconds until next 5m bar close (UTC)
# ---------------------------------------------------------------------------
def _seconds_until_next_5m(now: Optional[datetime] = None) -> int:
    """Return seconds until the next 5-minute boundary (UTC)."""
    if now is None:
        now = datetime.now(timezone.utc)

    epoch = int(now.timestamp())
    rem = epoch % 300  # 300 seconds = 5 minutes
    wait = 300 - rem + 1
    return max(wait, 1)

# ---------------------------------------------------------------------------
# 4. Bridge feature row -> strategy input dict
# ---------------------------------------------------------------------------
def _bridge_row_for_strategy(row: pd.Series) -> Dict[str, Any]:
    """
    Convert the latest institutional feature row into the dict the strategy expects.
    We support both naming styles (atr_5m vs atr_14, rvol_5m vs rvol_20, etc.)
    to stay compatible with your backtest pipeline.
    """
    d = row.to_dict()

    close = float(d.get("close", float("nan")))

    # ATR: try both possible column names
    atr_val = d.get("atr_5m", d.get("atr_14", float("nan")))
    atr_val = float(atr_val) if atr_val == atr_val else float("nan")

    # RVOL: try both possible column names
    rvol_val = d.get("rvol_5m", d.get("rvol_20", float("nan")))
    rvol_val = float(rvol_val) if rvol_val == rvol_val else float("nan")

    # Week position: try week_pos or pos_in_week_range
    week_pos_val = d.get("week_pos", d.get("pos_in_week_range", float("nan")))
    week_pos_val = float(week_pos_val) if week_pos_val == week_pos_val else float("nan")

    vwap_dist_atr = float(d.get("vwap_dist_atr", float("nan")))
    session = d.get("session", d.get("session_type", None))
    regime_trend_up = d.get("regime_trend_up", 0)

    if close == close and atr_val == atr_val and close != 0.0:
        atr_pct_5m = atr_val / close
    else:
        atr_pct_5m = float("nan")

    bridged = dict(d)
    bridged["atr_pct_5m"] = atr_pct_5m
    bridged["rvol_5m"] = rvol_val
    bridged["week_pos"] = week_pos_val
    bridged["vwap_dist_atr"] = vwap_dist_atr
    bridged["session_type"] = session
    bridged["regime_trend_up"] = int(regime_trend_up)

    return bridged

# ---------------------------------------------------------------------------
# 5. Strategy evaluation helper
# ---------------------------------------------------------------------------
def _evaluate_latest_bar(
    feat_df: pd.DataFrame,
    strategy: BTCPO3PowerStrategy,
) -> Dict[str, Any]:
    """Evaluate the strategy on the most recent feature row."""
    if feat_df.empty:
        raise RuntimeError("Feature frame is empty – nothing to evaluate.")

    latest_idx = feat_df.index[-1]
    latest_row = feat_df.iloc[-1]

    row_dict = _bridge_row_for_strategy(latest_row)
    decision = strategy.on_bar(row_dict)

    result: Dict[str, Any] = {
        "timestamp": latest_idx,
        "raw_decision": decision,
        "row_dict": row_dict,
    }

    if isinstance(decision, str):
        result["side"] = decision
    elif isinstance(decision, dict):
        result["side"] = decision.get("side") or decision.get("action")
        result["context"] = {
            k: v for k, v in decision.items() if k not in ("side", "action")
        }
    else:
        result["side"] = str(decision)

    return result

# ---------------------------------------------------------------------------
# 6. Simple Risk Manager
# ---------------------------------------------------------------------------
class SimpleRiskManager:
    """
    Very basic live risk manager.

    For now it does:
    - max_trades_per_day: cap how many trades we allow per UTC calendar day.

    You can extend this later with:
    - daily PnL limits
    - max concurrent positions
    - kill switch, etc.
    """

    def __init__(self, max_trades_per_day: int = 10):
        self.max_trades_per_day = max_trades_per_day
        self.trades_per_day: Dict[date, int] = {}

    def _day_key(self, ts: datetime) -> date:
        # Use UTC date as the key
        return ts.date()

    def register_filled_trade(self, ts: datetime) -> None:
        key = self._day_key(ts)
        self.trades_per_day[key] = self.trades_per_day.get(key, 0) + 1

    def trades_today(self, ts: datetime) -> int:
        key = self._day_key(ts)
        return self.trades_per_day.get(key, 0)

    def allow_trade(self, ts: datetime) -> bool:
        """
        Return True if we are allowed to open a new trade at this timestamp.
        """
        current = self.trades_today(ts)
        if current >= self.max_trades_per_day:
            logging.info(
                "RISK: Max trades per day reached (%d) – blocking new trade.",
                self.max_trades_per_day,
            )
            return False
        return True

# ---------------------------------------------------------------------------
# 7. Live stats helpers
# ---------------------------------------------------------------------------
def _init_live_stats() -> Dict[str, Any]:
    return {
        "start_time": datetime.now(timezone.utc),
        "bars_processed": 0,
        "signals": {"LONG": 0, "SHORT": 0, "HOLD": 0, "OTHER": 0},
        "orders_sent": 0,
        "last_signal": None,
        "last_signal_ts": None,
        "last_order": None,
        "last_order_ts": None,
    }

def _update_stats_on_signal(stats: Dict[str, Any], side: str, ts: Any) -> None:
    stats["bars_processed"] += 1
    key = side if side in ("LONG", "SHORT", "HOLD") else "OTHER"
    stats["signals"][key] += 1
    stats["last_signal"] = side
    stats["last_signal_ts"] = ts

def _update_stats_on_order(stats: Dict[str, Any], side: str, size: float, ts: Any, meta: Dict[str, Any]) -> None:
    stats["orders_sent"] += 1
    stats["last_order"] = {"side": side, "size": size, "meta": meta}
    stats["last_order_ts"] = ts

def _format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def _print_live_stats(stats: Dict[str, Any]) -> None:
    now = datetime.now(timezone.utc)
    elapsed = now - stats["start_time"]

    bars = stats["bars_processed"]
    sigs = stats["signals"]
    orders = stats["orders_sent"]

    last_sig = stats["last_signal"]
    last_sig_ts = stats["last_signal_ts"]
    last_order = stats["last_order"]
    last_order_ts = stats["last_order_ts"]

    print("\n" + "=" * 55)
    print("==== BTC PO3 LIVE STATS (DF PIPELINE + LLM) ===============")
    print("=" * 55)
    print(f"Session started : {stats['start_time'].isoformat(timespec='seconds')}")
    print(f"Uptime          : {_format_timedelta(elapsed)}")
    print("-" * 55)
    print(f"Bars processed  : {bars}")
    print(f"Signals (LONG)  : {sigs['LONG']}")
    print(f"Signals (SHORT) : {sigs['SHORT']}")
    print(f"Signals (HOLD)  : {sigs['HOLD']}")
    print(f"Signals (OTHER) : {sigs['OTHER']}")
    print(f"Orders sent     : {orders}")
    print("-" * 55)

    if last_sig is not None:
        print(f"Last signal     : {last_sig} @ {last_sig_ts}")
    else:
        print("Last signal     : None yet")

    if last_order is not None:
        print(
            f"Last order      : {last_order['side']} {last_order['size']} "
            f"@ {last_order_ts} (test order payload sent)"
        )
    else:
        print("Last order      : None sent yet")

    print("=" * 55 + "\n")

# ---------------------------------------------------------------------------
# 8. Main live PO3 + LLM + Risk + TradersPost loop (DF-based)
# ---------------------------------------------------------------------------
def run_btc_po3_traderspost_df_loop() -> None:
    # A) Load config
    cfg = load_config()

    trade_symbol = getattr(cfg, "trade_symbol", "BTCUSD")
    feed_symbol = getattr(cfg, "feed_symbol", "BTCUSD")
    acct_size = getattr(cfg, "account_size", 0)
    live_trading = bool(getattr(cfg, "live_trading", False))
    default_size = float(getattr(cfg, "default_position_size", 0.01))

    llm_mode = getattr(cfg, "llm_mode", "off")
    llm_api_key = getattr(cfg, "openai_api_key", "")
    llm_model = getattr(cfg, "llm_model", "gpt-4.1-mini")
    llm_temperature = float(getattr(cfg, "llm_temperature", 0.1))

    logging.info(
        "BTC_PO3_DF: Loaded config | trade_symbol=%s | feed_symbol=%s | "
        "account_size=%s | live_trading=%s | default_size=%s | llm_mode=%s",
        trade_symbol,
        feed_symbol,
        acct_size,
        live_trading,
        default_size,
        llm_mode,
    )

    # B) Wire up executor, strategy, LLM, risk
    executor = TradersPostExecutor(config=cfg)

    strat_config = BTCPO3PowerConfig(
        min_atr_pct=0.0008,
        max_atr_pct=0.06,
        min_rvol=1.0,
        allow_asia=True,
        allow_london=True,
        allow_ny=True,
        use_trend_regime=True,
        min_week_pos_for_longs=0.20,
        max_week_pos_for_shorts=0.80,
        max_vwap_dist_atr_entry=2.0,
        require_sweep=False,
        lookback_sweep_bars=5,
        verbose=True,
    )
    strategy = BTCPO3PowerStrategy(config=strat_config)
    logging.info("BTC_PO3_DF: BTCPO3PowerStrategy instantiated.")

    llm_agent = LLMApprovalAgent(
        mode=llm_mode,
        api_key=llm_api_key,
        model=llm_model,
        temperature=llm_temperature,
    )
    logging.info("BTC_PO3_DF: LLMApprovalAgent initialised (mode=%s).", llm_mode)

    risk_manager = SimpleRiskManager(max_trades_per_day=10)

    stats = _init_live_stats()

    # C) Bootstrap recent BTC 5m prices from TwelveData into DF
    logging.info(
        "BTC_PO3_DF: Bootstrapping BTCUSD 5m history from TwelveData (outputsize=2000, DF pipeline)..."
    )

    prices = fetch_btcusd_5m_history(outputsize=2000)
    if prices is None or prices.empty:
        logging.error("BTC_PO3_DF: No history returned from TwelveData. Exiting.")
        return

    prices = prices.sort_index()
    logging.info(
        "BTC_PO3_DF: Bootstrapped %d 5m bars from %s to %s.",
        len(prices),
        prices.index[0],
        prices.index[-1],
    )

    feat_df, n_rows = build_btc_5m_multiframe_features_institutional_from_df(prices)
    if feat_df.empty or n_rows == 0:
        logging.error(
            "BTC_PO3_DF: Institutional DF feature builder returned empty frame. Exiting."
        )
        return

    last_ts = feat_df.index[-1]
    logging.info(
        "BTC_PO3_DF: Feature frame (DF) built with %d rows (up to %s).",
        n_rows,
        last_ts,
    )

    # D) Initial decision
    initial_result = _evaluate_latest_bar(feat_df, strategy)
    side0 = initial_result.get("side")
    ts0 = initial_result.get("timestamp")

    _update_stats_on_signal(stats, side0, ts0)
    logging.info(
        "BTC_PO3_DF: Initial PO3 decision -> ts=%s | side=%s | raw=%r",
        ts0,
        side0,
        initial_result.get("raw_decision"),
    )
    _print_live_stats(stats)

    # E) Main live loop
    logging.info(
        "BTC_PO3_DF: Starting live 5m PO3 loop (DF pipeline + LLM + Risk + TradersPost)..."
    )

    while True:
        # 1) Sleep until next 5m boundary
        sleep_s = _seconds_until_next_5m()
        logging.info(
            "BTC_PO3_DF: Sleeping %d seconds until next 5m bar close...",
            sleep_s,
        )
        time.sleep(sleep_s)

        # 2) Fetch latest bar(s)
        try:
            latest_df = fetch_latest_btcusd_5m_bar()
        except Exception as exc:
            logging.warning("BTC_PO3_DF: Error fetching latest bar: %s", exc)
            continue

        if latest_df is None or latest_df.empty:
            logging.warning("BTC_PO3_DF: No latest bar returned from TwelveData – skipping.")
            continue

        latest_df = latest_df.sort_index()
        latest_ts = latest_df.index[-1]

        if latest_ts <= last_ts:
            logging.info(
                "BTC_PO3_DF: No new 5m bar yet (latest=%s, last_seen=%s).",
                latest_ts,
                last_ts,
            )
            continue

        new_prices = latest_df[latest_df.index > last_ts]
        if new_prices.empty:
            continue

        # 3) Update rolling prices DF
        prices = pd.concat([prices, new_prices]).sort_index()
        if len(prices) > 2500:
            prices = prices.tail(2500)

        # 4) Rebuild institutional features from DF
        feat_df, n_rows = build_btc_5m_multiframe_features_institutional_from_df(prices)
        if feat_df.empty or n_rows == 0:
            logging.warning(
                "BTC_PO3_DF: DF feature builder returned empty frame after update – skipping."
            )
            continue

        last_ts = feat_df.index[-1]
        latest_row = feat_df.iloc[-1]

        # 5) Evaluate strategy on latest bar (raw PO3)
        result = _evaluate_latest_bar(feat_df, strategy)
        side = result.get("side")
        raw_decision = result.get("raw_decision")
        ts = result.get("timestamp")
        row_dict = result.get("row_dict", {})

        close = float(row_dict.get("close", float("nan")))
        atr_pct = float(row_dict.get("atr_pct_5m", float("nan")))
        rvol = float(row_dict.get("rvol_5m", float("nan")))
        week_pos = float(row_dict.get("week_pos", float("nan")))
        vwap_dist_atr = float(row_dict.get("vwap_dist_atr", float("nan")))
        session = row_dict.get("session_type", None)
        regime_up = row_dict.get("regime_trend_up", None)

        logging.info(
            "BTC_PO3_DF: [PO3] ts=%s | close=%.2f | atr_pct=%.4f | "
            "rvol=%.2f | week_pos=%.2f | vwap_dist_atr=%.2f | "
            "session=%s | trend_up=%s | side=%s | raw=%r",
            ts,
            close,
            atr_pct,
            rvol,
            week_pos,
            vwap_dist_atr,
            session,
            regime_up,
            side,
            raw_decision,
        )

        _update_stats_on_signal(stats, side, ts)

        # 6) If PO3 says HOLD/OTHER → no trade; just print stats
        if side not in ("LONG", "SHORT"):
            _print_live_stats(stats)
            continue

        # 7) Build raw decision dict for LLM + Risk
        now_utc_dt = datetime.now(timezone.utc)
        now_utc_str = now_utc_dt.isoformat(timespec="seconds")

        decision: Dict[str, Any] = {
            "signal": side,              # LONG / SHORT
            "size": default_size,        # e.g. 0.01 BTC
            "meta": {
                "source": "btc_po3_traderspost_df_loop",
                "trade_symbol": trade_symbol,
                "feed_symbol": feed_symbol,
                "generated_at": now_utc_str,
                "bar_timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
                "features": {
                    "close": close,
                    "atr_pct_5m": atr_pct,
                    "rvol_5m": rvol,
                    "week_pos": week_pos,
                    "vwap_dist_atr": vwap_dist_atr,
                    "session": session,
                    "regime_trend_up": regime_up,
                },
            },
        }

        # 8) LLM approval
        logging.info(
            "BTC_PO3_DF: Sending decision to LLMApprovalAgent (mode=%s)...", llm_mode
        )
        try:
            approved = llm_agent.approve(decision)
        except Exception as exc:
            logging.error("BTC_PO3_DF: Error in LLMApprovalAgent.approve: %s", exc)
            approved = None

        if not approved or approved.get("signal") not in ("LONG", "SHORT"):
            logging.info(
                "BTC_PO3_DF: LLMApprovalAgent vetoed or returned non-trade signal -> no order sent."
            )
            _print_live_stats(stats)
            continue

        final_side = approved.get("signal", side)
        final_size = float(approved.get("size", default_size))
        final_meta = approved.get("meta", decision["meta"])

        logging.info(
            "BTC_PO3_DF: LLM-approved trade -> side=%s size=%s",
            final_side,
            final_size,
        )

        # 9) Simple risk check
        if not risk_manager.allow_trade(now_utc_dt):
            logging.info(
                "BTC_PO3_DF: RiskManager blocked trade (max trades per day reached)."
            )
            _print_live_stats(stats)
            continue

        # 10) Send order to TradersPost (test mode by default)
        test_flag = True  # keep True until you're ready for real paper/live

        logging.info(
            "BTC_PO3_DF: Sending %s size=%s to TradersPost (test=%s, live_trading=%s)...",
            final_side,
            final_size,
            test_flag,
            live_trading,
        )

        try:
            executor.send_order(final_side, final_size, test_flag, final_meta)
            risk_manager.register_filled_trade(now_utc_dt)
            _update_stats_on_order(stats, final_side, final_size, ts, final_meta)
        except Exception as exc:
            logging.error("BTC_PO3_DF: Error sending order to TradersPost: %s", exc)

        _print_live_stats(stats)

# ---------------------------------------------------------------------------
# 9. Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_btc_po3_traderspost_df_loop()
