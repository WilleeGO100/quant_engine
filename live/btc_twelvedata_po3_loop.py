from __future__ import annotations
# live/btc_twelvedata_po3_loop.py
#
# Live BTC PO3 loop using TwelveData 5m candles:
#   - Bootstrap last N 5m BTC bars from TwelveData
#   - Save them to data/btc_5m.csv (for feature builder)
#   - Also save/append them to data/live_btc_5m_history.csv (permanent log)
#   - On every new closed 5m bar:
#         * fetch latest bar from TwelveData
#         * if it's new -> update history, rebuild institutional features
#         * bridge features into BTCPO3PowerStrategy.on_bar()
#         * send LONG/SHORT signals through LLMApprovalAgent + TradersPostExecutor

import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

# --------------------------------------------------
# Path setup (project root)
# --------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --------------------------------------------------
# Local imports
# --------------------------------------------------
from config.config import load_config  # returns Config dataclass

from engine.data.twelvedata_btc import (
    fetch_btcusd_5m_history,
    fetch_latest_btcusd_5m_bar,
)

from engine.features.btc_multiframe_features import (
    build_btc_5m_multiframe_features_institutional,
)

from engine.strategies.smc_po3_power_btc import (
    BTCPO3PowerStrategy,
    BTCPO3PowerConfig,
)

from engine.agent.llm_approval import LLMApprovalAgent
from engine.execution.traderspost_executor import TradersPostExecutor

# --------------------------------------------------
# File paths
# --------------------------------------------------
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_5M_CSV = os.path.join(DATA_DIR, "btc_5m.csv")
LIVE_HISTORY_CSV = os.path.join(DATA_DIR, "live_btc_5m_history.csv")


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _seconds_until_next_5m(now: Optional[datetime] = None) -> int:
    """
    Compute number of seconds until the next 5-minute bar close.

    Bars close at hh:00, 05, 10, 15, ..., 55 (UTC).
    We wake up just after, so we add +1s.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    epoch = int(now.timestamp())
    rem = epoch % 300  # 300s = 5 minutes
    wait = 300 - rem + 1
    return max(wait, 1)


def _write_raw_5m_csv(df: pd.DataFrame) -> None:
    """
    Save current 5m BTC history into data/btc_5m.csv.

    Assumes index is datetime.
    """
    out = df.copy().reset_index()
    if "datetime" in out.columns:
        out = out.rename(columns={"datetime": "timestamp"})
    else:
        out = out.rename(columns={out.columns[0]: "timestamp"})

    os.makedirs(os.path.dirname(RAW_5M_CSV), exist_ok=True)
    out.to_csv(RAW_5M_CSV, index=False)


def _append_to_live_history(new_bars: pd.DataFrame) -> None:
    """
    Append new bars to data/live_btc_5m_history.csv.

    Assumes index is datetime.
    """
    if new_bars.empty:
        return

    out = new_bars.copy().reset_index()
    if "datetime" in out.columns:
        out = out.rename(columns={"datetime": "timestamp"})
    else:
        out = out.rename(columns={out.columns[0]: "timestamp"})

    os.makedirs(os.path.dirname(LIVE_HISTORY_CSV), exist_ok=True)

    file_exists = os.path.exists(LIVE_HISTORY_CSV)
    header = not file_exists

    out.to_csv(LIVE_HISTORY_CSV, mode="a", header=header, index=False)


def _bridge_row_for_strategy(row: pd.Series) -> Dict[str, Any]:
    """
    Bridge institutional feature row into dict expected by BTCPO3PowerStrategy.on_bar().
    """
    d = row.to_dict()

    close = float(d.get("close", float("nan")))
    atr14 = float(d.get("atr_14", float("nan")))
    rvol20 = float(d.get("rvol_20", float("nan")))
    week_pos = float(d.get("pos_in_week_range", float("nan")))
    vwap_dist_atr = float(d.get("vwap_dist_atr", float("nan")))

    session = d.get("session", None)
    regime_trend = d.get("regime_trend", None)  # "UP" / "DOWN" / None

    if close == close and atr14 == atr14 and close != 0.0:
        atr_pct_5m = atr14 / close
    else:
        atr_pct_5m = float("nan")

    regime_trend_up = 1 if str(regime_trend).upper() == "UP" else 0

    bridged = dict(d)
    bridged["atr_pct_5m"] = atr_pct_5m
    bridged["rvol_5m"] = rvol20
    bridged["week_pos"] = week_pos
    bridged["vwap_dist_atr"] = vwap_dist_atr
    bridged["session_type"] = session
    bridged["regime_trend_up"] = regime_trend_up

    return bridged


# --------------------------------------------------
# Main live loop
# --------------------------------------------------
def run_btc_po3_twelvedata_loop() -> None:
    cfg = load_config()

    print(
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} [INFO] BTC_TWELVEDATA_PO3_LOOP: "
        f"Loaded config: trade_symbol={getattr(cfg, 'trade_symbol', 'BTCUSD')} | "
        f"feed_symbol={getattr(cfg, 'feed_symbol', 'BTCUSD')} | "
        f"account_size={getattr(cfg, 'account_size', 0)} | "
        f"llm_mode={getattr(cfg, 'llm_mode', 'off')} | "
        f"live_trading={getattr(cfg, 'live_trading', False)}"
    )

    # LLM + TradersPost wiring
    llm_agent = LLMApprovalAgent(
        mode=getattr(cfg, "llm_mode", "off"),
        api_key=getattr(cfg, "openai_api_key", ""),
        model=getattr(cfg, "llm_model", "gpt-4.1-mini"),
        temperature=getattr(cfg, "llm_temperature", 0.1),
    )

    executor = TradersPostExecutor(config=cfg)
    default_size = float(getattr(cfg, "default_position_size", 0.01))

    # Strategy config (aligned with your good backtest)
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

    # 1) Bootstrap history
    print(
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} [INFO] BTC_TWELVEDATA_PO3_LOOP: "
        "Bootstrapping BTCUSD 5m history from TwelveData..."
    )

    hist_df = fetch_btcusd_5m_history()
    if hist_df.empty:
        print(
            f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} [ERROR] BTC_TWELVEDATA_PO3_LOOP: "
            "No history returned from TwelveData. Exiting."
        )
        return

    hist_df = hist_df.sort_index()

    _write_raw_5m_csv(hist_df)
    _append_to_live_history(hist_df)

    feat_df, n_rows = build_btc_5m_multiframe_features_institutional()
    last_ts = feat_df.index[-1]

    print(
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} [INFO] BTC_TWELVEDATA_PO3_LOOP: "
        f"Bootstrapped {n_rows} bars up to {last_ts}."
    )

    # 2) Main loop
    print(
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} [INFO] BTC_TWELVEDATA_PO3_LOOP: "
        "Starting BTC 5m PO3 live loop (TwelveData)..."
    )

    while True:
        sleep_s = _seconds_until_next_5m()
        print(
            f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} [INFO] BTC_TWELVEDATA_PO3_LOOP: "
            f"Sleeping {sleep_s} seconds until next 5m bar close..."
        )
        time.sleep(sleep_s)

        latest_df = fetch_latest_btcusd_5m_bar()
        if latest_df is None or latest_df.empty:
            print(
                f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} [WARN] BTC_TWELVEDATA_PO3_LOOP: "
                "No latest bar returned from TwelveData."
            )
            continue

        latest_df = latest_df.sort_index()
        latest_ts = latest_df.index[-1]

        if latest_ts <= last_ts:
            print(
                f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} [INFO] BTC_TWELVEDATA_PO3_LOOP: "
                f"No new closed 5m bar yet (latest={latest_ts}, last_seen={last_ts})."
            )
            continue

        new_bars = latest_df[latest_df.index > last_ts]
        if new_bars.empty:
            continue

        hist_df = pd.concat([hist_df, new_bars]).sort_index()

        _write_raw_5m_csv(hist_df)
        _append_to_live_history(new_bars)

        feat_df, n_rows = build_btc_5m_multiframe_features_institutional()
        last_ts = feat_df.index[-1]
        last_row = feat_df.iloc[-1]

        row_dict = _bridge_row_for_strategy(last_row)

        close = float(row_dict.get("close", float("nan")))
        atr_pct = float(row_dict.get("atr_pct_5m", float("nan")))
        rvol = float(row_dict.get("rvol_5m", float("nan")))
        week_pos = float(row_dict.get("week_pos", float("nan")))
        vwap_dist_atr = float(row_dict.get("vwap_dist_atr", float("nan")))
        session = row_dict.get("session_type", None)
        regime_up = row_dict.get("regime_trend_up", None)

        signal = strategy.on_bar(row_dict)  # "LONG" / "SHORT" / "HOLD"

        print(
            f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} [INFO] BTC_TWELVEDATA_PO3_LOOP: "
            f"[LIVE] ts={last_ts} | close={close:.2f} | atr_pct={atr_pct:.4f} | "
            f"rvol={rvol:.2f} | week_pos={week_pos:.2f} | vwap_dist_atr={vwap_dist_atr:.2f} | "
            f"session={session} | trend_up={regime_up} | signal={signal}"
        )

        if signal not in ("LONG", "SHORT"):
            continue

        now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        decision: Dict[str, Any] = {
            "signal": signal,
            "size": default_size,
            "meta": {
                "source": "btc_twelvedata_po3_loop",
                "trade_symbol": getattr(cfg, "trade_symbol", "BTCUSD"),
                "feed_symbol": getattr(cfg, "feed_symbol", "BTCUSD"),
                "generated_at": now_utc,
                "bar_timestamp": last_ts.isoformat(),
            },
        }

        approved = llm_agent.approve(decision)
        if not approved or approved.get("signal") not in ("LONG", "SHORT"):
            print(
                f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} [INFO] BTC_TWELVEDATA_PO3_LOOP: "
                "LLM approval vetoed or returned HOLD -> no order sent."
            )
            continue

        test_flag = True  # keep True until fully confident
        meta = approved.get("meta", decision["meta"])

        print(
            f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} [INFO] BTC_TWELVEDATA_PO3_LOOP: "
            f"Sending {approved['signal']} size={approved.get('size', default_size)} to TradersPost "
            f"(test={test_flag}, live_trading={getattr(cfg, 'live_trading', False)})..."
        )

        executor.send_order(
            approved["signal"],
            float(approved.get("size", default_size)),
            test_flag,
            meta,
        )


if __name__ == "__main__":
    run_btc_po3_twelvedata_loop()
