# live/main_live_btc_csv.py
from __future__ import annotations

import os
import csv
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable, List

import pandas as pd

from engine.features.btc_multiframe_features import build_btc_multiframe_features
from engine.strategies.smc_po3_power_btc import BTCPO3PowerStrategy, BTCPO3PowerConfig
strategy = BTCPO3PowerStrategy(BTCPO3PowerConfig())

# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")

BTC_FILE_NAME = os.path.join(DATA_DIR, "btc_5m.csv")
ORDERS_CSV = os.path.join(DATA_DIR, "paper_orders_btc.csv")
STATE_CSV = os.path.join(DATA_DIR, "paper_state_btc.csv")

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------
# Defaults / live parameters
# -----------------------------
EQUITY_USD_DEFAULT = 500.0
RISK_PCT_PER_TRADE_DEFAULT = 0.005  # 0.5%
MAX_OPEN_POSITIONS = 1

ATR_STOP_MULT_DEFAULT = 1.2
ATR_TP_MULT_DEFAULT = 3.0
MAX_HOLD_BARS_DEFAULT = 96
ATR_PERIOD_DEFAULT = 14

# Spot-safe mode
MODE_SPOT_LONG_ONLY = True


# -----------------------------
# State
# -----------------------------
def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_CSV):
        return {
            "position": "FLAT",
            "entry_price": None,
            "entry_time": None,
            "qty_btc": 0.0,
            "bars_in_trade": 0,
            "cooldown_remaining": 0,
            "last_bar_ts": None,
        }

    try:
        df = pd.read_csv(STATE_CSV)
        if df.empty:
            return {
                "position": "FLAT",
                "entry_price": None,
                "entry_time": None,
                "qty_btc": 0.0,
                "bars_in_trade": 0,
                "cooldown_remaining": 0,
                "last_bar_ts": None,
            }
        row = df.iloc[-1].to_dict()
        return {
            "position": str(row.get("position", "FLAT")),
            "entry_price": row.get("entry_price", None),
            "entry_time": row.get("entry_time", None),
            "qty_btc": float(row.get("qty_btc", 0.0) or 0.0),
            "bars_in_trade": int(row.get("bars_in_trade", 0) or 0),
            "cooldown_remaining": int(row.get("cooldown_remaining", 0) or 0),
            "last_bar_ts": row.get("last_bar_ts", None),
        }
    except Exception as e:
        logging.warning("Failed to load state CSV (%s). Using default.", e)
        return {
            "position": "FLAT",
            "entry_price": None,
            "entry_time": None,
            "qty_btc": 0.0,
            "bars_in_trade": 0,
            "cooldown_remaining": 0,
            "last_bar_ts": None,
        }


def save_state(state: Dict[str, Any]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = [
        "position",
        "entry_price",
        "entry_time",
        "qty_btc",
        "bars_in_trade",
        "cooldown_remaining",
        "last_bar_ts",
    ]
    write_header = not os.path.exists(STATE_CSV)
    with open(STATE_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({
            "position": state.get("position", "FLAT"),
            "entry_price": state.get("entry_price", None),
            "entry_time": state.get("entry_time", None),
            "qty_btc": state.get("qty_btc", 0.0),
            "bars_in_trade": state.get("bars_in_trade", 0),
            "cooldown_remaining": state.get("cooldown_remaining", 0),
            "last_bar_ts": state.get("last_bar_ts", None),
        })


def log_order(timestamp: str, action: str, side: str, qty_btc: float, reason: str,
              extra: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ["timestamp", "action", "side", "qty_btc", "reason", "extra_json"]
    write_header = not os.path.exists(ORDERS_CSV)

    extra_json = ""
    if extra:
        try:
            import json
            extra_json = json.dumps(extra, separators=(",", ":"))
        except Exception:
            extra_json = str(extra)

    with open(ORDERS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({
            "timestamp": timestamp,
            "action": action,
            "side": side,
            "qty_btc": f"{qty_btc:.8f}",
            "reason": reason,
            "extra_json": extra_json
        })


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None


def _coerce_utc_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _format_reason(parts: List[str]) -> str:
    parts = [p for p in parts if p]
    return " | ".join(parts) if parts else ""


def _get_float_attr(obj: object, names: Iterable[str], default: float) -> float:
    for name in names:
        if hasattr(obj, name):
            try:
                v = getattr(obj, name)
                if v is None:
                    continue
                return float(v)
            except Exception:
                pass
    return float(default)


def _get_int_attr(obj: object, names: Iterable[str], default: int) -> int:
    for name in names:
        if hasattr(obj, name):
            try:
                v = getattr(obj, name)
                if v is None:
                    continue
                return int(v)
            except Exception:
                pass
    return int(default)


def _resolve_atr_multipliers(config: BTCPO3PowerConfig) -> tuple[float, float]:
    stop_mult = _get_float_attr(
        config,
        ["atr_stop_mult", "stop_atr_mult", "stop_mult", "atr_mult_stop", "atr_stop"],
        ATR_STOP_MULT_DEFAULT
    )
    tp_mult = _get_float_attr(
        config,
        ["atr_tp_mult", "tp_atr_mult", "tp_mult", "atr_mult_tp", "atr_tp"],
        ATR_TP_MULT_DEFAULT
    )
    return float(stop_mult), float(tp_mult)


def _resolve_max_hold_bars(config: BTCPO3PowerConfig) -> int:
    return _get_int_attr(config, ["max_hold_bars", "max_hold", "hold_bars", "max_bars_in_trade"], MAX_HOLD_BARS_DEFAULT)


def _resolve_atr_period(config: BTCPO3PowerConfig) -> int:
    return _get_int_attr(config, ["atr_period", "atr_len", "atr_window"], ATR_PERIOD_DEFAULT)


@dataclass
class RiskParams:
    equity_usd: float
    risk_pct_per_trade: float
    max_open_positions: int


def _calc_qty_btc(equity_usd: float, risk_pct: float, entry_price: float, stop_price: float) -> float:
    risk_dollars = max(0.0, float(equity_usd) * float(risk_pct))
    stop_dist = abs(float(entry_price) - float(stop_price))
    if stop_dist <= 0:
        return 0.0
    qty = risk_dollars / stop_dist
    return float(qty) if qty > 0 else 0.0


def _normalize_decision(x: str) -> str:
    d = (x or "").strip().upper()
    if d in ["LONG", "ENTER_LONG", "BUY", "ENTER"]:
        return "LONG"
    if d in ["SHORT", "ENTER_SHORT", "SELL"]:
        return "SHORT"
    if d in ["EXIT", "CLOSE", "FLAT", "EXIT_LONG", "EXIT_SHORT"]:
        return "EXIT"
    return "HOLD"


def _extract_bar_ts_utc(df_feat: pd.DataFrame, df_raw: pd.DataFrame, raw_time_col: str) -> pd.Timestamp:
    for c in ["timestamp", "time", "datetime", "date"]:
        if c in df_feat.columns:
            ts = pd.to_datetime(df_feat.iloc[-1].get(c), utc=True, errors="coerce")
            if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
                return ts

    if isinstance(df_feat.index, pd.DatetimeIndex):
        ts = df_feat.index[-1]
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        return ts

    return pd.to_datetime(df_raw[raw_time_col].iloc[-1], utc=True, errors="coerce")


def _get_first_float(last_row: pd.Series, candidates: List[str]) -> Optional[float]:
    for c in candidates:
        if c in last_row.index:
            v = _safe_float(last_row.get(c))
            if v is not None:
                return v
    return None


def _make_reasons_diag(config: BTCPO3PowerConfig,
                       atr_pct: Optional[float],
                       rvol: Optional[float],
                       vwap_dist_atr: Optional[float]) -> List[str]:
    reasons: List[str] = []

    min_rvol = getattr(config, "min_rvol", None)
    max_rvol = getattr(config, "max_rvol", None)
    min_atr_pct = getattr(config, "min_atr_pct", None)
    max_atr_pct = getattr(config, "max_atr_pct", None)

    # ATR%
    if atr_pct is None:
        reasons.append("MISSING_ATR_PCT")
    else:
        if min_atr_pct is not None and atr_pct < float(min_atr_pct):
            reasons.append(f"ATR_PCT_TOO_LOW({atr_pct:.4f} < {float(min_atr_pct):.4f})")
        elif max_atr_pct is not None and atr_pct > float(max_atr_pct):
            reasons.append(f"ATR_PCT_TOO_HIGH({atr_pct:.4f} > {float(max_atr_pct):.4f})")
        else:
            reasons.append(f"ATR_PCT({atr_pct:.4f})")

    # RVOL
    if rvol is None:
        reasons.append("MISSING_RVOL")
    else:
        if min_rvol is not None and rvol < float(min_rvol):
            reasons.append(f"RVOL_TOO_LOW({rvol:.3f} < {float(min_rvol):.3f})")
        elif max_rvol is not None and rvol > float(max_rvol):
            reasons.append(f"RVOL_TOO_HIGH({rvol:.3f} > {float(max_rvol):.3f})")
        else:
            reasons.append(f"RVOL({rvol:.3f})")

    # VWAP distance in ATR units (this is what your code computes)
    if vwap_dist_atr is None:
        reasons.append("MISSING_VWAP_DIST_ATR")
    else:
        reasons.append(f"VWAP_DIST_ATR({vwap_dist_atr:.3f})")

    return reasons


# -----------------------------
# Main
# -----------------------------
def main(
    run_once: bool = True,
    poll_seconds: float = 2.0,
    equity_usd: float = EQUITY_USD_DEFAULT,
    risk_pct_per_trade: float = RISK_PCT_PER_TRADE_DEFAULT,
) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    logging.info("CSV Live Engine started. Using btc_5m.csv feed.")
    logging.info("Paper state file: %s", STATE_CSV)
    logging.info("Paper orders log: %s", ORDERS_CSV)

    config = BTCPO3PowerConfig()
    strategy = BTCPO3PowerStrategy(config)

    stop_mult, tp_mult = _resolve_atr_multipliers(config)
    max_hold_bars = _resolve_max_hold_bars(config)
    atr_period = _resolve_atr_period(config)

    logging.info("ATR multipliers: stop=%.3f tp=%.3f", stop_mult, tp_mult)
    logging.info("MODE: %s", "SPOT LONG/FLAT ONLY (no ENTER SHORT)" if MODE_SPOT_LONG_ONLY else "FULL LONG/SHORT")
    logging.info("Config knobs present: min_rvol=%s max_rvol=%s min_atr_pct=%s max_atr_pct=%s",
                 hasattr(config, "min_rvol"), hasattr(config, "max_rvol"),
                 hasattr(config, "min_atr_pct"), hasattr(config, "max_atr_pct"))

    risk = RiskParams(
        equity_usd=float(equity_usd),
        risk_pct_per_trade=float(risk_pct_per_trade),
        max_open_positions=MAX_OPEN_POSITIONS
    )

    last_seen_bar_ts: Optional[pd.Timestamp] = None

    while True:
        if not os.path.exists(BTC_FILE_NAME):
            logging.warning("Missing BTC file: %s", BTC_FILE_NAME)
            if run_once:
                return
            time.sleep(poll_seconds)
            continue

        # ------------------------------------------------------------
        # Load BTC CSV
        # ------------------------------------------------------------
        df_raw = pd.read_csv(BTC_FILE_NAME)
        if df_raw.empty:
            logging.warning("BTC CSV is empty.")
            if run_once:
                return
            time.sleep(poll_seconds)
            continue

        # ------------------------------------------------------------
        # Resolve timestamp column
        # ------------------------------------------------------------
        raw_time_col = None
        for c in ["timestamp", "time", "datetime", "date"]:
            if c in df_raw.columns:
                raw_time_col = c
                break

        if raw_time_col is None:
            logging.error(
                "BTC CSV missing timestamp column. Columns=%s",
                list(df_raw.columns),
            )
            return

        # ------------------------------------------------------------
        # Parse + sort timestamps
        # ------------------------------------------------------------
        df_raw[raw_time_col] = _coerce_utc_datetime(df_raw[raw_time_col])
        df_raw = df_raw.dropna(subset=[raw_time_col]).sort_values(raw_time_col)

        # ------------------------------------------------------------
        # PERFORMANCE GUARD — limit history size
        # ------------------------------------------------------------
        # 8000 x 5m bars ≈ ~27 days (more than enough for all indicators)
        df_raw = df_raw.tail(8000).copy()

        # ------------------------------------------------------------
        # Detect new closed bar
        # ------------------------------------------------------------
        latest_ts = df_raw[raw_time_col].iloc[-1]
        if last_seen_bar_ts is not None and latest_ts == last_seen_bar_ts:
            if run_once:
                logging.info("No new bar. Exiting (run_once=True).")
                return
            time.sleep(poll_seconds)
            continue

        last_seen_bar_ts = latest_ts


        # Build features
        try:
            df_feat = build_btc_multiframe_features(df_raw, atr_period=atr_period)
        except TypeError:
            df_feat = build_btc_multiframe_features(df_raw)

        if df_feat.empty:
            logging.warning("Feature build produced empty frame.")
            if run_once:
                return
            time.sleep(poll_seconds)
            continue

        last_row = df_feat.iloc[-1]
        bar_ts_utc = _extract_bar_ts_utc(df_feat, df_raw, raw_time_col)

        state = load_state()
        if not os.path.exists(STATE_CSV):
            save_state(state)

        # Pull features with broad name coverage
        close = _get_first_float(last_row, ["close", "Close", "c"])
        atr_5m = _get_first_float(last_row, ["atr_5m", "atr", "atr14", "atr_14"])
        atr_pct = _get_first_float(last_row, ["atr_pct_5m", "atr_pct", "atrp", "atr_percent"])
        rvol = _get_first_float(last_row, ["rvol", "rvol_5m", "rvol5m", "rel_vol", "relative_volume"])
        vwap_abs_pct = _get_first_float(last_row, ["vwap_dist_abs_pct"])
        vwap_dist_abs_pct = _get_first_float(last_row, ["vwap_dist_abs_pct"])
        vwap_dist_pct = _get_first_float(last_row, ["vwap_dist_pct"])
        vwap_dist_atr = _get_first_float(last_row,
                                         ["vwap_dist_atr", "vwap_distance_atr", "dist_atr_vwap", "vwap_atr_dist"])

        # ✅ correct strategy call
        row_dict = {k: (v.item() if hasattr(v, "item") else v) for k, v in last_row.to_dict().items()}
        decision_raw = strategy.on_bar(row_dict)
        decision = _normalize_decision(decision_raw)

        reasons = _make_reasons_diag(config, atr_pct, rvol, vwap_dist_atr)

        pos = str(state.get("position", "FLAT")).upper()

        # VWAP distance (percent preferred)
        if vwap_abs_pct is not None:
            reasons.append(f"VWAP_ABS_PCT({vwap_abs_pct:.4f})")
        elif vwap_dist_pct is not None:
            reasons.append(f"VWAP_PCT({vwap_dist_pct:.4f})")
        elif vwap_dist_atr is not None:
            reasons.append(f"VWAP_DIST_ATR({vwap_dist_atr:.3f})")
        else:
            reasons.append("MISSING_VWAP_DIST")

        # Spot-safe mapping
        if MODE_SPOT_LONG_ONLY and decision == "SHORT":
            if pos == "LONG":
                decision = "EXIT"
                reasons = reasons + ["SPOT_MODE_EXIT_ON_SHORT"]
            else:
                decision = "HOLD"
                reasons = reasons + ["SPOT_MODE_BLOCK_SHORT_ENTRY"]

        # ATR levels (for sizing + logging)
        sl_price = None
        tp_price = None
        if close is not None and atr_5m is not None:
            sl_price = float(close) - float(stop_mult) * float(atr_5m)
            tp_price = float(close) + float(tp_mult) * float(atr_5m)

        # Determine action
        action = "HOLD"
        if decision == "LONG" and pos == "FLAT":
            action = "ENTER"
        elif decision == "EXIT" and pos == "LONG":
            action = "EXIT"

        if action == "ENTER":
            if close is None or sl_price is None:
                reasons = reasons + ["NO_PRICE_OR_SL"]
            else:
                qty = _calc_qty_btc(
                    equity_usd=risk.equity_usd,
                    risk_pct=risk.risk_pct_per_trade,
                    entry_price=float(close),
                    stop_price=float(sl_price)
                )
                if qty <= 0:
                    reasons = reasons + ["QTY_ZERO"]
                else:
                    state["position"] = "LONG"
                    state["entry_price"] = float(close)
                    state["entry_time"] = str(bar_ts_utc)
                    state["qty_btc"] = float(qty)
                    state["bars_in_trade"] = 0
                    state["last_bar_ts"] = str(bar_ts_utc)
                    save_state(state)

                    log_order(
                        timestamp=str(bar_ts_utc),
                        action="ENTER",
                        side="LONG",
                        qty_btc=float(qty),
                        reason=_format_reason(reasons),
                        extra={
                            "close": close,
                            "atr_5m": atr_5m,
                            "atr_pct": atr_pct,
                            "rvol": rvol,
                            "vwap_dist_atr": vwap_dist_atr,
                            "sl_price": sl_price,
                            "tp_price": tp_price,
                            "decision_raw": decision_raw,
                        }
                    )

        elif action == "EXIT":
            qty = float(state.get("qty_btc", 0.0) or 0.0)
            entry_price = state.get("entry_price", None)
            pnl = None
            if close is not None and entry_price is not None:
                try:
                    pnl = (float(close) - float(entry_price)) * qty
                except Exception:
                    pnl = None

            state["position"] = "FLAT"
            state["entry_price"] = None
            state["entry_time"] = None
            state["qty_btc"] = 0.0
            state["bars_in_trade"] = 0
            state["cooldown_remaining"] = 0
            state["last_bar_ts"] = str(bar_ts_utc)
            save_state(state)

            log_order(
                timestamp=str(bar_ts_utc),
                action="EXIT",
                side="LONG",
                qty_btc=float(qty),
                reason=_format_reason(reasons),
                extra={
                    "close": close,
                    "entry_price": entry_price,
                    "pnl": pnl,
                    "atr_5m": atr_5m,
                    "atr_pct": atr_pct,
                    "rvol": rvol,
                    "vwap_dist_atr": vwap_dist_atr,
                    "decision_raw": decision_raw,
                }
            )

        logging.info("[LIVE] %s | %s | Decision=%s | %s",
                     str(bar_ts_utc), state.get("position", "FLAT"), decision, _format_reason(reasons))

        if run_once:
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main(run_once=True)
