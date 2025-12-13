from __future__ import annotations

import os
import sys
import uuid
import json
import logging
from typing import Optional, Dict, Any

import pandas as pd
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ORDERS_CSV = os.path.join(DATA_DIR, "paper_orders_btc.csv")
EXEC_LOG = os.path.join(DATA_DIR, "coinbase_exec_log.csv")
LAST_EXEC_JSON = os.path.join(DATA_DIR, "coinbase_last_exec.json")


def append_exec_log(row: dict) -> None:
    df = pd.DataFrame([row])
    header = not os.path.exists(EXEC_LOG)
    df.to_csv(EXEC_LOG, mode="a", header=header, index=False)


def _latest_action_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "action" not in df.columns or "timestamp" not in df.columns:
        raise RuntimeError(f"paper_orders_btc.csv missing required columns. Have: {df.columns.tolist()}")

    df = df[df["action"].astype(str).str.upper().str.startswith(("ENTER", "EXIT"))]
    if df.empty:
        return None

    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return None

    return df.iloc[-1]


def _map_intent_to_cb_side(action: str, side: str) -> str:
    action = action.upper().strip()
    side = side.upper().strip()

    if action == "ENTER":
        if side == "LONG":
            return "buy"
        if side == "SHORT":
            return "sell"
        raise RuntimeError(f"Unknown ENTER side: {side}")

    # EXIT_*
    if side == "LONG":
        return "sell"
    if side == "SHORT":
        return "buy"
    raise RuntimeError(f"Unknown EXIT side: {side}")


def _is_cdp_key(api_key: str) -> bool:
    return api_key.startswith("organizations/") and "/apiKeys/" in api_key


def _load_api_secret_from_env() -> str:
    secret_file = os.getenv("COINBASE_API_SECRET_FILE", "").strip()
    if secret_file:
        path = os.path.join(PROJECT_ROOT, secret_file) if not os.path.isabs(secret_file) else secret_file
        if not os.path.exists(path):
            raise RuntimeError(f"COINBASE_API_SECRET_FILE points to missing file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    return os.getenv("COINBASE_API_SECRET", "").strip()


def _load_last_exec() -> Optional[Dict[str, Any]]:
    if not os.path.exists(LAST_EXEC_JSON):
        return None
    try:
        with open(LAST_EXEC_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_last_exec(sig: Dict[str, Any]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LAST_EXEC_JSON, "w", encoding="utf-8") as f:
        json.dump(sig, f, indent=2, sort_keys=True)


def _make_intent_sig(ts: str, action: str, side: str, qty_btc: float) -> Dict[str, Any]:
    # Keep it simple + deterministic
    return {
        "timestamp": ts,
        "action": action.upper().strip(),
        "side": side.upper().strip(),
        "qty_btc": f"{qty_btc:.8f}",
    }


def _same_sig(a: Optional[Dict[str, Any]], b: Dict[str, Any]) -> bool:
    if not a:
        return False
    return (
        str(a.get("timestamp")) == str(b.get("timestamp"))
        and str(a.get("action")) == str(b.get("action"))
        and str(a.get("side")) == str(b.get("side"))
        and str(a.get("qty_btc")) == str(b.get("qty_btc"))
    )


def _execute_advanced_trade(api_key: str, api_secret_pem: str, product_id: str, cb_side: str, qty_btc: float, live_trading: bool) -> dict:
    try:
        from coinbase.rest import RESTClient
    except Exception as e:
        raise RuntimeError(
            "Missing Advanced Trade SDK. Install:\n"
            "  pip install coinbase-advanced-py\n"
            f"Import error: {e}"
        )

    client = RESTClient(api_key=api_key, api_secret=api_secret_pem)
    client_order_id = str(uuid.uuid4())

    if not live_trading:
        return {
            "live_trading": False,
            "would_send": True,
            "mode": "advanced_trade_sdk",
            "client_order_id": client_order_id,
            "product_id": product_id,
            "side": cb_side,
            "base_size": f"{qty_btc:.8f}",
        }

    # Note: Market BUY often wants quote_size; we approximate from current product price.
    if cb_side == "buy":
        product = client.get_product(product_id)
        price = float(product["price"])
        quote_size = str(max(1.0, qty_btc * price))  # $1 floor
        resp = client.market_order_buy(
            client_order_id=client_order_id,
            product_id=product_id,
            quote_size=quote_size,
        )
    else:
        resp = client.market_order_sell(
            client_order_id=client_order_id,
            product_id=product_id,
            base_size=f"{qty_btc:.8f}",
        )

    return {"live_trading": True, "mode": "advanced_trade_sdk", "response": resp}


def main():
    load_dotenv()

    api_key = os.getenv("COINBASE_API_KEY", "").strip()
    product_id = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD").strip()
    live_trading = os.getenv("COINBASE_LIVE_TRADING", "false").strip().lower() == "true"

    if not api_key:
        raise RuntimeError("COINBASE_API_KEY missing from .env")

    if not _is_cdp_key(api_key):
        raise RuntimeError(
            "COINBASE_API_KEY is not a CDP key (organizations/.../apiKeys/...).\n"
            "Your executor is in Advanced Trade mode; update the key."
        )

    api_secret_pem = _load_api_secret_from_env()
    if not api_secret_pem:
        raise RuntimeError(
            "Missing Advanced Trade secret.\n"
            "Set either:\n"
            "  COINBASE_API_SECRET_FILE=secrets/coinbase_private_key.pem\n"
            "or COINBASE_API_SECRET (inline PEM)."
        )

    if not os.path.exists(ORDERS_CSV):
        raise FileNotFoundError(f"Missing order intent file: {ORDERS_CSV}")

    df = pd.read_csv(ORDERS_CSV)
    row = _latest_action_row(df)
    if row is None:
        logging.info("No ENTER/EXIT actions found in %s", ORDERS_CSV)
        return

    action = str(row["action"]).upper().strip()
    side = str(row.get("side", "")).upper().strip()
    qty_btc = float(row.get("qty_btc", 0.0) or 0.0)
    ts = str(row.get("timestamp"))

    if qty_btc <= 0:
        raise RuntimeError(f"qty_btc <= 0 in latest intent: {qty_btc}")

    # DEDUP check
    new_sig = _make_intent_sig(ts, action, side, qty_btc)
    last_sig = _load_last_exec()
    if _same_sig(last_sig, new_sig):
        logging.info("DEDUP: latest intent already executed. sig=%s", new_sig)
        return

    cb_side = _map_intent_to_cb_side(action, side)

    logging.info(
        "Latest intent: ts=%s action=%s side=%s qty_btc=%.8f -> coinbase_side=%s live=%s",
        ts, action, side, qty_btc, cb_side, live_trading
    )

    result = _execute_advanced_trade(
        api_key=api_key,
        api_secret_pem=api_secret_pem,
        product_id=product_id,
        cb_side=cb_side,
        qty_btc=qty_btc,
        live_trading=live_trading,
    )

    # Save last executed signature ONLY if we got a clean 'would_send' or live response
    # (prevents freezing if it errors before sending)
    if result.get("would_send") or result.get("live_trading") is True:
        _save_last_exec(new_sig)

    append_exec_log({
        "timestamp": ts,
        "intent_action": action,
        "intent_side": side,
        "coinbase_side": cb_side,
        "qty_btc": f"{qty_btc:.8f}",
        "live_trading": str(live_trading),
        "mode": str(result.get("mode", "")),
        "response": str(result.get("response", result)),
    })

    logging.info("Execution result: %s", result)


if __name__ == "__main__":
    main()
