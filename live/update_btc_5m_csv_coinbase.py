from __future__ import annotations

import os
import sys
import json
import ast
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_CSV = os.path.join(DATA_DIR, "btc_5m.csv")
DEBUG_JSON = os.path.join(DATA_DIR, "coinbase_candles_debug.json")


# -------------------------
# Time helpers (CLOSED bars)
# -------------------------
def _last_closed_5m_end_dt() -> datetime:
    """
    Returns the most recent CLOSED 5-minute boundary end timestamp (UTC),
    i.e. we never include a still-forming bar.

    Example: if now is 20:33:xx, last closed 5m end is 20:30:00.
    """
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    # floor to minute already, now floor to 5-min
    minute = (now.minute // 5) * 5
    floored = now.replace(minute=minute)
    # if we're exactly on boundary, that's "end of last bar"; still safe.
    return floored


def _to_epoch_seconds(dt: datetime) -> int:
    return int(dt.timestamp())


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


# -------------------------
# CSV IO + sanitization
# -------------------------
def _load_existing() -> pd.DataFrame:
    if not os.path.exists(OUT_CSV):
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.read_csv(OUT_CSV)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.columns = [c.strip().lower() for c in df.columns]
    if "timestamp" not in df.columns:
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def _sanitize_existing(df: pd.DataFrame, last_closed_end: datetime) -> pd.DataFrame:
    """
    Drop any rows that are in the future relative to the last closed 5m end.
    This repairs the CSV if a bad fetch previously wrote future timestamps.
    """
    if df.empty:
        return df

    # last closed bar timestamp is <= last_closed_end
    # our bar "timestamp" represents bar start time for most feeds,
    # but the safe rule is: never allow bars whose timestamp is > last_closed_end
    cutoff = pd.Timestamp(last_closed_end)
    before = len(df)
    df2 = df[df["timestamp"] <= cutoff].copy()
    after = len(df2)

    dropped = before - after
    if dropped > 0:
        logging.warning("[SANITIZE] Dropped %d future rows beyond %s", dropped, cutoff)

    return df2.sort_values("timestamp")


def _save(df: pd.DataFrame) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df.to_csv(OUT_CSV, index=False)


# -------------------------
# Fetch window
# -------------------------
def _compute_time_window(
    existing_max: Optional[pd.Timestamp],
    last_closed_end: datetime,
    minutes_back: int = 1500,
) -> Tuple[datetime, datetime]:
    """
    Always compute a sane window within [past, last_closed_end].
    Never asks for the future.
    """
    end_dt = last_closed_end

    if existing_max is not None:
        existing_max = pd.to_datetime(existing_max, utc=True, errors="coerce")

        # If existing_max is somehow ahead of end_dt, clamp it.
        end_ts = pd.Timestamp(end_dt)  # end_dt already tz-aware
        if existing_max > end_ts:
            existing_max = end_ts

        existing_dt = existing_max.to_pydatetime()

        # Back up a bit so we can dedupe & fill gaps
        start_dt = existing_dt - timedelta(minutes=60)

        # also clamp start so we don’t request too much
        min_start = end_dt - timedelta(minutes=minutes_back)
        if start_dt < min_start:
            start_dt = min_start
    else:
        start_dt = end_dt - timedelta(minutes=minutes_back)

    # final safety: never let start exceed end
    if start_dt > end_dt:
        start_dt = end_dt - timedelta(minutes=60)

    return start_dt, end_dt


# -------------------------
# Coinbase response parsing
# -------------------------
def _extract_rows(resp: Any) -> List[Any]:
    if resp is None:
        return []
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        if isinstance(resp.get("candles"), list):
            return resp["candles"]
        data = resp.get("data")
        if isinstance(data, dict) and isinstance(data.get("candles"), list):
            return data["candles"]
        return []
    if hasattr(resp, "candles"):
        c = getattr(resp, "candles")
        if isinstance(c, list):
            return c
    return []


def _dump_debug(resp: Any) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    payload: Dict[str, Any] = {"type": str(type(resp))}
    try:
        payload["object_dict"] = getattr(resp, "__dict__", str(resp))
        if hasattr(resp, "candles"):
            c = getattr(resp, "candles")
            payload["candles_len"] = len(c) if isinstance(c, list) else None
            if isinstance(c, list) and c:
                payload["candles_item_type0"] = str(type(c[0]))
                payload["candles_item_str0"] = str(c[0])[:500]
                payload["candles_item_dict0"] = getattr(c[0], "__dict__", None)
    except Exception as e:
        payload["error_dumping"] = str(e)

    with open(DEBUG_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    logging.info("[DEBUG] Wrote raw response dump to: %s", DEBUG_JSON)


def _coerce_candle_row(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None

    if isinstance(raw, dict):
        return raw

    d = getattr(raw, "__dict__", None)
    if isinstance(d, dict) and d:
        if "start" in d or "open" in d or "close" in d:
            return d

    # As last resort: parse string representation (your SDK prints python dict literals)
    try:
        s = str(raw).strip()
        if not s:
            return None
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def _parse_start_to_ts(start_v: Any) -> Optional[pd.Timestamp]:
    try:
        if start_v is None:
            return None
        if isinstance(start_v, (int, float)):
            ts = pd.to_datetime(int(start_v), unit="s", utc=True, errors="coerce")
            return None if pd.isna(ts) else ts
        if isinstance(start_v, str):
            sv = start_v.strip()
            if sv.isdigit():
                ts = pd.to_datetime(int(sv), unit="s", utc=True, errors="coerce")
                return None if pd.isna(ts) else ts
            ts = pd.to_datetime(sv, utc=True, errors="coerce")
            return None if pd.isna(ts) else ts
        ts = pd.to_datetime(start_v, utc=True, errors="coerce")
        return None if pd.isna(ts) else ts
    except Exception:
        return None


def _fetch_coinbase_5m(product_id: str, start_dt: datetime, end_dt: datetime, limit: int = 300) -> pd.DataFrame:
    try:
        from coinbase.rest import RESTClient
    except Exception as e:
        raise RuntimeError(
            "Missing Coinbase Advanced SDK. Install:\n"
            "  pip install coinbase-advanced-py\n"
            f"Import error: {e}"
        )

    api_key = os.getenv("COINBASE_API_KEY", "").strip()
    if not api_key or not _is_cdp_key(api_key):
        raise RuntimeError("COINBASE_API_KEY must be CDP format (organizations/.../apiKeys/...)")

    api_secret = _load_api_secret_from_env()
    if not api_secret:
        raise RuntimeError("Missing COINBASE_API_SECRET or COINBASE_API_SECRET_FILE")

    client = RESTClient(api_key=api_key, api_secret=api_secret)

    start_epoch = _to_epoch_seconds(start_dt)
    end_epoch = _to_epoch_seconds(end_dt)

    candles_resp: Optional[Any] = None
    err: Optional[Exception] = None

    granularities = ["FIVE_MINUTE", "FIVE_MINUTES"]

    for fn_name in ("get_product_candles", "get_candles"):
        if not hasattr(client, fn_name):
            continue
        fn = getattr(client, fn_name)
        for gran in granularities:
            try:
                candles_resp = fn(
                    product_id=product_id,
                    start=str(start_epoch),
                    end=str(end_epoch),
                    granularity=gran,
                    limit=limit,
                )
                err = None
                break
            except Exception as e:
                err = e
                candles_resp = None
        if candles_resp is not None:
            break

    if candles_resp is None:
        raise RuntimeError(f"Could not fetch candles via SDK. Last error: {err}")

    rows = _extract_rows(candles_resp)

    norm: List[Dict[str, Any]] = []
    skipped = 0

    for raw in rows:
        r = _coerce_candle_row(raw)
        if r is None:
            skipped += 1
            continue

        start_v = r.get("start") or r.get("time") or r.get("timestamp")
        ts = _parse_start_to_ts(start_v)
        if ts is None:
            skipped += 1
            continue

        norm.append({
            "timestamp": ts,
            "open": float(r.get("open", 0) or 0),
            "high": float(r.get("high", 0) or 0),
            "low": float(r.get("low", 0) or 0),
            "close": float(r.get("close", 0) or 0),
            "volume": float(r.get("volume", 0) or 0),
        })

    if skipped:
        logging.info("[INFO] Skipped non-candle rows: %d", skipped)

    df = pd.DataFrame(norm)
    if df.empty:
        _dump_debug(candles_resp)
        return df

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def main():
    load_dotenv()
    product_id = os.getenv("COINBASE_PRODUCT_ID", "BTC-USD").strip()

    last_closed_end = _last_closed_5m_end_dt()

    existing = _load_existing()
    existing = _sanitize_existing(existing, last_closed_end=last_closed_end)

    # If we dropped anything, write the cleaned CSV immediately
    if not existing.empty:
        _save(existing)

    existing_max = existing["timestamp"].max() if not existing.empty else None

    start_dt, end_dt = _compute_time_window(existing_max, last_closed_end=last_closed_end, minutes_back=1500)
    logging.info("[INFO] Fetch window: %s -> %s", start_dt, end_dt)

    fresh = _fetch_coinbase_5m(product_id=product_id, start_dt=start_dt, end_dt=end_dt, limit=300)

    if fresh.empty:
        logging.warning("No candles fetched.")
        return

    # Safety: never allow fetched data beyond last_closed_end
    cutoff = pd.Timestamp(end_dt)
    fresh["timestamp"] = pd.to_datetime(fresh["timestamp"], utc=True, errors="coerce")
    fresh = fresh.dropna(subset=["timestamp"])
    fresh = fresh[fresh["timestamp"] <= cutoff].copy()

    if fresh.empty:
        logging.warning("Fetched candles were all beyond cutoff (unexpected).")
        return

    combined = pd.concat([existing, fresh], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["timestamp"]).sort_values("timestamp")
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")

    _save(combined)

    logging.info("[OK] Updated %s", OUT_CSV)
    logging.info("[OK] Range: %s -> %s", combined["timestamp"].min(), combined["timestamp"].max())

    if existing_max is None:
        logging.info("[OK] Rows: %d", len(combined))
    else:
        added = fresh[fresh["timestamp"] > pd.to_datetime(existing_max, utc=True)]
        logging.info("[OK] New bars appended: %d", int(len(added)))


if __name__ == "__main__":
    main()
