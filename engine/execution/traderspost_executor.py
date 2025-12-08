# engine/execution/traderspost_executor.py
#
# TradersPostExecutor
# -------------------
# Small wrapper that turns a signal into a TradersPost webhook payload
# and (optionally) POSTs it to your webhook URL.
#
# Supports both:
#   - dict config
#   - Config dataclass from config/config.py
#
# Expected config fields (either as keys or attributes):
#   - traderspost_webhook_url : str   (full webhook URL)
#   - trade_symbol            : str   (e.g. "BTCUSD")
#   - traderspost_ticker      : str   (fallback ticker name)
#   - traderspost_account_id  : str   (optional; your TradersPost account ID)
#   - entry_order_type        : str   ("market", "limit", etc.), default "market"
#   - live_trading            : bool  (False = always dry run)
#
# Public method:
#   send_order(signal: str, size: float, test: bool, meta: dict | None)
#
#   signal : "LONG" / "SHORT" / "BUY" / "SELL"
#   size   : position size (e.g. 0.01 BTC, 1 contract, etc.)
#   test   : if True, payload will include "test": true
#   meta   : optional dict -> goes into payload["meta"]
#

from __future__ import annotations

import json
from typing import Any, Dict, Optional


class TradersPostExecutor:
    def __init__(self, config: Any) -> None:
        """
        config can be:
          - a dict
          - your Config dataclass instance
        """
        if isinstance(config, dict):
            cfg = config
            self.live_trading: bool = bool(cfg.get("live_trading", False))
            self.webhook_url: str = cfg.get("traderspost_webhook_url", "")
            # prefer trade_symbol, fallback to traderspost_ticker, then BTCUSD
            self.ticker: str = cfg.get("trade_symbol") or cfg.get("traderspost_ticker", "BTCUSD")
            self.account_id: Optional[str] = (cfg.get("traderspost_account_id") or "").strip() or None
            self.order_type: str = cfg.get("entry_order_type", "market")
        else:
            # Assume Config dataclass with attributes
            self.live_trading = bool(getattr(config, "live_trading", False))
            self.webhook_url = getattr(config, "traderspost_webhook_url", "")
            self.ticker = getattr(
                config,
                "trade_symbol",
                getattr(config, "traderspost_ticker", "BTCUSD"),
            )
            self.account_id = (getattr(config, "traderspost_account_id", "") or "").strip() or None
            self.order_type = getattr(config, "entry_order_type", "market")

        if not self.webhook_url:
            print(
                "[TradersPostExecutor] WARNING: traderspost_webhook_url is empty. "
                "All sends will be treated as DRY RUN."
            )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def send_order(
        self,
        signal: str,
        size: float,
        test: bool,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Build and (optionally) POST a TradersPost webhook payload.

        signal : "LONG" / "SHORT" / "BUY" / "SELL"
        size   : numeric quantity
        test   : if True, payload includes "test": true
        meta   : optional dict -> payload["meta"]
        """
        if signal is None:
            print("[TradersPostExecutor] signal is None -> not sending.")
            return

        action_map = {
            "LONG": "buy",
            "BUY": "buy",
            "SHORT": "sell",
            "SELL": "sell",
        }

        sig_upper = str(signal).upper()
        action = action_map.get(sig_upper)
        if action is None:
            print(f"[TradersPostExecutor] Unknown signal='{signal}' -> not sending.")
            return

        qty = float(size)

        payload: Dict[str, Any] = {
            "ticker": self.ticker,
            "action": action,
            "quantity": qty,
            "orderType": self.order_type,
            "test": bool(test),
        }

        if self.account_id:
            payload["accountId"] = self.account_id

        if meta:
            payload["meta"] = meta

        print("[TradersPostExecutor] Prepared payload:")
        print(json.dumps(payload, indent=2))

        # Decide whether to actually POST
        if (not self.live_trading) or test or (not self.webhook_url):
            print(
                "[TradersPostExecutor] DRY RUN: "
                f"live_trading={self.live_trading} test={test} "
                f"url={'SET' if bool(self.webhook_url) else 'MISSING'} -> not sending webhook."
            )
            return

        # Live POST
        try:
            import requests
        except ImportError:
            print(
                "[TradersPostExecutor] ERROR: 'requests' is not installed. "
                "Install it with 'pip install requests' to enable live webhooks."
            )
            return

        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=5)
            print(
                f"[TradersPostExecutor] Webhook POST status={resp.status_code} "
                f"body={resp.text[:200]!r}"
            )
        except Exception as e:
            print(f"[TradersPostExecutor] ERROR sending webhook: {e}")
