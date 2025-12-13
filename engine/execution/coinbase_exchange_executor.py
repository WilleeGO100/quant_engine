from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class CoinbaseExchangeConfig:
    api_key: str
    api_secret_b64: str
    passphrase: str
    api_base: str = "https://api.exchange.coinbase.com"
    product_id: str = "BTC-USD"
    live_trading: bool = False
    timeout_sec: int = 15


class CoinbaseExchangeExecutor:
    """
    Coinbase Exchange REST API executor using CB-ACCESS-* headers.
    - Auth/signing: timestamp + method + requestPath + body, HMAC-SHA256, base64
    - Orders endpoint: POST /orders (market + limit)
    Docs:
      - Authentication/signing details :contentReference[oaicite:1]{index=1}
      - Create order endpoint :contentReference[oaicite:2]{index=2}
    """

    def __init__(self, cfg: CoinbaseExchangeConfig):
        self.cfg = cfg

    def _sign(self, timestamp: str, method: str, request_path: str, body: str) -> str:
        prehash = f"{timestamp}{method.upper()}{request_path}{body}"
        secret = base64.b64decode(self.cfg.api_secret_b64)
        sig = hmac.new(secret, prehash.encode("utf-8"), hashlib.sha256).digest()
        return base64.b64encode(sig).decode("utf-8")

    def _headers(self, method: str, request_path: str, body_obj: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        ts = str(int(time.time()))
        body = json.dumps(body_obj) if body_obj else ""
        signature = self._sign(ts, method, request_path, body)

        return {
            "CB-ACCESS-KEY": self.cfg.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": ts,
            "CB-ACCESS-PASSPHRASE": self.cfg.passphrase,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def place_market_order(self, side: str, base_size: str) -> Dict[str, Any]:
        """
        Places a MARKET order for product_id using base_size (BTC amount).
        side: "buy" or "sell"
        base_size: string numeric (example: "0.001")
        """
        side = side.lower().strip()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        request_path = "/orders"
        body = {
            "type": "market",
            "side": side,
            "product_id": self.cfg.product_id,
            "size": str(base_size),
        }

        if not self.cfg.live_trading:
            return {
                "live_trading": False,
                "would_send": True,
                "method": "POST",
                "url": f"{self.cfg.api_base}{request_path}",
                "body": body,
            }

        url = f"{self.cfg.api_base}{request_path}"
        headers = self._headers("POST", request_path, body)

        resp = requests.post(url, headers=headers, json=body, timeout=self.cfg.timeout_sec)
        try:
            data = resp.json()
        except Exception:
            data = {"raw_text": resp.text}

        return {
            "live_trading": True,
            "status_code": resp.status_code,
            "response": data,
            "request": {"url": url, "body": body},
        }
