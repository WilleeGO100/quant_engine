# engine/execution/traderspost_executor.py
from __future__ import annotations

import json
from typing import Optional, Dict, Any

import requests


class TradersPostExecutor:
    """
    Send LONG / SHORT signals to TradersPost.

    Config keys expected:
        - traderspost_webhook_url: str
        - traderspost_ticker: str        (e.g. "BTCUSD")
        - traderspost_account_id: str    (your TP paper account ID)
        - entry_order_type: str          ("market" or "limit")
        - live_trading: bool             (False = DRY RUN, True = actually hit webhook)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.url: str = config.get("traderspost_webhook_url", "")
        self.ticker: str = config.get("traderspost_ticker", "BTCUSD")
        self.account_id: str = config.get("traderspost_account_id", "")
        self.order_type: str = config.get("entry_order_type", "market")
        self.live_trading: bool = bool(config.get("live_trading", False))

        if not self.url:
            raise ValueError("TradersPostExecutor: 'traderspost_webhook_url' is missing.")
        if not self.account_id:
            raise ValueError("TradersPostExecutor: 'traderspost_account_id' (paper account) is missing.")

    def _map_signal_to_action(self, signal: str) -> str:
        """
        Convert our strategy signal into TradersPost 'action'.
        """
        s = signal.upper()
        if s == "LONG":
            return "buy"
        if s == "SHORT":
            return "sell"
        raise ValueError(f"Unsupported signal for TradersPost: {signal}")

    def send_order(
        self,
        signal: str,
        size: float,
        test: bool = True,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Main public method.

        signal : "LONG" or "SHORT"
        size   : numeric quantity (e.g. 0.01 BTC, 1 contract, etc.)
        test   : if True, marks the order as test/simulated (TradersPost 'test' flag)

        NOTE:
        - If self.live_trading is False, we DO NOT send an HTTP request.
        - If self.live_trading is True, we send the payload to TradersPost.
        """

        action = self._map_signal_to_action(signal)

        # 'test' flag is true if either:
        #   - we are not in live_trading mode, OR
        #   - the caller explicitly passes test=True
        test_flag = (not self.live_trading) or bool(test)

        payload: Dict[str, Any] = {
            "ticker": self.ticker,
            "action": action,
            "quantity": size,
            "orderType": self.order_type,
            "accountId": self.account_id,   # <- your TradersPost PAPER account
            "test": test_flag,
        }

        if meta:
            payload["meta"] = meta

        print("\n[TradersPostExecutor] Prepared payload:")
        print(json.dumps(payload, indent=2))

        # Safety: DRY RUN mode
        if not self.live_trading:
            print("[TradersPostExecutor] DRY RUN: live_trading=False -> not sending webhook.")
            return payload

        # Live HTTP POST (use with care!)
        try:
            resp = requests.post(
                self.url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=10,
            )
            print(f"[TradersPostExecutor] Webhook response: {resp.status_code}")
            print(resp.text)
            return resp
        except Exception as e:
            print(f"[TradersPostExecutor] ERROR sending webhook: {e}")
            return None
