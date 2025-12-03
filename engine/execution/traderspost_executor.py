from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests


class TradersPostExecutor:
    """
    Simple TradersPost webhook executor.

    Unified signature so we can call:

        executor.send_order(
            signal="LONG" or "SHORT",
            size=1.0,
            test=True,
            meta={"source": "..."},
        )

    from BOTH ES and BTC pipelines.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.webhook_url: str = getattr(config, "traderspost_webhook_url", "").strip()

        # Symbol we actually send to TradersPost (what their strategy expects)
        self.trade_symbol: str = getattr(
            config,
            "trade_symbol",
            getattr(config, "symbol", "MNQ"),
        )

        # Master switch for live vs dry-run
        self.live_trading: bool = bool(getattr(config, "live_trading", False))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _signal_to_action(self, signal: str) -> str:
        """
        Map our internal signal ('LONG'/'SHORT') to TradersPost action ('buy'/'sell').
        """
        s = (signal or "").upper()
        if s == "LONG":
            return "buy"
        if s == "SHORT":
            return "sell"
        # Anything else: we consider it a no-op – but build payload anyway.
        return "none"

    def _build_payload(
        self,
        signal: str,
        size: float,
        test: bool,
        meta: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        action = self._signal_to_action(signal)

        payload: Dict[str, Any] = {
            "ticker": self.trade_symbol,
            "action": action,
            "quantity": float(size),
            "orderType": "market",
            "test": bool(test),
        }

        if meta:
            payload["meta"] = meta

        return payload

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send_order(
        self,
        signal: str,
        size: float,
        test: bool = True,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send a single order to TradersPost.

        Parameters
        ----------
        signal : str
            'LONG' or 'SHORT' (anything else will map to action='none').
        size : float
            Position size / quantity to send.
        test : bool
            Whether to mark this as a TradersPost test order.
        meta : dict | None
            Extra metadata dictionary to attach to the payload.
        """
        # Build JSON payload
        payload = self._build_payload(signal, size, test, meta)

        print("[TradersPostExecutor] Prepared payload:")
        print(json.dumps(payload, indent=2))

        if not self.webhook_url:
            print("[TradersPostExecutor] WARNING: No webhook URL configured – not sending.")
            return

        # Respect live_trading master switch
        if not self.live_trading:
            print(
                "[TradersPostExecutor] DRY RUN: live_trading is False "
                "-> not sending webhook."
            )
            return

        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            print(
                f"[TradersPostExecutor] Webhook response: "
                f"status={resp.status_code}, body={resp.text[:300]!r}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[TradersPostExecutor] ERROR sending webhook: {exc}")
