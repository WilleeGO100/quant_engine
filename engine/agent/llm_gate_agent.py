from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

# Optional deps: we try openai first, then fall back to requests.
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


# -----------------------------
# Data contracts
# -----------------------------

@dataclass
class TradeProposal:
    symbol: str
    timeframe: str  # e.g. "5m"
    timestamp_utc: str  # ISO string
    action: str  # "ENTER_LONG", "EXIT", "HOLD"
    side_mode: str  # "SPOT_LONG_ONLY" etc.

    # Core numbers (keep small + stable)
    price: float
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    rvol: Optional[float] = None
    vwap_dist: Optional[float] = None
    session: Optional[str] = None  # "asia" | "london" | "ny" | None

    # Strategy diagnostics (strings are best; LLMs handle them well)
    reasons: Optional[List[str]] = None  # e.g. ["RVOL_TOO_LOW", "ATR_PCT_TOO_LOW"]
    extra: Optional[Dict[str, Any]] = None  # any other tiny facts


@dataclass
class GateDecision:
    approve: bool
    confidence: float  # 0.0 - 1.0
    reason: str
    risk_flags: List[str]
    overrides: Dict[str, Any]
    raw_model: str
    latency_ms: int


# -----------------------------
# LLM Gate Agent
# -----------------------------

class LLMGateAgent:
    """
    LLM approval gate that NEVER generates signals.
    It only approves/vetoes a provided TradeProposal.

    mode:
      - "off": approve always, no API calls
      - "shadow": call LLM + log, but do not block upstream (you enforce this in caller)
      - "on": call LLM and enforce veto upstream
    """

    def __init__(
        self,
        mode: str = "off",
        model: str = "gpt-4.1-mini",
        temperature: float = 0.1,
        timeout_s: int = 12,
        max_retries: int = 2,
    ) -> None:
        self.mode = (mode or "off").strip().lower()
        self.model = model
        self.temperature = float(temperature)
        self.timeout_s = int(timeout_s)
        self.max_retries = int(max_retries)

        self._openai_client = None
        if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def decide(self, proposal: TradeProposal, account_state: Optional[Dict[str, Any]] = None) -> GateDecision:
        # Off mode: fast pass-through
        if self.mode == "off":
            return GateDecision(
                approve=True,
                confidence=1.0,
                reason="LLM gate disabled (mode=off).",
                risk_flags=[],
                overrides={},
                raw_model=self.model,
                latency_ms=0,
            )

        payload = {
            "proposal": asdict(proposal),
            "account_state": account_state or {},
            "constraints": {
                "never_generate_signals": True,
                "allowed_actions": ["APPROVE", "VETO"],
                "spot_long_only_expected": True,
            },
            "output_schema": {
                "approve": "boolean",
                "confidence": "number 0..1",
                "reason": "string (short, specific)",
                "risk_flags": "array of strings",
                "overrides": {
                    "reduce_size_pct": "number 0..100 (optional)",
                    "tighten_stop_mult": "number (optional)",
                    "take_profit_mult": "number (optional)",
                    "max_hold_bars": "integer (optional)"
                }
            }
        }

        system = (
            "You are a strict risk-control approval gate for an algorithmic trading engine.\n"
            "You NEVER create trade ideas or signals.\n"
            "You ONLY approve or veto the given proposal.\n"
            "You must respond with VALID JSON only (no markdown, no extra text).\n"
            "Veto only for concrete risk/context reasons using the supplied fields.\n"
            "If data is missing or ambiguous, prefer VETO with reason 'INSUFFICIENT_CONTEXT'."
        )

        user = (
            "Evaluate this proposed action. Return JSON matching the schema.\n"
            f"INPUT_JSON={json.dumps(payload, separators=(',', ':'))}"
        )

        start = time.time()
        last_err = None

        for attempt in range(self.max_retries + 1):
            try:
                text = self._call_llm(system, user)
                decision = self._parse_decision(text)
                latency_ms = int((time.time() - start) * 1000)
                decision.latency_ms = latency_ms
                decision.raw_model = self.model
                return decision
            except Exception as e:
                last_err = e
                # brief backoff
                time.sleep(0.4 * (attempt + 1))

        # Fail-safe: if LLM fails, VETO in on/shadow modes (caller can choose to ignore veto in shadow)
        latency_ms = int((time.time() - start) * 1000)
        return GateDecision(
            approve=False,
            confidence=0.0,
            reason=f"LLM_ERROR: {type(last_err).__name__}",
            risk_flags=["LLM_ERROR"],
            overrides={},
            raw_model=self.model,
            latency_ms=latency_ms,
        )

    def _call_llm(self, system: str, user: str) -> str:
        # Preferred: official OpenAI python client
        if self._openai_client is not None:
            resp = self._openai_client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                timeout=self.timeout_s,
            )
            return resp.choices[0].message.content or ""

        # Fallback: REST call (requires requests)
        if requests is None:
            raise RuntimeError("No OpenAI client and 'requests' not installed. Install openai or requests.")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        r = requests.post(url, headers=headers, json=body, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "")

    @staticmethod
    def _parse_decision(text: str) -> GateDecision:
        # Hard strip common wrappers
        t = (text or "").strip()
        # If model returns leading/trailing junk, try to recover first JSON object
        if not (t.startswith("{") and t.endswith("}")):
            start = t.find("{")
            end = t.rfind("}")
            if start != -1 and end != -1 and end > start:
                t = t[start : end + 1]

        obj = json.loads(t)

        approve = bool(obj.get("approve", False))
        confidence = float(obj.get("confidence", 0.0))
        reason = str(obj.get("reason", "")).strip()[:400]
        risk_flags = obj.get("risk_flags") or []
        if not isinstance(risk_flags, list):
            risk_flags = [str(risk_flags)]

        overrides = obj.get("overrides") or {}
        if not isinstance(overrides, dict):
            overrides = {}

        # Clamp confidence
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0

        return GateDecision(
            approve=approve,
            confidence=confidence,
            reason=reason or ("APPROVED" if approve else "VETOED"),
            risk_flags=[str(x)[:80] for x in risk_flags][:12],
            overrides=overrides,
            raw_model=str(obj.get("model", "")) or "unknown",
            latency_ms=0,
        )
