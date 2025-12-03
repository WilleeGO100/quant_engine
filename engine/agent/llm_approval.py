# engine/agent/llm_approval.py

from __future__ import annotations
import json
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMApprovalAgent:
    """
    Universal LLM approval wrapper.

    Modes:
        - "off": always approve exactly as-is (bypass LLM)
        - "echo": return same decision but log it (no API call)
        - "live": call actual OpenAI model

    Required external input for live mode:
        api_key must be provided.
    """

    def __init__(
        self,
        mode: str = "echo",
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.1,
    ):
        self.mode = (mode or "echo").lower()
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)

        print(
            f"[LLMApprovalAgent] Initialised with mode={self.mode}, "
            f"model={self.model}, temperature={self.temperature}"
        )

        # Build client lazily only if in live mode with valid key
        self.client = None
        if self.mode == "live":
            if not self.api_key:
                print("[LLMApprovalAgent] No API key provided for live mode -> rejecting trade.")
            else:
                try:
                    self.client = OpenAI(api_key=self.api_key)
                except Exception as e:
                    print(f"[LLMApprovalAgent] Failed to initialise OpenAI client: {e}")
                    self.client = None

    # ------------------------------------------------------------------
    # PUBLIC METHOD EXPECTED BY ALL SCRIPTS:
    #
    #     approve(raw_decision: dict) -> dict | None
    # ------------------------------------------------------------------
    def approve(self, raw_decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a raw decision dict through LLM filter.

        raw_decision example:
            {
                "signal": "LONG",
                "size": 1.0,
                "stop_loss": None,
                "take_profit": None,
                "meta": {...}
            }
        """

        # Normalize signal
        signal = (raw_decision.get("signal") or "").upper()

        # -----------------------------
        # Mode: OFF  -> always approve
        # -----------------------------
        if self.mode == "off":
            return raw_decision

        # -----------------------------
        # Mode: ECHO -> return unchanged
        # -----------------------------
        if self.mode == "echo":
            return raw_decision

        # -----------------------------
        # Mode: LIVE -> actual API call
        # -----------------------------
        if self.mode == "live":
            # Cannot run live LLM without a working client
            if not self.client or not self.api_key:
                print("[LLMApprovalAgent] No API key provided or client failed -> rejecting trade.")
                return {"signal": "HOLD", "size": 0, "meta": {"reason": "no_api_key"}}

            # Build system + user messages
            system_prompt = """
You are a trading risk validator. Your job is:
 - Ensure the proposed trade is logically valid.
 - Block trades that look unreasonable or risky.
 - You may return: LONG, SHORT, or HOLD.
 - You may adjust position size if needed.
"""

            user_prompt = f"""
Raw trade decision:
{json.dumps(raw_decision, indent=2)}

Respond ONLY in JSON with this format:
{{
  "signal": "LONG" | "SHORT" | "HOLD",
  "size": <float>,
  "stop_loss": null,
  "take_profit": null,
  "meta": {{"llm_checked": true}}
}}
"""

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                text = response.choices[0].message.content
                cleaned = json.loads(text)

                return cleaned

            except Exception as e:
                print(f"[LLMApprovalAgent] ERROR during API call: {e}")
                return {"signal": "HOLD", "size": 0, "meta": {"error": str(e)}}

        # -----------------------------
        # Unknown mode
        # -----------------------------
        print(f"[LLMApprovalAgent] Unknown mode='{self.mode}' -> HOLD")
        return {"signal": "HOLD", "size": 0}
