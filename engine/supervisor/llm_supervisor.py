# engine/supervisor/llm_supervisor.py
#
# LLM Supervisor for the GOLD PO3 engine.
# - We feed it a structured snapshot of market + risk + candidate signal.
# - It returns a JSON decision: approve / reject / adjust size / override side.

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, List, Literal, Optional, Dict, Any
import json


Side = Literal["LONG", "SHORT", "HOLD"]


@dataclass
class MarketSnapshot:
    # High-level identifiers
    time: str
    symbol: str
    session: str

    # Price + volatility
    price: float
    atr_points: float
    range_pct: float

    # SMC / structure context (from strategy)
    htf_bias: int          # -1, 0, +1
    mtf_bias: int          # -1, 0, +1
    smc_phase: str         # ACCUMULATION / MANIPULATION / EXPANSION
    struct_trend: int      # -1, 0, +1

    # Risk state
    daily_realised_pnl: float
    daily_trades: int
    daily_locked: bool
    open_trades: int

    # Candidate decision
    candidate_signal: Side

    # Recent bars + trades for extra context
    recent_bars: List[Dict[str, Any]]   # e.g. [{time, close, range_pct}, ...]
    recent_trades: List[Dict[str, Any]] # your last N closed trades summary


@dataclass
class SupervisorDecision:
    approved: bool
    adjusted_signal: Side
    size_multiplier: float
    reason: str


@dataclass
class LLMSupervisorConfig:
    enabled: bool = True
    model_name: str = "gpt-4.1-mini"  # only used in prompt text
    temperature: float = 0.2
    max_tokens: int = 300


class LLMSupervisor:
    """
    Wraps an LLM and turns it into a strict, JSON-only supervisor.

    You pass in an `llm_fn` that takes a prompt string and returns
    a *string containing JSON*.
    """

    def __init__(
        self,
        config: LLMSupervisorConfig,
        llm_fn: Callable[[str], str],
    ):
        self.config = config
        self.llm_fn = llm_fn

    # -------------------------------------------------------------
    # Public entrypoint
    # -------------------------------------------------------------
    def evaluate(self, snapshot: MarketSnapshot) -> SupervisorDecision:
        """
        Main method: given a MarketSnapshot, either auto-approve
        (if disabled) or send to LLM and parse the JSON result.
        """
        if not self.config.enabled:
            # Bypass: just approve whatever the strategy wants
            return SupervisorDecision(
                approved=True,
                adjusted_signal=snapshot.candidate_signal,
                size_multiplier=1.0,
                reason="LLM supervisor disabled (auto-approve)",
            )

        prompt = self._build_prompt(snapshot)
        raw = self.llm_fn(prompt)

        decision = self._parse_response(raw, snapshot)
        return decision

    # -------------------------------------------------------------
    # Prompt construction
    # -------------------------------------------------------------
    def _build_prompt(self, snapshot: MarketSnapshot) -> str:
        """
        Build a clear instruction prompt. We ask the model to return
        a strict JSON object with a fixed schema.
        """

        snap_json = json.dumps(asdict(snapshot), indent=2, default=str)

        instructions = f"""
You are an institutional-grade trading risk supervisor sitting ON TOP of a deterministic algorithmic strategy.

The underlying strategy is a Smart Money Concepts (SMC) + PO3 (Accumulation / Manipulation / Expansion) model
that already has an edge. Your job is NOT to invent trades, but to:

  1. APPROVE or REJECT the candidate signal from the strategy.
  2. Optionally ADJUST the side (LONG/SHORT) if the signal is clearly against higher-timeframe bias or structure.
  3. Optionally SCALE the size via a multiplier in [0.5, 1.5] based on conviction and risk context.
  4. Always avoid obvious overtrading, trading directly into extreme risk, or trading against strong HTF trend.

IMPORTANT RULES:

- You operate only on ***conservative, professional*** principles.
- When in doubt, be STRICT and reject the trade.
- NEVER create trades out of thin air; only refine the candidate_signal.
- Do NOT try to predict exact prices. Just judge the quality of the setup.

You must respond ONLY with a single JSON object with this exact schema:

{{
  "approved": true or false,
  "adjusted_signal": "LONG" or "SHORT" or "HOLD",
  "size_multiplier": float between 0.5 and 1.5,
  "reason": "short human-readable explanation"
}}

Guidelines:

- If daily_realised_pnl is already very negative or daily_locked is true,
  you should usually reject new trades.
- If open_trades >= 1, you should be very reluctant to approve new trades.
- If htf_bias and mtf_bias both agree with candidate_signal, and smc_phase is EXPANSION,
  you are more willing to approve and slightly size up (e.g. 1.1).
- If candidate_signal goes AGAINST combined bias or struct_trend, you should reject or flip it.
- In low volatility (small range_pct, low atr_points), focus on cleaner setups or pass.

Here is the current market snapshot in JSON:

{snap_json}

Return ONLY the JSON object, nothing else.
"""

        return instructions.strip()

    # -------------------------------------------------------------
    # Response parsing + safety
    # -------------------------------------------------------------
    def _parse_response(
        self,
        raw: str,
        snapshot: MarketSnapshot,
    ) -> SupervisorDecision:
        """
        Try to parse the LLM response as JSON. If it fails for any reason,
        fall back to "approve original signal".
        """
        try:
            data = json.loads(raw)
        except Exception:
            return SupervisorDecision(
                approved=True,
                adjusted_signal=snapshot.candidate_signal,
                size_multiplier=1.0,
                reason="LLM returned non-JSON or parse error; auto-approved original signal.",
            )

        approved = bool(data.get("approved", True))
        adjusted_signal = data.get("adjusted_signal", snapshot.candidate_signal)

        if adjusted_signal not in ("LONG", "SHORT", "HOLD"):
            adjusted_signal = snapshot.candidate_signal

        try:
            size_multiplier = float(data.get("size_multiplier", 1.0))
        except Exception:
            size_multiplier = 1.0

        # Clamp multiplier
        if size_multiplier < 0.5:
            size_multiplier = 0.5
        if size_multiplier > 1.5:
            size_multiplier = 1.5

        reason = str(data.get("reason", "LLM decision"))

        return SupervisorDecision(
            approved=approved,
            adjusted_signal=adjusted_signal,  # type: ignore[arg-type]
            size_multiplier=size_multiplier,
            reason=reason,
        )
