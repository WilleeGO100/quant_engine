# engine/agents/types.py
#
# Core data structures that flow through the multi-agent engine.
# Everything (strategy, LLM, risk, execution) speaks in terms of these.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class DecisionType(str, Enum):
    HOLD = "HOLD"              # no trade
    ENTER = "ENTER"            # open new position
    EXIT = "EXIT"              # close / reduce existing position


@dataclass
class StrategySignal:
    """
    Raw output from the strategy (no risk, no LLM).
    """

    decision: DecisionType         # ENTER / EXIT / HOLD
    side: Optional[Side] = None    # LONG / SHORT (for ENTER)
    confidence: float = 0.0        # 0–1 subjective confidence
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    notes: str = ""                # free-text debug info

    extra: Dict[str, Any] = None   # anything else (PO3 phase, regime, etc.)

    def is_trade(self) -> bool:
        return self.decision in {DecisionType.ENTER, DecisionType.EXIT}


@dataclass
class TradeIntent:
    """
    Trade proposal after strategy (and optionally LLM) have spoken,
    but before risk sizing is applied.
    """

    decision: DecisionType
    side: Optional[Side]
    quantity_pct: float           # % of account to risk/move (pre-risk-agent)
    stop_price: Optional[float]
    take_profit_price: Optional[float]
    reason: str                   # textual rationale (for logs / LLM)
    meta: Dict[str, Any]          # context (features, regime, etc.)


@dataclass
class RiskAdjustedOrder:
    """
    Final order after risk rules are applied.
    This is what actually goes to the execution layer.
    """

    should_trade: bool            # False = NO_TRADE
    decision: Optional[DecisionType] = None
    side: Optional[Side] = None
    quantity: Optional[float] = None      # actual position size in units/contracts
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    reason: str = ""              # why risk engine allowed/blocked
    meta: Dict[str, Any] = None   # extra fields for logging


@dataclass
class ExecutionResult:
    """
    Result from the execution layer.
    """

    success: bool
    mode: str                     # "dry-run" or "live"
    broker_order_id: Optional[str]
    description: str              # human-readable description / error
    payload: Dict[str, Any]       # raw payload sent to broker / simulated
