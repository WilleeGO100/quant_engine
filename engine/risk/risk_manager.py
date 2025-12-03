# engine/risk/risk_manager.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RiskState:
    """
    Simple container for current risk state.
    For now we track only daily PnL, but you can extend this later.
    """
    daily_pnl: float = 0.0  # realised PnL for the day


class RiskManager:
    """
    Very simple risk layer used by both:
      - main_live.py
      - test_force_trade.py

    It expects a config object with at least:
        - account_size (float)
        - max_daily_loss_pct (float)
        - default_position_size (float)

    And it exposes ONE main method:

        apply(decision: dict, current_state: Optional[dict]) -> Optional[dict]

    Where `decision` is the trade dict coming from strategy/LLM, like:
        {
            "signal": "LONG" | "SHORT" | "HOLD",
            "size": 1.0,
            "stop_loss": None,
            "take_profit": None,
            "meta": {...}
        }

    If risk is violated, we return None.
    If risk is OK, we return a possibly-adjusted trade dict.
    """

    def __init__(self, config: Any):
        # Config is your ConfigObj from config/config.py
        self.config = config
        self.account_size: float = float(getattr(config, "account_size", 0.0))
        self.max_daily_loss_pct: float = float(getattr(config, "max_daily_loss_pct", 0.0))
        self.default_position_size: float = float(getattr(config, "default_position_size", 1.0))

        # Pre-compute max daily loss in currency terms
        self._max_daily_loss_value: float = (
            self.account_size * self.max_daily_loss_pct / 100.0
            if self.account_size and self.max_daily_loss_pct
            else 0.0
        )

    # ------------------------------------------------------------------
    # Core helper: check if we are blown out on the day
    # ------------------------------------------------------------------
    def _daily_loss_exceeded(self, state: Optional[Dict[str, Any]]) -> bool:
        if state is None:
            return False

        # state may be a dict or RiskState
        if isinstance(state, RiskState):
            daily_pnl = state.daily_pnl
        else:
            daily_pnl = float(state.get("daily_pnl", 0.0))

        if self._max_daily_loss_value <= 0:
            # No limit configured
            return False

        # If we've lost more than the allowed amount, block trades
        return daily_pnl <= -self._max_daily_loss_value

    # ------------------------------------------------------------------
    # PUBLIC API used by code:
    #     risk_manager.apply(decision_dict, current_state=...)
    # ------------------------------------------------------------------
    def apply(
        self,
        decision: Dict[str, Any],
        current_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Apply basic risk checks & sizing to the proposed trade decision.

        Returns:
            - dict (possibly modified) if trade is allowed
            - None if trade should be blocked
        """

        signal = (decision.get("signal") or "").upper()

        # 1) If we have no trade, just propagate HOLD / None
        if signal not in ("LONG", "SHORT"):
            # Explicitly normalise HOLD semantics => caller treats None as "no trade"
            return None

        # 2) Daily loss guardrail
        if self._daily_loss_exceeded(current_state):
            print(
                "[RiskManager] Daily loss limit exceeded -> blocking new trades."
            )
            return None

        # 3) Position sizing: if size missing, use default from config
        size = decision.get("size")
        if size is None:
            size = self.default_position_size

        try:
            size = float(size)
        except Exception:
            # Fallback if something weird comes from LLM
            print(
                f"[RiskManager] Invalid size '{size}' from decision, "
                f"falling back to default_position_size={self.default_position_size}"
            )
            size = self.default_position_size

        if size <= 0:
            print("[RiskManager] Non-positive size after processing -> blocking trade.")
            return None

        # 4) Build a clean, normalised trade dict to send downstream
        cleaned: Dict[str, Any] = {
            "signal": signal,  # LONG / SHORT
            "size": size,
            "stop_loss": decision.get("stop_loss"),
            "take_profit": decision.get("take_profit"),
            "meta": decision.get("meta", {}),
        }

        print(f"[RiskManager] Trade approved: {cleaned}")
        return cleaned
