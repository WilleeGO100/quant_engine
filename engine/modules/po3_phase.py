# engine/modules/po3_phase.py

from __future__ import annotations

from enum import Enum


class PO3Phase(Enum):
    ACCUMULATION = 1
    MANIPULATION = 2
    EXPANSION = 3


def update_phase(
    prev_phase: PO3Phase | None,
    bos_up: bool,
    bos_down: bool,
    sweep_up: bool,
    sweep_down: bool,
    bias: int,
    struct_trend: int,
) -> PO3Phase:
    """
    Simple PO3 phase state machine.

    ACCUMULATION:
        - No clear liquidity event yet
        - Wait for a sweep in direction opposite bias

    MANIPULATION:
        - Liquidity taken (sweep)
        - Wait for BOS in direction of bias to confirm expansion

    EXPANSION:
        - Trending in direction of bias after manipulation
        - Reset back to ACCUMULATION when opposing BOS or bias conflict

    Parameters
    ----------
    prev_phase : current/previous PO3Phase
    bos_up / bos_down : structure breaks
    sweep_up / sweep_down : liquidity sweeps
    bias : combined HTF+MTF bias (+1 / -1 / 0)
    struct_trend : structural trend (+1 / -1 / 0)

    Returns
    -------
    new_phase : PO3Phase
    """

    phase = prev_phase or PO3Phase.ACCUMULATION

    # ---------------- ACCUMULATION ----------------
    if phase == PO3Phase.ACCUMULATION:
        # Looking for manipulation sweeps against the bias
        if sweep_up and bias <= 0:
            # swept above to fuel potential downside
            return PO3Phase.MANIPULATION
        if sweep_down and bias >= 0:
            # swept below to fuel potential upside
            return PO3Phase.MANIPULATION
        return phase

    # ---------------- MANIPULATION ----------------
    if phase == PO3Phase.MANIPULATION:
        # Confirm expansion once BOS aligns with bias / structural trend
        if (bias <= 0 and bos_down) or (struct_trend == -1 and bos_down):
            return PO3Phase.EXPANSION
        if (bias >= 0 and bos_up) or (struct_trend == 1 and bos_up):
            return PO3Phase.EXPANSION
        return phase

    # ---------------- EXPANSION ----------------
    if phase == PO3Phase.EXPANSION:
        # Kill expansion when we see structure or bias against us
        if bias == 0:
            return PO3Phase.ACCUMULATION

        # Opposing BOS
        if bias >= 0 and bos_down:
            return PO3Phase.ACCUMULATION
        if bias <= 0 and bos_up:
            return PO3Phase.ACCUMULATION

        # Structural trend flipped hard against us
        if (bias >= 0 and struct_trend == -1) or (bias <= 0 and struct_trend == 1):
            return PO3Phase.ACCUMULATION

        return phase

    return PO3Phase.ACCUMULATION
