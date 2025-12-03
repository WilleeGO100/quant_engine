# engine/strategies/smc_po3_power_v2.py
#
# EXPERIMENTAL VERSION – SAFE TO MODIFY
# Start from GOLD SMCPo3PowerStrategy behaviour and tweak here
# without ever touching smc_po3_power.py (the confirmed good version).

from __future__ import annotations

from typing import Optional

from .base import BaseStrategy
from engine.modules.liquidity import detect_sweep
from engine.modules.fvg import detect_fvg
from engine.modules.structure import detect_structure
from engine.modules.po3_phase import PO3Phase, update_phase


def _update_ema(prev: Optional[float], alpha: float, price: float) -> float:
    if prev is None:
        return price
    return alpha * price + (1 - alpha) * prev


def _is_nan(x) -> bool:
    return isinstance(x, float) and x != x


class SMCPo3PowerV2Strategy(BaseStrategy):
    """
    POWER SMC PO3 Engine – v2 experimental

    Start identical to GOLD version, but any new logic / exits / filters
    should be implemented here so we can directly compare:

      - SMCPo3PowerStrategy   (gold)
      - SMCPo3PowerV2Strategy (experimental)

    Ingredients:
      - HTF (2H) + MTF (1H) EMA bias
      - 5M structure (BOS / CHOCH)
      - Liquidity sweeps (wick-based)
      - Fair Value Gaps (imbalances)
      - PO3 phase machine (Accumulation / Manipulation / Expansion)
      - Volatility + session filters
    """

    def __init__(
        self,
        fast_1h: int = 8,
        slow_1h: int = 24,
        fast_2h: int = 8,
        slow_2h: int = 24,
        sweep_lookback: int = 10,
        vol_floor_pct: float = 0.25,
        rv_floor_1h: Optional[float] = None,
        pivot_len: int = 5,
    ):
        super().__init__()

        # EMA params
        self.alpha_1h_fast = 2 / (fast_1h + 1)
        self.alpha_1h_slow = 2 / (slow_1h + 1)
        self.alpha_2h_fast = 2 / (fast_2h + 1)
        self.alpha_2h_slow = 2 / (slow_2h + 1)

        # EMA states
        self.ema_1h_fast: Optional[float] = None
        self.ema_1h_slow: Optional[float] = None
        self.ema_2h_fast: Optional[float] = None
        self.ema_2h_slow: Optional[float] = None

        # 5m structure series
        self.highs: list[float] = []
        self.lows: list[float] = []
        self.closes: list[float] = []

        self.sweep_lookback = sweep_lookback
        self.vol_floor_pct = vol_floor_pct
        self.rv_floor_1h = rv_floor_1h
        self.pivot_len = pivot_len

        # PO3 phase + bias memory
        self.phase: PO3Phase = PO3Phase.ACCUMULATION
        self.prev_bias: int = 0

        # ---- NEW: store last structural trend for LLM snapshot ----
        # +1 = bullish structure, -1 = bearish, 0 = neutral / unknown
        self.last_struct_trend: int = 0

    # ---- helpers ----
    def _htf_bias(self) -> int:
        if self.ema_2h_fast is None or self.ema_2h_slow is None:
            return 0
        return 1 if self.ema_2h_fast > self.ema_2h_slow else -1

    def _mtf_bias(self) -> int:
        if self.ema_1h_fast is None or self.ema_1h_slow is None:
            return 0
        return 1 if self.ema_1h_fast > self.ema_1h_slow else -1

    def _combined_bias(self) -> int:
        htf = self._htf_bias()
        mtf = self._mtf_bias()

        if htf == 0 and mtf == 0:
            return 0
        if htf == mtf:
            return htf
        # disagreement -> neutral
        return 0

    # ---- main ----
    def on_bar(self, row):
        """
        row: one row of ES 5m multiframe features with columns incl:
          time, open, high, low, close, volume,
          atr_14, ret_log_1, rv_1h, rv_1d,
          range_total, range_total_pct, session, trend_flag,
          h1_close, h2_close, etc.
        """
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        session = str(row.get("session", "OFF"))
        range_pct = float(row.get("range_total_pct", 0.0))
        rv_1h = row.get("rv_1h", None)

        # -------- update HTF EMAs from 1H / 2H close --------
        h1_close = row.get("h1_close", None)
        if h1_close is not None and not _is_nan(h1_close):
            h1 = float(h1_close)
            self.ema_1h_fast = _update_ema(self.ema_1h_fast, self.alpha_1h_fast, h1)
            self.ema_1h_slow = _update_ema(self.ema_1h_slow, self.alpha_1h_slow, h1)

        h2_close = row.get("h2_close", None)
        if h2_close is not None and not _is_nan(h2_close):
            h2 = float(h2_close)
            self.ema_2h_fast = _update_ema(self.ema_2h_fast, self.alpha_2h_fast, h2)
            self.ema_2h_slow = _update_ema(self.ema_2h_slow, self.alpha_2h_slow, h2)

        # Need HTF EMAs initialized
        if any(
            v is None
            for v in [
                self.ema_1h_fast,
                self.ema_1h_slow,
                self.ema_2h_fast,
                self.ema_2h_slow,
            ]
        ):
            return "HOLD"

        bias = self._combined_bias()

        # 5m structure series
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(price)

        min_len = max(self.sweep_lookback + 3, self.pivot_len * 2 + 3)
        if len(self.highs) < min_len:
            self.prev_bias = bias
            return "HOLD"

        # -------- Volatility filter --------
        if range_pct < self.vol_floor_pct:
            self.prev_bias = bias
            return "HOLD"

        if (
            self.rv_floor_1h is not None
            and rv_1h is not None
            and not _is_nan(rv_1h)
        ):
            try:
                rv_val = float(rv_1h)
                if rv_val < self.rv_floor_1h:
                    self.prev_bias = bias
                    return "HOLD"
            except Exception:
                # if weird value, just ignore the rv filter
                pass

        # -------- Session filter (focus on London/NY) --------
        if session not in ("LONDON", "NY"):
            self.prev_bias = bias
            return "HOLD"

        # -------- Structural info (BOS / CHOCH) --------
        bos_up, bos_down, choch_up, choch_down, struct_trend = detect_structure(
            self.highs, self.lows, self.closes, pivot_len=self.pivot_len
        )

        # NEW: persist the structural trend for external use (LLM snapshot, logging, etc.)
        self.last_struct_trend = struct_trend

        # -------- SMC modules: sweeps + FVG --------
        sweep_up, sweep_down = detect_sweep(
            self.highs, self.lows, self.closes, lookback=self.sweep_lookback
        )
        bull_fvg, bear_fvg = detect_fvg(self.highs, self.lows)

        # -------- PO3 Phase update --------
        self.phase = update_phase(
            prev_phase=self.phase,
            bos_up=bos_up,
            bos_down=bos_down,
            sweep_up=sweep_up,
            sweep_down=sweep_down,
            bias=bias,
            prev_bias=self.prev_bias,
            struct_trend=struct_trend,  # <── use real structure trend here
            bull_fvg=bull_fvg,
            bear_fvg=bear_fvg,
        )
        self.prev_bias = bias

        # =====================================================
        # ENTRY LOGIC (same as GOLD, for now)
        # =====================================================

        # ===== LONG CONDITIONS =====
        long_struct_ok = bos_up or choch_up or struct_trend == 1
        long_of_ok = sweep_down or bull_fvg

        long_phase_ok = (
            self.phase in (PO3Phase.MANIPULATION, PO3Phase.EXPANSION) and bias >= 0
        )

        if long_phase_ok and long_struct_ok and long_of_ok:
            return "LONG"

        # ===== SHORT CONDITIONS =====
        short_struct_ok = bos_down or choch_down or struct_trend == -1
        short_of_ok = sweep_up or bear_fvg

        short_phase_ok = (
            self.phase in (PO3Phase.MANIPULATION, PO3Phase.EXPANSION) and bias <= 0
        )

        if short_phase_ok and short_struct_ok and short_of_ok:
            return "SHORT"

        return "HOLD"
