# engine/strategies/smc_po3_power.py

from __future__ import annotations

from typing import Optional

from .base import BaseStrategy
from engine.modules.liquidity import detect_sweep
from engine.modules.fvg import detect_fvg
from engine.modules.structure import detect_structure


def _update_ema(prev: Optional[float], alpha: float, price: float) -> float:
    if prev is None:
        return price
    return alpha * price + (1 - alpha) * prev


def _is_nan(x) -> bool:
    return isinstance(x, float) and x != x


class SMCPo3PowerStrategy(BaseStrategy):
    """
    SMC + Volatility + Multi-Timeframe PO3-style engine.

    Uses:
      - 2H & 1H EMAs for directional bias
      - Structural events (BOS / CHOCH) from swing highs/lows
      - Liquidity sweeps (wick-based) above/below recent highs/lows
      - Fair Value Gaps (3-candle imbalance)
      - Volatility filters (range percentile, realized vol)
      - Session filter (LONDON / NY focus)

    Entry logic (simplified):

      LONG if:
        - HTF+MTF bias >= 0 (bullish/neutral)
        - Structure supports upside (BOS_UP or CHOCH_UP or trend == +1)
        - And we have either:
            * liquidity sweep DOWN (grab below) OR
            * bullish FVG

      SHORT if:
        - HTF+MTF bias <= 0 (bearish/neutral)
        - Structure supports downside (BOS_DOWN or CHOCH_DOWN or trend == -1)
        - And we have either:
            * liquidity sweep UP OR
            * bearish FVG
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

        # Debug / logging
        self.last_reason: str = "INIT"

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

    def get_last_reason(self) -> str:
        """Return the last decision reason (for logging/backtests)."""
        return self.last_reason

    # ---- main ----
    def on_bar(self, row):
        """
        row: one row of ES 5m multiframe features with columns incl:
          time, open, high, low, close, volume,
          atr_14, ret_log_1, rv_1h, rv_1d,
          range_total, range_total_pct, session, trend_flag,
          h1_close, h2_close, etc.
        """
        self.last_reason = "HOLD_DEFAULT"

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
            self.last_reason = "WAITING_FOR_EMA_WARMUP"
            return "HOLD"

        bias = self._combined_bias()

        # 5m structure series
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(price)

        if len(self.highs) < max(self.sweep_lookback + 3, self.pivot_len * 2 + 3):
            self.last_reason = "NOT_ENOUGH_5M_HISTORY"
            return "HOLD"

        # -------- Volatility filter --------
        if range_pct < self.vol_floor_pct:
            self.last_reason = f"LOW_RANGE_VOL ({range_pct:.3f} < {self.vol_floor_pct:.3f})"
            return "HOLD"

        if (
            self.rv_floor_1h is not None
            and rv_1h is not None
            and not _is_nan(rv_1h)
        ):
            try:
                rv_val = float(rv_1h)
                if rv_val < self.rv_floor_1h:
                    self.last_reason = f"LOW_RV_1H ({rv_val:.3f} < {self.rv_floor_1h:.3f})"
                    return "HOLD"
            except Exception:
                # if it's junk, just ignore the rv_1h filter
                pass

        # -------- Session filter (focus on London/NY) --------
        if session not in ("LONDON", "NY"):
            self.last_reason = f"OFF_SESSION_{session}"
            return "HOLD"

        # -------- Structural info (BOS / CHOCH) --------
        bos_up, bos_down, choch_up, choch_down, struct_trend = detect_structure(
            self.highs, self.lows, self.closes, pivot_len=self.pivot_len
        )

        # -------- SMC modules: sweeps + FVG --------
        sweep_up, sweep_down = detect_sweep(
            self.highs, self.lows, self.closes, lookback=self.sweep_lookback
        )
        bull_fvg, bear_fvg = detect_fvg(self.highs, self.lows)

        # ===== LONG CONDITIONS =====
        long_struct_ok = bos_up or choch_up or struct_trend == 1
        long_of_ok = sweep_down or bull_fvg

        if bias >= 0 and long_struct_ok and long_of_ok:
            self.last_reason = (
                "LONG: bias>=0, "
                f"struct_ok(bos_up={bos_up}, choch_up={choch_up}, trend={struct_trend}), "
                f"of_ok(sweep_down={sweep_down}, bull_fvg={bull_fvg})"
            )
            return "LONG"

        # ===== SHORT CONDITIONS =====
        short_struct_ok = bos_down or choch_down or struct_trend == -1
        short_of_ok = sweep_up or bear_fvg

        if bias <= 0 and short_struct_ok and short_of_ok:
            self.last_reason = (
                "SHORT: bias<=0, "
                f"struct_ok(bos_down={bos_down}, choch_down={choch_down}, trend={struct_trend}), "
                f"of_ok(sweep_up={sweep_up}, bear_fvg={bear_fvg})"
            )
            return "SHORT"

        # If we reached here, we explicitly explain why no trade:
        if bias > 0 and not long_struct_ok:
            self.last_reason = "NO_LONG: BULLISH_BIAS_BUT_STRUCT_NOT_UP"
        elif bias > 0 and long_struct_ok and not long_of_ok:
            self.last_reason = "NO_LONG: BULLISH_BIAS+STRUCT_UP_BUT_NO_SWEEP_OR_BULL_FVG"
        elif bias < 0 and not short_struct_ok:
            self.last_reason = "NO_SHORT: BEARISH_BIAS_BUT_STRUCT_NOT_DOWN"
        elif bias < 0 and short_struct_ok and not short_of_ok:
            self.last_reason = "NO_SHORT: BEARISH_BIAS+STRUCT_DOWN_BUT_NO_SWEEP_OR_BEAR_FVG"
        else:
            self.last_reason = "NO_SETUP: BIAS_NEUTRAL_OR_CONFLICTING"

        return "HOLD"
