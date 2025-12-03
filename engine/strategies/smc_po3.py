# engine/strategies/smc_po3.py

from .base import BaseStrategy
from engine.modules.liquidity import detect_sweep
from engine.modules.fvg import detect_fvg


def _update_ema(prev, alpha, price):
    if prev is None:
        return price
    return alpha * price + (1 - alpha) * prev


def _is_nan(x) -> bool:
    return isinstance(x, float) and x != x


class SMCPo3Strategy(BaseStrategy):
    """
    Balanced SMC – v2 (more realistic trade frequency)

    Improvements:
    - Sweep lookback lowered for ES (default 10)
    - FVG detection broadened (any displacement in last 3 bars)
    - HTF bias optional (prefer agreement but not required)
    - Accept either: sweep→FVG OR FVG→sweep
    - This matches real-world ES SMC behavior more accurately
    """

    def __init__(
        self,
        fast_5m: int = 10,
        slow_5m: int = 40,
        fast_1h: int = 8,
        slow_1h: int = 24,
        fast_2h: int = 8,
        slow_2h: int = 24,
        sweep_lookback: int = 10,   # LOWER = more sweeps found
        fvg_window: int = 3,        # NEW: look back 3 bars for FVG
    ):
        super().__init__()

        # EMA parameters
        self.alpha_5m_fast = 2 / (fast_5m + 1)
        self.alpha_5m_slow = 2 / (slow_5m + 1)
        self.alpha_1h_fast = 2 / (fast_1h + 1)
        self.alpha_1h_slow = 2 / (slow_1h + 1)
        self.alpha_2h_fast = 2 / (fast_2h + 1)
        self.alpha_2h_slow = 2 / (slow_2h + 1)

        # EMA state
        self.ema_5m_fast = None
        self.ema_5m_slow = None
        self.ema_1h_fast = None
        self.ema_1h_slow = None
        self.ema_2h_fast = None
        self.ema_2h_slow = None

        # 5m structure
        self.highs = []
        self.lows = []
        self.closes = []

        self.sweep_lookback = sweep_lookback
        self.fvg_window = fvg_window

    # Helpers
    def _htf_bias(self):
        if self.ema_2h_fast is None:
            return 0
        return 1 if self.ema_2h_fast > self.ema_2h_slow else -1

    def _mtf_bias(self):
        if self.ema_1h_fast is None:
            return 0
        return 1 if self.ema_1h_fast > self.ema_1h_slow else -1

    # Main
    def on_bar(self, row):
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        # Update 5m EMAs first
        self.ema_5m_fast = _update_ema(self.ema_5m_fast, self.alpha_5m_fast, price)
        self.ema_5m_slow = _update_ema(self.ema_5m_slow, self.alpha_5m_slow, price)

        # Update 1h EMA
        h1 = row.get("h1_close", None)
        if h1 is not None and not _is_nan(h1):
            h1 = float(h1)
            self.ema_1h_fast = _update_ema(self.ema_1h_fast, self.alpha_1h_fast, h1)
            self.ema_1h_slow = _update_ema(self.ema_1h_slow, self.alpha_1h_slow, h1)

        # Update 2h EMA
        h2 = row.get("h2_close", None)
        if h2 is not None and not _is_nan(h2):
            h2 = float(h2)
            self.ema_2h_fast = _update_ema(self.ema_2h_fast, self.alpha_2h_fast, h2)
            self.ema_2h_slow = _update_ema(self.ema_2h_slow, self.alpha_2h_slow, h2)

        # Need EMAs to start
        if any(v is None for v in [self.ema_5m_fast, self.ema_5m_slow, self.ema_1h_fast, self.ema_1h_slow, self.ema_2h_fast, self.ema_2h_slow]):
            return "HOLD"

        htf = self._htf_bias()
        mtf = self._mtf_bias()

        # NEW: bias logic
        if htf == mtf:
            bias = htf       # strongest
        elif htf == 0 or mtf == 0:
            bias = 0        # neutral
        else:
            bias = 0        # disagreement

        # Update structure arrays
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(price)

        if len(self.highs) < max(self.sweep_lookback + 5, 5):
            return "HOLD"

        # MODULES
        sweep_up, sweep_down = detect_sweep(self.highs, self.lows, self.closes, lookback=self.sweep_lookback)
        bull_fvg, bear_fvg = detect_fvg(self.highs, self.lows)

        # BALANCED ENTRY CONDITIONS (v2)

        # SHORT scenarios
        if (
            (bias <= 0) and                # bearish or neutral
            (sweep_up and bear_fvg) or     # classic sweep->FVG
            (bear_fvg and sweep_up)        # OR FVG->sweep
        ):
            return "SHORT"

        # LONG scenarios
        if (
            (bias >= 0) and                 # bullish or neutral
            (sweep_down and bull_fvg) or    # sweep->FVG
            (bull_fvg and sweep_down)       # FVG->sweep
        ):
            return "LONG"

        return "HOLD"
