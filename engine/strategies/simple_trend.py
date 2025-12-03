# engine/strategies/simple_trend.py

from .base import BaseStrategy


class SimpleTrendStrategy(BaseStrategy):
    """
    Very simple trend-following strategy:
      - Uses two EMAs (fast & slow)
      - When fast EMA crosses ABOVE slow EMA -> go LONG
      - When fast EMA crosses BELOW slow EMA -> go SHORT

    The backtests engine manages the actual position.
    We just return signals: "LONG", "SHORT", "FLAT" or "HOLD".
    """

    def __init__(self, fast_len: int = 20, slow_len: int = 50):
        self.fast_len = fast_len
        self.slow_len = slow_len

        # EMA smoothing factors
        self.alpha_fast = 2 / (fast_len + 1)
        self.alpha_slow = 2 / (slow_len + 1)

        # Running EMA values
        self.ema_fast = None
        self.ema_slow = None

        # Track which side fast EMA was on last bar (above/below slow)
        self.prev_fast_above = None

    def on_bar(self, row):
        """
        row: pandas Series with at least:
            - close

        Returns:
            "LONG"  -> open/flip to long
            "SHORT" -> open/flip to short
            "FLAT"  -> flat (we don't use this here)
            "HOLD"  -> keep current position
        """

        price = float(row["close"])

        # First bar: initialize EMAs, no signal yet
        if self.ema_fast is None or self.ema_slow is None:
            self.ema_fast = price
            self.ema_slow = price
            self.prev_fast_above = None
            return "HOLD"

        # Update EMAs
        self.ema_fast = (
            self.alpha_fast * price + (1 - self.alpha_fast) * self.ema_fast
        )
        self.ema_slow = (
            self.alpha_slow * price + (1 - self.alpha_slow) * self.ema_slow
        )

        fast_above = self.ema_fast > self.ema_slow
        signal = "HOLD"

        # Look for crossovers
        if self.prev_fast_above is not None:
            # Bear -> Bull: go long
            if fast_above and not self.prev_fast_above:
                signal = "LONG"
            # Bull -> Bear: go short
            elif (not fast_above) and self.prev_fast_above:
                signal = "SHORT"

        # Save state for next bar
        self.prev_fast_above = fast_above
        return signal
