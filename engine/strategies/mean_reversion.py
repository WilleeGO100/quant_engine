# engine/strategies/mean_reversion.py

from collections import deque
from .base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Simple mean-reversion strategy:

      - Maintain rolling window of closes
      - Compute mean & std
      - If price > mean + z_entry*std  -> SHORT
      - If price < mean - z_entry*std  -> LONG
      - Exit when z-score returns to near 0
    """

    def __init__(self, lookback: int = 50, z_entry: float = 1.0, z_exit: float = 0.3):
        super().__init__()
        self.lookback = lookback
        self.z_entry = z_entry
        self.z_exit = z_exit

        # 👇 THIS is the deque that caused the NameError
        self.prices = deque(maxlen=lookback)

        # internal tracking: long=1, short=-1, flat=0
        self.in_position = 0

    def _mean_std(self):
        n = len(self.prices)
        if n == 0:
            return None, None
        mean = sum(self.prices) / n
        var = sum((p - mean) ** 2 for p in self.prices) / n
        std = var ** 0.5
        return mean, std

    def on_bar(self, row):
        price = float(row["close"])
        self.prices.append(price)

        # Not enough samples for stats
        if len(self.prices) < self.lookback:
            return "HOLD"

        mean, std = self._mean_std()
        if std is None or std == 0:
            return "HOLD"

        z = (price - mean) / std

        # ------------------------------
        # ENTRY SIGNALS
        # ------------------------------
        if self.in_position == 0:
            if z > self.z_entry:
                self.in_position = -1
                return "SHORT"
            elif z < -self.z_entry:
                self.in_position = 1
                return "LONG"
            else:
                return "HOLD"

        # ------------------------------
        # EXIT SIGNALS
        # ------------------------------
        if abs(z) < self.z_exit:
            self.in_position = 0
            return "FLAT"

        return "HOLD"
