# engine/execution/mt5_executor.py

class MT5Executor:
    """
    Placeholder MT5 execution wrapper.
    Right now it does NOT send real orders, it just prints what it would do.
    Later we'll replace the print with real MetaTrader 5 API calls.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.symbol = self.config.get("symbol", "XAUUSD")
        # later: initialize MT5 connection here

    def send_order(self, decision: dict) -> None:
        """
        decision: dictionary like:
        {
            "signal": "LONG" | "SHORT" | "FLAT",
            "size": 0.1,
            "stop_loss": 0.5,
            "take_profit": 1.0
        }
        For now, we only print it to verify flow.
        """
        print(f"[MT5Executor] Would execute on {self.symbol}: {decision}")
