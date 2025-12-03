# engine/strategies/base.py

class BaseStrategy:
    """
    Base class for all strategies.
    The backtests engine expects each strategy to implement:
        - on_bar(row): returns a signal: "LONG", "SHORT", "FLAT", or "HOLD"

    Subclasses can store parameters, indicators, running states, etc.
    """

    def __init__(self):
        pass

    def on_bar(self, row):
        """
        row: pandas Series
        Should be overridden by child strategies.
        """
        raise NotImplementedError("on_bar() must be implemented by strategy classes.")
