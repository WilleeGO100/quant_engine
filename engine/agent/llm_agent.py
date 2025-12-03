# engine/agent/llm_agent.py

class LLMAgent:
    """
    Placeholder for the LLM-based decision maker.
    Right now it just returns a simple decision based on the candle:
      - LONG  if close > open
      - SHORT if close < open
      - FLAT  otherwise
    Later this will call the LLM with a prompt and parse JSON.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def decide(self, context: dict) -> dict:
        """
        Very simple placeholder logic:
        - If close > open  -> LONG
        - If close < open  -> SHORT
        - Else             -> FLAT
        """

        bar = context.get("bar", {})
        open_price = bar.get("open")
        close_price = bar.get("close")

        # Default decision (if data is missing)
        decision = {
            "signal": "FLAT",
            "size": 0.0,
            "stop_loss": None,
            "take_profit": None,
        }

        # Only do logic if we actually have prices
        if open_price is not None and close_price is not None:
            if close_price > open_price:
                decision["signal"] = "LONG"
                decision["size"] = self.config.get("default_position_size", 0.1)
            elif close_price < open_price:
                decision["signal"] = "SHORT"
                decision["size"] = self.config.get("default_position_size", 0.1)

        return decision
