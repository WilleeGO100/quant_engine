# test_traderspost_signal.py
#
# Minimal sanity check that TradersPostExecutor builds the right JSON
# and (in live mode) can reach your webhook.

from engine.execution.traderspost_executor import TradersPostExecutor


def main() -> None:
    # >>>> IMPORTANT <<<<
    # Paste YOUR webhook URL string between the quotes below.
    # Example: "https://webhooks.traderspost.io/trading/webhook/6d52.../ef54..."
    webhook_url = "https://webhooks.traderspost.io/trading/webhook/6d521a51-75c0-49fc-acda-427f9af5e6e2/ef543ebe65285b354d3844419465db9a"

    config = {
        "traderspost_webhook_url": webhook_url,
        "traderspost_ticker": "MNQ",   # matches your curl example
        "entry_order_type": "market",

        # START WITH DRY RUN = False to only print payload.
        # When you're absolutely sure, set this to True to hit the webhook.
        "live_trading": True,
    }

    executor = TradersPostExecutor(config=config)

    # This mimics the decision dict coming from your strategy
    decision = {
        "signal": "LONG",    # -> action "buy"
        "size": 1.0,         # quantity
        # optional extras:
        # "signal_price": 17450.25,
        # "take_profit_pct": 10.0,
        # "stop_loss_pct": 5.0,
    }

    executor.send_order(decision, current_state=None)


if __name__ == "__main__":
    main()
