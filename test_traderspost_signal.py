# test_traderspost_signal.py
#
# Minimal test to verify TradersPostExecutor builds the correct JSON
# and (optionally) reaches your webhook.

from engine.execution.traderspost_executor import TradersPostExecutor


def main() -> None:
    # >>> IMPORTANT <<<
    # Paste YOUR webhook URL here.
    webhook_url = "https://webhooks.traderspost.io/trading/webhook/6d521a51-75c0-49fc-acda-427f9af5e6e2/ef543ebe65285b354d3844419465db9a"

    config = {
        "traderspost_webhook_url": webhook_url,
        "trade_symbol": "MNQ",        # symbol TP expects
        "live_trading": False,        # START FALSE (dry run)
    }

    executor = TradersPostExecutor(config)

    # Define the trade you want to send
    signal = "LONG"          # or "SHORT"
    size = 1.0               # quantity
    test_flag = True         # TradersPost test mode
    meta = {
        "source": "test_traderspost_signal.py",
        "note": "sanity test"
    }

    executor.send_order(
        signal=signal,
        size=size,
        test=test_flag,
        meta=meta
    )


if __name__ == "__main__":
    main()
