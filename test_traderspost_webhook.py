# test_traderspost_webhook.py
#
# One-off sanity check that your TradersPost webhook
# URL is reachable and accepts a JSON payload.
#
# This uses "test": true so it should not place a real trade.

import json
import requests

WEBHOOK_URL = (
    "https://webhooks.traderspost.io/trading/webhook/"
    "6d521a51-75c0-49fc-acda-427f9af5e6e2/ef543ebe65285b354d3844419465db9a"
)

def main() -> None:
    payload = {
        "ticker": "MNQ",
        "action": "buy",
        "quantity": 1.0,
        "orderType": "market",
        "test": False,  # <- TradersPost "paper" mode
    }

    print("Sending payload to TradersPost (test mode):")
    print(json.dumps(payload, indent=2))

    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"\n[ERROR] Request failed: {e!r}")
        return

    print("\nResponse from TradersPost:")
    print(f"Status: {resp.status_code}")
    print(resp.text)

if __name__ == "__main__":
    main()
