# live/inspect_btcpo3_strategy.py
import inspect
from engine.strategies.smc_po3_power_btc import BTCPO3PowerStrategy, BTCPO3PowerConfig

def main():
    cfg = BTCPO3PowerConfig()
    strat = BTCPO3PowerStrategy(cfg)

    print("=== BTCPO3PowerStrategy: public methods ===")
    methods = []
    for name in dir(strat):
        if name.startswith("_"):
            continue
        attr = getattr(strat, name)
        if callable(attr):
            methods.append(name)

    for name in sorted(methods):
        try:
            sig = str(inspect.signature(getattr(strat, name)))
        except Exception:
            sig = "(signature unavailable)"
        print(f"- {name}{sig}")

    print("\n=== Class source location ===")
    try:
        print(inspect.getsourcefile(BTCPO3PowerStrategy))
    except Exception as e:
        print("Could not resolve source file:", e)

if __name__ == "__main__":
    main()
