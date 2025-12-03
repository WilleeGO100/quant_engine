# backtests/inspect_btc_features.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

print("[INSPECT_BTC] Module imported...")

from engine.features.institutional_features import (
    build_btc_5m_multiframe_features_institutional,
)


def main() -> None:
    print("[INSPECT_BTC] Building BTC institutional features...")
    df, n_rows = build_btc_5m_multiframe_features_institutional()
    print(f"[INSPECT_BTC] BTC institutional features built. Rows: {n_rows}")

    print("[INSPECT_BTC] First 5 columns:", list(df.columns)[:10])
    print("[INSPECT_BTC] First 5 rows:")
    print(df.head())


if __name__ == "__main__":
    print("[INSPECT_BTC] __main__ guard hit, running main()")
    main()
