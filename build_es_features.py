# build_es_features.py

"""
Offline feature generation for ES 5m data.

This DOES NOT change your backtests behavior.
It simply reads data/raw/es_5m_clean.csv
and writes data/features/es_5m_features.csv
"""

from pathlib import Path

import pandas as pd

from engine.data_loader import ROOT  # already defined in data_loader.py
from engine.features import build_es_5m_features
from engine.features import build_es_5m_features


def main():
    data_raw = ROOT / "data" / "raw" / "es_5m_clean.csv"
    out_dir = ROOT / "data" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "es_5m_features.csv"

    print(f"Loading raw 5m ES data from: {data_raw}")
    df_raw = pd.read_csv(data_raw)

    print("Building features...")
    df_feat = build_es_5m_features(df_raw)

    print(f"Saving features to: {out_path}")
    df_feat.to_csv(out_path, index=False)

    print("Done. Preview:")
    print(df_feat.head())


if __name__ == "__main__":
    main()
