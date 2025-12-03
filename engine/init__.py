# engine/__init__.py

"""
Core engine package.

Subpackages:
- strategies   -> all strategy classes and registry
- agent        -> LLM agent logic
- execution    -> order execution backends (e.g., MT5)
- risk         -> risk management tools

Utilities:
- data_loader  -> CSV / data loading helpers

IMPORTANT:
We intentionally do NOT import any strategies or other submodules here.
That avoids circular imports and keeps package resolution clean.

Use them explicitly like:

    from engine.data_loader import load_csv
    from engine.strategies import get_strategy_class
"""

# Do NOT put any imports here.
# No `from .simple_trend import ...`
# No `from .strategies import ...`
# This file only marks the package and documents structure.
