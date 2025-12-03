# engine/strategy_registry.py

"""
Backward-compat wrapper for the strategy registry.

Old code:
    from engine.strategy_registry import get_strategy

New preferred usage:
    from engine.strategies import get_strategy_class
"""

from engine.strategies import get_strategy_class as get_strategy
