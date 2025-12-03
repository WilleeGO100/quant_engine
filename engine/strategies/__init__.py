# engine/strategies/__init__.py

"""
Strategy registry and public interface.

Usage:
    from engine.strategies import get_strategy_class

    StrategyClass = get_strategy_class("simple_trend")
    strategy = StrategyClass()
"""

from .simple_trend import SimpleTrendStrategy
from .mean_reversion import MeanReversionStrategy
from .smc_po3 import SMCPo3Strategy


_STRATEGY_REGISTRY = {
    "simple_trend": SimpleTrendStrategy,
    "mean_reversion": MeanReversionStrategy,
    "smc_po3": SMCPo3Strategy,
}


def get_strategy_class(name: str):
    """
    Returns the strategy class associated with the given name string.
    """
    cls = _STRATEGY_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy_name: {name}. Available: {available}")
    return cls
