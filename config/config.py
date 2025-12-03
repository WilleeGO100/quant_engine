# config/config.py

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict


CONFIG_PATH = Path(__file__).resolve().parent / "settings.yaml"


class Config:
    """
    Simple container for all config values.
    Accepts arbitrary keyword arguments and stores them as attributes.
    This allows us to freely add new settings in settings.yaml
    without breaking the code.
    """

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"


# ----------------------------
# YAML Loader
# ----------------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"settings.yaml not found at: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ----------------------------
# Public load function
# ----------------------------
def load_config() -> Config:
    raw = _load_yaml(CONFIG_PATH)

    # raw is already a dict mapping keys → values from YAML
    # We don't enforce specific fields — Config allows arbitrary keys.
    return Config(**raw)
