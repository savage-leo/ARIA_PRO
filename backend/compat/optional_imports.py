"""
Optional import utilities for heavy/optional dependencies.

Example usage:
    torch, TORCH_AVAILABLE = optional_import("torch")
    PPO, SB3_AVAILABLE = optional_attr("stable_baselines3", "PPO")
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)


def optional_import(name: str, default: Any | None = None, warn: bool = True) -> Tuple[Any | None, bool]:
    """
    Try to import a module by name.
    Returns (module_or_default, available_flag).
    """
    try:
        module = importlib.import_module(name)
        return module, True
    except Exception as e:  # pragma: no cover - only hit when module missing
        if warn:
            logger.debug("Optional import '%s' not available: %s", name, e)
        return default, False


def optional_attr(module_name: str, attr: str, default: Any | None = None, warn: bool = True) -> Tuple[Any | None, bool]:
    """
    Try to import an attribute from a module.
    Returns (attribute_or_default, available_flag).
    """
    mod, ok = optional_import(module_name, warn=warn)
    if not ok or mod is None:
        return default, False
    try:
        return getattr(mod, attr), True
    except AttributeError:  # pragma: no cover
        if warn:
            logger.debug("Optional attribute '%s.%s' not found", module_name, attr)
        return default, False


__all__ = ["optional_import", "optional_attr"]
