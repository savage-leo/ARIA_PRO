"""
Pydantic compatibility shim (v1 and v2)
Re-exports BaseModel and provides a create_model wrapper that works across versions.
Usage:
    from backend.compat.pydantic_compat import BaseModel, create_model
"""
from __future__ import annotations

from typing import Any, Type
import logging

logger = logging.getLogger(__name__)

try:
    # Preferred import path (works in v1 and v2)
    from pydantic import BaseModel as _BaseModel  # type: ignore
    from pydantic import create_model as _create_model  # type: ignore
    from pydantic import Field, ValidationError  # type: ignore
    _PYDANTIC_IMPORT = "pydantic"
except Exception:  # pragma: no cover - fallback only when v2-only packaging differs
    # Fallback for environments where v2 provides a v1 shim namespace
    from pydantic.v1 import BaseModel as _BaseModel  # type: ignore
    from pydantic.v1 import create_model as _create_model  # type: ignore
    from pydantic.v1 import Field, ValidationError  # type: ignore
    _PYDANTIC_IMPORT = "pydantic.v1"

# Public aliases
BaseModel: Type[_BaseModel] = _BaseModel


def create_model(name: str, __config__: Any | None = None, __base__: Type[_BaseModel] | None = None,
                 __module__: str | None = None, **fields: Any) -> Type[_BaseModel]:
    """
    Compatibility wrapper for pydantic.create_model.
    - Tries full v1-style signature first.
    - Falls back to v2 minimal signature if needed.
    """
    try:
        return _create_model(
            name,
            __config__=__config__,
            __base__=__base__ or BaseModel,
            __module__=__module__ or __name__,
            **fields,
        )
    except TypeError as e:
        # Some v2 builds may not accept __config__/__module__
        logger.debug("pydantic.create_model fallback (%s): %s", _PYDANTIC_IMPORT, e)
        return _create_model(
            name,
            __base__=__base__ or BaseModel,
            **fields,
        )

__all__ = ["BaseModel", "create_model", "Field", "ValidationError"]
