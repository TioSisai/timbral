"""Audio model components.

Concrete Transform and Encoder classes are imported from
:mod:`timbral.models.transforms` and :mod:`timbral.models.encoders`
respectively; this module only re-exports the caller-facing symbols from
:mod:`timbral.models.registry`.
"""

from .registry import (
    ModelPair,
    ModelSpec,
    create_model,
    list_models,
    register_model,
)

__all__ = (
    "ModelPair",
    "ModelSpec",
    "create_model",
    "list_models",
    "register_model",
)
