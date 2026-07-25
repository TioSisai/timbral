"""Embedding extraction orchestration package: raw cache ->
Transform/Encoder batched forward pass -> embedding DatasetDict cache.

Orchestration layer that is allowed to import both the timbral.datasets and
timbral.models components (the components themselves still do not import
each other); internally split into config / labels / builder.
"""

from timbral.embeddings.builder import prepare_embeddings
from timbral.embeddings.config import EmbPrepConfig, resolve_config

__all__ = ["EmbPrepConfig", "prepare_embeddings", "resolve_config"]
