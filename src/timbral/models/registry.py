"""Model registry: builds a matched Transform/Encoder pair by registered name."""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from .encoders import (
    AstEncoder,
    BaseEncoder,
    BeatsEncoder,
    ClapHtsatEncoder,
    Granularity,
    PannsCnn14Encoder,
    Wav2Vec2Encoder,
)
from .helpers.ast_helpers import AST_CHECKPOINT
from .helpers.beats import BEATS_CHECKPOINTS
from .helpers.clap import CLAP_CHECKPOINT
from .helpers.panns import PANNS_CHECKPOINTS, PANNS_OFFICIAL_FRONTENDS
from .helpers.wav2vec2 import WAV2VEC2_CHECKPOINT
from .transforms import (
    AstKaldiFbankTransform,
    BaseTransform,
    BeatsKaldiFbankTransform,
    ClapLogmelTransform,
    PannsLogmelTransform,
    Wav2Vec2WaveformTransform,
)

_PUBLIC_PARAMETER_NAMES = frozenset(
    ("granularity", "pretrained", "pretrained_dir")
)


class ModelPair(NamedTuple):
    """A matched Transform/Encoder pair."""

    transform: BaseTransform
    encoder: BaseEncoder


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """The pairing declaration for one registered name.

    Constructor arguments are routed according to each component
    constructor's signature, relying on the existing contract that component
    constructors explicitly declare all their parameters and do not accept
    ``**kwargs``.

    Attributes:
        transform_cls: The concrete Transform class.
        encoder_cls: The concrete Encoder class.
        fixed_kwargs: The constructor arguments fixed for this registered
            name.
    """

    transform_cls: type[BaseTransform]
    encoder_cls: type[BaseEncoder]
    fixed_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a fixed configuration that conflicts with
        ``create_model``'s public parameters.
        """
        conflicts = sorted(
            self.fixed_kwargs.keys() & _PUBLIC_PARAMETER_NAMES
        )
        if conflicts:
            raise ValueError(
                f"fixed_kwargs must not contain the public parameters "
                f"{conflicts}."
            )


MODELS: dict[str, ModelSpec] = {
    AST_CHECKPOINT.repo_id: ModelSpec(
        transform_cls=AstKaldiFbankTransform,
        encoder_cls=AstEncoder,
    ),
    CLAP_CHECKPOINT.repo_id: ModelSpec(
        transform_cls=ClapLogmelTransform,
        encoder_cls=ClapHtsatEncoder,
    ),
    WAV2VEC2_CHECKPOINT.repo_id: ModelSpec(
        transform_cls=Wav2Vec2WaveformTransform,
        encoder_cls=Wav2Vec2Encoder,
        fixed_kwargs={"do_normalize": True},
    ),
    **{
        metadata.model_name: ModelSpec(
            transform_cls=PannsLogmelTransform,
            encoder_cls=PannsCnn14Encoder,
            fixed_kwargs={
                "target_sample_rate": target_sample_rate,
                "variant": variant,
                **PANNS_OFFICIAL_FRONTENDS[target_sample_rate],
            },
        )
        for (
            target_sample_rate,
            variant,
        ), metadata in PANNS_CHECKPOINTS.items()
    },
    **{
        entry: ModelSpec(
            transform_cls=BeatsKaldiFbankTransform,
            encoder_cls=BeatsEncoder,
            fixed_kwargs={"checkpoint": entry},
        )
        for entry in BEATS_CHECKPOINTS
    },
}


def register_model(name: str, spec: ModelSpec) -> None:
    """Register a new model pairing declaration (for new model extensions
    or test injection).
    """
    MODELS[name] = spec


def create_model(
    name: str,
    *,
    granularity: Granularity,
    pretrained: bool = True,
    pretrained_dir: str | Path | None = None,
    **kwargs: Any,
) -> ModelPair:
    """Build a matched Transform/Encoder pair by registered name.

    The public parameters, the registered name's fixed parameters, and
    ``**kwargs`` are merged and then routed according to the two component
    constructors' signatures: a component whose constructor declares a given
    parameter name receives that parameter; a shared parameter declared by
    both constructors is passed the same value to both; a public parameter
    not declared by either component is not passed at all.

    Args:
        name: The registered name; see :func:`list_models` for all available
            values.
        granularity: The Encoder's output granularity, ``"clip"`` or
            ``"frame"``.
        pretrained: Whether to load the official pretrained weights.
        pretrained_dir: An explicit weights directory; when ``None``, each
            component's default cache path is used.
        **kwargs: Model-specific parameters, routed together with the fixed
            parameters.

    Returns:
        A matched :class:`ModelPair`.

    Raises:
        KeyError: ``name`` is not registered.
        TypeError: ``**kwargs`` overrides a fixed parameter, or a
            model-specific parameter is not declared by either component
            constructor.
    """
    if name not in MODELS:
        raise KeyError(f"Model {name} is not registered; registered: "
                       f"{sorted(MODELS)}")
    spec = MODELS[name]

    overridden = sorted(spec.fixed_kwargs.keys() & kwargs.keys())
    if overridden:
        raise TypeError(
            f"Parameters {overridden} are fixed by registered name {name} "
            "and cannot be overridden at the call site."
        )

    transform_params = frozenset(
        inspect.signature(spec.transform_cls).parameters
    )
    encoder_params = frozenset(
        inspect.signature(spec.encoder_cls).parameters
    )
    specific = {**spec.fixed_kwargs, **kwargs}
    unrouted = sorted(
        param
        for param in specific
        if param not in transform_params and param not in encoder_params
    )
    if unrouted:
        raise TypeError(
            f"Parameters {unrouted} are not declared by either "
            f"{spec.transform_cls.__name__}'s or "
            f"{spec.encoder_cls.__name__}'s constructor."
        )

    candidates = {
        "granularity": granularity,
        "pretrained": pretrained,
        "pretrained_dir": pretrained_dir,
        **specific,
    }
    transform = spec.transform_cls(
        **{k: v for k, v in candidates.items() if k in transform_params}
    )
    encoder = spec.encoder_cls(
        **{k: v for k, v in candidates.items() if k in encoder_params}
    )
    return ModelPair(transform=transform, encoder=encoder)


def list_models() -> list[str]:
    """Return all registered names sorted alphabetically."""
    return sorted(MODELS)


__all__ = (
    "ModelPair",
    "ModelSpec",
    "create_model",
    "list_models",
    "register_model",
)
