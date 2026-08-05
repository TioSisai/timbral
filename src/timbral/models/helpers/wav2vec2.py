"""Identity, download, and safe-loading logic for the fixed wav2vec2
checkpoint, plus the conv frontend geometry shared by the Transform and
the Encoder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import Wav2Vec2Config

from .common import ensure_hf_snapshot, verify_file_sha256


@dataclass(frozen=True, slots=True)
class Wav2Vec2CheckpointMetadata:
    """Describes a fixed Hugging Face wav2vec2 checkpoint."""

    repo_id: str
    revision: str
    filenames: tuple[str, ...]
    sha256: dict[str, str]


WAV2VEC2_CHECKPOINT = Wav2Vec2CheckpointMetadata(
    repo_id="facebook/wav2vec2-base",
    revision="0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
    filenames=(
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
    ),
    sha256={
        "config.json": (
            "4937977e24d12d1bba70cdce8709c3c0"
            "4807a8e4ae8ddac4229c48c436ae99ae"
        ),
        "preprocessor_config.json": (
            "b225d617c025463b9e157e06afea8b90"
            "dc7078fc70b013c533328423e0486b4a"
        ),
        "pytorch_model.bin": (
            "3249fe98bfc62fcbc26067f724716a6e"
            "c49d12c4728a2af1df659013905dff21"
        ),
    },
)

WAV2VEC2_CONFIG_FIELDS: dict[str, Any] = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "feat_extract_norm": "group",
    "feat_extract_activation": "gelu",
    "conv_dim": [512, 512, 512, 512, 512, 512, 512],
    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
    "conv_bias": False,
    "num_feat_extract_layers": 7,
    "num_conv_pos_embeddings": 128,
    "num_conv_pos_embedding_groups": 16,
    "do_stable_layer_norm": False,
    "layer_norm_eps": 1e-5,
    "apply_spec_augment": True,
    "mask_time_prob": 0.05,
    "mask_feature_prob": 0.0,
    "hidden_dropout": 0.1,
    "attention_dropout": 0.1,
    "activation_dropout": 0.0,
    "feat_proj_dropout": 0.1,
    "final_dropout": 0.0,
    "layerdrop": 0.0,
    "initializer_range": 0.02,
}

# Conv frontend geometry derived from conv_kernel/conv_stride: the product
# of the strides is the hop, and the receptive field of one output frame
# is 400 input samples; both are fixed by the architecture contract above.
WAV2VEC2_HOP_SAMPLES = 320
WAV2VEC2_MIN_TARGET_SAMPLES = 400

_BACKBONE_PREFIX = "wav2vec2."
_EXPECTED_BACKBONE_TENSORS = 211
_EXPECTED_PRETRAINING_HEAD_KEYS = frozenset(
    {
        "project_hid.bias",
        "project_hid.weight",
        "project_q.bias",
        "project_q.weight",
        "quantizer.codevectors",
        "quantizer.weight_proj.bias",
        "quantizer.weight_proj.weight",
    }
)
_POS_CONV_KEY_RENAMES = {
    "encoder.pos_conv_embed.conv.weight_g": (
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
    ),
    "encoder.pos_conv_embed.conv.weight_v": (
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1"
    ),
}


def verify_wav2vec2_file(path: Path, expected_sha256: str) -> None:
    """Verify the SHA-256 of a wav2vec2 file.

    Args:
        path: File to verify.
        expected_sha256: Fixed hexadecimal digest.

    Raises:
        ValueError: The file digest does not match the fixed identity.
    """
    verify_file_sha256(path, expected_sha256, label="wav2vec2 file")


def ensure_wav2vec2_checkpoint(
    pretrained_dir: str | Path | None,
) -> Path:
    """Prepare and verify the fixed wav2vec2 snapshot (download to a temp
    dir first, then atomically move in after verification).

    Args:
        pretrained_dir: Explicit snapshot directory; when ``None``, uses
            the project-specific directory under the HF cache.

    Returns:
        The directory containing all three fixed files, each verified
        against its digest.
    """
    return ensure_hf_snapshot(
        repo_id=WAV2VEC2_CHECKPOINT.repo_id,
        revision=WAV2VEC2_CHECKPOINT.revision,
        filenames=WAV2VEC2_CHECKPOINT.filenames,
        sha256=WAV2VEC2_CHECKPOINT.sha256,
        pretrained_dir=pretrained_dir,
        label="wav2vec2 file",
    )


def fixed_wav2vec2_config() -> Wav2Vec2Config:
    """Construct the supported wav2vec2 configuration from the fields
    fixed in code.
    """
    return Wav2Vec2Config(**WAV2VEC2_CONFIG_FIELDS)


def load_and_validate_wav2vec2_config(config_path: Path) -> Wav2Vec2Config:
    """Load the official config and validate all key architecture fields.

    Args:
        config_path: The ``config.json`` in the fixed snapshot.

    Returns:
        The validated ``Wav2Vec2Config`` built from the fixed fields.

    Raises:
        ValueError: A config field is missing or conflicts with the local
            fixed contract.
    """
    with config_path.open(encoding="utf-8") as file:
        configuration = json.load(file)

    mismatches = {
        name: (expected, configuration.get(name))
        for name, expected in WAV2VEC2_CONFIG_FIELDS.items()
        if configuration.get(name) != expected
    }
    if mismatches:
        details = "; ".join(
            f"{name}: expected {expected!r}, got {actual!r}"
            for name, (expected, actual) in mismatches.items()
        )
        raise ValueError(
            "wav2vec2 config does not match the fixed architecture: "
            f"{details}."
        )
    return fixed_wav2vec2_config()


def load_wav2vec2_backbone_state(
    checkpoint_path: Path,
) -> dict[str, Tensor]:
    """Safely load the checkpoint and extract the full backbone state.

    The official checkpoint stores a ``Wav2Vec2ForPreTraining`` model; the
    backbone tensors carry the ``wav2vec2.`` prefix and the pos_conv
    weight-norm tensors use the legacy ``weight_g``/``weight_v`` names.

    Args:
        checkpoint_path: The fixed ``pytorch_model.bin``.

    Returns:
        A state dict with the backbone prefix stripped and the pos_conv
        keys renamed to the parametrize convention used by the fixed
        Transformers ``Wav2Vec2Model``.

    Raises:
        ValueError: The backbone tensor count is wrong, or the excluded
            keys are not the fixed pretraining heads.
    """
    checkpoint_state = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location="cpu",
    )
    excluded_keys = frozenset(
        key
        for key in checkpoint_state
        if not key.startswith(_BACKBONE_PREFIX)
    )
    if excluded_keys != _EXPECTED_PRETRAINING_HEAD_KEYS:
        raise ValueError(
            "Non-backbone tensor set in wav2vec2 checkpoint does not "
            f"match: expected {sorted(_EXPECTED_PRETRAINING_HEAD_KEYS)}, "
            f"got {sorted(excluded_keys)}."
        )

    backbone_state = {}
    for checkpoint_key, value in checkpoint_state.items():
        if not checkpoint_key.startswith(_BACKBONE_PREFIX):
            continue
        model_key = checkpoint_key.removeprefix(_BACKBONE_PREFIX)
        model_key = _POS_CONV_KEY_RENAMES.get(model_key, model_key)
        backbone_state[model_key] = value
    if len(backbone_state) != _EXPECTED_BACKBONE_TENSORS:
        raise ValueError(
            "wav2vec2 backbone tensor count does not match: "
            f"expected {_EXPECTED_BACKBONE_TENSORS}, "
            f"got {len(backbone_state)}."
        )
    return backbone_state


def wav2vec2_feature_frames(target_valid_samples: Tensor) -> Tensor:
    """Map 16 kHz valid sample counts to conv frontend output frame counts.

    Applies the official per-layer formula
    ``floor((length - kernel) / stride) + 1`` over the seven fixed conv
    layers, matching ``Wav2Vec2Model._get_feat_extract_output_lengths``.

    Args:
        target_valid_samples: Integer tensor of valid 16 kHz sample
            counts.

    Returns:
        An integer tensor of valid output frame counts with the same
        shape.
    """
    lengths = target_valid_samples
    for kernel, stride in zip(
        WAV2VEC2_CONFIG_FIELDS["conv_kernel"],
        WAV2VEC2_CONFIG_FIELDS["conv_stride"],
    ):
        lengths = torch.div(
            lengths - kernel,
            stride,
            rounding_mode="floor",
        ) + 1
    return lengths


__all__ = (
    "WAV2VEC2_CHECKPOINT",
    "WAV2VEC2_CONFIG_FIELDS",
    "WAV2VEC2_HOP_SAMPLES",
    "WAV2VEC2_MIN_TARGET_SAMPLES",
    "Wav2Vec2CheckpointMetadata",
    "ensure_wav2vec2_checkpoint",
    "fixed_wav2vec2_config",
    "load_and_validate_wav2vec2_config",
    "load_wav2vec2_backbone_state",
    "verify_wav2vec2_file",
    "wav2vec2_feature_frames",
)
