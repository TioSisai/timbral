"""Identity, download, and safe-loading logic for the fixed AST checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file
from torch import Tensor
from transformers import ASTConfig

from .common import ensure_hf_snapshot, verify_file_sha256


@dataclass(frozen=True, slots=True)
class AstCheckpointMetadata:
    """Describes a fixed Hugging Face AST checkpoint."""

    repo_id: str
    revision: str
    filenames: tuple[str, ...]
    sha256: dict[str, str]


AST_CHECKPOINT = AstCheckpointMetadata(
    repo_id="MIT/ast-finetuned-audioset-10-10-0.4593",
    revision="f826b80d28226b62986cc218e5cec390b1096902",
    filenames=(
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ),
    sha256={
        "config.json": (
            "a93d525511d77e8ecc933d09674b8509"
            "9815bbbb417c228a4edd655e252fb9ff"
        ),
        "preprocessor_config.json": (
            "8d04ba5a9c6fca5d39d0de2b1fd05ec"
            "f79deb589fbba279728bbebac39934231"
        ),
        "model.safetensors": (
            "ae0c1e2ad4e1381d851fa9bf298ba13e"
            "bc9c5a914cdee2dbe427a6583869924d"
        ),
    },
)

AST_CONFIG_FIELDS: dict[str, Any] = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "patch_size": 16,
    "frequency_stride": 10,
    "time_stride": 10,
    "num_mel_bins": 128,
    "max_length": 1024,
    "qkv_bias": True,
}

_BACKBONE_PREFIX = "audio_spectrogram_transformer."
_EXPECTED_BACKBONE_TENSORS = 199
_EXPECTED_CLASSIFIER_KEYS = frozenset(
    {
        "classifier.dense.bias",
        "classifier.dense.weight",
        "classifier.layernorm.bias",
        "classifier.layernorm.weight",
    }
)
_BACKBONE_KEY_RENAMES = (
    ("encoder.layer.", "layers."),
    ("attention.query", "q_proj"),
    ("attention.key", "k_proj"),
    ("attention.value", "v_proj"),
    ("attention.output.dense", "attention.o_proj"),
    ("intermediate.dense", "mlp.fc1"),
    ("output.dense", "mlp.fc2"),
)


def verify_ast_file(path: Path, expected_sha256: str) -> None:
    """Verify the SHA-256 of an AST file.

    Args:
        path: File to verify.
        expected_sha256: Fixed hexadecimal digest.

    Raises:
        ValueError: The file digest does not match the fixed identity.
    """
    verify_file_sha256(path, expected_sha256, label="AST file")


def ensure_ast_checkpoint(
    pretrained_dir: str | Path | None,
) -> Path:
    """Prepare and verify the fixed AST snapshot (download to a temp dir
    first, then atomically move in after verification).

    Args:
        pretrained_dir: Explicit snapshot directory; when ``None``, uses
            the project-specific directory under the HF cache.

    Returns:
        The directory containing all three fixed files, each verified
        against its digest.
    """
    return ensure_hf_snapshot(
        repo_id=AST_CHECKPOINT.repo_id,
        revision=AST_CHECKPOINT.revision,
        filenames=AST_CHECKPOINT.filenames,
        sha256=AST_CHECKPOINT.sha256,
        pretrained_dir=pretrained_dir,
        label="AST file",
    )


def fixed_ast_config() -> ASTConfig:
    """Construct the supported AST configuration from the fields fixed in
    code.
    """
    return ASTConfig(**AST_CONFIG_FIELDS)


def load_and_validate_ast_config(config_path: Path) -> ASTConfig:
    """Load the official config and validate all key architecture fields.

    Args:
        config_path: The ``config.json`` in the fixed snapshot.

    Returns:
        The validated official ``ASTConfig``.

    Raises:
        ValueError: A config field is missing or conflicts with the local
            fixed contract.
    """
    with config_path.open(encoding="utf-8") as file:
        configuration = json.load(file)

    mismatches = {
        name: (expected, configuration.get(name))
        for name, expected in AST_CONFIG_FIELDS.items()
        if configuration.get(name) != expected
    }
    if mismatches:
        details = "; ".join(
            f"{name}: expected {expected!r}, got {actual!r}"
            for name, (expected, actual) in mismatches.items()
        )
        raise ValueError(
            f"AST config does not match the fixed architecture: {details}."
        )
    return ASTConfig.from_json_file(config_path)


def load_ast_backbone_state(
    checkpoint_path: Path,
) -> dict[str, Tensor]:
    """Safely load the checkpoint and extract the full AST backbone state.

    Args:
        checkpoint_path: The fixed ``model.safetensors``.

    Returns:
        A state dict with the checkpoint's backbone prefix stripped and
        keys renamed to match the fixed Transformers 5.13.1 ``ASTModel``
        parameter names.

    Raises:
        ValueError: The backbone tensor count is wrong, or the excluded
            keys are not the fixed classifier head.
    """
    checkpoint_state = load_file(checkpoint_path, device="cpu")
    excluded_keys = frozenset(
        key
        for key in checkpoint_state
        if not key.startswith(_BACKBONE_PREFIX)
    )
    if excluded_keys != _EXPECTED_CLASSIFIER_KEYS:
        raise ValueError(
            "Non-backbone tensor set in AST checkpoint does not match: "
            f"expected {sorted(_EXPECTED_CLASSIFIER_KEYS)}, "
            f"got {sorted(excluded_keys)}."
        )

    backbone_state = {}
    for checkpoint_key, value in checkpoint_state.items():
        if not checkpoint_key.startswith(_BACKBONE_PREFIX):
            continue
        model_key = checkpoint_key.removeprefix(_BACKBONE_PREFIX)
        for source, target in _BACKBONE_KEY_RENAMES:
            model_key = model_key.replace(source, target)
        if model_key in backbone_state:
            raise ValueError(
                f"Conflict while renaming AST backbone key: {model_key}."
            )
        backbone_state[model_key] = value
    if len(backbone_state) != _EXPECTED_BACKBONE_TENSORS:
        raise ValueError(
            "AST backbone tensor count does not match: "
            f"expected {_EXPECTED_BACKBONE_TENSORS}, "
            f"got {len(backbone_state)}."
        )
    return backbone_state


__all__ = (
    "AST_CHECKPOINT",
    "AST_CONFIG_FIELDS",
    "AstCheckpointMetadata",
    "ensure_ast_checkpoint",
    "fixed_ast_config",
    "load_and_validate_ast_config",
    "load_ast_backbone_state",
    "verify_ast_file",
)
