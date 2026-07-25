"""Identity, download, and safe-loading logic for the fixed CLAP checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open
from torch import Tensor
from transformers import ClapAudioConfig

from .common import ensure_hf_snapshot, verify_file_sha256


@dataclass(frozen=True, slots=True)
class ClapCheckpointMetadata:
    """Describes the fixed Hugging Face CLAP checkpoint."""

    repo_id: str
    revision: str
    filenames: tuple[str, ...]
    sha256: dict[str, str]


CLAP_CHECKPOINT = ClapCheckpointMetadata(
    repo_id="laion/clap-htsat-fused",
    revision="365dea6ef167def6676140ed93bbc43f84dabb28",
    filenames=(
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ),
    sha256={
        "config.json": (
            "b1d63489dc5061da229c23d2b11e9ca"
            "731639574449f82319fabb01da7fcf480"
        ),
        "preprocessor_config.json": (
            "072bdd9ba771b6d213c56f15c0f765e3"
            "3192b92e481581b52271cf16c9013684"
        ),
        "model.safetensors": (
            "3f648de6d030e17494be455d323b8d19"
            "1233fbae0c7ce0ba745fd21a926a63a6"
        ),
    },
)

CLAP_AUDIO_CONFIG_FIELDS: dict[str, Any] = {
    "window_size": 8,
    "num_mel_bins": 64,
    "spec_size": 256,
    "hidden_act": "gelu",
    "patch_size": 4,
    "patch_stride": [4, 4],
    "num_classes": 527,
    "hidden_size": 768,
    "projection_dim": 512,
    "depths": [2, 2, 6, 2],
    "num_attention_heads": [4, 8, 16, 32],
    "enable_fusion": True,
    "hidden_dropout_prob": 0.1,
    "fusion_type": None,
    "patch_embed_input_channels": 1,
    "flatten_patch_embeds": True,
    "patch_embeds_hidden_size": 96,
    "enable_patch_layer_norm": True,
    "drop_path_rate": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "qkv_bias": True,
    "mlp_ratio": 4.0,
    "aff_block_r": 4,
    "num_hidden_layers": 4,
    "projection_hidden_act": "relu",
    "layer_norm_eps": 1e-5,
    "initializer_factor": 1.0,
}

CLAP_PREPROCESSOR_FIELDS: dict[str, Any] = {
    "feature_extractor_type": "ClapFeatureExtractor",
    "feature_size": 64,
    "sampling_rate": 48000,
    "hop_length": 480,
    "fft_window_size": 1024,
    "frequency_min": 50,
    "frequency_max": 14000,
    "max_length_s": 10,
    "nb_max_samples": 480000,
    "padding": "repeatpad",
    "truncation": "fusion",
}

_AUDIO_MODEL_PREFIX = "audio_model."
_AUDIO_PROJECTION_PREFIX = "audio_projection."
_EXPECTED_AUDIO_MODEL_TENSORS = 266
_EXPECTED_AUDIO_PROJECTION_TENSORS = 4


def verify_clap_file(path: Path, expected_sha256: str) -> None:
    """Verify the SHA-256 of a CLAP file.

    Args:
        path: File to verify.
        expected_sha256: Fixed hexadecimal digest.

    Raises:
        ValueError: The file digest does not match the fixed identity.
    """
    verify_file_sha256(path, expected_sha256, label="CLAP file")


def ensure_clap_checkpoint(
    pretrained_dir: str | Path | None,
) -> Path:
    """Prepare and verify the fixed CLAP snapshot (download to a temp dir
    first, then atomically move in after verification).

    Args:
        pretrained_dir: Explicit snapshot directory; when ``None``, uses
            the project-specific directory under the HF cache.

    Returns:
        The directory containing all three fixed files, each verified
        against its digest.
    """
    return ensure_hf_snapshot(
        repo_id=CLAP_CHECKPOINT.repo_id,
        revision=CLAP_CHECKPOINT.revision,
        filenames=CLAP_CHECKPOINT.filenames,
        sha256=CLAP_CHECKPOINT.sha256,
        pretrained_dir=pretrained_dir,
        label="CLAP file",
    )


def fixed_clap_audio_config() -> ClapAudioConfig:
    """Construct the CLAP audio configuration from the fields fixed in code."""
    return ClapAudioConfig(**CLAP_AUDIO_CONFIG_FIELDS)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"CLAP JSON top level must be an object: {path}.")
    return value


def _validate_fields(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    source_name: str,
) -> None:
    """Validate fixed JSON fields."""
    mismatches = {
        name: (expected_value, actual.get(name))
        for name, expected_value in expected.items()
        if actual.get(name) != expected_value
    }
    if mismatches:
        details = "; ".join(
            f"{name}: expected {expected_value!r}, got {actual_value!r}"
            for name, (expected_value, actual_value) in mismatches.items()
        )
        raise ValueError(f"{source_name} fields do not match: {details}.")


def load_and_validate_clap_audio_config(
    config_path: Path,
    preprocessor_path: Path,
) -> ClapAudioConfig:
    """Load and validate the fixed CLAP configuration.

    Args:
        config_path: The snapshot's ``config.json``.
        preprocessor_path: The snapshot's ``preprocessor_config.json``.

    Returns:
        The fixed CLAP audio configuration.
    """
    configuration = _load_json(config_path)
    audio_configuration = configuration.get("audio_config")
    if not isinstance(audio_configuration, dict):
        raise ValueError("CLAP config is missing the audio_config object.")
    _validate_fields(
        audio_configuration,
        CLAP_AUDIO_CONFIG_FIELDS,
        source_name="CLAP audio_config",
    )
    _validate_fields(
        _load_json(preprocessor_path),
        CLAP_PREPROCESSOR_FIELDS,
        source_name="CLAP preprocessor_config",
    )
    return fixed_clap_audio_config()


def load_clap_audio_state(
    checkpoint_path: Path,
) -> dict[str, Tensor]:
    """Safely load the CLAP audio tower and projection state.

    Args:
        checkpoint_path: The fixed snapshot's ``model.safetensors``.

    Returns:
        A state dict with names matching
        ``ClapAudioModelWithProjection.state_dict``.

    Raises:
        ValueError: The audio tower or projection tensor count does not
            match.
    """
    with safe_open(
        checkpoint_path,
        framework="pt",
        device="cpu",
    ) as checkpoint:
        keys = tuple(checkpoint.keys())
        audio_model_keys = tuple(
            key
            for key in keys
            if key.startswith(_AUDIO_MODEL_PREFIX)
        )
        audio_projection_keys = tuple(
            key
            for key in keys
            if key.startswith(_AUDIO_PROJECTION_PREFIX)
        )
        if len(audio_model_keys) != _EXPECTED_AUDIO_MODEL_TENSORS:
            raise ValueError(
                "CLAP audio_model tensor count does not match: "
                f"expected {_EXPECTED_AUDIO_MODEL_TENSORS}, "
                f"got {len(audio_model_keys)}."
            )
        if (
            len(audio_projection_keys)
            != _EXPECTED_AUDIO_PROJECTION_TENSORS
        ):
            raise ValueError(
                "CLAP audio_projection tensor count does not match: "
                f"expected {_EXPECTED_AUDIO_PROJECTION_TENSORS}, "
                f"got {len(audio_projection_keys)}."
            )
        selected_keys = audio_model_keys + audio_projection_keys
        return {
            key: checkpoint.get_tensor(key)
            for key in selected_keys
        }


__all__ = (
    "CLAP_AUDIO_CONFIG_FIELDS",
    "CLAP_CHECKPOINT",
    "CLAP_PREPROCESSOR_FIELDS",
    "ClapCheckpointMetadata",
    "ensure_clap_checkpoint",
    "fixed_clap_audio_config",
    "load_and_validate_clap_audio_config",
    "load_clap_audio_state",
    "verify_clap_file",
)
