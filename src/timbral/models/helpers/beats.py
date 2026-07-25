"""Checkpoint identity, expected cfg, and loading logic shared by BEATs
components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub.constants import HF_HUB_CACHE
from torch import Tensor

from timbral.paths import project_root

from .common import sha256_file

_PREDICTOR_KEYS = frozenset(("predictor.weight", "predictor.bias"))
_BACKBONE_TENSOR_COUNT = 250
# The official checkpoint uses legacy weight_norm key names; the local
# module uses the parametrize version.
_POS_CONV_KEY_RENAMES = {
    "encoder.pos_conv.0.weight_g": (
        "encoder.pos_conv.0.parametrizations.weight.original0"
    ),
    "encoder.pos_conv.0.weight_v": (
        "encoder.pos_conv.0.parametrizations.weight.original1"
    ),
}


@dataclass(frozen=True, slots=True)
class BeatsCheckpointMetadata:
    """Describes a fixed official BEATs checkpoint."""

    filename: str
    sha256: str
    finetuned: bool


# Entry names come from the cell text of the last three columns in the
# official README table: lowercased; hyphens/spaces/parentheses converted
# to underscores; ``+`` converted to ``_plus``; consecutive underscores
# collapsed and leading/trailing underscores stripped.
_BEATS_CHECKPOINT_IDENTITIES: dict[str, tuple[str, bool]] = {
    "beats_iter1": (
        "b5f4cc10bcbff63a437c695f33389e6411513b3f7d5cdae8fb62b5005f4a1fcd",
        False,
    ),
    "fine_tuned_beats_iter1_cpt1": (
        "e0e739e3670bfbb93c51adefb1d02981621397addc979d392aefd3dc53c22cab",
        True,
    ),
    "fine_tuned_beats_iter1_cpt2": (
        "2f3a7b65ab232c4f75570d4d17e21e5ebc34b3c40fe1a074f27d199e81354960",
        True,
    ),
    "beats_iter2": (
        "81a23e00aa4878d7e8627ded87ea697fb347c8ceffed21223e0398ed0fa34ad8",
        False,
    ),
    "fine_tuned_beats_iter2_cpt1": (
        "3a120810c0f6dbfd50a7f48dc03ed077971a50cb2dbb7999695d5c700d03da45",
        True,
    ),
    "fine_tuned_beats_iter2_cpt2": (
        "08363b9b5eabeb47b0879c84145b27c603e7e50c116a633fa5b98ade119fc354",
        True,
    ),
    "beats_iter3": (
        "8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34",
        False,
    ),
    "fine_tuned_beats_iter3_cpt1": (
        "379369a41d0b3749f746cdcea8036de506cb3aedecce84de7db0a75fda2a4fe7",
        True,
    ),
    "fine_tuned_beats_iter3_cpt2": (
        "08374f1cbd49143900b351bc81cd307de386a11f8e609eb3862634e992068b55",
        True,
    ),
    "beats_iter3_plus_as20k": (
        "8008b126bb5e8ab08912c60c58847ed676d32e64a5864c922356b7c2522fb2f8",
        False,
    ),
    "fine_tuned_beats_iter3_plus_as20k_cpt1": (
        "2c366278dcf835e9bdefad4f7147b0edba4b940c59146fd05dc49a401fa82ff8",
        True,
    ),
    "fine_tuned_beats_iter3_plus_as20k_cpt2": (
        "6d28b32bfa7bcaaf84ab834186581c2a360c6669e372e808d054cf0ef4d5c2d2",
        True,
    ),
    "beats_iter3_plus_as2m": (
        "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34",
        False,
    ),
    "fine_tuned_beats_iter3_plus_as2m_cpt1": (
        "7f9362028ac6e5c049e8dc314d87e90e4f82a15a8e472deb56af55d7f9b34d6a",
        True,
    ),
    "fine_tuned_beats_iter3_plus_as2m_cpt2": (
        "e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c",
        True,
    ),
}

BEATS_CHECKPOINTS: dict[str, BeatsCheckpointMetadata] = {
    entry: BeatsCheckpointMetadata(
        filename=f"{entry}.pt",
        sha256=sha256,
        finetuned=finetuned,
    )
    for entry, (sha256, finetuned) in _BEATS_CHECKPOINT_IDENTITIES.items()
}

# The inference-related architecture fields are identical across all 15
# checkpoints, verified empirically; differences are only in training-time
# fields. Two full-field expectation tables are kept, split by
# pretrained/finetuned, and strictly compared against the checkpoint's cfg
# at load time.
_BEATS_COMMON_CFG: dict[str, Any] = {
    "input_patch_size": 16,
    "embed_dim": 512,
    "conv_bias": False,
    "encoder_layers": 12,
    "encoder_embed_dim": 768,
    "encoder_ffn_embed_dim": 3072,
    "encoder_attention_heads": 12,
    "activation_fn": "gelu",
    "layer_norm_first": False,
    "deep_norm": True,
    "activation_dropout": 0.0,
    "encoder_layerdrop": 0.05,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "relative_position_embedding": True,
    "num_buckets": 320,
    "max_distance": 800,
    "gru_rel_pos": True,
}

BEATS_EXPECTED_CFG_PRETRAINED: dict[str, Any] = {
    **_BEATS_COMMON_CFG,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "dropout_input": 0.1,
    "layer_wise_gradient_decay_ratio": 1.0,
}

BEATS_EXPECTED_CFG_FINETUNED: dict[str, Any] = {
    **_BEATS_COMMON_CFG,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "dropout_input": 0.0,
    "layer_wise_gradient_decay_ratio": 0.6,
    "finetuned_model": True,
    "predictor_dropout": 0.0,
    "predictor_class": 527,
}


def _lookup_metadata(entry: str) -> BeatsCheckpointMetadata:
    """Look up checkpoint metadata by entry name."""
    if entry not in BEATS_CHECKPOINTS:
        raise ValueError(
            f"Unknown BEATs entry {entry!r}, "
            f"available: {sorted(BEATS_CHECKPOINTS)}."
        )
    return BEATS_CHECKPOINTS[entry]


def ensure_beats_checkpoint(
    entry: str,
    pretrained_dir: str | Path | None,
) -> Path:
    """Resolve and verify an official BEATs checkpoint (does not download).

    Args:
        entry: One of the 15 entry names.
        pretrained_dir: Explicit checkpoint directory; when ``None``, uses
            the HF cache.

    Returns:
        The checkpoint path, verified against its SHA-256.

    Raises:
        ValueError: The entry is unknown, or the SHA-256 does not match.
        FileNotFoundError: The file is missing; the message includes the
            invocation command for the download script.
    """
    metadata = _lookup_metadata(entry)
    if pretrained_dir is None:
        directory = Path(HF_HUB_CACHE) / "audioencoders" / "beats"
    else:
        directory = Path(pretrained_dir)
    checkpoint_path = directory / metadata.filename

    if not checkpoint_path.exists():
        script_path = project_root() / "scripts" / "extra" / "beats_dl.py"
        raise FileNotFoundError(
            f"BEATs checkpoint is missing: {checkpoint_path}. Please run "
            f"`python {script_path} --dest {directory} --entries {entry}` "
            "to download it (this script requires playwright and requests)."
        )
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != metadata.sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: {checkpoint_path}, "
            f"expected {metadata.sha256}, got {actual_sha256}."
        )
    return checkpoint_path


def load_beats_checkpoint_state(
    entry: str,
    checkpoint_path: Path,
) -> dict[str, Tensor]:
    """Safely load the checkpoint and return the backbone state with the
    predictor dropped.

    Args:
        entry: One of the 15 entry names.
        checkpoint_path: The verified checkpoint file.

    Returns:
        A state dict with exactly 250 tensors that can be loaded strictly
        into ``BeatsEncoder``; the two pos_conv weight_norm keys have been
        mapped to their parametrize names.

    Raises:
        ValueError: The cfg does not match the expectation table, or the
            state_dict key set is anomalous.
    """
    metadata = _lookup_metadata(entry)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    expected_cfg = (
        BEATS_EXPECTED_CFG_FINETUNED
        if metadata.finetuned
        else BEATS_EXPECTED_CFG_PRETRAINED
    )
    actual_cfg = checkpoint["cfg"]
    if actual_cfg != expected_cfg:
        mismatched = sorted(
            (set(actual_cfg) ^ set(expected_cfg))
            | {
                field
                for field in set(actual_cfg) & set(expected_cfg)
                if actual_cfg[field] != expected_cfg[field]
            }
        )
        raise ValueError(
            "checkpoint cfg does not match the expectation table: "
            f"{checkpoint_path}, differing fields {mismatched}."
        )

    state: dict[str, Tensor] = checkpoint["model"]
    predictor_keys = {
        key for key in state if key.startswith("predictor.")
    }
    if metadata.finetuned:
        if predictor_keys != set(_PREDICTOR_KEYS):
            raise ValueError(
                f"Anomalous predictor keys in finetuned checkpoint: "
                f"{sorted(predictor_keys)}."
            )
        for key in predictor_keys:
            del state[key]
    elif predictor_keys:
        raise ValueError(
            f"Pretrained checkpoint should not contain predictor keys: "
            f"{sorted(predictor_keys)}."
        )
    if len(state) != _BACKBONE_TENSOR_COUNT:
        raise ValueError(
            f"Anomalous backbone tensor count: expected "
            f"{_BACKBONE_TENSOR_COUNT}, got {len(state)}."
        )
    for legacy_key, parametrize_key in _POS_CONV_KEY_RENAMES.items():
        state[parametrize_key] = state.pop(legacy_key)
    return state


__all__ = (
    "BEATS_CHECKPOINTS",
    "BEATS_EXPECTED_CFG_FINETUNED",
    "BEATS_EXPECTED_CFG_PRETRAINED",
    "BeatsCheckpointMetadata",
    "ensure_beats_checkpoint",
    "load_beats_checkpoint_state",
)
