"""Alignment tests between the local AST and pinned Hugging Face feature extraction, backbone, and real weights."""

from __future__ import annotations

import hashlib
import inspect
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import transformers
from torch import Tensor
from transformers import ASTFeatureExtractor, ASTModel

from timbral.models.encoders import AstEncoder
from timbral.models.helpers.ast_helpers import (
    AST_CHECKPOINT,
    ensure_ast_checkpoint,
)
from timbral.models.transforms import AstKaldiFbankTransform

pytestmark = pytest.mark.alignment("ast")

_TRANSFORMERS_VERSION = "5.13.1"
_EXPECTED_REPO_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
_EXPECTED_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
_EXPECTED_CHECKPOINT_SHA256 = {
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
}
_SOURCE_SHA256 = {
    ASTFeatureExtractor: (
        "ab4957749b5113067413dcd662dc2129"
        "52b9a610d297e8b4515e2cab1ff1fce4"
    ),
    ASTModel: (
        "5ef9fe1c7847400453095c158c761919"
        "13226788eaa1f4ba6afbb378b9e70547"
    ),
}
_DURATION_CASES = (
    ("random", 0.025),
    ("sine", 0.9999375),
    ("impulse", 4.03),
    ("silence", 10.0),
    ("multisine", 10.245),
    ("random_invalid_tail", 10.255),
)
_TRANSFORM_ATOL = 1e-4
_ENCODER_ATOL = 1e-4
_ENCODER_RTOL = 1e-4
_MIN_COSINE = 0.99999
_MAX_RELATIVE_L2 = 1e-4


@dataclass
class _Metric:
    """Hold one set of alignment audit metrics."""

    max_absolute: float
    cosine: float
    absolute_l2: float
    relative_l2: float


@pytest.fixture(scope="module")
def snapshot_directory() -> Path:
    """Prepare the pinned model snapshot under TMPDIR."""
    temporary_root = Path(
        os.environ.get("TMPDIR", tempfile.gettempdir())
    )
    return ensure_ast_checkpoint(
        temporary_root / "timbral-ast-alignment" / "snapshot"
    )


@pytest.fixture(scope="module")
def reference_feature_extractor(
    snapshot_directory: Path,
) -> ASTFeatureExtractor:
    """Load the pinned official ASTFeatureExtractor."""
    return ASTFeatureExtractor.from_pretrained(
        snapshot_directory,
        local_files_only=True,
    )


@pytest.fixture(scope="module")
def aligned_models(snapshot_directory: Path):
    """Load the local Encoder and the official ASTModel."""
    local_encoder = AstEncoder(
        granularity="clip",
        pretrained=True,
        pretrained_dir=snapshot_directory,
    ).eval()
    reference_model = ASTModel.from_pretrained(
        snapshot_directory,
        local_files_only=True,
    ).eval()
    return local_encoder, reference_model


def _sha256(path: Path) -> str:
    """Compute the file's SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signal(case_name: str, valid_samples: int) -> Tensor:
    """Build a fixed mono test signal."""
    if case_name.startswith("random"):
        generator = torch.Generator().manual_seed(valid_samples)
        return torch.randn(valid_samples, generator=generator) * 0.1
    if case_name == "silence":
        return torch.zeros(valid_samples)
    if case_name == "impulse":
        waveform = torch.zeros(valid_samples)
        waveform[valid_samples // 2] = 1.0
        return waveform

    time = torch.arange(valid_samples, dtype=torch.float32) / 16000
    if case_name == "sine":
        return 0.1 * torch.sin(2 * torch.pi * 997 * time)
    generator = torch.Generator().manual_seed(valid_samples)
    return (
        0.31 * torch.sin(2 * torch.pi * 321 * time)
        + 0.17 * torch.cos(2 * torch.pi * 1123 * time)
        + 0.05 * torch.randn(valid_samples, generator=generator)
        + 0.03
    )


def _mixed_waveforms() -> tuple[Tensor, Tensor, tuple[str, ...]]:
    """Build a mixed-length batch with a nonzero invalid tail."""
    physical_samples = 164080 + 257
    waveforms = []
    durations = []
    names = []
    for index, (case_name, duration) in enumerate(_DURATION_CASES):
        valid_samples = round(duration * 16000)
        waveform = torch.full((physical_samples,), 3.0 + index)
        waveform[:valid_samples] = _signal(case_name, valid_samples)
        waveforms.append(waveform)
        durations.append(duration)
        names.append(f"{case_name}:{duration}")
    return (
        torch.stack(waveforms),
        torch.tensor(durations, dtype=torch.float32),
        tuple(names),
    )


def _reference_features(
    extractor: ASTFeatureExtractor,
    waveforms: Tensor,
    valid_seconds: Tensor,
) -> Tensor:
    """Run the official CPU feature extractor sample by sample."""
    features = []
    for waveform, seconds in zip(waveforms, valid_seconds, strict=True):
        valid_samples = round(float(seconds) * 16000)
        output = extractor(
            waveform[:valid_samples].numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        features.append(output.input_values[0])
    return torch.stack(features)


def _metric(actual: Tensor, expected: Tensor) -> _Metric:
    """Compute the max difference, cosine, and L2 norms in float64."""
    actual_flat = actual.detach().to(device="cpu", dtype=torch.float64).flatten()
    expected_flat = (
        expected.detach().to(device="cpu", dtype=torch.float64).flatten()
    )
    difference = actual_flat - expected_flat
    absolute_l2 = torch.linalg.vector_norm(difference)
    expected_l2 = torch.linalg.vector_norm(expected_flat)
    return _Metric(
        max_absolute=float(difference.abs().max()),
        cosine=float(
            torch.nn.functional.cosine_similarity(
                actual_flat,
                expected_flat,
                dim=0,
            )
        ),
        absolute_l2=float(absolute_l2),
        relative_l2=float(absolute_l2 / expected_l2),
    )


def _assert_metric(
    actual: Tensor,
    expected: Tensor,
    *,
    atol: float,
    rtol: float,
) -> _Metric:
    """Enforce both the pointwise and the unified audit gates."""
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
    )
    metric = _metric(actual, expected)
    assert metric.cosine >= _MIN_COSINE
    assert metric.relative_l2 <= _MAX_RELATIVE_L2
    return metric


def _assert_global_metric(
    actual: Tensor,
    expected: Tensor,
) -> _Metric:
    """Enforce the cosine and relative-L2 gates on a non-public intermediate state."""
    metric = _metric(actual, expected)
    assert metric.cosine >= _MIN_COSINE
    assert metric.relative_l2 <= _MAX_RELATIVE_L2
    return metric


def _worst(metrics: list[tuple[str, _Metric]]) -> dict[str, object]:
    """Aggregate metrics and keep the worst case for each item."""
    return {
        "max_absolute": max(
            metrics,
            key=lambda item: item[1].max_absolute,
        ),
        "min_cosine": min(
            metrics,
            key=lambda item: item[1].cosine,
        ),
        "max_absolute_l2": max(
            metrics,
            key=lambda item: item[1].absolute_l2,
        ),
        "max_relative_l2": max(
            metrics,
            key=lambda item: item[1].relative_l2,
        ),
    }


def _expected_frame_output(
    hidden: Tensor,
    valid_seconds: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the project's frame-output contract from the official hidden state."""
    embedding = hidden[:, 2:].reshape(-1, 12, 101, 768).mean(dim=1)
    valid_samples = torch.round(valid_seconds * 16000).long()
    valid_frames = torch.div(
        valid_samples + 1599,
        1600,
        rounding_mode="floor",
    ).clamp(max=101)
    indices = torch.arange(101, device=hidden.device)
    valid_mask = indices.unsqueeze(0) < valid_frames.unsqueeze(1)
    boundaries = torch.arange(
        102,
        device=hidden.device,
        dtype=torch.float32,
    ) * 0.1
    starts = boundaries[:-1].unsqueeze(0).expand(hidden.shape[0], -1)
    ends = torch.minimum(
        boundaries[1:].unsqueeze(0),
        valid_seconds.float().unsqueeze(1),
    )
    ends = ends.scatter(
        1,
        (valid_frames - 1).unsqueeze(1),
        valid_seconds.float().unsqueeze(1),
    )
    geometry = torch.stack((starts, ends), dim=2)
    return (
        embedding * valid_mask.unsqueeze(2),
        geometry * valid_mask.unsqueeze(2),
        valid_mask,
    )


def test_fixed_transformers_and_checkpoint_identity(
    snapshot_directory: Path,
):
    assert transformers.__version__ == _TRANSFORMERS_VERSION
    assert AST_CHECKPOINT.repo_id == _EXPECTED_REPO_ID
    assert AST_CHECKPOINT.revision == _EXPECTED_REVISION
    assert AST_CHECKPOINT.filenames == tuple(
        _EXPECTED_CHECKPOINT_SHA256
    )
    assert AST_CHECKPOINT.sha256 == _EXPECTED_CHECKPOINT_SHA256
    for implementation, expected_sha256 in _SOURCE_SHA256.items():
        source_path = Path(inspect.getsourcefile(implementation))
        assert _sha256(source_path) == expected_sha256
    for filename, expected_sha256 in _EXPECTED_CHECKPOINT_SHA256.items():
        assert (
            _sha256(snapshot_directory / filename)
            == expected_sha256
        )


def test_exact_backbone_state_matches_official_loader(aligned_models):
    local_encoder, reference_model = aligned_models
    local_state = local_encoder.backbone.state_dict()
    reference_state = reference_model.state_dict()

    assert len(local_state) == len(reference_state) == 199
    assert local_state.keys() == reference_state.keys()
    for key in local_state:
        assert torch.equal(local_state[key], reference_state[key]), key


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="No CUDA available in the current environment.",
            ),
        ),
    ],
)
def test_official_transform_encoder_and_project_outputs_align(
    device_name: str,
    reference_feature_extractor: ASTFeatureExtractor,
    aligned_models,
):
    device = torch.device(device_name)
    local_encoder, reference_model = aligned_models
    local_encoder.to(device)
    reference_model.to(device)
    local_transform = AstKaldiFbankTransform().to(device)
    waveforms, valid_seconds, case_names = _mixed_waveforms()
    reference_features = _reference_features(
        reference_feature_extractor,
        waveforms,
        valid_seconds,
    )

    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.inference_mode():
            local_features = local_transform(
                waveforms,
                sample_rate=16000,
                valid_seconds=valid_seconds,
            )["input_features"]
            transform_metrics = [
                (
                    name,
                    _assert_metric(
                        local_features[index],
                        reference_features[index].to(device),
                        atol=_TRANSFORM_ATOL,
                        rtol=0.0,
                    ),
                )
                for index, name in enumerate(case_names)
            ]

            hidden_metrics = []
            clip_metrics = []
            frame_metrics = []
            for index, name in enumerate(case_names):
                seconds = valid_seconds[index : index + 1].to(device)
                local_input = local_features[index : index + 1]
                reference_input = reference_features[
                    index : index + 1
                ].to(device)
                local_backbone_output = local_encoder.backbone(
                    input_values=local_input
                )
                reference_output = reference_model(
                    input_values=reference_input
                )

                with patch.object(
                    local_encoder,
                    "_backbone_outputs",
                    return_value=local_backbone_output,
                ):
                    local_encoder.granularity = "clip"
                    local_clip = local_encoder(
                        local_input,
                        valid_seconds=seconds,
                    )
                    local_encoder.granularity = "frame"
                    local_frame = local_encoder(
                        local_input,
                        valid_seconds=seconds,
                    )

                reference_frame = _expected_frame_output(
                    reference_output.last_hidden_state,
                    seconds,
                )
                hidden_metrics.append(
                    (
                        name,
                        _assert_global_metric(
                            local_backbone_output.last_hidden_state,
                            reference_output.last_hidden_state,
                        ),
                    )
                )
                clip_metrics.append(
                    (
                        name,
                        _assert_metric(
                            local_clip["embedding"],
                            reference_output.pooler_output,
                            atol=_ENCODER_ATOL,
                            rtol=_ENCODER_RTOL,
                        ),
                    )
                )
                frame_metrics.append(
                    (
                        name,
                        _assert_metric(
                            local_frame["embedding"],
                            reference_frame[0],
                            atol=_ENCODER_ATOL,
                            rtol=_ENCODER_RTOL,
                        ),
                    )
                )
                assert torch.equal(
                    local_frame["geometry"],
                    reference_frame[1],
                )
                assert local_frame["geometry"].dtype is torch.float32
                assert torch.equal(
                    local_frame["valid_mask"],
                    reference_frame[2],
                )
                assert local_frame["valid_mask"].dtype is torch.bool
                assert torch.count_nonzero(
                    local_frame["embedding"][
                        ~local_frame["valid_mask"]
                    ]
                ) == 0

            tail_waveform = waveforms[-1:].clone()
            tail_changed_waveform = tail_waveform.clone()
            tail_start_sample = 162800
            tail_changed_waveform[:, tail_start_sample:164080] += torch.linspace(
                -0.5,
                0.5,
                164080 - tail_start_sample,
            )
            tail_seconds = valid_seconds[-1:]
            tail_features = local_transform(
                tail_waveform,
                sample_rate=16000,
                valid_seconds=tail_seconds,
            )["input_features"]
            tail_changed = local_transform(
                tail_changed_waveform,
                sample_rate=16000,
                valid_seconds=tail_seconds,
            )["input_features"]
            assert torch.equal(
                tail_changed[:, :1016],
                tail_features[:, :1016],
            )
            assert not torch.equal(
                tail_changed[:, 1016:],
                tail_features[:, 1016:],
            )
            assert torch.equal(
                local_encoder.backbone.embeddings.patch_embeddings(
                    tail_changed
                ),
                local_encoder.backbone.embeddings.patch_embeddings(
                    tail_features
                ),
            )
            tail_output = local_encoder.backbone(
                input_values=tail_changed
            )
            original_output = local_encoder.backbone(
                input_values=tail_features
            )
            assert torch.equal(
                tail_output.last_hidden_state,
                original_output.last_hidden_state,
            )
            assert torch.equal(
                tail_output.pooler_output,
                original_output.pooler_output,
            )
            tail_seconds = tail_seconds.to(device)
            with patch.object(
                local_encoder,
                "_backbone_outputs",
                return_value=original_output,
            ):
                local_encoder.granularity = "clip"
                original_clip = local_encoder(
                    tail_features,
                    valid_seconds=tail_seconds,
                )
                local_encoder.granularity = "frame"
                original_frame = local_encoder(
                    tail_features,
                    valid_seconds=tail_seconds,
                )
            with patch.object(
                local_encoder,
                "_backbone_outputs",
                return_value=tail_output,
            ):
                local_encoder.granularity = "clip"
                changed_clip = local_encoder(
                    tail_changed,
                    valid_seconds=tail_seconds,
                )
                local_encoder.granularity = "frame"
                changed_frame = local_encoder(
                    tail_changed,
                    valid_seconds=tail_seconds,
                )
            assert torch.equal(
                changed_clip["embedding"],
                original_clip["embedding"],
            )
            assert torch.equal(
                changed_frame["embedding"],
                original_frame["embedding"],
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32

    print(
        f"AST {device_name} alignment:",
        {
            "transform": _worst(transform_metrics),
            "hidden": _worst(hidden_metrics),
            "clip": _worst(clip_metrics),
            "frame": _worst(frame_metrics),
        },
    )


@pytest.mark.parametrize(
    "valid_samples",
    [400, 559, 560, 164079, 164080],
)
def test_discrete_transform_boundaries_match_official(
    valid_samples: int,
    reference_feature_extractor: ASTFeatureExtractor,
):
    waveform = _signal("random", valid_samples)
    local = AstKaldiFbankTransform()(
        waveform.unsqueeze(0),
        sample_rate=16000,
    )["input_features"]
    reference = reference_feature_extractor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    ).input_values

    _assert_metric(
        local,
        reference,
        atol=_TRANSFORM_ATOL,
        rtol=0.0,
    )


def test_project_only_length_differences(aligned_models):
    transform = AstKaldiFbankTransform()
    for valid_samples in (1, 399):
        output = transform(
            torch.zeros(1, valid_samples),
            sample_rate=16000,
        )
        assert output["input_features"].shape == (1, 1024, 128)

    sub_target_sample = transform(
        torch.zeros(1, 1),
        sample_rate=48000,
    )
    assert torch.round(
        sub_target_sample["valid_seconds"] * 16000
    ).item() == 0
    local_encoder, _ = aligned_models
    local_encoder.granularity = "frame"
    with torch.inference_mode():
        frame_output = local_encoder(**sub_target_sample)
    assert frame_output["valid_mask"].sum().item() == 1
    assert frame_output["geometry"][0, 0, 1].item() > 0

    with pytest.raises(ValueError, match="10.255"):
        transform(
            torch.zeros(1, 164081),
            sample_rate=16000,
        )
