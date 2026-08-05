"""Alignment tests between the local wav2vec2 and the pinned Hugging Face official stack."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torchaudio
import transformers
from torch import Tensor
from torch.nn import functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from timbral.models.encoders import Wav2Vec2Encoder
from timbral.models.helpers.wav2vec2 import (
    WAV2VEC2_CHECKPOINT,
    WAV2VEC2_CONFIG_FIELDS,
    ensure_wav2vec2_checkpoint,
    wav2vec2_feature_frames,
)
from timbral.models.transforms import Wav2Vec2WaveformTransform

pytestmark = pytest.mark.alignment("wav2vec2")

_TRANSFORMERS_VERSION = "5.14.1"
_EXPECTED_REPO_ID = "facebook/wav2vec2-base"
_EXPECTED_REVISION = "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
_EXPECTED_CHECKPOINT_SHA256 = {
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
}
_SOURCE_SHA256 = {
    "feature_extraction_wav2vec2.py": (
        "e5e9a0baf70716fee503f4f66a7a6131"
        "2a132be989b2d7e2649e057ccbefa2cc"
    ),
    "modeling_wav2vec2.py": (
        "c6ee256c01c9c640f7e00dabe6bb480d"
        "6e6f8aae532671f46acbbe95c198cd2d"
    ),
}
_DURATION_CASES = (
    ("random", 0.025),              # 400 samples  -> 1 frame
    ("sine", 0.0449375),            # 719 samples  -> 1 frame
    ("impulse", 0.045625),          # 730 samples  -> 2 frames
    ("multisine", 1.0),             # 16000 samples -> 49 frames
    ("silence", 2.0),               # 32000 samples -> 99 frames
    ("random_invalid_tail", 10.0),  # 160000 samples -> 499 frames
)
_EXPECTED_VALID_SAMPLES = [400, 719, 730, 16000, 32000, 160000]
_EXPECTED_VALID_FRAMES = [1, 1, 2, 49, 99, 499]
_BOUNDARY_FRAMES = {400: 1, 719: 1, 720: 2, 730: 2, 16000: 49}
_FOREIGN_SAMPLE_RATE = 44100
_FOREIGN_DURATION_CASES = (("multisine", 0.9), ("random", 2.5))
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
    """Prepare the pinned wav2vec2 snapshot in an explicit directory or under TMPDIR."""
    explicit = os.environ.get("TIMBRAL_WAV2VEC2_SNAPSHOT")
    if explicit:
        return ensure_wav2vec2_checkpoint(Path(explicit))
    temporary_root = Path(
        os.environ.get("TMPDIR", tempfile.gettempdir())
    )
    return ensure_wav2vec2_checkpoint(
        temporary_root / "timbral-wav2vec2-alignment" / "snapshot"
    )


@pytest.fixture(scope="module")
def reference_feature_extractor(
    snapshot_directory: Path,
) -> Wav2Vec2FeatureExtractor:
    """Load the pinned official Wav2Vec2FeatureExtractor."""
    return Wav2Vec2FeatureExtractor.from_pretrained(
        snapshot_directory,
        local_files_only=True,
    )


@pytest.fixture(scope="module")
def aligned_models(snapshot_directory: Path):
    """Load the local Encoder and the official Wav2Vec2Model.

    ``from_pretrained`` on the pretraining checkpoint prints a load
    report with 7 UNEXPECTED keys (``quantizer.*``, ``project_q.*``,
    ``project_hid.*``); that is the expected pretraining-head remainder,
    not an error.
    """
    local_encoder = Wav2Vec2Encoder(
        granularity="clip",
        pretrained=True,
        pretrained_dir=snapshot_directory,
    ).eval()
    reference_model = Wav2Vec2Model.from_pretrained(
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


def _signal(
    case_name: str,
    valid_samples: int,
    *,
    sample_rate: int = 16000,
) -> Tensor:
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

    time = torch.arange(valid_samples, dtype=torch.float32) / sample_rate
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
    """Build a mixed-length 16 kHz batch with a nonzero invalid tail."""
    physical_samples = 160000 + 257
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
    extractor: Wav2Vec2FeatureExtractor,
    waveforms: Tensor,
    valid_seconds: Tensor,
) -> list[Tensor]:
    """Run the official CPU extractor sample by sample on exact lengths."""
    features = []
    for waveform, seconds in zip(waveforms, valid_seconds, strict=True):
        valid_samples = round(float(seconds) * 16000)
        output = extractor(
            waveform[:valid_samples].numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        features.append(output.input_values[0])
    return features


def _metric(actual: Tensor, expected: Tensor) -> _Metric:
    """Compute the max difference, cosine, and L2 norms in float64."""
    actual_flat = actual.detach().to(
        device="cpu", dtype=torch.float64
    ).flatten()
    expected_flat = expected.detach().to(
        device="cpu", dtype=torch.float64
    ).flatten()
    difference = actual_flat - expected_flat
    absolute_l2 = torch.linalg.vector_norm(difference)
    expected_l2 = torch.linalg.vector_norm(expected_flat)
    if float(expected_l2) == 0.0 and float(
        torch.linalg.vector_norm(actual_flat)
    ) == 0.0:
        # Silence maps to exact zeros on both sides, so the direction
        # metrics degenerate to their perfect values.
        return _Metric(
            max_absolute=0.0,
            cosine=1.0,
            absolute_l2=0.0,
            relative_l2=0.0,
        )
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


def _expected_frame_geometry(
    valid_frames: Tensor,
    valid_seconds: Tensor,
    *,
    total_frames: int,
) -> tuple[Tensor, Tensor]:
    """Rebuild the project's 0.02 s frame-ownership contract from scratch."""
    device = valid_seconds.device
    frame_indices = torch.arange(total_frames, device=device)
    valid_mask = frame_indices.unsqueeze(0) < valid_frames.unsqueeze(1)
    boundaries = torch.arange(
        total_frames + 1,
        device=device,
        dtype=torch.float32,
    ) * (320 / 16000)
    starts = boundaries[:-1].unsqueeze(0).expand(
        valid_seconds.shape[0], -1
    )
    ends = torch.minimum(
        boundaries[1:].unsqueeze(0),
        valid_seconds.unsqueeze(1),
    )
    ends = ends.scatter(
        1,
        (valid_frames - 1).unsqueeze(1),
        valid_seconds.unsqueeze(1),
    )
    geometry = torch.stack((starts, ends), dim=2) * valid_mask.unsqueeze(2)
    return geometry, valid_mask


def test_fixed_transformers_and_checkpoint_identity(
    snapshot_directory: Path,
):
    assert transformers.__version__ == _TRANSFORMERS_VERSION
    assert WAV2VEC2_CHECKPOINT.repo_id == _EXPECTED_REPO_ID
    assert WAV2VEC2_CHECKPOINT.revision == _EXPECTED_REVISION
    assert WAV2VEC2_CHECKPOINT.filenames == tuple(
        _EXPECTED_CHECKPOINT_SHA256
    )
    assert WAV2VEC2_CHECKPOINT.sha256 == _EXPECTED_CHECKPOINT_SHA256

    source_paths = {
        "feature_extraction_wav2vec2.py": Path(
            inspect.getsourcefile(Wav2Vec2FeatureExtractor)
        ),
        "modeling_wav2vec2.py": Path(
            inspect.getsourcefile(Wav2Vec2Model)
        ),
    }
    for filename, path in source_paths.items():
        assert path.name == filename
        assert _sha256(path) == _SOURCE_SHA256[filename]
    for filename, expected_sha256 in _EXPECTED_CHECKPOINT_SHA256.items():
        assert (
            _sha256(snapshot_directory / filename)
            == expected_sha256
        )

    with (snapshot_directory / "config.json").open(
        encoding="utf-8"
    ) as file:
        configuration = json.load(file)
    for name, expected in WAV2VEC2_CONFIG_FIELDS.items():
        assert configuration.get(name) == expected, name

    with (snapshot_directory / "preprocessor_config.json").open(
        encoding="utf-8"
    ) as file:
        preprocessor = json.load(file)
    assert preprocessor["do_normalize"] is True
    assert preprocessor["sampling_rate"] == 16000
    assert preprocessor["return_attention_mask"] is False


def test_exact_backbone_state_matches_official_loader(aligned_models):
    local_encoder, reference_model = aligned_models
    local_state = local_encoder.backbone.state_dict()
    reference_state = reference_model.state_dict()

    assert len(local_state) == len(reference_state) == 211
    assert local_state.keys() == reference_state.keys()
    assert "masked_spec_embed" in local_state
    assert (
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
        in local_state
    )
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
    reference_feature_extractor: Wav2Vec2FeatureExtractor,
    aligned_models,
):
    device = torch.device(device_name)
    local_encoder, reference_model = aligned_models
    local_encoder.to(device)
    reference_model.to(device)
    local_transform = Wav2Vec2WaveformTransform().to(device)
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
            local_output = local_transform(
                waveforms,
                sample_rate=16000,
                valid_seconds=valid_seconds,
            )
            local_features = local_output["input_features"]
            valid_samples = local_output["valid_samples"]
            seconds_device = local_output["valid_seconds"]
            assert local_features.dtype is torch.float32
            assert valid_samples.dtype is torch.int64
            assert valid_samples.tolist() == _EXPECTED_VALID_SAMPLES
            assert local_features.shape == (
                len(case_names),
                max(_EXPECTED_VALID_SAMPLES),
            )

            transform_metrics = []
            for index, name in enumerate(case_names):
                count = _EXPECTED_VALID_SAMPLES[index]
                assert torch.count_nonzero(
                    local_features[index, count:]
                ) == 0
                transform_metrics.append(
                    (
                        name,
                        _assert_metric(
                            local_features[index, :count],
                            reference_features[index].to(device),
                            atol=_TRANSFORM_ATOL,
                            rtol=0.0,
                        ),
                    )
                )
                # Mixed batch vs a per-sample crop must be bit-identical.
                single_features = local_transform(
                    waveforms[index : index + 1, :count],
                    sample_rate=16000,
                )["input_features"]
                assert torch.equal(
                    single_features[0],
                    local_features[index, :count],
                )

            local_encoder.granularity = "clip"
            local_clip = local_encoder(
                local_features,
                valid_seconds=seconds_device,
                valid_samples=valid_samples,
            )
            local_encoder.granularity = "frame"
            local_frame = local_encoder(
                local_features,
                valid_seconds=seconds_device,
                valid_samples=valid_samples,
            )
            assert local_clip["embedding"].shape == (len(case_names), 768)
            assert local_frame["embedding"].shape == (
                len(case_names),
                max(_EXPECTED_VALID_FRAMES),
                768,
            )

            hidden_metrics = []
            clip_metrics = []
            frame_metrics = []
            reference_frame_counts = []
            for index, name in enumerate(case_names):
                count = _EXPECTED_VALID_SAMPLES[index]
                # Both sides consume the same exact-length input, so the
                # encoder gate is isolated from transform micro-errors.
                exact_input = local_features[index : index + 1, :count]
                reference_hidden = reference_model(
                    input_values=exact_input
                ).last_hidden_state
                local_hidden = local_encoder._backbone_hidden(exact_input)
                frames = reference_hidden.shape[1]
                reference_frame_counts.append(frames)

                hidden_metrics.append(
                    (
                        name,
                        _assert_global_metric(
                            local_hidden,
                            reference_hidden,
                        ),
                    )
                )
                clip_metrics.append(
                    (
                        name,
                        _assert_metric(
                            local_clip["embedding"][index : index + 1],
                            reference_hidden.mean(dim=1),
                            atol=_ENCODER_ATOL,
                            rtol=_ENCODER_RTOL,
                        ),
                    )
                )
                frame_metrics.append(
                    (
                        name,
                        _assert_metric(
                            local_frame["embedding"][
                                index : index + 1, :frames
                            ],
                            reference_hidden,
                            atol=_ENCODER_ATOL,
                            rtol=_ENCODER_RTOL,
                        ),
                    )
                )

                single_frame = local_encoder(
                    exact_input,
                    valid_seconds=seconds_device[index : index + 1],
                    valid_samples=valid_samples[index : index + 1],
                )
                assert torch.equal(
                    single_frame["embedding"],
                    local_frame["embedding"][index : index + 1, :frames],
                )

            assert reference_frame_counts == _EXPECTED_VALID_FRAMES
            reference_frames = torch.tensor(
                reference_frame_counts,
                device=device,
            )
            assert torch.equal(
                wav2vec2_feature_frames(valid_samples),
                reference_frames,
            )

            expected_geometry, expected_mask = _expected_frame_geometry(
                reference_frames,
                seconds_device,
                total_frames=local_frame["embedding"].shape[1],
            )
            assert torch.equal(local_frame["geometry"], expected_geometry)
            assert local_frame["geometry"].dtype is torch.float32
            assert torch.equal(local_frame["valid_mask"], expected_mask)
            assert local_frame["valid_mask"].dtype is torch.bool
            assert torch.count_nonzero(
                local_frame["embedding"][~expected_mask]
            ) == 0

            assert torch.equal(
                local_clip["geometry"],
                torch.stack(
                    (
                        torch.zeros_like(seconds_device),
                        seconds_device,
                    ),
                    dim=1,
                ),
            )
            assert local_clip["geometry"].dtype is torch.float32
            assert local_clip["valid_mask"].dtype is torch.bool
            assert bool(local_clip["valid_mask"].all())

            # Rewriting the nonzero invalid tails must not change any
            # output bit.
            tainted_waveforms = waveforms.clone()
            for index in range(len(case_names)):
                count = _EXPECTED_VALID_SAMPLES[index]
                tainted_waveforms[index, count:] = -5.0 - index
            tainted_output = local_transform(
                tainted_waveforms,
                sample_rate=16000,
                valid_seconds=valid_seconds,
            )
            assert torch.equal(
                tainted_output["input_features"],
                local_features,
            )
            local_encoder.granularity = "clip"
            tainted_clip = local_encoder(
                tainted_output["input_features"],
                valid_seconds=seconds_device,
                valid_samples=valid_samples,
            )
            assert torch.equal(
                tainted_clip["embedding"],
                local_clip["embedding"],
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32

    print(
        f"wav2vec2 {device_name} alignment:",
        {
            "transform": _worst(transform_metrics),
            "hidden": _worst(hidden_metrics),
            "clip": _worst(clip_metrics),
            "frame": _worst(frame_metrics),
        },
    )


@pytest.mark.parametrize(
    "valid_samples",
    [400, 719, 720, 730, 16000],
)
def test_discrete_transform_boundaries_match_official(
    valid_samples: int,
    reference_feature_extractor: Wav2Vec2FeatureExtractor,
    aligned_models,
):
    _, reference_model = aligned_models
    reference_model.to("cpu")
    waveform = _signal("random", valid_samples)
    local_output = Wav2Vec2WaveformTransform()(
        waveform.unsqueeze(0),
        sample_rate=16000,
    )
    reference = reference_feature_extractor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    ).input_values

    assert local_output["valid_samples"].tolist() == [valid_samples]
    _assert_metric(
        local_output["input_features"],
        reference,
        atol=_TRANSFORM_ATOL,
        rtol=0.0,
    )

    with torch.inference_mode():
        reference_hidden = reference_model(
            input_values=reference
        ).last_hidden_state
    expected_frames = _BOUNDARY_FRAMES[valid_samples]
    assert reference_hidden.shape[1] == expected_frames
    assert wav2vec2_feature_frames(
        local_output["valid_samples"]
    ).tolist() == [expected_frames]


def test_foreign_sample_rate_matches_official_per_sample(
    reference_feature_extractor: Wav2Vec2FeatureExtractor,
    aligned_models,
):
    local_encoder, reference_model = aligned_models
    local_encoder.to("cpu")
    reference_model.to("cpu")
    local_transform = Wav2Vec2WaveformTransform()

    physical_samples = round(2.5 * _FOREIGN_SAMPLE_RATE) + 123
    waveforms = []
    durations = []
    for index, (case_name, duration) in enumerate(
        _FOREIGN_DURATION_CASES
    ):
        source_samples = round(duration * _FOREIGN_SAMPLE_RATE)
        waveform = torch.full((physical_samples,), 4.0 + index)
        waveform[:source_samples] = _signal(
            case_name,
            source_samples,
            sample_rate=_FOREIGN_SAMPLE_RATE,
        )
        waveforms.append(waveform)
        durations.append(duration)
    batch = torch.stack(waveforms)
    valid_seconds = torch.tensor(durations, dtype=torch.float32)

    transform_metrics = []
    hidden_metrics = []
    clip_metrics = []
    with torch.inference_mode():
        local_output = local_transform(
            batch,
            sample_rate=_FOREIGN_SAMPLE_RATE,
            valid_seconds=valid_seconds,
        )
        local_features = local_output["input_features"]
        assert local_output["valid_samples"].tolist() == [14400, 40000]

        local_encoder.granularity = "clip"
        local_clip = local_encoder(
            local_features,
            valid_seconds=local_output["valid_seconds"],
            valid_samples=local_output["valid_samples"],
        )

        for index, (case_name, duration) in enumerate(
            _FOREIGN_DURATION_CASES
        ):
            source_samples = round(duration * _FOREIGN_SAMPLE_RATE)
            target_samples = round(duration * 16000)
            name = f"{case_name}:{duration}"
            # The official reference: identical torchaudio resampling
            # of the exact valid prefix, then the official extractor.
            resampled = torchaudio.functional.resample(
                batch[index, :source_samples].unsqueeze(0),
                orig_freq=_FOREIGN_SAMPLE_RATE,
                new_freq=16000,
            )[0][:target_samples]
            if resampled.shape[0] < target_samples:
                resampled = F.pad(
                    resampled,
                    (0, target_samples - resampled.shape[0]),
                )
            reference = reference_feature_extractor(
                resampled.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
            ).input_values

            transform_metrics.append(
                (
                    name,
                    _assert_metric(
                        local_features[index, :target_samples],
                        reference[0],
                        atol=_TRANSFORM_ATOL,
                        rtol=0.0,
                    ),
                )
            )

            # End-to-end: each side consumes its own transform output,
            # so this case audits the compounded pipeline error.
            reference_hidden = reference_model(
                input_values=reference
            ).last_hidden_state
            local_hidden = local_encoder._backbone_hidden(
                local_features[index : index + 1, :target_samples]
            )
            assert (
                int(
                    wav2vec2_feature_frames(
                        local_output["valid_samples"]
                    )[index]
                )
                == reference_hidden.shape[1]
            )
            hidden_metrics.append(
                (
                    name,
                    _assert_global_metric(
                        local_hidden,
                        reference_hidden,
                    ),
                )
            )
            clip_metrics.append(
                (
                    name,
                    _assert_metric(
                        local_clip["embedding"][index : index + 1],
                        reference_hidden.mean(dim=1),
                        atol=_ENCODER_ATOL,
                        rtol=_ENCODER_RTOL,
                    ),
                )
            )

    print(
        "wav2vec2 44100 Hz alignment:",
        {
            "transform": _worst(transform_metrics),
            "hidden": _worst(hidden_metrics),
            "clip": _worst(clip_metrics),
        },
    )


def test_below_min_receptive_field_raises_value_error():
    transform = Wav2Vec2WaveformTransform()
    boundary = transform(
        _signal("random", 400).unsqueeze(0),
        sample_rate=16000,
    )
    assert boundary["input_features"].shape == (1, 400)
    assert boundary["valid_samples"].tolist() == [400]

    with pytest.raises(ValueError, match="400 samples"):
        transform(
            _signal("random", 399).unsqueeze(0),
            sample_rate=16000,
        )

    with pytest.raises(ValueError, match="400 samples"):
        transform(
            torch.zeros(1, _FOREIGN_SAMPLE_RATE),
            sample_rate=_FOREIGN_SAMPLE_RATE,
            valid_seconds=torch.tensor([0.0249375]),
        )
