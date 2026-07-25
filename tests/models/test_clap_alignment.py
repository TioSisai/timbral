"""Alignment tests between the local CLAP and pinned Hugging Face frontend, audio tower, and weights."""

from __future__ import annotations

import hashlib
import inspect
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio
import transformers
from torch import Tensor
from transformers import ClapFeatureExtractor, ClapModel

from timbral.models.encoders import ClapHtsatEncoder
from timbral.models.helpers.clap import (
    CLAP_CHECKPOINT,
    ensure_clap_checkpoint,
)
from timbral.models.transforms import ClapLogmelTransform

pytestmark = pytest.mark.alignment("clap")

_TRANSFORMERS_VERSION = "5.13.1"
_EXPECTED_REPO_ID = "laion/clap-htsat-fused"
_EXPECTED_REVISION = "365dea6ef167def6676140ed93bbc43f84dabb28"
_EXPECTED_CHECKPOINT_SHA256 = {
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
}
_SOURCE_SHA256 = {
    "feature_extraction_clap.py": (
        "a2bc74b2f7e3d11bb704b9e7699705e"
        "2d5bfe62400375f18020dda6f7382db45"
    ),
    "modeling_clap.py": (
        "2e1739468cd53541dcb53a985a66b585"
        "8ac2be8047cc75fc5a1dcc2fd268f1c8"
    ),
}
_TRANSFORM_ATOL = 1e-4
_TRANSFORM_RTOL = 1e-5
_ENCODER_ATOL = 1e-5
_ENCODER_RTOL = 1e-5
_MIN_COSINE = 0.999999


@pytest.fixture(scope="module")
def snapshot_directory() -> Path:
    """Prepare the pinned CLAP snapshot in an explicit directory or under TMPDIR."""
    explicit = os.environ.get("TIMBRAL_CLAP_SNAPSHOT")
    if explicit:
        return ensure_clap_checkpoint(Path(explicit))
    temporary_root = Path(
        os.environ.get("TMPDIR", tempfile.gettempdir())
    )
    return ensure_clap_checkpoint(
        temporary_root / "timbral-clap-alignment" / "snapshot"
    )


@pytest.fixture(scope="module")
def reference_feature_extractor(
    snapshot_directory: Path,
) -> ClapFeatureExtractor:
    """Load the pinned official CLAP frontend."""
    return ClapFeatureExtractor.from_pretrained(
        snapshot_directory,
        local_files_only=True,
    )


@pytest.fixture(scope="module")
def aligned_models(snapshot_directory: Path):
    """Load the local audio-only Encoder and the full official ClapModel."""
    local = ClapHtsatEncoder(
        granularity="clip",
        pretrained=True,
        pretrained_dir=snapshot_directory,
    ).eval()
    reference = ClapModel.from_pretrained(
        snapshot_directory,
        local_files_only=True,
    ).eval()
    return local, reference


def _sha256(path: Path) -> str:
    """Compute the file's SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rng_states_equal(left, right) -> bool:
    """Compare NumPy legacy RNG states."""
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _signal(num_samples: int) -> Tensor:
    """Build a fixed, non-degenerate alignment signal."""
    time = torch.arange(num_samples, dtype=torch.float32) / 48000
    generator = torch.Generator().manual_seed(num_samples)
    return (
        0.31 * torch.sin(2 * torch.pi * 321 * time)
        + 0.17 * torch.cos(2 * torch.pi * 1123 * time)
        + 0.03 * torch.randn(
            num_samples,
            generator=generator,
        )
        + 0.01
    )


def _official_single_features(
    extractor: ClapFeatureExtractor,
    waveform: Tensor,
) -> tuple[Tensor, bool]:
    """Call the official per-sample feature path, bypassing the all-short-batch forced flag."""
    features, is_longer = extractor._get_input_mel(
        waveform.numpy().astype(np.float64),
        480000,
        "fusion",
        "repeatpad",
    )
    return torch.from_numpy(features), bool(is_longer)


def _official_features_with_anchored_crops(
    extractor: ClapFeatureExtractor,
    waveform: Tensor,
) -> tuple[Tensor, bool]:
    """Compute official features after forcing the official crop selection to the project's deterministic anchor starts."""
    num_samples = waveform.shape[0]
    if num_samples < 480480:
        return _official_single_features(extractor, waveform)
    starts = list(
        ClapLogmelTransform._anchored_crop_starts(
            num_samples // 480 + 1
        )
    )
    original_choice = np.random.choice
    np.random.choice = lambda values: starts.pop(0)
    try:
        features = _official_single_features(extractor, waveform)
    finally:
        np.random.choice = original_choice
    assert not starts
    return features


def _devices() -> list[torch.device]:
    """Return the devices to align against this run."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def test_fixed_identity_and_source_hashes(snapshot_directory: Path):
    assert transformers.__version__ == _TRANSFORMERS_VERSION
    assert CLAP_CHECKPOINT.repo_id == _EXPECTED_REPO_ID
    assert CLAP_CHECKPOINT.revision == _EXPECTED_REVISION
    assert CLAP_CHECKPOINT.sha256 == _EXPECTED_CHECKPOINT_SHA256
    for filename, expected in _EXPECTED_CHECKPOINT_SHA256.items():
        assert _sha256(snapshot_directory / filename) == expected

    source_paths = {
        "feature_extraction_clap.py": Path(
            inspect.getsourcefile(ClapFeatureExtractor)
        ),
        "modeling_clap.py": Path(
            inspect.getsourcefile(ClapModel)
        ),
    }
    for filename, path in source_paths.items():
        assert _sha256(path) == _SOURCE_SHA256[filename]


def test_audio_only_weights_equal_full_official_model(aligned_models):
    local, reference = aligned_models
    local_state = local.backbone.state_dict()
    reference_state = reference.state_dict()

    assert len(local_state) == 270
    assert sum(
        key.startswith("audio_model.")
        for key in local_state
    ) == 266
    assert sum(
        key.startswith("audio_projection.")
        for key in local_state
    ) == 4
    assert set(local_state).issubset(reference_state)
    for key, value in local_state.items():
        assert torch.equal(value, reference_state[key]), key


@pytest.mark.parametrize("device", _devices(), ids=lambda value: value.type)
def test_transform_matches_official_boundaries_with_anchored_crops(
    device: torch.device,
    reference_feature_extractor: ClapFeatureExtractor,
):
    transform = ClapLogmelTransform().to(device).eval()
    sample_counts = (
        1,
        480000,
        480001,
        480479,
        480480,
        960000,
    )
    state_before = np.random.get_state()
    for num_samples in sample_counts:
        waveform = _signal(num_samples)
        expected, expected_is_longer = (
            _official_features_with_anchored_crops(
                reference_feature_extractor,
                waveform,
            )
        )
        actual = transform(
            waveform.unsqueeze(0).to(device),
            sample_rate=48000,
        )["input_features"][0]

        torch.testing.assert_close(
            actual.cpu(),
            expected,
            atol=_TRANSFORM_ATOL,
            rtol=_TRANSFORM_RTOL,
        )
        assert expected_is_longer == (num_samples >= 480480)
    state_after = np.random.get_state()
    assert _rng_states_equal(state_before, state_after)


def test_none_valid_seconds_preserves_long_physical_length(
    reference_feature_extractor: ClapFeatureExtractor,
):
    num_samples = 6_147_839
    waveform = _signal(num_samples)
    expected, expected_is_longer = (
        _official_features_with_anchored_crops(
            reference_feature_extractor,
            waveform,
        )
    )

    state_before = np.random.get_state()
    actual = ClapLogmelTransform()(
        waveform.unsqueeze(0),
        sample_rate=48000,
    )["input_features"][0]
    state_after = np.random.get_state()

    torch.testing.assert_close(
        actual,
        expected,
        atol=_TRANSFORM_ATOL,
        rtol=_TRANSFORM_RTOL,
    )
    assert expected_is_longer
    assert _rng_states_equal(state_before, state_after)


def test_float64_waveform_precision_matches_official(
    reference_feature_extractor: ClapFeatureExtractor,
):
    waveform = (
        0.123456789
        + torch.arange(48000, dtype=torch.float64) * 1e-11
    )
    expected, expected_is_longer = _official_single_features(
        reference_feature_extractor,
        waveform,
    )
    actual = ClapLogmelTransform()(
        waveform.unsqueeze(0),
        sample_rate=48000,
    )["input_features"][0]

    torch.testing.assert_close(
        actual,
        expected,
        atol=_TRANSFORM_ATOL,
        rtol=_TRANSFORM_RTOL,
    )
    assert not expected_is_longer


def test_all_short_batch_uses_length_routing_without_rng_hack(
    reference_feature_extractor: ClapFeatureExtractor,
):
    waveforms = [_signal(48000), _signal(240000), _signal(480000)]
    padded = torch.stack(
        (
            torch.nn.functional.pad(waveforms[0], (0, 432000)),
            torch.nn.functional.pad(waveforms[1], (0, 240000)),
            waveforms[2],
        )
    )
    valid_seconds = torch.tensor([1.0, 5.0, 10.0])

    external_state = np.random.get_state()
    try:
        np.random.seed(1729)
        official_public = reference_feature_extractor(
            [waveform.numpy() for waveform in waveforms],
            sampling_rate=48000,
            return_tensors="pt",
        )
        assert official_public["is_longer"].sum().item() == 1

        np.random.seed(1729)
        state_before = np.random.get_state()
        local = ClapLogmelTransform()(
            padded,
            sample_rate=48000,
            valid_seconds=valid_seconds,
        )
        state_after = np.random.get_state()

        torch.testing.assert_close(
            local["input_features"],
            official_public["input_features"],
            atol=_TRANSFORM_ATOL,
            rtol=_TRANSFORM_RTOL,
        )
        assert _rng_states_equal(state_before, state_after)
        local_routing = (
            torch.round(local["valid_seconds"] * 48000)
            >= 480480
        )
        assert not local_routing.any()
    finally:
        np.random.set_state(external_state)


def test_mixed_long_batch_matches_per_sample_official(
    reference_feature_extractor: ClapFeatureExtractor,
):
    sample_counts = (960000, 600000, 720000)
    waveforms = [_signal(count) for count in sample_counts]
    padded = torch.stack(
        tuple(
            torch.nn.functional.pad(
                waveform,
                (0, max(sample_counts) - waveform.numel()),
            )
            for waveform in waveforms
        )
    )
    valid_seconds = (
        torch.tensor(sample_counts, dtype=torch.float32) / 48000
    )

    expected = torch.stack(
        tuple(
            _official_features_with_anchored_crops(
                reference_feature_extractor,
                waveform,
            )[0]
            for waveform in waveforms
        )
    )

    state_before = np.random.get_state()
    actual = ClapLogmelTransform()(
        padded,
        sample_rate=48000,
        valid_seconds=valid_seconds,
    )["input_features"]
    state_after = np.random.get_state()

    torch.testing.assert_close(
        actual,
        expected,
        atol=_TRANSFORM_ATOL,
        rtol=_TRANSFORM_RTOL,
    )
    assert _rng_states_equal(state_before, state_after)


def test_project_multichannel_resampling_and_invalid_tail(
    reference_feature_extractor: ClapFeatureExtractor,
):
    generator = torch.Generator().manual_seed(41)
    mono = torch.randn(2, 8820, generator=generator)
    stereo = torch.stack((mono - 0.2, mono + 0.2), dim=1)
    changed_tail = stereo.clone()
    changed_tail[0, :, 4410:] = 7.0
    valid_seconds = torch.tensor([0.1, 0.2])
    transform = ClapLogmelTransform()

    mono_output = transform(
        mono,
        sample_rate=44100,
        valid_seconds=valid_seconds,
    )
    stereo_output = transform(
        stereo,
        sample_rate=44100,
        valid_seconds=valid_seconds,
    )
    changed_output = transform(
        changed_tail,
        sample_rate=44100,
        valid_seconds=valid_seconds,
    )

    torch.testing.assert_close(
        stereo_output["input_features"],
        mono_output["input_features"],
    )
    torch.testing.assert_close(
        changed_output["input_features"][0],
        stereo_output["input_features"][0],
    )
    for sample_index, source_samples in enumerate((4410, 8820)):
        resampled = torchaudio.functional.resample(
            mono[sample_index, :source_samples].to(torch.float64),
            orig_freq=44100,
            new_freq=48000,
        )
        target_samples = int(
            torch.round(valid_seconds[sample_index] * 48000).item()
        )
        resampled = resampled[:target_samples]
        if resampled.shape[0] < target_samples:
            resampled = torch.nn.functional.pad(
                resampled,
                (0, target_samples - resampled.shape[0]),
            )
        expected, expected_is_longer = _official_single_features(
            reference_feature_extractor,
            resampled,
        )
        torch.testing.assert_close(
            mono_output["input_features"][sample_index],
            expected,
            atol=_TRANSFORM_ATOL,
            rtol=_TRANSFORM_RTOL,
        )
        assert not expected_is_longer


@pytest.mark.parametrize("device", _devices(), ids=lambda value: value.type)
def test_audio_tower_projection_and_e2e_match_official(
    device: torch.device,
    aligned_models,
):
    local, reference = aligned_models
    local = local.to(device).eval()
    reference = reference.to(device).eval()
    waveform = torch.stack(
        (
            torch.nn.functional.pad(_signal(240000), (0, 720000)),
            _signal(960000),
        )
    ).to(device)
    valid_seconds = torch.tensor(
        [5.0, 20.0],
        dtype=torch.float32,
        device=device,
    )

    previous_tf32 = None
    if device.type == "cuda":
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        transformed = ClapLogmelTransform().to(device)(
            waveform,
            sample_rate=48000,
            valid_seconds=valid_seconds,
        )
        routing = (
            torch.round(valid_seconds * 48000) >= 480480
        ).unsqueeze(1)

        with torch.inference_mode():
            local_backbone = local.backbone(
                input_features=transformed["input_features"],
                is_longer=routing,
            )
            reference_backbone = reference.audio_model(
                input_features=transformed["input_features"],
                is_longer=routing,
            )
            reference_projected = reference.audio_projection(
                reference_backbone.pooler_output
            )
            local_output = local(**transformed)
            reference_output = reference.get_audio_features(
                input_features=transformed["input_features"],
                is_longer=routing,
            )
            reference_embedding = (
                reference_output.pooler_output
                if hasattr(reference_output, "pooler_output")
                else reference_output
            )

        torch.testing.assert_close(
            local_backbone.audio_embeds,
            reference_projected,
            atol=_ENCODER_ATOL,
            rtol=_ENCODER_RTOL,
        )
        torch.testing.assert_close(
            local_output["embedding"],
            reference_embedding,
            atol=_ENCODER_ATOL,
            rtol=_ENCODER_RTOL,
        )
        cosine = torch.nn.functional.cosine_similarity(
            local_output["embedding"],
            reference_embedding,
        )
        assert torch.all(cosine >= _MIN_COSINE)
        torch.testing.assert_close(
            local_output["embedding"].norm(dim=1),
            torch.ones(2, device=device),
        )
        torch.testing.assert_close(
            local_output["geometry"],
            torch.tensor(
                [[0.0, 5.0], [0.0, 20.0]],
                device=device,
            ),
        )
        assert local_output["valid_mask"].tolist() == [True, True]
    finally:
        if previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
