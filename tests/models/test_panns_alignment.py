"""Full alignment tests between the local PANNs implementation and the pinned official source code and real weights."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from timbral.models.encoders import PannsCnn14Encoder
from timbral.models.helpers.panns import (
    PANNS_CHECKPOINTS,
    load_panns_checkpoint_model,
)
from timbral.models.transforms import PannsLogmelTransform
from timbral.models.transforms import panns as transform_module

_OFFICIAL_REPOSITORY = (
    "https://github.com/qiuqiangkong/audioset_tagging_cnn"
)
_OFFICIAL_COMMIT = "d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4"
_OFFICIAL_SOURCE_SHA256 = {
    "pytorch/models.py": (
        "7f9af440395ace5160bbb51d654a0dc35"
        "fb887fbf5edecb12da61ff6efb306d9"
    ),
    "pytorch/pytorch_utils.py": (
        "1464fcfbfc0fe4c55f690f6b39e1c80ee"
        "ed5de1e7fd1b7fd30334d304de7dbe9"
    ),
}
_DURATIONS = (0.02, 0.32, 1.0, 4.03, 10.0, 20.0)
_ATOL = 1e-4
_RTOL = 1e-4

_CONFIGURATIONS = (
    {
        "model_name": "panns-16k-cnn14-max_mean",
        "filename": "Cnn14_16k_mAP=0.438.pth",
        "target_sample_rate": 16000,
        "n_fft": 512,
        "win_length": 512,
        "hop_length": 160,
        "n_mels": 64,
        "f_min": 50.0,
        "f_max": 8000.0,
        "variant": "max_mean",
        "official_class": "Cnn14_16k",
    },
    {
        "model_name": "panns-32k-cnn14-max_mean",
        "filename": "Cnn14_mAP=0.431.pth",
        "target_sample_rate": 32000,
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 320,
        "n_mels": 64,
        "f_min": 50.0,
        "f_max": 14000.0,
        "variant": "max_mean",
        "official_class": "Cnn14",
    },
    {
        "model_name": "panns-32k-cnn14-decision_level_max",
        "filename": "Cnn14_DecisionLevelMax_mAP=0.385.pth",
        "target_sample_rate": 32000,
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 320,
        "n_mels": 64,
        "f_min": 50.0,
        "f_max": 14000.0,
        "variant": "decision_level_max",
        "official_class": "Cnn14_DecisionLevelMax",
    },
)


def _sha256(path: Path) -> str:
    """Compute the file's SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(*arguments: str) -> str:
    """Run Git and return its standard output."""
    completed = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _prepare_official_repository() -> Path:
    """Prepare and verify the pinned official source code under TMPDIR."""
    temporary_root = Path(
        os.environ.get("TMPDIR", tempfile.gettempdir())
    )
    alignment_root = temporary_root / "timbral-panns-alignment"
    repository = alignment_root / "audioset_tagging_cnn"
    alignment_root.mkdir(parents=True, exist_ok=True)

    if not repository.exists():
        _run_git(
            "clone",
            "--no-checkout",
            _OFFICIAL_REPOSITORY,
            str(repository),
        )
    if not (repository / ".git").is_dir():
        raise RuntimeError(
            f"Official source directory is not a Git repository: {repository}"
        )

    try:
        _run_git(
            "-C",
            str(repository),
            "cat-file",
            "-e",
            f"{_OFFICIAL_COMMIT}^{{commit}}",
        )
    except subprocess.CalledProcessError:
        _run_git(
            "-C",
            str(repository),
            "fetch",
            "--depth",
            "1",
            "origin",
            _OFFICIAL_COMMIT,
        )
    _run_git(
        "-C",
        str(repository),
        "checkout",
        "--detach",
        "--force",
        _OFFICIAL_COMMIT,
    )
    head = _run_git("-C", str(repository), "rev-parse", "HEAD")
    if head != _OFFICIAL_COMMIT:
        raise RuntimeError(
            f"Official source revision mismatch: expected {_OFFICIAL_COMMIT}, "
            f"got {head}"
        )

    for relative_path, expected_sha256 in _OFFICIAL_SOURCE_SHA256.items():
        source_path = repository / relative_path
        actual_sha256 = _sha256(source_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Official source SHA-256 mismatch: {relative_path}, "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
    return repository


class _ReferenceStft(nn.Module):
    """torchlibrosa-compatible fixed convolutional STFT."""

    def __init__(
        self,
        *,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        fft_window = librosa.filters.get_window(
            window,
            win_length,
            fftbins=True,
        )
        fft_window = librosa.util.pad_center(fft_window, size=n_fft)
        grid_x, grid_y = np.meshgrid(np.arange(n_fft), np.arange(n_fft))
        dft = np.power(
            np.exp(-2.0 * np.pi * 1j / n_fft),
            grid_x * grid_y,
        )
        num_bins = n_fft // 2 + 1
        real_weight = np.real(
            dft[:, :num_bins] * fft_window[:, None]
        ).T
        imag_weight = np.imag(
            dft[:, :num_bins] * fft_window[:, None]
        ).T
        self.conv_real = nn.Conv1d(
            1,
            num_bins,
            kernel_size=n_fft,
            stride=hop_length,
            bias=False,
        )
        self.conv_imag = nn.Conv1d(
            1,
            num_bins,
            kernel_size=n_fft,
            stride=hop_length,
            bias=False,
        )
        self.conv_real.weight.data.copy_(
            torch.from_numpy(real_weight).float().unsqueeze(1)
        )
        self.conv_imag.weight.data.copy_(
            torch.from_numpy(imag_weight).float().unsqueeze(1)
        )
        self.conv_real.weight.requires_grad_(False)
        self.conv_imag.weight.requires_grad_(False)

    def forward(self, waveform: Tensor) -> Tensor:
        """Compute the power spectrum."""
        padded = F.pad(
            waveform.unsqueeze(1),
            (self.n_fft // 2, self.n_fft // 2),
            mode="reflect",
        )
        real = self.conv_real(padded)
        imag = self.conv_imag(padded)
        return (
            (real.square() + imag.square())
            .transpose(1, 2)
            .unsqueeze(1)
        )


class _ReferenceSpectrogram(nn.Module):
    """Spectrogram compatibility shim required by the official constructor."""

    def __init__(
        self,
        *,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
        center: bool,
        pad_mode: str,
        freeze_parameters: bool,
    ) -> None:
        super().__init__()
        if not center or pad_mode != "reflect" or not freeze_parameters:
            raise ValueError(
                "The reference shim only implements PANNs' fixed parameters."
            )
        self.stft = _ReferenceStft(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        """Compute the power spectrum in the official layout."""
        return self.stft(waveform)


class _ReferenceLogmelFilterBank(nn.Module):
    """LogmelFilterBank compatibility shim required by the official constructor."""

    def __init__(
        self,
        *,
        sr: int,
        n_fft: int,
        n_mels: int,
        fmin: float,
        fmax: float,
        ref: float,
        amin: float,
        top_db: float | None,
        freeze_parameters: bool,
    ) -> None:
        super().__init__()
        if ref != 1.0 or amin != 1e-10 or top_db is not None:
            raise ValueError(
                "The reference shim only implements PANNs' fixed dB parameters."
            )
        mel_weight = librosa.filters.mel(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        ).T
        self.melW = nn.Parameter(
            torch.from_numpy(mel_weight.astype(np.float32)),
            requires_grad=not freeze_parameters,
        )

    def forward(self, spectrogram: Tensor) -> Tensor:
        """Apply the mel projection and the fixed dB transform."""
        mel = torch.clamp(spectrogram @ self.melW, min=1e-10)
        return 10.0 * torch.log10(mel)


class _ReferenceSpecAugmentation(nn.Module):
    """Minimal SpecAugmentation interface, not exercised by eval alignment."""

    def __init__(self, **_: Any) -> None:
        super().__init__()

    def forward(self, features: Tensor) -> Tensor:
        """Leave the input unchanged; the eval path never calls this method."""
        return features


def _install_torchlibrosa_shim() -> None:
    """Install a compatibility module for the pinned official source that does not depend on torchlibrosa."""
    package = types.ModuleType("torchlibrosa")
    package.__path__ = []
    stft_module = types.ModuleType("torchlibrosa.stft")
    augmentation_module = types.ModuleType("torchlibrosa.augmentation")
    stft_module.Spectrogram = _ReferenceSpectrogram
    stft_module.LogmelFilterBank = _ReferenceLogmelFilterBank
    augmentation_module.SpecAugmentation = _ReferenceSpecAugmentation
    package.stft = stft_module
    package.augmentation = augmentation_module
    sys.modules["torchlibrosa"] = package
    sys.modules["torchlibrosa.stft"] = stft_module
    sys.modules["torchlibrosa.augmentation"] = augmentation_module


def _load_official_models_module(repository: Path):
    """Import the official models.py at the pinned revision."""
    _install_torchlibrosa_shim()
    pytorch_directory = repository / "pytorch"
    sys.modules.pop("pytorch_utils", None)
    sys.path.insert(0, str(pytorch_directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "_timbral_official_panns_models",
            pytorch_directory / "models.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                "Failed to create an import spec for the official models.py."
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(pytorch_directory))
    return module


def _reference_features(model: nn.Module, waveform: Tensor) -> Tensor:
    """Run the official frontend, applying the project's minimum-padding contract for very short inputs."""
    spectrogram = model.spectrogram_extractor(waveform)
    logmel = model.logmel_extractor(spectrogram)
    native_frames = logmel.shape[2]
    if native_frames < 32:
        logmel = F.pad(
            logmel,
            (0, 0, 0, 32 - native_frames),
            value=-100.0,
        )
    features = model.bn0(logmel.transpose(1, 3)).transpose(1, 3)
    return features.squeeze(1)


def _reference_backbone(model: nn.Module, features: Tensor) -> Tensor:
    """Run the pinned official convolutional backbone and export the frequency-averaged sequence."""
    outputs = features.unsqueeze(1)
    for block_index in range(1, 6):
        block = getattr(model, f"conv_block{block_index}")
        outputs = block(
            outputs,
            pool_size=(2, 2),
            pool_type="avg",
        )
        outputs = F.dropout(outputs, p=0.2, training=False)
    outputs = model.conv_block6(
        outputs,
        pool_size=(1, 1),
        pool_type="avg",
    )
    outputs = F.dropout(outputs, p=0.2, training=False)
    return outputs.mean(dim=3)


def _reference_embeddings(
    model: nn.Module,
    *,
    variant: str,
    backbone: Tensor,
) -> tuple[Tensor, Tensor]:
    """Generate the project-defined clip and frame embeddings from the official backbone."""
    if variant == "max_mean":
        pooled = backbone.amax(dim=2) + backbone.mean(dim=2)
        pooled = F.dropout(pooled, p=0.5, training=False)
        clip_embedding = F.relu(model.fc1(pooled))
        clip_embedding = F.dropout(
            clip_embedding,
            p=0.5,
            training=False,
        )
        frame_features = F.dropout(
            backbone,
            p=0.5,
            training=False,
        )
        frame_embedding = F.relu(
            model.fc1(frame_features.transpose(1, 2))
        )
        frame_embedding = F.dropout(
            frame_embedding,
            p=0.5,
            training=False,
        )
        return clip_embedding, frame_embedding

    smoothed = F.max_pool1d(
        backbone,
        kernel_size=3,
        stride=1,
        padding=1,
    ) + F.avg_pool1d(
        backbone,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    smoothed = F.dropout(smoothed, p=0.5, training=False)
    frame_embedding = F.relu(model.fc1(smoothed.transpose(1, 2)))
    frame_embedding = F.dropout(
        frame_embedding,
        p=0.5,
        training=False,
    )
    return frame_embedding.amax(dim=1), frame_embedding


def _make_waveform(
    *,
    sample_rate: int,
    durations: list[float],
    case_index: int,
    silence: bool = False,
) -> tuple[Tensor, Tensor]:
    """Build a waveform combining random, sine, and impulse signals with nonzero invalid padding."""
    valid_seconds = torch.tensor(durations, dtype=torch.float32)
    valid_samples = torch.round(
        valid_seconds * sample_rate
    ).to(torch.long)
    max_samples = int(valid_samples.max().item())
    generator = torch.Generator().manual_seed(1000 + case_index)
    waveform = 0.5 + 0.1 * torch.randn(
        len(durations),
        max_samples,
        generator=generator,
    )

    for sample_index, num_samples_tensor in enumerate(valid_samples):
        num_samples = int(num_samples_tensor.item())
        time = torch.arange(num_samples, dtype=torch.float32) / sample_rate
        if silence:
            signal = torch.zeros(num_samples)
        elif sample_index % 3 == 0:
            signal = (
                0.15
                * torch.randn(num_samples, generator=generator)
                + 0.2 * torch.sin(2 * torch.pi * 440.0 * time)
            )
        elif sample_index % 3 == 1:
            signal = (
                0.25 * torch.sin(2 * torch.pi * 317.0 * time)
                + 0.1 * torch.sin(2 * torch.pi * 911.0 * time)
            )
        else:
            signal = torch.zeros(num_samples)
            signal[0] = 1.0
            signal[num_samples // 2] = -0.75
        waveform[sample_index, :num_samples] = signal
    return waveform, valid_seconds


def _alignment_cases() -> list[tuple[str, list[float], bool]]:
    """Build the full duration and batch test matrix."""
    cases = [
        (f"single-{duration:g}", [duration], False)
        for duration in _DURATIONS
    ]
    cases.extend(
        (
            f"batch3-{duration:g}",
            [duration, duration, duration],
            False,
        )
        for duration in _DURATIONS
    )
    cases.extend(
        (
            ("mixed-short", [0.02, 1.0, 4.03], False),
            ("mixed-long", [1.0, 4.03, 10.0], False),
            ("silence-1", [1.0], True),
        )
    )
    return cases


def _update_metrics(
    metrics: dict[str, dict[str, float]],
    *,
    stage: str,
    actual: Tensor,
    expected: Tensor,
) -> None:
    """Accumulate alignment error and enforce the allclose gate."""
    difference = (actual - expected).abs().float()
    relative = difference / expected.abs().float().clamp_min(1e-12)
    flattened_actual = actual.float().reshape(-1)
    flattened_expected = expected.float().reshape(-1)
    expected_norm = torch.linalg.vector_norm(flattened_expected)
    if expected_norm > 0:
        relative_l2 = (
            torch.linalg.vector_norm(
                flattened_actual - flattened_expected
            )
            / expected_norm
        )
        cosine = F.cosine_similarity(
            flattened_actual,
            flattened_expected,
            dim=0,
        )
    else:
        relative_l2 = torch.linalg.vector_norm(flattened_actual)
        cosine = torch.tensor(1.0, device=actual.device)

    current = metrics.setdefault(
        stage,
        {
            "max_abs": 0.0,
            "max_mean_abs": 0.0,
            "max_p99_abs": 0.0,
            "max_rel": 0.0,
            "max_mean_rel": 0.0,
            "max_p99_rel": 0.0,
            "max_relative_l2": 0.0,
            "min_cosine": 1.0,
        },
    )
    current["max_abs"] = max(
        current["max_abs"],
        float(difference.max().item()),
    )
    current["max_mean_abs"] = max(
        current["max_mean_abs"],
        float(difference.mean().item()),
    )
    current["max_p99_abs"] = max(
        current["max_p99_abs"],
        float(torch.quantile(difference, 0.99).item()),
    )
    current["max_rel"] = max(
        current["max_rel"],
        float(relative.max().item()),
    )
    current["max_mean_rel"] = max(
        current["max_mean_rel"],
        float(relative.mean().item()),
    )
    current["max_p99_rel"] = max(
        current["max_p99_rel"],
        float(torch.quantile(relative, 0.99).item()),
    )
    current["max_relative_l2"] = max(
        current["max_relative_l2"],
        float(relative_l2.item()),
    )
    current["min_cosine"] = min(
        current["min_cosine"],
        float(cosine.item()),
    )
    torch.testing.assert_close(
        actual,
        expected,
        atol=_ATOL,
        rtol=_RTOL,
    )


def _expected_geometry(
    valid_feature_frames: Tensor,
    valid_seconds: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build the PANNs frame ownership geometry and mask."""
    valid_embedding_frames = torch.clamp(
        torch.div(
            valid_feature_frames,
            32,
            rounding_mode="floor",
        ),
        min=1,
    )
    max_frames = int(valid_embedding_frames.max().item())
    frame_indices = torch.arange(
        max_frames,
        device=valid_seconds.device,
    )
    valid_mask = (
        frame_indices.unsqueeze(0)
        < valid_embedding_frames.unsqueeze(1)
    )
    boundaries = torch.arange(
        max_frames + 1,
        device=valid_seconds.device,
        dtype=torch.float32,
    ) * 0.32
    starts = boundaries[:-1].unsqueeze(0).expand(
        valid_seconds.shape[0],
        -1,
    )
    ends = torch.minimum(
        boundaries[1:].unsqueeze(0),
        valid_seconds.unsqueeze(1),
    )
    ends = ends.scatter(
        1,
        (valid_embedding_frames - 1).unsqueeze(1),
        valid_seconds.unsqueeze(1),
    )
    geometry = torch.stack((starts, ends), dim=2)
    return geometry * valid_mask.unsqueeze(2), valid_mask


def _assert_weight_mapping(
    transform: PannsLogmelTransform,
    encoder: PannsCnn14Encoder,
    checkpoint_state: dict[str, Tensor],
) -> None:
    """Verify the Transform's and Encoder's checkpoint mappings value by value."""
    for checkpoint_key, local_key in transform_module._FRONTEND_KEY_MAP.items():
        assert torch.equal(
            transform.state_dict()[local_key].cpu(),
            checkpoint_state[checkpoint_key],
        )
    for local_key, local_value in encoder.state_dict().items():
        assert torch.equal(
            local_value.cpu(),
            checkpoint_state[local_key],
        )


@pytest.mark.alignment("panns")
def test_panns_official_alignment():
    """Run the full alignment matrix across three checkpoints, two granularities, and CPU/CUDA."""
    debug_root_value = os.environ.get("DEBUG_ROOT")
    if not debug_root_value:
        pytest.fail(
            "Explicit PANNs alignment requires the DEBUG_ROOT environment "
            "variable."
        )
    debug_root = Path(debug_root_value)
    repository = _prepare_official_repository()
    official_module = _load_official_models_module(repository)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    summary: dict[str, Any] = {
        "official_commit": _OFFICIAL_COMMIT,
        "official_source_sha256": _OFFICIAL_SOURCE_SHA256,
        "devices": {},
        "case_count_per_configuration": len(_alignment_cases()),
        "configuration_count": len(_CONFIGURATIONS),
        "granularities": ["clip", "frame"],
        "atol": _ATOL,
        "rtol": _RTOL,
    }

    with torch.inference_mode():
        for device in devices:
            device_metrics: dict[str, dict[str, float]] = {}
            executed_cases = 0
            for configuration in _CONFIGURATIONS:
                model_directory = (
                    debug_root
                    / "panns"
                    / configuration["model_name"]
                )
                checkpoint_path = (
                    model_directory / configuration["filename"]
                )
                metadata = PANNS_CHECKPOINTS[
                    (
                        configuration["target_sample_rate"],
                        configuration["variant"],
                    )
                ]
                checkpoint_state = load_panns_checkpoint_model(
                    checkpoint_path,
                    requires_numpy_allowlist=(
                        metadata.requires_numpy_allowlist
                    ),
                )

                transform = PannsLogmelTransform(
                    target_sample_rate=configuration[
                        "target_sample_rate"
                    ],
                    n_fft=configuration["n_fft"],
                    win_length=configuration["win_length"],
                    hop_length=configuration["hop_length"],
                    n_mels=configuration["n_mels"],
                    f_min=configuration["f_min"],
                    f_max=configuration["f_max"],
                    variant=configuration["variant"],
                    pretrained=True,
                    pretrained_dir=model_directory,
                ).eval().to(device)
                frame_encoder = PannsCnn14Encoder(
                    granularity="frame",
                    target_sample_rate=configuration[
                        "target_sample_rate"
                    ],
                    variant=configuration["variant"],
                    pretrained=True,
                    pretrained_dir=model_directory,
                ).eval().to(device)
                clip_encoder = PannsCnn14Encoder(
                    granularity="clip",
                    target_sample_rate=configuration[
                        "target_sample_rate"
                    ],
                    variant=configuration["variant"],
                    pretrained=True,
                    pretrained_dir=model_directory,
                ).eval().to(device)

                official_class = getattr(
                    official_module,
                    configuration["official_class"],
                )
                official_model = official_class(
                    sample_rate=configuration["target_sample_rate"],
                    window_size=configuration["n_fft"],
                    hop_size=configuration["hop_length"],
                    mel_bins=configuration["n_mels"],
                    fmin=configuration["f_min"],
                    fmax=configuration["f_max"],
                    classes_num=527,
                )
                official_model.load_state_dict(
                    checkpoint_state,
                    strict=True,
                )
                official_model.eval().to(device)

                _assert_weight_mapping(
                    transform,
                    frame_encoder,
                    checkpoint_state,
                )
                for case_index, (
                    case_name,
                    durations,
                    silence,
                ) in enumerate(_alignment_cases()):
                    waveform_cpu, valid_seconds_cpu = _make_waveform(
                        sample_rate=configuration[
                            "target_sample_rate"
                        ],
                        durations=durations,
                        case_index=case_index,
                        silence=silence,
                    )
                    waveform = waveform_cpu.to(device)
                    valid_seconds = valid_seconds_cpu.to(device)
                    local_features = transform(
                        waveform,
                        sample_rate=configuration[
                            "target_sample_rate"
                        ],
                        valid_seconds=valid_seconds,
                    )
                    valid_feature_frames = local_features[
                        "valid_feature_frames"
                    ]
                    target_samples = torch.round(
                        valid_seconds
                        * configuration["target_sample_rate"]
                    ).to(torch.long)
                    reference_features = torch.zeros_like(
                        local_features["input_features"]
                    )
                    reference_clip = waveform.new_zeros(
                        (len(durations), 2048)
                    )
                    valid_embedding_frames = torch.clamp(
                        torch.div(
                            valid_feature_frames,
                            32,
                            rounding_mode="floor",
                        ),
                        min=1,
                    )
                    max_embedding_frames = int(
                        valid_embedding_frames.max().item()
                    )
                    reference_frame = waveform.new_zeros(
                        (
                            len(durations),
                            max_embedding_frames,
                            2048,
                        )
                    )

                    unique_lengths, inverse = torch.unique(
                        target_samples,
                        sorted=True,
                        return_inverse=True,
                    )
                    reference_backbones: dict[int, Tensor] = {}
                    for group_index in range(unique_lengths.shape[0]):
                        batch_indices = torch.nonzero(
                            inverse == group_index,
                            as_tuple=False,
                        ).squeeze(1)
                        target_length = int(
                            unique_lengths[group_index].item()
                        )
                        group_waveform = waveform.index_select(
                            0,
                            batch_indices,
                        )[:, :target_length]
                        group_features = _reference_features(
                            official_model,
                            group_waveform,
                        )
                        reference_features[
                            batch_indices,
                            : group_features.shape[1],
                        ] = group_features
                        group_backbone = _reference_backbone(
                            official_model,
                            group_features,
                        )
                        group_clip, group_frame = _reference_embeddings(
                            official_model,
                            variant=configuration["variant"],
                            backbone=group_backbone,
                        )
                        reference_clip[batch_indices] = group_clip
                        reference_frame[
                            batch_indices,
                            : group_frame.shape[1],
                        ] = group_frame
                        feature_length = int(
                            valid_feature_frames[
                                batch_indices[0]
                            ].item()
                        )
                        reference_backbones[feature_length] = (
                            group_backbone
                        )

                    _update_metrics(
                        device_metrics,
                        stage="transform",
                        actual=local_features["input_features"],
                        expected=reference_features,
                    )

                    captured_block6: list[Tensor] = []

                    def _capture_block6(_, __, output):
                        captured_block6.append(output.mean(dim=3))

                    hook = frame_encoder.conv_block6.register_forward_hook(
                        _capture_block6
                    )
                    try:
                        local_frame = frame_encoder(**local_features)
                    finally:
                        hook.remove()
                    local_clip = clip_encoder(**local_features)

                    unique_feature_lengths = torch.unique(
                        valid_feature_frames,
                        sorted=True,
                    )
                    for feature_length, local_backbone in zip(
                        unique_feature_lengths.tolist(),
                        captured_block6,
                        strict=True,
                    ):
                        _update_metrics(
                            device_metrics,
                            stage="backbone",
                            actual=local_backbone,
                            expected=reference_backbones[
                                feature_length
                            ],
                        )
                    _update_metrics(
                        device_metrics,
                        stage="frame_embedding",
                        actual=local_frame["embedding"],
                        expected=reference_frame,
                    )
                    _update_metrics(
                        device_metrics,
                        stage="clip_embedding",
                        actual=local_clip["embedding"],
                        expected=reference_clip,
                    )

                    expected_geometry, expected_mask = (
                        _expected_geometry(
                            valid_feature_frames,
                            valid_seconds,
                        )
                    )
                    torch.testing.assert_close(
                        local_frame["geometry"],
                        expected_geometry,
                    )
                    assert torch.equal(
                        local_frame["valid_mask"],
                        expected_mask,
                    )
                    assert torch.isfinite(
                        local_features["input_features"]
                    ).all()
                    assert torch.isfinite(
                        local_frame["embedding"]
                    ).all()
                    assert torch.isfinite(
                        local_clip["embedding"]
                    ).all()
                    executed_cases += 1

                del (
                    official_model,
                    transform,
                    frame_encoder,
                    clip_encoder,
                    checkpoint_state,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            summary["devices"][device.type] = {
                "executed_cases": executed_cases,
                "metrics": device_metrics,
            }

    output_path = (
        Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
        / "timbral-panns-alignment"
        / "panns-alignment-summary.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
