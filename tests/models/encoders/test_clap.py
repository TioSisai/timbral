"""Offline unit tests for ClapHtsatEncoder and the checkpoint helpers."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from timbral.models.encoders import ClapHtsatEncoder
from timbral.models.encoders import clap as encoder_module
from timbral.models.encoders.base import BaseEncoder
from timbral.models.helpers import clap as clap_helpers


class _FakeClapBackbone(nn.Module):
    """Lightweight backbone that records routing and produces a checkable embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.audio_projection = nn.Module()
        self.audio_projection.linear1 = nn.Linear(1, 1, bias=False)
        self.received_is_longer: Tensor | None = None

    def forward(
        self,
        *,
        input_features: Tensor,
        is_longer: Tensor,
    ):
        """Return an object compatible with ClapAudioModelOutput."""
        self.received_is_longer = is_longer
        scale = self.audio_projection.linear1.weight.reshape(())
        summary = input_features.mean(dim=(1, 2, 3)) * scale
        audio_embeds = torch.stack(
            (
                summary,
                summary + 1,
                is_longer[:, 0].to(summary.dtype) + 1,
            ),
            dim=1,
        )
        return SimpleNamespace(audio_embeds=audio_embeds)


def _fake_encoder() -> ClapHtsatEncoder:
    """Bypass the official constructor and inject a lightweight backbone."""
    encoder = ClapHtsatEncoder.__new__(ClapHtsatEncoder)
    BaseEncoder.__init__(encoder, "clip")
    encoder.pretrained = False
    encoder.pretrained_dir = None
    encoder.backbone = _FakeClapBackbone()
    return encoder


def test_public_export_keyword_only_constructor_and_capability():
    from timbral.models.encoders.clap import (
        ClapHtsatEncoder as DirectEncoder,
    )

    assert ClapHtsatEncoder is DirectEncoder
    assert ClapHtsatEncoder.supported_granularities == frozenset(("clip",))
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in inspect.signature(
            ClapHtsatEncoder
        ).parameters.values()
    )


def test_frame_rejected_before_checkpoint_side_effect(monkeypatch):
    monkeypatch.setattr(
        encoder_module,
        "ensure_clap_checkpoint",
        lambda _: pytest.fail("An unsupported granularity should not prepare the checkpoint"),
    )

    with pytest.raises(ValueError, match="does not support 'frame'.*clip"):
        ClapHtsatEncoder(granularity="frame")


def test_pretrained_false_uses_fixed_config_without_checkpoint(
    monkeypatch,
):
    monkeypatch.setattr(
        encoder_module,
        "ensure_clap_checkpoint",
        lambda _: pytest.fail("pretrained=False should not resolve the checkpoint"),
    )
    encoder = ClapHtsatEncoder(
        granularity="clip",
        pretrained=False,
    )

    assert encoder.backbone.config.enable_fusion is True
    assert encoder.backbone.config.hidden_size == 768
    assert encoder.backbone.config.projection_dim == 512
    assert len(encoder.backbone.state_dict()) == 270
    assert encoder.training
    assert encoder.backbone.training


def test_pretrained_requires_bool():
    with pytest.raises(TypeError, match="bool"):
        ClapHtsatEncoder(
            granularity="clip",
            pretrained=1,
        )


def test_clip_output_derives_fusion_mask_and_normalizes():
    encoder = _fake_encoder()
    features = torch.randn(3, 4, 1001, 64)
    valid_seconds = (
        torch.tensor([480479, 480480, 960000], dtype=torch.float32)
        / 48000
    )

    output = encoder(
        features,
        valid_seconds=valid_seconds,
    )

    assert encoder.backbone.received_is_longer.tolist() == [
        [False],
        [True],
        [True],
    ]
    assert output["embedding"].shape == (3, 3)
    torch.testing.assert_close(
        output["embedding"].norm(dim=1),
        torch.ones(3),
    )
    torch.testing.assert_close(
        output["geometry"],
        torch.stack(
            (torch.zeros(3), valid_seconds),
            dim=1,
        ),
    )
    assert output["valid_mask"].tolist() == [True, True, True]


@pytest.mark.parametrize(
    ("features", "valid_seconds", "error"),
    [
        (torch.ones(2, 1, 1001, 64), torch.ones(2), ValueError),
        (torch.ones(2, 4, 1000, 64), torch.ones(2), ValueError),
        (torch.ones(2, 4, 1001, 63), torch.ones(2), ValueError),
        (
            torch.ones(2, 4, 1001, 64, dtype=torch.float64),
            torch.ones(2),
            TypeError,
        ),
        (torch.ones(2, 4, 1001, 64), torch.ones(2, 1), ValueError),
        (torch.ones(2, 4, 1001, 64), torch.ones(1), ValueError),
    ],
)
def test_fixed_input_contract_raises(
    features,
    valid_seconds,
    error,
):
    with pytest.raises(error):
        _fake_encoder()(
            features,
            valid_seconds=valid_seconds,
        )


def test_unknown_model_parameter_raises():
    with pytest.raises(TypeError, match="unknown"):
        _fake_encoder()(
            torch.ones(1, 4, 1001, 64),
            valid_seconds=torch.ones(1),
            unknown=True,
        )


def test_gradient_reaches_backbone_and_features():
    encoder = _fake_encoder()
    features = torch.randn(
        1,
        4,
        1001,
        64,
        requires_grad=True,
    )

    output = encoder(
        features,
        valid_seconds=torch.tensor([1.0]),
    )
    output["embedding"].sum().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    parameter_grad = (
        encoder.backbone.audio_projection.linear1.weight.grad
    )
    assert parameter_grad is not None
    assert torch.isfinite(parameter_grad).all()


def test_config_validation_and_fixed_config(tmp_path: Path):
    config_path = tmp_path / "config.json"
    preprocessor_path = tmp_path / "preprocessor_config.json"
    config_path.write_text(
        json.dumps(
            {
                "audio_config": (
                    clap_helpers.CLAP_AUDIO_CONFIG_FIELDS
                )
            }
        ),
        encoding="utf-8",
    )
    preprocessor_path.write_text(
        json.dumps(clap_helpers.CLAP_PREPROCESSOR_FIELDS),
        encoding="utf-8",
    )

    config = clap_helpers.load_and_validate_clap_audio_config(
        config_path,
        preprocessor_path,
    )

    for name, expected in (
        clap_helpers.CLAP_AUDIO_CONFIG_FIELDS.items()
    ):
        assert getattr(config, name) == expected


def test_config_validation_rejects_mismatch(tmp_path: Path):
    config_path = tmp_path / "config.json"
    preprocessor_path = tmp_path / "preprocessor_config.json"
    audio_fields = dict(clap_helpers.CLAP_AUDIO_CONFIG_FIELDS)
    audio_fields["enable_fusion"] = False
    config_path.write_text(
        json.dumps({"audio_config": audio_fields}),
        encoding="utf-8",
    )
    preprocessor_path.write_text(
        json.dumps(clap_helpers.CLAP_PREPROCESSOR_FIELDS),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="enable_fusion"):
        clap_helpers.load_and_validate_clap_audio_config(
            config_path,
            preprocessor_path,
        )


def test_verify_clap_file(tmp_path: Path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"clap")

    clap_helpers.verify_clap_file(
        path,
        "8cceaf5c89e63591aea2e0a16fd98363"
        "477c370764ffa98fa7f00a60928576e1",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        clap_helpers.verify_clap_file(path, "0" * 64)
