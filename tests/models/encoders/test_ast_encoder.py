"""Offline unit tests for AstEncoder and the AST checkpoint helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from timbral.models.encoders import AstEncoder
from timbral.models.encoders import ast_encoder as encoder_module
from timbral.models.encoders.base import BaseEncoder
from timbral.models.helpers import ast_helpers
from timbral.models.transforms import AstKaldiFbankTransform

_EXPECTED_AST_CONFIG_FIELDS = {
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


class _FakeAstBackbone(nn.Module):
    """Lightweight backbone for checking token order, output shape, and gradients."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = nn.Module()
        self.embeddings.patch_embeddings.projection = nn.Conv2d(
            1,
            1,
            kernel_size=1,
            bias=False,
        )

    def forward(self, *, input_values: Tensor):
        """Return a minimal object compatible with ASTModelOutputs."""
        scale = (
            self.embeddings.patch_embeddings.projection.weight.reshape(())
        )
        input_summary = input_values.mean(dim=(1, 2)) * scale
        frequency = torch.arange(
            12,
            device=input_values.device,
            dtype=input_values.dtype,
        ).view(12, 1)
        time = torch.arange(
            101,
            device=input_values.device,
            dtype=input_values.dtype,
        ).view(1, 101)
        patch_values = (frequency * 1000 + time).reshape(1, 1212, 1)
        patch_tokens = (
            patch_values
            + input_summary.view(-1, 1, 1)
        ).expand(-1, -1, 768)
        special_tokens = input_summary.view(-1, 1, 1).expand(-1, 2, 768)
        hidden = torch.cat((special_tokens, patch_tokens), dim=1)
        return SimpleNamespace(
            last_hidden_state=hidden,
            pooler_output=input_summary.unsqueeze(1).expand(-1, 768),
        )


def _fake_encoder(granularity: str) -> AstEncoder:
    """Bypass the official constructor to inject a lightweight backbone for output-contract tests."""
    encoder = AstEncoder.__new__(AstEncoder)
    BaseEncoder.__init__(encoder, granularity)
    encoder.pretrained = False
    encoder.pretrained_dir = None
    encoder.backbone = _FakeAstBackbone()
    return encoder


def test_public_export_and_keyword_only_constructor():
    from timbral.models.encoders.ast_encoder import AstEncoder as DirectEncoder

    assert AstEncoder is DirectEncoder
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in inspect.signature(AstEncoder).parameters.values()
    )


def test_pretrained_false_uses_fixed_architecture_without_checkpoint(
    monkeypatch,
):
    monkeypatch.setattr(
        encoder_module,
        "ensure_ast_checkpoint",
        lambda _: pytest.fail("pretrained=False should not resolve the checkpoint"),
    )
    encoder = AstEncoder(granularity="clip", pretrained=False)
    config = encoder.backbone.config

    assert ast_helpers.AST_CONFIG_FIELDS == _EXPECTED_AST_CONFIG_FIELDS
    for name, expected in _EXPECTED_AST_CONFIG_FIELDS.items():
        assert getattr(config, name) == expected
    assert encoder.training
    assert encoder.backbone.training


def test_pretrained_requires_bool():
    with pytest.raises(TypeError, match="bool"):
        AstEncoder(granularity="clip", pretrained=1)


def test_clip_output_uses_pooler_and_base_geometry():
    encoder = _fake_encoder("clip")
    features = torch.randn(2, 1024, 128)
    valid_seconds = torch.tensor([0.025, 10.255])

    output = encoder(features, valid_seconds=valid_seconds)

    assert output["embedding"].shape == (2, 768)
    torch.testing.assert_close(
        output["embedding"][:, 0],
        features.mean(dim=(1, 2))
        * encoder.backbone.embeddings.patch_embeddings.projection.weight
        .reshape(()),
    )
    torch.testing.assert_close(
        output["geometry"],
        torch.tensor([[0.0, 0.025], [0.0, 10.255]]),
    )
    assert output["valid_mask"].tolist() == [True, True]
    assert output["geometry"].dtype is torch.float32
    assert output["valid_mask"].dtype is torch.bool


def test_frame_export_preserves_frequency_major_token_order():
    encoder = _fake_encoder("frame")
    features = torch.zeros(1, 1024, 128)

    output = encoder(
        features,
        valid_seconds=torch.tensor([10.255]),
    )

    assert output["embedding"].shape == (1, 101, 768)
    expected = torch.arange(101, dtype=torch.float32) + 5500.0
    torch.testing.assert_close(output["embedding"][0, :, 0], expected)
    torch.testing.assert_close(
        output["embedding"][0, :, 0],
        output["embedding"][0, :, -1],
    )
    assert output["geometry"].dtype is torch.float32
    assert output["valid_mask"].dtype is torch.bool


@pytest.mark.parametrize(
    ("seconds", "valid_frames", "last_geometry"),
    [
        (0.025, 1, (0.0, 0.025)),
        (10.0, 100, (9.9, 10.0)),
        (10.245, 101, (10.0, 10.245)),
        (10.255, 101, (10.0, 10.255)),
    ],
)
def test_frame_geometry_and_ceil_valid_count(
    seconds,
    valid_frames,
    last_geometry,
):
    encoder = _fake_encoder("frame")

    output = encoder(
        torch.zeros(1, 1024, 128),
        valid_seconds=torch.tensor([seconds]),
    )

    assert output["valid_mask"].sum().item() == valid_frames
    torch.testing.assert_close(
        output["geometry"][0, valid_frames - 1],
        torch.tensor(last_geometry),
    )
    assert torch.equal(
        output["embedding"][0, valid_frames:],
        torch.zeros_like(output["embedding"][0, valid_frames:]),
    )
    assert torch.equal(
        output["geometry"][0, valid_frames:],
        torch.zeros_like(output["geometry"][0, valid_frames:]),
    )


def test_mixed_length_frame_batch_zeros_invalid_positions():
    encoder = _fake_encoder("frame")
    output = encoder(
        torch.randn(3, 1024, 128),
        valid_seconds=torch.tensor([0.025, 0.2, 10.255]),
    )

    assert output["valid_mask"].sum(dim=1).tolist() == [1, 2, 101]
    assert torch.count_nonzero(output["embedding"][0, 1:]) == 0
    assert torch.count_nonzero(output["geometry"][1, 2:]) == 0
    torch.testing.assert_close(
        output["geometry"][1, :2],
        torch.tensor([[0.0, 0.1], [0.1, 0.2]]),
    )


def test_frame_geometry_reuses_bit_exact_adjacent_boundaries():
    output = _fake_encoder("frame")(
        torch.zeros(1, 1024, 128),
        valid_seconds=torch.tensor([1.0]),
    )
    valid_geometry = output["geometry"][0, output["valid_mask"][0]]

    assert torch.equal(
        valid_geometry[:-1, 1],
        valid_geometry[1:, 0],
    )


@pytest.mark.parametrize("sample_rate", [32000, 48000, 96000])
def test_positive_duration_rounding_to_zero_keeps_one_frame(sample_rate):
    transform_output = AstKaldiFbankTransform()(
        torch.ones(1, 1),
        sample_rate=sample_rate,
    )
    assert torch.round(
        transform_output["valid_seconds"] * 16000
    ).item() == 0

    output = _fake_encoder("frame")(**transform_output)

    assert output["valid_mask"].sum().item() == 1
    torch.testing.assert_close(
        output["geometry"][0, 0],
        torch.stack(
            (
                torch.tensor(0.0),
                transform_output["valid_seconds"][0],
            )
        ),
    )
    assert torch.count_nonzero(output["embedding"][0, 1:]) == 0
    assert torch.count_nonzero(output["geometry"][0, 1:]) == 0


def test_last_eight_fbank_rows_do_not_enter_patch_projection():
    encoder = AstEncoder(granularity="clip", pretrained=False).eval()
    base = torch.randn(1, 1024, 128)
    changed = base.clone()
    changed[:, 1016:] += 100.0

    with torch.inference_mode():
        base_patches = encoder.backbone.embeddings.patch_embeddings(base)
        changed_patches = encoder.backbone.embeddings.patch_embeddings(
            changed
        )

    assert torch.equal(base_patches, changed_patches)


def test_unknown_model_input_raises():
    encoder = _fake_encoder("clip")
    with pytest.raises(TypeError, match="unknown"):
        encoder(
            torch.randn(1, 1024, 128),
            valid_seconds=torch.tensor([1.0]),
            unknown=True,
        )


def test_training_path_preserves_input_and_parameter_gradients():
    encoder = _fake_encoder("frame")
    features = torch.randn(2, 1024, 128, requires_grad=True)

    output = encoder(
        features,
        valid_seconds=torch.tensor([0.1, 0.2]),
    )
    output["embedding"].mean().backward()

    projection = (
        encoder.backbone.embeddings.patch_embeddings.projection.weight
    )
    assert features.grad is not None
    assert projection.grad is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No CUDA available in the current environment.",
)
def test_base_encoder_moves_cpu_inputs_to_cuda():
    encoder = _fake_encoder("clip").cuda()

    output = encoder(
        torch.randn(1, 1024, 128),
        valid_seconds=torch.tensor([0.025]),
    )

    assert output["embedding"].is_cuda
    assert output["geometry"].is_cuda
    assert output["valid_mask"].is_cuda


def test_pretrained_path_loads_backbone_strictly(
    monkeypatch,
    tmp_path: Path,
):
    loaded: dict[str, object] = {}

    class _LoadingBackbone(_FakeAstBackbone):
        def __init__(self, config) -> None:
            super().__init__()
            loaded["config"] = config

        def load_state_dict(self, state_dict, strict=True):
            loaded["state"] = state_dict
            loaded["strict"] = strict
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    config = ast_helpers.fixed_ast_config()
    expected_state = {"embeddings.cls_token": torch.ones(1)}
    monkeypatch.setattr(
        encoder_module,
        "ensure_ast_checkpoint",
        lambda _: tmp_path,
    )
    monkeypatch.setattr(
        encoder_module,
        "load_and_validate_ast_config",
        lambda _: config,
    )
    monkeypatch.setattr(
        encoder_module,
        "load_ast_backbone_state",
        lambda _: expected_state,
    )
    monkeypatch.setattr(encoder_module, "ASTModel", _LoadingBackbone)

    AstEncoder(
        granularity="clip",
        pretrained=True,
        pretrained_dir=tmp_path,
    )

    assert loaded == {
        "config": config,
        "state": expected_state,
        "strict": True,
    }


def test_checkpoint_state_filter_requires_exact_classifier_set(
    monkeypatch,
):
    backbone_state = {
        f"audio_spectrogram_transformer.tensor_{index}": torch.ones(1)
        for index in range(199)
    }
    classifier_state = {
        key: torch.ones(1)
        for key in (
            "classifier.dense.bias",
            "classifier.dense.weight",
            "classifier.layernorm.bias",
            "classifier.layernorm.weight",
        )
    }
    monkeypatch.setattr(
        ast_helpers,
        "load_file",
        lambda *_args, **_kwargs: backbone_state | classifier_state,
    )

    result = ast_helpers.load_ast_backbone_state(Path("unused"))

    assert len(result) == 199
    assert "tensor_0" in result

    monkeypatch.setattr(
        ast_helpers,
        "load_file",
        lambda *_args, **_kwargs: (
            backbone_state | classifier_state | {"unexpected": torch.ones(1)}
        ),
    )
    with pytest.raises(ValueError, match="Non-backbone"):
        ast_helpers.load_ast_backbone_state(Path("unused"))


def test_checkpoint_legacy_keys_follow_official_ast_conversion(
    monkeypatch,
):
    backbone_state = {
        (
            "audio_spectrogram_transformer.encoder.layer.0."
            "attention.attention.query.weight"
        ): torch.ones(1),
        **{
            f"audio_spectrogram_transformer.tensor_{index}": torch.ones(1)
            for index in range(198)
        },
    }
    classifier_state = {
        key: torch.ones(1)
        for key in (
            "classifier.dense.bias",
            "classifier.dense.weight",
            "classifier.layernorm.bias",
            "classifier.layernorm.weight",
        )
    }
    monkeypatch.setattr(
        ast_helpers,
        "load_file",
        lambda *_args, **_kwargs: backbone_state | classifier_state,
    )

    result = ast_helpers.load_ast_backbone_state(Path("unused"))

    assert "layers.0.attention.q_proj.weight" in result
