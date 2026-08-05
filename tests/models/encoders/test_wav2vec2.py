"""Offline unit tests for ``Wav2Vec2Encoder`` and the wav2vec2 helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from timbral.models.encoders import Wav2Vec2Encoder
from timbral.models.encoders import wav2vec2 as encoder_module
from timbral.models.encoders.base import BaseEncoder
from timbral.models.helpers import wav2vec2 as wav2vec2_helpers
from timbral.models.helpers.wav2vec2 import (
    WAV2VEC2_CONFIG_FIELDS,
    fixed_wav2vec2_config,
    load_wav2vec2_backbone_state,
    wav2vec2_feature_frames,
)

_EXPECTED_WAV2VEC2_CONFIG_FIELDS = {
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
_PRETRAINING_HEAD_KEYS = (
    "project_hid.bias",
    "project_hid.weight",
    "project_q.bias",
    "project_q.weight",
    "quantizer.codevectors",
    "quantizer.weight_proj.bias",
    "quantizer.weight_proj.weight",
)
_LEGACY_POS_CONV_KEYS = (
    "wav2vec2.encoder.pos_conv_embed.conv.weight_g",
    "wav2vec2.encoder.pos_conv_embed.conv.weight_v",
)
_RENAMED_POS_CONV_KEYS = (
    "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
)


class _FakeWav2Vec2Backbone(nn.Module):
    """Lightweight backbone exposing the length-to-frame mapping.

    The last hidden state of a ``[B, L]`` waveform group is
    ``summary[b] + t`` broadcast over the 768 channels, where ``summary``
    is the scaled mean of the exact input prefix and ``t`` is the frame
    index over ``wav2vec2_feature_frames(L)`` frames. Any padding leak or
    frame-count drift therefore changes the output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_projection = nn.Module()
        self.feature_projection.projection = nn.Linear(1, 1, bias=False)

    def forward(self, *, input_values: Tensor) -> SimpleNamespace:
        """Return a minimal object with ``last_hidden_state``."""
        scale = self.feature_projection.projection.weight.reshape(())
        num_frames = int(
            wav2vec2_feature_frames(torch.tensor(input_values.shape[1]))
        )
        input_summary = input_values.mean(dim=1) * scale
        frame_index = torch.arange(
            num_frames,
            device=input_values.device,
            dtype=input_values.dtype,
        )
        hidden = (
            input_summary.view(-1, 1, 1) + frame_index.view(1, -1, 1)
        ).expand(-1, -1, 768)
        return SimpleNamespace(last_hidden_state=hidden)


def _fake_encoder(granularity: str) -> Wav2Vec2Encoder:
    """Bypass the official constructor to inject a lightweight backbone."""
    encoder = Wav2Vec2Encoder.__new__(Wav2Vec2Encoder)
    BaseEncoder.__init__(encoder, granularity)
    encoder.pretrained = False
    encoder.pretrained_dir = None
    encoder.backbone = _FakeWav2Vec2Backbone()
    return encoder


def _make_inputs(
    valid_samples: list[int],
    seed: int = 3,
) -> dict[str, Tensor]:
    """Build a zero-padded waveform canvas plus its length tensors."""
    generator = torch.Generator().manual_seed(seed)
    features = torch.zeros(len(valid_samples), max(valid_samples))
    for index, samples in enumerate(valid_samples):
        features[index, :samples] = torch.randn(
            (samples,),
            generator=generator,
        )
    return {
        "input_features": features,
        "valid_samples": torch.tensor(valid_samples, dtype=torch.int64),
        "valid_seconds": torch.tensor(
            [samples / 16000 for samples in valid_samples],
            dtype=torch.float32,
        ),
    }


def _fake_checkpoint_state() -> dict[str, Tensor]:
    """Mimic the official ``Wav2Vec2ForPreTraining`` checkpoint layout."""
    state = {key: torch.ones(1) for key in _PRETRAINING_HEAD_KEYS}
    state["wav2vec2.masked_spec_embed"] = torch.ones(1)
    for key in _LEGACY_POS_CONV_KEYS:
        state[key] = torch.ones(1)
    for index in range(208):
        state[f"wav2vec2.tensor_{index}"] = torch.ones(1)
    return state


@pytest.fixture(scope="module")
def random_frame_encoder() -> Wav2Vec2Encoder:
    """Build the single real random-weight backbone used by real tests."""
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            encoder_module,
            "ensure_wav2vec2_checkpoint",
            lambda *_args, **_kwargs: pytest.fail(
                "pretrained=False should not resolve the checkpoint"
            ),
        )
        torch.manual_seed(0)
        encoder = Wav2Vec2Encoder(granularity="frame", pretrained=False)
    return encoder.eval()


@pytest.fixture(scope="module")
def random_clip_encoder(random_frame_encoder) -> Wav2Vec2Encoder:
    """Share the frame fixture's backbone under clip granularity."""
    encoder = Wav2Vec2Encoder.__new__(Wav2Vec2Encoder)
    BaseEncoder.__init__(encoder, "clip")
    encoder.pretrained = False
    encoder.pretrained_dir = None
    encoder.backbone = random_frame_encoder.backbone
    return encoder.eval()


def test_public_export_and_keyword_only_constructor():
    from timbral.models.encoders.wav2vec2 import (
        Wav2Vec2Encoder as DirectEncoder,
    )

    assert Wav2Vec2Encoder is DirectEncoder
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in inspect.signature(
            Wav2Vec2Encoder
        ).parameters.values()
    )
    assert Wav2Vec2Encoder.supported_granularities == frozenset(
        ("clip", "frame")
    )
    assert Wav2Vec2Encoder.embedding_dim == 768


def test_invalid_construction():
    with pytest.raises(ValueError, match="granularity"):
        Wav2Vec2Encoder(granularity="chunk", pretrained=False)
    with pytest.raises(TypeError, match="bool"):
        Wav2Vec2Encoder(granularity="clip", pretrained=1)


def test_pretrained_false_uses_fixed_architecture_without_checkpoint(
    monkeypatch,
):
    monkeypatch.setattr(
        encoder_module,
        "ensure_wav2vec2_checkpoint",
        lambda *_args, **_kwargs: pytest.fail(
            "pretrained=False should not resolve the checkpoint"
        ),
    )
    encoder = Wav2Vec2Encoder(granularity="clip", pretrained=False)
    config = encoder.backbone.config

    assert WAV2VEC2_CONFIG_FIELDS == _EXPECTED_WAV2VEC2_CONFIG_FIELDS
    for name, expected in _EXPECTED_WAV2VEC2_CONFIG_FIELDS.items():
        assert getattr(config, name) == expected
    assert encoder.pretrained is False
    assert encoder.pretrained_dir is None
    # The constructor must not touch the default training lifecycle.
    assert encoder.training
    assert encoder.backbone.training


@pytest.mark.parametrize(
    ("samples", "frames"),
    [(400, 1), (719, 1), (720, 2), (16000, 49)],
)
def test_feature_frames_conv_formula_boundaries(samples, frames):
    result = wav2vec2_feature_frames(
        torch.tensor([samples], dtype=torch.int64)
    )
    assert result.tolist() == [frames]


def test_clip_output_contract():
    encoder = _fake_encoder("clip")
    inputs = _make_inputs([720, 16000])

    output = encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_samples=inputs["valid_samples"],
    )

    assert output["embedding"].shape == (2, 768)
    assert output["embedding"].dtype is torch.float32
    scale = (
        encoder.backbone.feature_projection.projection.weight.reshape(())
    )
    # 720 samples give 2 frames (mean index 0.5); 16000 give 49 (mean 24).
    expected = torch.stack(
        (
            inputs["input_features"][0, :720].mean() * scale + 0.5,
            inputs["input_features"][1].mean() * scale + 24.0,
        )
    )
    torch.testing.assert_close(output["embedding"][:, 0], expected)
    assert torch.equal(output["embedding"][:, 0], output["embedding"][:, -1])
    torch.testing.assert_close(
        output["geometry"],
        torch.tensor([[0.0, 0.045], [0.0, 1.0]]),
    )
    assert output["geometry"].dtype is torch.float32
    assert output["valid_mask"].dtype is torch.bool
    assert output["valid_mask"].tolist() == [True, True]


def test_clip_matches_frame_valid_mean():
    frame_encoder = _fake_encoder("frame")
    clip_encoder = _fake_encoder("clip")
    clip_encoder.backbone = frame_encoder.backbone
    inputs = _make_inputs([400, 720, 16000])

    frame_output = frame_encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_samples=inputs["valid_samples"],
    )
    clip_output = clip_encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_samples=inputs["valid_samples"],
    )

    for index in range(3):
        valid = frame_output["valid_mask"][index]
        torch.testing.assert_close(
            clip_output["embedding"][index],
            frame_output["embedding"][index][valid].mean(dim=0),
        )


def test_frame_output_contract():
    encoder = _fake_encoder("frame")
    inputs = _make_inputs([400, 16000, 16000])

    output = encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_samples=inputs["valid_samples"],
    )

    assert output["embedding"].shape == (3, 49, 768)
    assert output["geometry"].shape == (3, 49, 2)
    assert output["valid_mask"].shape == (3, 49)
    assert output["embedding"].dtype is torch.float32
    assert output["geometry"].dtype is torch.float32
    assert output["valid_mask"].dtype is torch.bool
    assert output["valid_mask"].sum(dim=1).tolist() == [1, 49, 49]
    # Invalid slots are exactly 0, not merely small.
    assert torch.count_nonzero(output["embedding"][0, 1:]) == 0
    assert torch.count_nonzero(output["geometry"][0, 1:]) == 0
    scale = (
        encoder.backbone.feature_projection.projection.weight.reshape(())
    )
    torch.testing.assert_close(
        output["embedding"][1, :, 0],
        inputs["input_features"][1].mean() * scale + torch.arange(49.0),
    )
    assert torch.equal(
        output["embedding"][..., 0],
        output["embedding"][..., -1],
    )


def test_frame_geometry_follows_20ms_grid():
    encoder = _fake_encoder("frame")
    inputs = _make_inputs([720, 16000])

    output = encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_samples=inputs["valid_samples"],
    )

    torch.testing.assert_close(
        output["geometry"][0, :2],
        torch.tensor([[0.0, 0.02], [0.02, 0.045]]),
    )
    assert torch.count_nonzero(output["geometry"][0, 2:]) == 0
    valid_geometry = output["geometry"][1][output["valid_mask"][1]]
    torch.testing.assert_close(
        valid_geometry[:, 0],
        torch.arange(49.0) * 0.02,
    )
    # Adjacent valid boundaries are bit-identical; the last slot
    # stretches from the nominal grid to valid_seconds.
    assert torch.equal(valid_geometry[:-1, 1], valid_geometry[1:, 0])
    assert valid_geometry[0, 0].item() == 0.0
    assert valid_geometry[-1, 1].item() == 1.0
    assert valid_geometry[-1, 0].item() == pytest.approx(0.96)


def test_mixed_length_batch_matches_single_calls_bitwise():
    encoder = _fake_encoder("frame")
    inputs = _make_inputs([16000, 400, 720])

    output_batch = encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_samples=inputs["valid_samples"],
    )

    for index, samples in enumerate((16000, 400, 720)):
        output_single = encoder(
            inputs["input_features"][index : index + 1, :samples],
            valid_seconds=inputs["valid_seconds"][index : index + 1],
            valid_samples=inputs["valid_samples"][index : index + 1],
        )
        frames = output_single["embedding"].shape[1]
        assert torch.equal(
            output_batch["embedding"][index, :frames],
            output_single["embedding"][0],
        )
        assert torch.equal(
            output_batch["geometry"][index, :frames],
            output_single["geometry"][0],
        )
        assert torch.count_nonzero(
            output_batch["embedding"][index, frames:]
        ) == 0


def test_real_backbone_mixed_batch_matches_single_calls(
    random_frame_encoder,
):
    inputs = _make_inputs([1120, 400, 720])

    output_batch = random_frame_encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_samples=inputs["valid_samples"],
    )

    assert output_batch["embedding"].shape == (3, 3, 768)
    assert output_batch["valid_mask"].sum(dim=1).tolist() == [3, 1, 2]
    for index, samples in enumerate((1120, 400, 720)):
        output_single = random_frame_encoder(
            inputs["input_features"][index : index + 1, :samples],
            valid_seconds=inputs["valid_seconds"][index : index + 1],
            valid_samples=inputs["valid_samples"][index : index + 1],
        )
        frames = output_single["embedding"].shape[1]
        torch.testing.assert_close(
            output_batch["embedding"][index, :frames],
            output_single["embedding"][0],
        )
        assert torch.count_nonzero(
            output_batch["embedding"][index, frames:]
        ) == 0


def test_real_backbone_clip_matches_frame_valid_mean(
    random_frame_encoder,
    random_clip_encoder,
):
    inputs = _make_inputs([1120])

    frame_output = random_frame_encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_samples=inputs["valid_samples"],
    )
    clip_output = random_clip_encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_samples=inputs["valid_samples"],
    )

    assert frame_output["embedding"].shape == (1, 3, 768)
    torch.testing.assert_close(
        clip_output["embedding"][0],
        frame_output["embedding"][0].mean(dim=0),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        clip_output["geometry"],
        torch.tensor([[0.0, 0.07]]),
    )


def test_unknown_forward_input_raises():
    encoder = _fake_encoder("clip")
    inputs = _make_inputs([720])

    with pytest.raises(TypeError):
        encoder(
            inputs["input_features"],
            valid_seconds=inputs["valid_seconds"],
            valid_samples=inputs["valid_samples"],
            unknown_input=torch.zeros(1),
        )
    with pytest.raises(TypeError):
        encoder(
            inputs["input_features"],
            valid_seconds=inputs["valid_seconds"],
        )


@pytest.mark.parametrize("granularity", ("clip", "frame"))
@pytest.mark.parametrize("input_name", ("valid_seconds", "valid_samples"))
def test_rejects_mismatched_batch_cardinality(granularity, input_name):
    encoder = _fake_encoder(granularity)
    inputs = _make_inputs([400, 720])
    inputs[input_name] = inputs[input_name][:1]

    with pytest.raises(
        ValueError,
        match=rf"{input_name} must have shape \[B\]",
    ):
        encoder(
            inputs["input_features"],
            valid_seconds=inputs["valid_seconds"],
            valid_samples=inputs["valid_samples"],
        )


def test_training_path_preserves_input_and_parameter_gradients():
    encoder = _fake_encoder("frame")
    inputs = _make_inputs([400, 800])
    features = inputs["input_features"].requires_grad_()

    output = encoder(
        features,
        valid_seconds=inputs["valid_seconds"],
        valid_samples=inputs["valid_samples"],
    )
    output["embedding"].mean().backward()

    assert features.grad is not None
    assert (
        encoder.backbone.feature_projection.projection.weight.grad
        is not None
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No CUDA available in the current environment.",
)
def test_base_encoder_moves_cpu_inputs_to_cuda():
    encoder = _fake_encoder("frame").cuda()
    inputs = _make_inputs([400, 720])

    output = encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_samples=inputs["valid_samples"],
    )

    assert encoder.device.type == "cuda"
    assert output["embedding"].is_cuda
    assert output["geometry"].is_cuda
    assert output["valid_mask"].is_cuda


def test_pretrained_path_loads_backbone_strictly(
    monkeypatch,
    tmp_path: Path,
):
    loaded: dict[str, object] = {}

    class _LoadingBackbone(_FakeWav2Vec2Backbone):
        def __init__(self, config) -> None:
            super().__init__()
            loaded["config"] = config

        def load_state_dict(self, state_dict, strict=True):
            loaded["state"] = state_dict
            loaded["strict"] = strict
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    config = fixed_wav2vec2_config()
    expected_state = {"masked_spec_embed": torch.ones(1)}
    monkeypatch.setattr(
        encoder_module,
        "ensure_wav2vec2_checkpoint",
        lambda _: tmp_path,
    )
    monkeypatch.setattr(
        encoder_module,
        "load_and_validate_wav2vec2_config",
        lambda _: config,
    )
    monkeypatch.setattr(
        encoder_module,
        "load_wav2vec2_backbone_state",
        lambda _: expected_state,
    )
    monkeypatch.setattr(encoder_module, "Wav2Vec2Model", _LoadingBackbone)

    encoder = Wav2Vec2Encoder(
        granularity="clip",
        pretrained=True,
        pretrained_dir=tmp_path,
    )

    assert loaded == {
        "config": config,
        "state": expected_state,
        "strict": True,
    }
    assert encoder.pretrained is True
    assert encoder.pretrained_dir == tmp_path


def test_checkpoint_state_filter_and_pos_conv_rename(monkeypatch):
    recorded: dict[str, object] = {}
    state = _fake_checkpoint_state()

    def _fake_torch_load(path, weights_only=False, map_location=None):
        recorded["weights_only"] = weights_only
        recorded["map_location"] = map_location
        return state

    monkeypatch.setattr(wav2vec2_helpers.torch, "load", _fake_torch_load)

    result = load_wav2vec2_backbone_state(Path("unused"))

    assert recorded == {"weights_only": True, "map_location": "cpu"}
    assert len(result) == 211
    assert "masked_spec_embed" in result
    assert "tensor_0" in result
    for legacy, renamed in zip(
        _LEGACY_POS_CONV_KEYS,
        _RENAMED_POS_CONV_KEYS,
    ):
        assert legacy.removeprefix("wav2vec2.") not in result
        assert renamed in result
    for head_key in _PRETRAINING_HEAD_KEYS:
        assert head_key not in result


def test_checkpoint_state_filter_rejects_unexpected_layouts(monkeypatch):
    extra_state = _fake_checkpoint_state() | {"unexpected": torch.ones(1)}
    monkeypatch.setattr(
        wav2vec2_helpers.torch,
        "load",
        lambda *_args, **_kwargs: extra_state,
    )
    with pytest.raises(ValueError, match="Non-backbone"):
        load_wav2vec2_backbone_state(Path("unused"))

    truncated_state = _fake_checkpoint_state()
    del truncated_state["wav2vec2.tensor_0"]
    monkeypatch.setattr(
        wav2vec2_helpers.torch,
        "load",
        lambda *_args, **_kwargs: truncated_state,
    )
    with pytest.raises(ValueError, match="tensor count"):
        load_wav2vec2_backbone_state(Path("unused"))


def test_train_mode_spec_augment_rejects_sub_mask_length_samples(
    random_frame_encoder,
):
    """Fewer than 10 frames raise upstream SpecAugment's ValueError in
    train mode, while eval mode accepts the same Transform-legal input.
    """
    features = torch.zeros(1, 400)
    valid_seconds = torch.tensor([0.025])
    valid_samples = torch.tensor([400])

    random_frame_encoder.train()
    try:
        with pytest.raises(ValueError, match="mask_length"):
            random_frame_encoder(
                features,
                valid_seconds=valid_seconds,
                valid_samples=valid_samples,
            )
    finally:
        random_frame_encoder.eval()

    output = random_frame_encoder(
        features,
        valid_seconds=valid_seconds,
        valid_samples=valid_samples,
    )
    assert output["embedding"].shape == (1, 1, 768)
