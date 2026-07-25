"""Unit tests for PannsCnn14Encoder without weights."""

from __future__ import annotations

import inspect

import pytest
import torch

from timbral.models.encoders import PannsCnn14Encoder, PannsVariant
from timbral.models.encoders import panns as encoder_module
from timbral.models.helpers.panns import (
    PANNS_CHECKPOINTS,
    PannsVariant as HelperPannsVariant,
)


def _encoder(
    *,
    granularity: str = "frame",
    target_sample_rate: int = 16000,
    variant: str = "max_mean",
) -> PannsCnn14Encoder:
    return PannsCnn14Encoder(
        granularity=granularity,
        target_sample_rate=target_sample_rate,
        variant=variant,
        pretrained=False,
    )


def test_public_export_metadata_and_keyword_only_constructor():
    from timbral.models.encoders.panns import (
        PannsCnn14Encoder as DirectEncoder,
        PannsVariant as DirectVariant,
    )

    assert PannsCnn14Encoder is DirectEncoder
    assert PannsVariant is DirectVariant
    assert PannsVariant is HelperPannsVariant
    assert set(PANNS_CHECKPOINTS) == {
        (16000, "max_mean"),
        (32000, "max_mean"),
        (32000, "decision_level_max"),
    }
    parameters = inspect.signature(PannsCnn14Encoder).parameters
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )


def test_random_16k_decision_level_max_is_allowed():
    encoder = _encoder(variant="decision_level_max")

    assert encoder.variant == "decision_level_max"
    assert not encoder.pretrained


def test_pretrained_16k_decision_level_max_is_rejected():
    with pytest.raises(ValueError, match="No official weights exist"):
        PannsCnn14Encoder(
            granularity="frame",
            target_sample_rate=16000,
            variant="decision_level_max",
            pretrained=True,
        )


def test_architecture_has_fixed_downsample_ratio_and_shapes():
    encoder = _encoder()

    assert encoder_module._ENCODER_DOWNSAMPLE_RATIO == 32
    channels = [
        encoder.conv_block1.conv1.out_channels,
        encoder.conv_block2.conv1.out_channels,
        encoder.conv_block3.conv1.out_channels,
        encoder.conv_block4.conv1.out_channels,
        encoder.conv_block5.conv1.out_channels,
        encoder.conv_block6.conv1.out_channels,
    ]
    assert channels == [64, 128, 256, 512, 1024, 2048]
    assert encoder.fc1.in_features == encoder.fc1.out_features == 2048


@pytest.mark.parametrize(
    ("variant", "granularity"),
    [
        ("max_mean", "clip"),
        ("max_mean", "frame"),
        ("decision_level_max", "clip"),
        ("decision_level_max", "frame"),
    ],
)
def test_all_variant_granularity_combinations(
    variant,
    granularity,
):
    encoder = _encoder(
        variant=variant,
        granularity=granularity,
    ).eval()
    input_features = torch.randn(1, 33, 64)

    with torch.inference_mode():
        output = encoder(
            input_features,
            valid_feature_frames=torch.tensor([33]),
            valid_seconds=torch.tensor([0.32]),
        )

    if granularity == "clip":
        assert output["embedding"].shape == (1, 2048)
        assert output["geometry"].shape == (1, 2)
        assert output["valid_mask"].tolist() == [True]
    else:
        assert output["embedding"].shape == (1, 1, 2048)
        assert output["geometry"].shape == (1, 1, 2)
        assert output["valid_mask"].tolist() == [[True]]


def test_short_metadata_requires_and_uses_physical_32_frames():
    encoder = _encoder().eval()
    input_features = torch.randn(2, 32, 64)

    with torch.inference_mode():
        output = encoder(
            input_features,
            valid_feature_frames=torch.tensor([1, 31]),
            valid_seconds=torch.tensor([0.01, 0.30]),
        )

    assert output["embedding"].shape == (2, 1, 2048)
    assert output["valid_mask"].tolist() == [[True], [True]]
    torch.testing.assert_close(
        output["geometry"],
        torch.tensor([[[0.0, 0.01]], [[0.0, 0.30]]]),
    )


def test_remainder_feature_can_affect_retained_embedding():
    encoder = _encoder().eval()
    base = torch.randn(1, 33, 64)
    changed = base.clone()
    changed[:, -1] += 10.0

    with torch.inference_mode():
        base_output = encoder(
            base,
            valid_feature_frames=torch.tensor([33]),
            valid_seconds=torch.tensor([0.32]),
        )["embedding"]
        changed_output = encoder(
            changed,
            valid_feature_frames=torch.tensor([33]),
            valid_seconds=torch.tensor([0.32]),
        )["embedding"]

    assert not torch.equal(base_output, changed_output)


def test_mixed_length_batch_matches_individual_calls_and_zeros_padding():
    encoder = _encoder().eval()
    input_features = torch.randn(2, 65, 64)
    input_features[0, 33:] = 123.0
    valid_frames = torch.tensor([33, 65])
    valid_seconds = torch.tensor([0.32, 0.65])

    with torch.inference_mode():
        batch_output = encoder(
            input_features,
            valid_feature_frames=valid_frames,
            valid_seconds=valid_seconds,
        )
        individual_outputs = [
            encoder(
                input_features[index : index + 1, :length],
                valid_feature_frames=valid_frames[index : index + 1],
                valid_seconds=valid_seconds[index : index + 1],
            )
            for index, length in enumerate((33, 65))
        ]

    torch.testing.assert_close(
        batch_output["embedding"][0, :1],
        individual_outputs[0]["embedding"][0],
    )
    torch.testing.assert_close(
        batch_output["embedding"][1, :2],
        individual_outputs[1]["embedding"][0],
    )
    assert batch_output["valid_mask"].tolist() == [
        [True, False],
        [True, True],
    ]
    torch.testing.assert_close(
        batch_output["embedding"][0, 1],
        torch.zeros(2048),
    )
    torch.testing.assert_close(
        batch_output["geometry"],
        torch.tensor(
            [
                [[0.0, 0.32], [0.0, 0.0]],
                [[0.0, 0.32], [0.32, 0.65]],
            ]
        ),
    )


def test_frame_geometry_reuses_bit_exact_adjacent_boundaries():
    encoder = _encoder().eval()
    with torch.inference_mode():
        output = encoder(
            torch.randn(1, 224, 64),
            valid_feature_frames=torch.tensor([224]),
            valid_seconds=torch.tensor([2.24]),
        )
    valid_geometry = output["geometry"][0, output["valid_mask"][0]]

    assert torch.equal(
        valid_geometry[:-1, 1],
        valid_geometry[1:, 0],
    )


def test_max_mean_frame_branch_applies_linear_on_embedding_dimension():
    encoder = _encoder().eval()
    backbone_features = torch.randn(2, 2048, 3)

    with torch.inference_mode():
        output = encoder._max_mean_frames(backbone_features)
        expected = torch.relu(
            torch.nn.functional.linear(
                backbone_features.transpose(1, 2),
                encoder.fc1.weight,
                encoder.fc1.bias,
            )
        )

    assert output.shape == (2, 3, 2048)
    torch.testing.assert_close(output, expected)


def test_decision_level_clip_is_frame_amax():
    clip_encoder = _encoder(
        granularity="clip",
        variant="decision_level_max",
    ).eval()
    frame_encoder = _encoder(
        granularity="frame",
        variant="decision_level_max",
    ).eval()
    frame_encoder.load_state_dict(clip_encoder.state_dict())
    features = torch.randn(1, 65, 64)
    common = {
        "valid_feature_frames": torch.tensor([65]),
        "valid_seconds": torch.tensor([0.65]),
    }

    with torch.inference_mode():
        clip_output = clip_encoder(features, **common)
        frame_output = frame_encoder(features, **common)

    torch.testing.assert_close(
        clip_output["embedding"],
        frame_output["embedding"].amax(dim=1),
    )


def test_training_dropout_is_active_and_eval_is_deterministic():
    encoder = _encoder()
    features = torch.randn(2, 33, 64)
    arguments = {
        "valid_feature_frames": torch.tensor([33, 33]),
        "valid_seconds": torch.tensor([0.32, 0.32]),
    }

    training_a = encoder(features, **arguments)["embedding"]
    training_b = encoder(features, **arguments)["embedding"]
    assert not torch.equal(training_a, training_b)

    encoder.eval()
    with torch.inference_mode():
        evaluation_a = encoder(features, **arguments)["embedding"]
        evaluation_b = encoder(features, **arguments)["embedding"]
    torch.testing.assert_close(evaluation_a, evaluation_b)


def test_unknown_model_input_raises():
    encoder = _encoder()

    with pytest.raises(TypeError, match="unknown"):
        encoder(
            torch.randn(1, 32, 64),
            valid_feature_frames=torch.tensor([3]),
            valid_seconds=torch.tensor([0.02]),
            unknown=True,
        )


def test_training_path_preserves_backbone_gradients():
    encoder = _encoder(granularity="clip")
    input_features = torch.randn(2, 32, 64, requires_grad=True)

    output = encoder(
        input_features,
        valid_feature_frames=torch.tensor([3, 31]),
        valid_seconds=torch.tensor([0.02, 0.30]),
    )
    output["embedding"].mean().backward()

    assert input_features.grad is not None
    assert encoder.conv_block1.conv1.weight.grad is not None
    assert encoder.fc1.weight.grad is not None
