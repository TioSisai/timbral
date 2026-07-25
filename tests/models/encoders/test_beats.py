"""Unit tests for ``BeatsEncoder`` and ``helpers/beats`` without weights."""

from __future__ import annotations

import pytest
import torch

from timbral.models.encoders import BeatsEncoder
from timbral.models.encoders import beats as beats_encoder_module
from timbral.models.helpers.beats import (
    BEATS_CHECKPOINTS,
    BEATS_EXPECTED_CFG_FINETUNED,
    BEATS_EXPECTED_CFG_PRETRAINED,
    ensure_beats_checkpoint,
)

_EXPECTED_TOP_KEYS = (
    "patch_embedding.weight",
    "layer_norm.weight",
    "layer_norm.bias",
    "post_extract_proj.weight",
    "post_extract_proj.bias",
    "encoder.pos_conv.0.bias",
    "encoder.pos_conv.0.parametrizations.weight.original0",
    "encoder.pos_conv.0.parametrizations.weight.original1",
    "encoder.layer_norm.weight",
    "encoder.layer_norm.bias",
)
_EXPECTED_LAYER_KEY_SUFFIXES = (
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.out_proj.weight",
    "self_attn.out_proj.bias",
    "self_attn.grep_linear.weight",
    "self_attn.grep_linear.bias",
    "self_attn.grep_a",
    "self_attn.relative_attention_bias.weight",
    "self_attn_layer_norm.weight",
    "self_attn_layer_norm.bias",
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "final_layer_norm.weight",
    "final_layer_norm.bias",
)


@pytest.fixture(scope="module")
def frame_encoder() -> BeatsEncoder:
    torch.manual_seed(0)
    return BeatsEncoder(
        granularity="frame",
        checkpoint="beats_iter1",
        pretrained=False,
    ).eval()


@pytest.fixture(scope="module")
def clip_encoder(frame_encoder) -> BeatsEncoder:
    encoder = BeatsEncoder(
        granularity="clip",
        checkpoint="beats_iter1",
        pretrained=False,
    ).eval()
    encoder.load_state_dict(frame_encoder.state_dict())
    return encoder


def _make_inputs(
    valid_feature_frames: list[int],
    seed: int = 3,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    max_frames = max(valid_feature_frames)
    features = torch.zeros(len(valid_feature_frames), max_frames, 128)
    for index, frames in enumerate(valid_feature_frames):
        features[index, :frames] = torch.randn(
            (frames, 128),
            generator=generator,
        )
    return {
        "input_features": features,
        "valid_feature_frames": torch.tensor(
            valid_feature_frames,
            dtype=torch.int64,
        ),
        "valid_seconds": torch.tensor(
            [frames / 100 for frames in valid_feature_frames],
            dtype=torch.float32,
        ),
    }


def test_invalid_construction():
    with pytest.raises(ValueError, match="Unknown BEATs entry"):
        BeatsEncoder(
            granularity="clip",
            checkpoint="beats_iter9",
            pretrained=False,
        )
    with pytest.raises(ValueError, match="granularity"):
        BeatsEncoder(
            granularity="chunk",
            checkpoint="beats_iter1",
            pretrained=False,
        )
    with pytest.raises(TypeError, match="bool"):
        BeatsEncoder(
            granularity="clip",
            checkpoint="beats_iter1",
            pretrained=1,
        )


def test_pretrained_false_does_not_resolve(monkeypatch):
    monkeypatch.setattr(
        beats_encoder_module,
        "ensure_beats_checkpoint",
        lambda *args, **kwargs: pytest.fail(
            "pretrained=False should not resolve the checkpoint"
        ),
    )
    encoder = BeatsEncoder(
        granularity="clip",
        checkpoint="fine_tuned_beats_iter1_cpt1",
        pretrained=False,
    )
    assert encoder.checkpoint == "fine_tuned_beats_iter1_cpt1"
    assert encoder.pretrained is False
    assert encoder.embedding_dim == 768


def test_state_dict_matches_official_keys(frame_encoder):
    # The two weight_norm keys of pos_conv are named per the parametrize
    # convention, corresponding to the official checkpoint's weight_g/weight_v,
    # mapped by the helpers at load time.
    expected = set(_EXPECTED_TOP_KEYS)
    for layer_index in range(12):
        for suffix in _EXPECTED_LAYER_KEY_SUFFIXES:
            expected.add(f"encoder.layers.{layer_index}.{suffix}")
    assert len(expected) == 250
    assert set(frame_encoder.state_dict().keys()) == expected


def test_relative_attention_bias_shared_across_layers(frame_encoder):
    shared = (
        frame_encoder.encoder.layers[0]
        .self_attn.relative_attention_bias.weight
    )
    for layer in frame_encoder.encoder.layers:
        assert layer.self_attn.relative_attention_bias.weight is shared


def test_frame_output_contract(frame_encoder):
    inputs = _make_inputs([16, 98, 98])
    output = frame_encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_feature_frames=inputs["valid_feature_frames"],
    )

    assert output["embedding"].shape == (3, 6, 768)
    assert output["geometry"].shape == (3, 6, 2)
    assert output["geometry"].dtype == torch.float32
    assert output["valid_mask"].dtype == torch.bool
    assert output["valid_mask"].tolist() == [
        [True, False, False, False, False, False],
        [True] * 6,
        [True] * 6,
    ]
    # embedding and geometry at invalid positions are exactly 0
    assert torch.all(output["embedding"][0, 1:] == 0)
    assert torch.all(output["geometry"][0, 1:] == 0)
    # adjacent valid boundaries are exactly equal; the last frame's end is valid_seconds
    geometry = output["geometry"][1]
    assert torch.equal(geometry[:-1, 1], geometry[1:, 0])
    assert geometry[0, 0].item() == 0.0
    assert geometry[-1, 1].item() == pytest.approx(0.98)
    assert output["geometry"][0, 0, 1].item() == pytest.approx(0.16)


def test_clip_output_contract(clip_encoder):
    inputs = _make_inputs([98, 98])
    output = clip_encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_feature_frames=inputs["valid_feature_frames"],
    )

    assert output["embedding"].shape == (2, 768)
    assert torch.all(output["valid_mask"])
    torch.testing.assert_close(
        output["geometry"],
        torch.stack(
            (
                torch.zeros(2),
                inputs["valid_seconds"],
            ),
            dim=1,
        ),
    )


def test_clip_matches_frame_time_mean(frame_encoder, clip_encoder):
    inputs = _make_inputs([98])
    frame_output = frame_encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_feature_frames=inputs["valid_feature_frames"],
    )
    clip_output = clip_encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_feature_frames=inputs["valid_feature_frames"],
    )

    torch.testing.assert_close(
        clip_output["embedding"][0],
        frame_output["embedding"][0].mean(dim=0),
        atol=1e-5,
        rtol=1e-5,
    )


def test_token_flatten_order_time_outer_freq_inner():
    torch.manual_seed(5)
    encoder = BeatsEncoder(
        granularity="frame",
        checkpoint="beats_iter1",
        pretrained=False,
    ).eval()
    # After replacing the transformer with identity, the patch conv's
    # kernel=stride=16 makes each frame depend only on its own 16-frame time
    # block; if the flatten order were not time-outer/frequency-inner,
    # _encode_frame's view-based grouping would mix tokens from different
    # time blocks into the same frame, and the following per-block locality
    # assertions would fail.
    encoder.encoder = torch.nn.Identity()

    generator = torch.Generator().manual_seed(23)
    features = torch.randn((1, 48, 128), generator=generator)
    full_output = encoder(
        features,
        valid_seconds=torch.tensor([0.48]),
        valid_feature_frames=torch.tensor([48], dtype=torch.int64),
    )

    for block in range(3):
        block_output = encoder(
            features[:, block * 16 : (block + 1) * 16],
            valid_seconds=torch.tensor([0.16]),
            valid_feature_frames=torch.tensor([16], dtype=torch.int64),
        )
        torch.testing.assert_close(
            full_output["embedding"][:, block],
            block_output["embedding"][:, 0],
        )

    # Discriminative input: only time block 1 is perturbed, other frames are unchanged elementwise
    perturbed = features.clone()
    perturbed[:, 16:32] += 1.0
    perturbed_output = encoder(
        perturbed,
        valid_seconds=torch.tensor([0.48]),
        valid_feature_frames=torch.tensor([48], dtype=torch.int64),
    )
    assert torch.equal(
        perturbed_output["embedding"][:, [0, 2]],
        full_output["embedding"][:, [0, 2]],
    )
    assert not torch.allclose(
        perturbed_output["embedding"][:, 1],
        full_output["embedding"][:, 1],
    )


def test_frame_and_clip_match_manual_reference(
    frame_encoder,
    clip_encoder,
):
    generator = torch.Generator().manual_seed(29)
    features = torch.randn((1, 98, 128), generator=generator)
    valid_seconds = torch.tensor([0.98])
    valid_feature_frames = torch.tensor([98], dtype=torch.int64)

    frame_output = frame_encoder(
        features,
        valid_seconds=valid_seconds,
        valid_feature_frames=valid_feature_frames,
    )
    clip_output = clip_encoder(
        features,
        valid_seconds=valid_seconds,
        valid_feature_frames=valid_feature_frames,
    )

    # Manual reference: explicitly stack tokens time-outer/frequency-inner, then average over frequency per block slice
    with torch.no_grad():
        conv_out = frame_encoder.patch_embedding(features.unsqueeze(1))
        manual_tokens = torch.stack(
            [
                conv_out[:, :, block, patch]
                for block in range(6)
                for patch in range(8)
            ],
            dim=1,
        )
        manual_tokens = frame_encoder.post_extract_proj(
            frame_encoder.layer_norm(manual_tokens)
        )
        manual_tokens = frame_encoder.encoder(manual_tokens)
        manual_frame = torch.stack(
            [
                manual_tokens[:, block * 8 : (block + 1) * 8].mean(dim=1)
                for block in range(6)
            ],
            dim=1,
        )
        manual_clip = manual_tokens.mean(dim=1)

    torch.testing.assert_close(frame_output["embedding"], manual_frame)
    torch.testing.assert_close(clip_output["embedding"], manual_clip)


def test_mixed_batch_matches_single_calls(frame_encoder):
    inputs = _make_inputs([16, 50, 98])
    output_batch = frame_encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_feature_frames=inputs["valid_feature_frames"],
    )

    for index in range(3):
        frames = int(inputs["valid_feature_frames"][index])
        output_single = frame_encoder(
            inputs["input_features"][index : index + 1, :frames],
            valid_seconds=inputs["valid_seconds"][index : index + 1],
            valid_feature_frames=inputs["valid_feature_frames"][
                index : index + 1
            ],
        )
        num_valid = frames // 16
        torch.testing.assert_close(
            output_batch["embedding"][index, :num_valid],
            output_single["embedding"][0, :num_valid],
        )


def test_unknown_forward_input_raises(frame_encoder):
    inputs = _make_inputs([16])
    with pytest.raises(TypeError):
        frame_encoder(
            inputs["input_features"],
            valid_seconds=inputs["valid_seconds"],
            valid_feature_frames=inputs["valid_feature_frames"],
            unknown_input=torch.zeros(1),
        )
    with pytest.raises(TypeError):
        frame_encoder(
            inputs["input_features"],
            valid_seconds=inputs["valid_seconds"],
        )


def test_checkpoint_table_complete():
    assert len(BEATS_CHECKPOINTS) == 15
    finetuned_flags = [
        metadata.finetuned for metadata in BEATS_CHECKPOINTS.values()
    ]
    assert sum(finetuned_flags) == 10
    for entry, metadata in BEATS_CHECKPOINTS.items():
        assert metadata.filename == f"{entry}.pt"
        assert len(metadata.sha256) == 64
        assert metadata.finetuned == entry.startswith("fine_tuned_")


def test_expected_cfg_tables():
    assert len(BEATS_EXPECTED_CFG_PRETRAINED) == 22
    assert len(BEATS_EXPECTED_CFG_FINETUNED) == 25
    for table in (
        BEATS_EXPECTED_CFG_PRETRAINED,
        BEATS_EXPECTED_CFG_FINETUNED,
    ):
        assert table["input_patch_size"] == 16
        assert table["encoder_embed_dim"] == 768
        assert table["max_distance"] == 800
        assert table["gru_rel_pos"] is True
        assert table["deep_norm"] is True
    assert BEATS_EXPECTED_CFG_PRETRAINED["dropout"] == 0.1
    assert BEATS_EXPECTED_CFG_FINETUNED["dropout"] == 0.0
    assert BEATS_EXPECTED_CFG_FINETUNED["predictor_class"] == 527


def test_ensure_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        ensure_beats_checkpoint("beats_iter1", tmp_path)

    message = str(exc_info.value)
    assert "beats_dl.py" in message
    assert str(tmp_path) in message
    assert "--entries beats_iter1" in message


def test_ensure_sha_mismatch(tmp_path):
    (tmp_path / "beats_iter1.pt").write_bytes(b"bogus")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ensure_beats_checkpoint("beats_iter1", tmp_path)


def test_ensure_unknown_entry(tmp_path):
    with pytest.raises(ValueError, match="Unknown BEATs entry"):
        ensure_beats_checkpoint("beats_iter9", tmp_path)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_device_transfer(frame_encoder):
    encoder = BeatsEncoder(
        granularity="frame",
        checkpoint="beats_iter1",
        pretrained=False,
    ).to("cuda").eval()
    inputs = _make_inputs([16, 98])
    output = encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_feature_frames=inputs["valid_feature_frames"],
    )

    assert encoder.device.type == "cuda"
    assert output["embedding"].device.type == "cuda"
    assert output["geometry"].device.type == "cuda"
    assert output["valid_mask"].device.type == "cuda"
