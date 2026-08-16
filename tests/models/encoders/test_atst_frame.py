"""Unit tests for ``AtstFrameEncoder`` without loading any weights."""

from __future__ import annotations

import pytest
import torch

from timbral.models.encoders import AtstFrameEncoder
from timbral.models.encoders import atst as atst_encoder_module
from timbral.models.helpers.atst import (
    ATST_CHUNK_FRAMES,
    ATST_EMBED_DIMS,
    ATST_FRAME_STEP_SECONDS,
    ATST_NUM_BLOCKS,
    ATST_NUM_MELS,
    ATST_PATCH_WIDTH,
    ATST_POSITION_SLOTS,
)

_ARCH = "small"
_EMBED_DIM = ATST_EMBED_DIMS[_ARCH]
_N_BLOCKS = 2
_OUTPUT_DIM = _N_BLOCKS * _EMBED_DIM

# ATST-Frame carries no cls token, and its final norm is named
# norm_frame; mask_embed is pretraining-only and kept for state_dict
# parity with the official checkpoint.
_EXPECTED_TOP_KEYS = (
    "mask_embed",
    "pos_embed",
    "patch_embed.patch_embed.weight",
    "patch_embed.patch_embed.bias",
    "norm_frame.weight",
    "norm_frame.bias",
)
# The official blocks run qkv without a bias term.
_EXPECTED_BLOCK_KEY_SUFFIXES = (
    "norm1.weight",
    "norm1.bias",
    "attn.qkv.weight",
    "attn.proj.weight",
    "attn.proj.bias",
    "norm2.weight",
    "norm2.bias",
    "mlp.fc1.weight",
    "mlp.fc1.bias",
    "mlp.fc2.weight",
    "mlp.fc2.bias",
)


@pytest.fixture(scope="module")
def frame_encoder() -> AtstFrameEncoder:
    torch.manual_seed(0)
    return AtstFrameEncoder(
        granularity="frame",
        arch=_ARCH,
        n_blocks=_N_BLOCKS,
        pretrained=False,
    ).eval()


@pytest.fixture(scope="module")
def clip_encoder(frame_encoder) -> AtstFrameEncoder:
    encoder = AtstFrameEncoder(
        granularity="clip",
        arch=_ARCH,
        n_blocks=_N_BLOCKS,
        pretrained=False,
    ).eval()
    encoder.load_state_dict(frame_encoder.state_dict())
    return encoder


def _clone_encoder(
    frame_encoder: AtstFrameEncoder,
    *,
    n_blocks: int = _N_BLOCKS,
) -> AtstFrameEncoder:
    """Build a frame encoder sharing ``frame_encoder``'s weights.

    ``n_blocks`` only selects which block outputs are collected, so it
    leaves the parameter set untouched and ``strict=True`` still holds.
    """
    encoder = AtstFrameEncoder(
        granularity="frame",
        arch=_ARCH,
        n_blocks=n_blocks,
        pretrained=False,
    ).eval()
    encoder.load_state_dict(frame_encoder.state_dict())
    return encoder


def _make_inputs(
    valid_feature_frames: list[int],
    seed: int = 3,
) -> dict[str, torch.Tensor]:
    """Build a zero-padded ``[B, T, 64]`` canvas plus its length tensors.

    The transform emits ``valid_samples // 160 + 1`` mel frames, so a
    valid frame count of ``F`` corresponds to ``(F - 1) * 10 ms`` of
    valid audio.
    """
    generator = torch.Generator().manual_seed(seed)
    max_frames = max(valid_feature_frames)
    features = torch.zeros(
        len(valid_feature_frames), max_frames, ATST_NUM_MELS)
    for index, frames in enumerate(valid_feature_frames):
        features[index, :frames] = torch.randn(
            (frames, ATST_NUM_MELS),
            generator=generator,
        )
    return {
        "input_features": features,
        "valid_feature_frames": torch.tensor(
            valid_feature_frames,
            dtype=torch.int64,
        ),
        "valid_seconds": torch.tensor(
            [(frames - 1) / 100 for frames in valid_feature_frames],
            dtype=torch.float32,
        ),
    }


def _forward(
    encoder: AtstFrameEncoder,
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run one forward pass over a full ``_make_inputs`` batch."""
    return encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_feature_frames=inputs["valid_feature_frames"],
    )


def test_state_dict_matches_official_keys(frame_encoder):
    expected = set(_EXPECTED_TOP_KEYS)
    for block_index in range(ATST_NUM_BLOCKS):
        for suffix in _EXPECTED_BLOCK_KEY_SUFFIXES:
            expected.add(f"blocks.{block_index}.{suffix}")
    assert len(expected) == 138
    assert set(frame_encoder.state_dict().keys()) == expected


def test_state_dict_omits_clip_only_parameters(frame_encoder):
    keys = set(frame_encoder.state_dict().keys())
    # This family pools over patch tokens only, so there is neither a
    # cls token nor the clip family's final norm.
    assert "cls_token" not in keys
    assert not any(key.startswith("norm.") for key in keys)
    assert {"norm_frame.weight", "norm_frame.bias"} <= keys
    # The official qkv projection is bias-free.
    assert not any(key.endswith("attn.qkv.bias") for key in keys)


def test_parameter_shapes(frame_encoder):
    state = frame_encoder.state_dict()
    # 250 patch slots plus the slot 0 the official FrameAST leaves
    # unused, as in the four official checkpoints.
    assert ATST_POSITION_SLOTS == 251
    assert state["pos_embed"].shape == (1, 251, _EMBED_DIM)
    assert state["mask_embed"].shape == (1, 1, _EMBED_DIM)
    assert state["patch_embed.patch_embed.weight"].shape == (
        _EMBED_DIM,
        ATST_NUM_MELS * ATST_PATCH_WIDTH,
    )
    assert state["blocks.0.attn.qkv.weight"].shape == (
        3 * _EMBED_DIM, _EMBED_DIM)


@pytest.mark.parametrize(
    ("arch", "n_blocks"),
    [("small", 1), ("small", 12), ("base", 3)],
)
def test_embedding_dim_has_no_cls_branch(arch, n_blocks):
    encoder = AtstFrameEncoder(
        granularity="frame",
        arch=arch,
        n_blocks=n_blocks,
        pretrained=False,
    )
    # Only the block outputs are concatenated: no factor of two.
    assert encoder.embedding_dim == n_blocks * ATST_EMBED_DIMS[arch]
    # The width depends on the constructor, so it lives on the instance.
    assert "embedding_dim" in vars(encoder)
    assert "embedding_dim" not in vars(AtstFrameEncoder)


def test_supported_granularities():
    assert AtstFrameEncoder.supported_granularities == frozenset(
        ("clip", "frame"))
    for granularity in ("clip", "frame"):
        encoder = AtstFrameEncoder(
            granularity=granularity,
            arch=_ARCH,
            pretrained=False,
        )
        assert encoder.granularity == granularity
        assert encoder.embedding_dim == _EMBED_DIM


def test_invalid_construction():
    with pytest.raises(ValueError, match="Unknown ATST arch"):
        AtstFrameEncoder(
            granularity="frame",
            arch="tiny",
            pretrained=False,
        )
    with pytest.raises(ValueError, match="granularity"):
        AtstFrameEncoder(
            granularity="chunk",
            arch=_ARCH,
            pretrained=False,
        )
    with pytest.raises(TypeError, match="n_blocks must be an int"):
        AtstFrameEncoder(
            granularity="frame",
            arch=_ARCH,
            n_blocks=True,
            pretrained=False,
        )
    for n_blocks in (0, ATST_NUM_BLOCKS + 1):
        with pytest.raises(ValueError, match="n_blocks must be within"):
            AtstFrameEncoder(
                granularity="frame",
                arch=_ARCH,
                n_blocks=n_blocks,
                pretrained=False,
            )
    with pytest.raises(TypeError, match="pretrained must be a bool"):
        AtstFrameEncoder(
            granularity="frame",
            arch=_ARCH,
            pretrained=1,
        )


def test_pretrained_false_does_not_resolve(monkeypatch):
    monkeypatch.setattr(
        atst_encoder_module,
        "ensure_atst_checkpoint",
        lambda *args, **kwargs: pytest.fail(
            "pretrained=False should not resolve the checkpoint"
        ),
    )
    encoder = AtstFrameEncoder(
        granularity="frame",
        arch=_ARCH,
        pretrained=False,
        pretrained_dir="/nonexistent",
    )
    assert encoder.arch == _ARCH
    assert encoder.pretrained is False
    assert encoder.pretrained_dir.name == "nonexistent"


def test_patch_embedding_uses_the_official_element_order(frame_encoder):
    """One patch flattens mel bin major, frame minor.

    The official ``PatchEmbed_v2`` rearranges ``b c (h p1) (w p2) ->
    b (w h) (p1 p2 c)`` over a mel-major ``[B, 64, T]`` spectrogram with
    ``c = 1``, so patch ``w`` is the column block ``4w .. 4w + 3`` read
    out mel bin by mel bin. That layout is derived here from the
    mel-major view rather than from the encoder's own reshape chain.
    """
    generator = torch.Generator().manual_seed(23)
    num_frames = 12
    features = torch.randn(
        (2, num_frames, ATST_NUM_MELS), generator=generator)
    spectrogram = features.transpose(1, 2)
    patches = torch.stack(
        [
            spectrogram[
                :, :, start : start + ATST_PATCH_WIDTH
            ].reshape(2, -1)
            for start in range(0, num_frames, ATST_PATCH_WIDTH)
        ],
        dim=1,
    )

    assert patches.shape == (
        2, num_frames // ATST_PATCH_WIDTH,
        ATST_NUM_MELS * ATST_PATCH_WIDTH,
    )
    torch.testing.assert_close(
        frame_encoder.patch_embed(features),
        frame_encoder.patch_embed.patch_embed(patches),
    )


def test_position_slot_zero_is_unused(frame_encoder):
    """Slot 0 belongs to a cls token this family does not have.

    Zeroing an unused slot cannot move the output, while zeroing a slot
    the patches actually consume must; comparing the two pins the
    ``pos_embed[:, 1 : P + 1]`` slice rather than restating it.
    """
    encoder = _clone_encoder(frame_encoder)
    inputs = _make_inputs([16], seed=5)
    reference = _forward(encoder, inputs)["embedding"].clone()

    with torch.no_grad():
        encoder.pos_embed[:, 0] = 0.0
    assert torch.equal(_forward(encoder, inputs)["embedding"], reference)

    # 16 mel frames are 4 patches, which consume slots 1 through 4.
    encoder.load_state_dict(frame_encoder.state_dict())
    with torch.no_grad():
        encoder.pos_embed[:, 4] = 0.0
    assert not torch.equal(
        _forward(encoder, inputs)["embedding"], reference)


def test_n_blocks_selects_the_trailing_blocks(frame_encoder):
    """``n_blocks`` takes the last k blocks, shallowest slice first."""
    narrow_encoder = _clone_encoder(frame_encoder, n_blocks=1)
    inputs = _make_inputs([8, 27, 40])
    wide = _forward(frame_encoder, inputs)["embedding"]
    narrow = _forward(narrow_encoder, inputs)["embedding"]

    assert narrow.shape[-1] == _EMBED_DIM
    assert wide.shape[-1] == 2 * _EMBED_DIM
    # n_blocks=1 is the deepest block alone, so it must coincide with the
    # trailing slice of the n_blocks=2 output, never the leading one.
    assert torch.equal(narrow, wide[..., -_EMBED_DIM:])
    assert not torch.equal(narrow, wide[..., :_EMBED_DIM])


def test_block_outputs_pass_through_the_frame_norm(frame_encoder):
    """Every block slice leaves norm_frame standardized.

    The encoder is untrained here and ``_init_module`` leaves each
    LayerNorm at weight 1 and bias 0, so norm_frame reduces to plain
    standardization over the width and the property is checkable
    without any weights.
    """
    inputs = _make_inputs([40], seed=17)
    embedding = _forward(frame_encoder, inputs)["embedding"][0]

    for block_index in range(_N_BLOCKS):
        block_slice = embedding[
            :,
            block_index * _EMBED_DIM : (block_index + 1) * _EMBED_DIM,
        ]
        torch.testing.assert_close(
            block_slice.mean(dim=-1),
            torch.zeros(block_slice.shape[0]),
            atol=1e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            block_slice.std(dim=-1, unbiased=False),
            torch.ones(block_slice.shape[0]),
            atol=1e-5,
            rtol=0,
        )


def test_frame_output_contract(frame_encoder):
    frame_counts = [8, 27, 40]
    inputs = _make_inputs(frame_counts)
    output = _forward(frame_encoder, inputs)

    assert set(output) == {"embedding", "geometry", "valid_mask"}
    # One 40 ms frame per patch, and T is the batch maximum.
    valid_patches = [count // ATST_PATCH_WIDTH for count in frame_counts]
    assert valid_patches == [2, 6, 10]
    total_frames = max(valid_patches)
    assert output["embedding"].shape == (3, total_frames, _OUTPUT_DIM)
    assert output["embedding"].dtype is torch.float32
    assert output["geometry"].shape == (3, total_frames, 2)
    assert output["geometry"].dtype is torch.float32
    assert output["valid_mask"].dtype is torch.bool
    assert output["valid_mask"].tolist() == [
        [index < patches for index in range(total_frames)]
        for patches in valid_patches
    ]

    for index, patches in enumerate(valid_patches):
        # Padding frames are exactly 0 in both tensors, never merely small.
        assert torch.all(output["embedding"][index, patches:] == 0)
        assert torch.all(output["geometry"][index, patches:] == 0)
        assert torch.any(output["embedding"][index, :patches] != 0)


def test_frame_geometry_semantics(frame_encoder):
    frame_counts = [8, 27, 40]
    inputs = _make_inputs(frame_counts)
    output = _forward(frame_encoder, inputs)
    geometry = output["geometry"]

    for index, frame_count in enumerate(frame_counts):
        patches = frame_count // ATST_PATCH_WIDTH
        seconds = inputs["valid_seconds"][index]
        sample_geometry = geometry[index, :patches]
        # Every start sits on the exact 40 ms grid.
        expected_starts = (
            torch.arange(patches, dtype=torch.float32)
            * ATST_FRAME_STEP_SECONDS
        )
        torch.testing.assert_close(
            sample_geometry[:, 0], expected_starts)
        # Interior ends are the next frame's start, so the intervals are
        # contiguous and non-overlapping.
        torch.testing.assert_close(
            sample_geometry[:-1, 1],
            expected_starts[1:],
        )
        torch.testing.assert_close(
            sample_geometry[:-1, 1],
            sample_geometry[1:, 0],
        )
        assert torch.all(sample_geometry[:, 1] > sample_geometry[:, 0])
        # The covered span is exactly [0, valid_seconds].
        assert sample_geometry[0, 0].item() == 0.0
        assert sample_geometry[-1, 1].item() == pytest.approx(
            seconds.item())


def test_clip_output_contract(clip_encoder):
    inputs = _make_inputs([8, 27, 40])
    output = _forward(clip_encoder, inputs)

    assert output["embedding"].shape == (3, _OUTPUT_DIM)
    assert output["valid_mask"].shape == (3,)
    assert output["valid_mask"].dtype == torch.bool
    assert torch.all(output["valid_mask"])
    torch.testing.assert_close(
        output["geometry"],
        torch.stack(
            (
                torch.zeros(3),
                inputs["valid_seconds"],
            ),
            dim=1,
        ),
    )


def test_clip_matches_frame_mean(frame_encoder, clip_encoder):
    # A single chunk gives every patch the same weight in both paths.
    inputs = _make_inputs([40], seed=7)
    frame_output = _forward(frame_encoder, inputs)
    clip_output = _forward(clip_encoder, inputs)

    # The official pooling divides by the patch count offset by 1e-6, so
    # the two paths agree only up to rounding, not bit for bit; the
    # default float32 tolerances are already far tighter than that gap.
    torch.testing.assert_close(
        clip_output["embedding"][0],
        frame_output["embedding"][0].mean(dim=0),
    )


def test_frame_batch_composition_independence(frame_encoder):
    frame_counts = [8, 27, 40]
    inputs = _make_inputs(frame_counts)
    output_batch = _forward(frame_encoder, inputs)

    for index, frame_count in enumerate(frame_counts):
        single = frame_encoder(
            inputs["input_features"][index : index + 1, :frame_count],
            valid_seconds=inputs["valid_seconds"][index : index + 1],
            valid_feature_frames=inputs["valid_feature_frames"][
                index : index + 1
            ],
        )
        patches = frame_count // ATST_PATCH_WIDTH
        assert torch.equal(
            output_batch["embedding"][index, :patches],
            single["embedding"][0, :patches],
        )
        assert torch.equal(
            output_batch["geometry"][index, :patches],
            single["geometry"][0, :patches],
        )


def test_clip_batch_composition_independence(clip_encoder):
    frame_counts = [8, 27, 40]
    inputs = _make_inputs(frame_counts)
    output_batch = _forward(clip_encoder, inputs)

    for index, frame_count in enumerate(frame_counts):
        single = clip_encoder(
            inputs["input_features"][index : index + 1, :frame_count],
            valid_seconds=inputs["valid_seconds"][index : index + 1],
            valid_feature_frames=inputs["valid_feature_frames"][
                index : index + 1
            ],
        )
        assert torch.equal(
            output_batch["embedding"][index],
            single["embedding"][0],
        )


def test_chunk_split_fills_the_position_slots(frame_encoder, monkeypatch):
    """The backbone must see 1000-frame chunks, i.e. 250 patch slots.

    Recording the frame counts the backbone is actually handed pins the
    split itself; comparing chunked output against manually chunked
    input cannot, because a wrong chunk length would re-split the manual
    chunks the same wrong way.
    """
    assert ATST_CHUNK_FRAMES // ATST_PATCH_WIDTH == ATST_POSITION_SLOTS - 1

    seen_frames = []
    original = AtstFrameEncoder._chunk_tokens

    def recording(self, features):
        seen_frames.append(features.shape[1])
        return original(self, features)

    monkeypatch.setattr(AtstFrameEncoder, "_chunk_tokens", recording)
    # 1004 frames are one full chunk plus a trailing chunk of exactly
    # one patch, which is short but still long enough to be kept.
    inputs = _make_inputs([ATST_CHUNK_FRAMES + ATST_PATCH_WIDTH], seed=11)
    output = _forward(frame_encoder, inputs)

    assert seen_frames == [ATST_CHUNK_FRAMES, ATST_PATCH_WIDTH]
    # The chunks tile the time axis, so no patch is lost or duplicated.
    assert output["embedding"].shape[1] == sum(
        frames // ATST_PATCH_WIDTH for frames in seen_frames)


def test_chunking_concatenates_chunk_token_sequences(frame_encoder):
    num_frames = 2500
    inputs = _make_inputs([num_frames], seed=11)
    output = _forward(frame_encoder, inputs)

    assert output["embedding"].shape == (
        1, num_frames // ATST_PATCH_WIDTH, _OUTPUT_DIM)
    assert output["embedding"].shape[1] == 625

    # Every chunk re-uses the same positional slots and attends only to
    # itself, so feeding the chunks in separately must reproduce the
    # concatenated sequence in the same order.
    bounds = [(0, 1000), (1000, 2000), (2000, 2500)]
    assert bounds[0][1] == ATST_CHUNK_FRAMES
    chunk_embeddings = []
    for start, end in bounds:
        chunk_frames = end - start
        chunk_output = frame_encoder(
            inputs["input_features"][:, start:end],
            valid_seconds=torch.tensor(
                [(chunk_frames - 1) / 100], dtype=torch.float32),
            valid_feature_frames=torch.tensor(
                [chunk_frames], dtype=torch.int64),
        )
        assert chunk_output["embedding"].shape[1] == (
            chunk_frames // ATST_PATCH_WIDTH)
        chunk_embeddings.append(chunk_output["embedding"])

    manual = torch.cat(chunk_embeddings, dim=1)
    assert manual.shape == output["embedding"].shape
    assert torch.equal(output["embedding"], manual)


def test_clip_averages_chunks_with_equal_weight(frame_encoder, clip_encoder):
    """A short trailing chunk still counts as one whole chunk."""
    inputs = _make_inputs([ATST_CHUNK_FRAMES + ATST_PATCH_WIDTH], seed=11)
    frame_tokens = _forward(frame_encoder, inputs)["embedding"][0]
    clip_embedding = _forward(clip_encoder, inputs)["embedding"][0]

    # 1004 frames are 250 patches plus a trailing chunk of one patch.
    chunk_patches = ATST_CHUNK_FRAMES // ATST_PATCH_WIDTH
    assert frame_tokens.shape[0] == chunk_patches + 1
    expected = torch.stack(
        (
            frame_tokens[:chunk_patches].mean(dim=0),
            frame_tokens[chunk_patches:].mean(dim=0),
        )
    ).mean(dim=0)

    torch.testing.assert_close(clip_embedding, expected)
    # A plain mean over all 251 patches would drown the trailing chunk,
    # so equal chunk weighting has to be observable.
    assert not torch.allclose(clip_embedding, frame_tokens.mean(dim=0))


def test_minimum_length_yields_one_frame(frame_encoder):
    # 513 samples is the transform's physical minimum: 4 mel frames,
    # i.e. exactly one patch.
    inputs = _make_inputs([4], seed=13)
    output = _forward(frame_encoder, inputs)

    assert output["embedding"].shape == (1, 1, _OUTPUT_DIM)
    assert output["valid_mask"].tolist() == [[True]]
    assert output["geometry"][0, 0, 0].item() == 0.0
    assert output["geometry"][0, 0, 1].item() == pytest.approx(0.03)


def test_unknown_forward_input_raises(frame_encoder):
    inputs = _make_inputs([8])
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_device_transfer():
    encoder = AtstFrameEncoder(
        granularity="frame",
        arch=_ARCH,
        n_blocks=_N_BLOCKS,
        pretrained=False,
    ).to("cuda").eval()
    inputs = _make_inputs([8, 40])
    output = _forward(encoder, inputs)

    assert encoder.device.type == "cuda"
    assert output["embedding"].device.type == "cuda"
    assert output["geometry"].device.type == "cuda"
    assert output["valid_mask"].device.type == "cuda"
