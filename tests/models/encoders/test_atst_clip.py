"""Unit tests for ``AtstClipEncoder`` without any pretrained weights."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from timbral.models.encoders import AtstClipEncoder
from timbral.models.encoders import atst as atst_encoder_module
from timbral.models.encoders.base import BaseEncoder
from timbral.models.helpers.atst import (
    ATST_CHUNK_FRAMES,
    ATST_EMBED_DIMS,
    ATST_NUM_BLOCKS,
    ATST_PATCH_WIDTH,
    ATST_POSITION_SLOTS,
)

_NUM_MELS = 64
_EXPECTED_TOP_KEYS = (
    "mask_embed",
    "cls_token",
    "pos_embed",
    "patch_embed.patch_embed.weight",
    "patch_embed.patch_embed.bias",
    "norm.weight",
    "norm.bias",
)
# qkv is built with bias=False, so no ``attn.qkv.bias`` key exists; the
# official checkpoints carry none either.
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
def small_encoder() -> AtstClipEncoder:
    torch.manual_seed(0)
    return AtstClipEncoder(
        granularity="clip",
        arch="small",
        pretrained=False,
    ).eval()


@pytest.fixture(scope="module")
def base_encoder() -> AtstClipEncoder:
    torch.manual_seed(0)
    return AtstClipEncoder(
        granularity="clip",
        arch="base",
        pretrained=False,
    ).eval()


def _make_inputs(
    valid_feature_frames: list[int],
    seed: int = 3,
) -> dict[str, torch.Tensor]:
    """Build a zero-padded mel canvas plus its length tensors."""
    generator = torch.Generator().manual_seed(seed)
    max_frames = max(valid_feature_frames)
    features = torch.zeros(len(valid_feature_frames), max_frames, _NUM_MELS)
    for index, frames in enumerate(valid_feature_frames):
        features[index, :frames] = torch.randn(
            (frames, _NUM_MELS),
            generator=generator,
        )
    return {
        "input_features": features,
        "valid_feature_frames": torch.tensor(
            valid_feature_frames,
            dtype=torch.int64,
        ),
        # The transform emits ``valid_samples // 160 + 1`` frames at
        # 16 kHz, so F frames stand for ``(F - 1) * 10 ms`` of audio.
        "valid_seconds": torch.tensor(
            [(frames - 1) / 100 for frames in valid_feature_frames],
            dtype=torch.float32,
        ),
    }


def _forward(
    encoder: AtstClipEncoder,
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run one forward pass over a full ``_make_inputs`` batch."""
    return encoder(
        inputs["input_features"],
        valid_seconds=inputs["valid_seconds"],
        valid_feature_frames=inputs["valid_feature_frames"],
    )


def test_public_export_and_keyword_only_constructor():
    from timbral.models.encoders.atst import (
        AtstClipEncoder as DirectEncoder,
    )

    assert AtstClipEncoder is DirectEncoder
    assert issubclass(AtstClipEncoder, BaseEncoder)
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in inspect.signature(
            AtstClipEncoder
        ).parameters.values()
    )


def test_only_clip_granularity_is_supported():
    assert AtstClipEncoder.supported_granularities == frozenset(("clip",))
    with pytest.raises(ValueError, match="does not support 'frame'"):
        AtstClipEncoder(
            granularity="frame",
            arch="small",
            pretrained=False,
        )
    with pytest.raises(ValueError, match="granularity"):
        AtstClipEncoder(
            granularity="chunk",
            arch="small",
            pretrained=False,
        )


def test_invalid_construction():
    with pytest.raises(ValueError, match="Unknown ATST arch"):
        AtstClipEncoder(
            granularity="clip",
            arch="tiny",
            pretrained=False,
        )
    for n_blocks in (0, ATST_NUM_BLOCKS + 1):
        with pytest.raises(ValueError, match="n_blocks must be within"):
            AtstClipEncoder(
                granularity="clip",
                arch="small",
                n_blocks=n_blocks,
                pretrained=False,
            )
    # bool is a subclass of int, so an exact type check is required.
    for n_blocks in (1.5, "1", True):
        with pytest.raises(TypeError, match="n_blocks must be an int"):
            AtstClipEncoder(
                granularity="clip",
                arch="small",
                n_blocks=n_blocks,
                pretrained=False,
            )
    with pytest.raises(TypeError, match="bool"):
        AtstClipEncoder(
            granularity="clip",
            arch="small",
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
    encoder = AtstClipEncoder(
        granularity="clip",
        arch="small",
        n_blocks=2,
        pretrained=False,
    )

    assert encoder.arch == "small"
    assert encoder.n_blocks == 2
    assert encoder.pretrained is False
    assert encoder.pretrained_dir is None
    assert encoder.embedding_dim == 2 * 2 * 384
    # The constructor must not touch the default training lifecycle.
    assert encoder.training

    # A directory is only stored, never read, while pretrained is False.
    explicit = AtstClipEncoder(
        granularity="clip",
        arch="small",
        pretrained=False,
        pretrained_dir="/nonexistent",
    )

    assert explicit.pretrained_dir == Path("/nonexistent")


def test_state_dict_matches_official_keys(small_encoder):
    state_dict = small_encoder.state_dict()
    expected = set(_EXPECTED_TOP_KEYS)
    for block_index in range(ATST_NUM_BLOCKS):
        for suffix in _EXPECTED_BLOCK_KEY_SUFFIXES:
            expected.add(f"blocks.{block_index}.{suffix}")

    assert len(expected) == 139
    assert set(state_dict.keys()) == expected


def test_qkv_has_no_bias(small_encoder):
    state_dict = small_encoder.state_dict()
    for block_index in range(ATST_NUM_BLOCKS):
        assert f"blocks.{block_index}.attn.qkv.bias" not in state_dict
    for block in small_encoder.blocks:
        assert block.attn.qkv.bias is None


@pytest.mark.parametrize("arch", ["small", "base"])
def test_positional_and_patch_shapes(arch, small_encoder, base_encoder):
    encoder = small_encoder if arch == "small" else base_encoder
    embed_dim = ATST_EMBED_DIMS[arch]

    # 250 patch slots plus the cls slot, as in the official checkpoints.
    assert ATST_POSITION_SLOTS == 251
    assert encoder.pos_embed.shape == (1, 251, embed_dim)
    assert encoder.cls_token.shape == (1, 1, embed_dim)
    assert encoder.mask_embed.shape == (1, 1, embed_dim)
    # One patch is 64 mel bins by 4 frames, flattened to 256 features.
    assert encoder.patch_embed.patch_embed.weight.shape == (embed_dim, 256)
    assert encoder.patch_embed.patch_embed.bias.shape == (embed_dim,)
    assert len(encoder.blocks) == ATST_NUM_BLOCKS


@pytest.mark.parametrize(
    ("arch", "n_blocks", "expected"),
    [
        ("small", 1, 768),
        ("small", 3, 2304),
        ("small", 12, 9216),
        ("base", 1, 1536),
        ("base", 12, 18432),
    ],
)
def test_embedding_dim_is_twice_blocks_times_width(arch, n_blocks, expected):
    encoder = AtstClipEncoder(
        granularity="clip",
        arch=arch,
        n_blocks=n_blocks,
        pretrained=False,
    )

    assert expected == 2 * n_blocks * ATST_EMBED_DIMS[arch]
    assert encoder.embedding_dim == expected


def test_embedding_dim_is_an_instance_attribute(small_encoder):
    # The width depends on n_blocks and arch, so it cannot be a ClassVar;
    # callers must read it off the instance.
    assert "embedding_dim" in vars(small_encoder)
    assert "embedding_dim" not in vars(AtstClipEncoder)
    assert "embedding_dim" not in vars(BaseEncoder)
    assert not hasattr(AtstClipEncoder, "embedding_dim")


def test_clip_output_contract(small_encoder):
    inputs = _make_inputs([4, 60, 100])
    output = _forward(small_encoder, inputs)

    assert set(output) == {"embedding", "geometry", "valid_mask"}
    assert output["embedding"].shape == (3, 768)
    assert output["embedding"].dtype is torch.float32
    assert output["geometry"].shape == (3, 2)
    assert output["geometry"].dtype is torch.float32
    assert output["valid_mask"].shape == (3,)
    assert output["valid_mask"].dtype is torch.bool
    assert torch.all(output["valid_mask"])
    assert torch.equal(
        output["geometry"],
        torch.stack(
            (torch.zeros(3), inputs["valid_seconds"]),
            dim=1,
        ),
    )
    # The embedding must be a live function of the input: finite, not a
    # zero canvas, and different for samples carrying different audio.
    assert torch.isfinite(output["embedding"]).all()
    assert torch.all(output["embedding"].abs().sum(dim=1) > 0)
    for first, second in ((0, 1), (0, 2), (1, 2)):
        assert not torch.equal(
            output["embedding"][first],
            output["embedding"][second],
        )


def test_clip_pooling_concatenates_cls_and_patch_mean(small_encoder):
    # 100 frames make 25 patches, i.e. a single chunk, so the pooled
    # vector must be readable straight off the last block: the official
    # extraction norms that block, takes the cls token, and averages the
    # patch tokens. The implementation divides by ``P + 1e-6`` instead of
    # ``P``, a relative offset of 4e-8 that stays far below the float32
    # tolerance used here.
    inputs = _make_inputs([100], seed=5)
    embed_dim = ATST_EMBED_DIMS["small"]
    captured: list[torch.Tensor] = []
    handle = small_encoder.blocks[-1].register_forward_hook(
        lambda module, args, output: captured.append(output)
    )
    try:
        with torch.no_grad():
            output = _forward(small_encoder, inputs)
            normed = small_encoder.norm(captured[0])
    finally:
        handle.remove()

    assert len(captured) == 1
    assert normed.shape == (1, 26, embed_dim)
    torch.testing.assert_close(
        output["embedding"][:, :embed_dim],
        normed[:, 0],
    )
    torch.testing.assert_close(
        output["embedding"][:, embed_dim:],
        normed[:, 1:].mean(dim=1),
    )


def test_n_blocks_concatenates_the_trailing_blocks(small_encoder):
    # Weight sharing with the n_blocks=1 fixture pins both the block
    # selection and the layout: the official extraction concatenates
    # every cls branch first and every patch-mean branch second, each in
    # block order, so the single-block output must reappear as the last
    # slice of either half.
    torch.manual_seed(0)
    deep_encoder = AtstClipEncoder(
        granularity="clip",
        arch="small",
        n_blocks=3,
        pretrained=False,
    ).eval()
    deep_encoder.load_state_dict(small_encoder.state_dict())
    inputs = _make_inputs([120], seed=9)
    embed_dim = ATST_EMBED_DIMS["small"]

    with torch.no_grad():
        shallow_output = _forward(small_encoder, inputs)
        deep_output = _forward(deep_encoder, inputs)

    assert deep_output["embedding"].shape == (1, 6 * embed_dim)
    torch.testing.assert_close(
        deep_output["embedding"][:, 2 * embed_dim : 3 * embed_dim],
        shallow_output["embedding"][:, :embed_dim],
    )
    torch.testing.assert_close(
        deep_output["embedding"][:, 5 * embed_dim :],
        shallow_output["embedding"][:, embed_dim:],
    )
    # The other slices carry the earlier blocks, not copies of the last.
    assert not torch.equal(
        deep_output["embedding"][:, :embed_dim],
        deep_output["embedding"][:, 2 * embed_dim : 3 * embed_dim],
    )
    assert not torch.equal(
        deep_output["embedding"][:, 2 * embed_dim : 3 * embed_dim],
        deep_output["embedding"][:, 5 * embed_dim :],
    )


def test_trailing_frames_below_one_patch_are_dropped(small_encoder):
    # The patch count is ``frames // 4``, so frames 4 to 6 of a 7-frame
    # clip never reach the backbone.
    generator = torch.Generator().manual_seed(23)
    features = torch.randn(
        (1, 7, _NUM_MELS), generator=generator).repeat(2, 1, 1)
    features[1, ATST_PATCH_WIDTH:] = torch.randn(
        (3, _NUM_MELS), generator=generator)

    with torch.no_grad():
        output = small_encoder(
            features,
            valid_seconds=torch.tensor([0.06, 0.06]),
            valid_feature_frames=torch.tensor([7, 7]),
        )
        # 4 frames are the transform's physical minimum, one patch; the
        # 7-frame clip must reduce to exactly that patch.
        shortest = small_encoder(
            features[:1, :ATST_PATCH_WIDTH],
            valid_seconds=torch.tensor([0.03]),
            valid_feature_frames=torch.tensor([ATST_PATCH_WIDTH]),
        )

    assert torch.equal(output["embedding"][0], output["embedding"][1])
    torch.testing.assert_close(
        shortest["embedding"][0],
        output["embedding"][0],
    )


def test_mixed_batch_matches_single_calls(small_encoder):
    # The three lengths land in three distinct length groups, one of
    # which is long enough to be chunked.
    inputs = _make_inputs([4, 640, 1200])
    output_batch = _forward(small_encoder, inputs)

    for index in range(3):
        frames = int(inputs["valid_feature_frames"][index])
        output_single = small_encoder(
            inputs["input_features"][index : index + 1, :frames],
            valid_seconds=inputs["valid_seconds"][index : index + 1],
            valid_feature_frames=inputs["valid_feature_frames"][
                index : index + 1
            ],
        )
        assert torch.equal(
            output_batch["embedding"][index],
            output_single["embedding"][0],
        )


def test_long_input_is_chunked_and_averaged(small_encoder):
    # 2500 frames exceed the 250 patch slots of pos_embed and are split
    # into 1000 + 1000 + 500 frames; the short trailing chunk carries the
    # same weight as the full ones. Each chunk is re-encoded through the
    # public API, which reuses the same positional slots.
    inputs = _make_inputs([2500], seed=11)
    features = inputs["input_features"]
    bounds = [(0, 1000), (1000, 2000), (2000, 2500)]

    assert ATST_CHUNK_FRAMES == 1000
    with torch.no_grad():
        output = _forward(small_encoder, inputs)
        chunk_embeddings = [
            small_encoder(
                features[:, start:end],
                valid_seconds=torch.tensor([(end - start - 1) / 100]),
                valid_feature_frames=torch.tensor([end - start]),
            )["embedding"]
            for start, end in bounds
        ]

    expected = torch.stack(chunk_embeddings, dim=0).mean(dim=0)
    assert output["embedding"].shape == (1, 768)
    assert torch.equal(output["embedding"], expected)
    # Averaging really mixes the chunks: no single chunk reproduces the
    # result, so dropping or reweighting one would change it.
    for chunk_embedding in chunk_embeddings:
        assert not torch.equal(output["embedding"], chunk_embedding)


def test_forward_requires_the_paired_transform_inputs(small_encoder):
    inputs = _make_inputs([8])
    with pytest.raises(TypeError):
        small_encoder(
            inputs["input_features"],
            valid_seconds=inputs["valid_seconds"],
        )
    with pytest.raises(TypeError):
        small_encoder(
            inputs["input_features"],
            valid_seconds=inputs["valid_seconds"],
            valid_feature_frames=inputs["valid_feature_frames"],
            unknown_input=torch.zeros(1),
        )


def test_device_follows_patch_embedding_weight(small_encoder):
    assert small_encoder.device == (
        small_encoder.patch_embed.patch_embed.weight.device
    )
    assert small_encoder.device.type == "cpu"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_device_transfer():
    torch.manual_seed(0)
    encoder = AtstClipEncoder(
        granularity="clip",
        arch="small",
        pretrained=False,
    ).to("cuda").eval()
    inputs = _make_inputs([4, 100])
    output = _forward(encoder, inputs)

    assert encoder.device.type == "cuda"
    assert output["embedding"].device.type == "cuda"
    assert output["geometry"].device.type == "cuda"
    assert output["valid_mask"].device.type == "cuda"
