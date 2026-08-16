# `AtstFrameEncoder` Design

This document freezes the design of `timbral.models.encoders.AtstFrameEncoder`, the local rewrite of the
official `FrameAST` backbone of ATST-Frame (IEEE TASLP, arXiv:2306.04186). The companion Transform is
[`../transforms/atst.md`](../transforms/atst.md); the sibling family, which shares that Transform, the
checkpoint helpers, and the whole backbone body, is [`atst_clip.md`](atst_clip.md); the official alignment
contract is [`../extra/atst_frame-alignment.md`](../extra/atst_frame-alignment.md).

This document describes the behavior the current implementation must satisfy, and follows the common
contract in [`base.md`](base.md). Everything the two ATST families share is stated once in
[`atst_clip.md`](atst_clip.md) and only cross-referenced here; the detail below is spent on what
ATST-Frame does differently.

## Design goals

`AtstFrameEncoder` is responsible for:

- reproducing the official `FrameAST` backbone via an inference-only local rewrite that is
  operator-for-operator numerically equivalent, with `state_dict` keys matching the official checkpoint
  one-to-one (138 keys, `strict=True`);
- supporting both official architectures, `small` (D=384, 6 heads) and `base` (D=768, 12 heads);
- exposing the official frame-level output — one 40 ms frame per patch, `n_blocks * D` wide — as the
  family's primary granularity, and the official `scene=True` patch mean as its clip granularity;
- splitting inputs longer than the 250 positional slots into chunks on a grid that keeps every patch on
  the global 40 ms time base, concatenating chunks along time for frame granularity and averaging them
  with equal weight for clip granularity;
- grouping mixed-length batches by `valid_feature_frames`, keeping the output independent of batch
  composition;
- producing geometry, valid_mask, and zero padding conforming to `BaseEncoder`.

This component does not hold a Transform, does not accept waveforms, does not compute mel features, and
imposes no duration cap. Self-attention cost is bounded per chunk (at most 250 tokens), but the number of
chunks grows linearly with duration; resource constraints for long audio are the caller's responsibility.

## Supported checkpoints

The family ships exactly two official files, registered as `atst-frame-small` and `atst-frame-base` (see
[`../registry.md`](../registry.md)). Both are PyTorch-Lightning archives, so unlike ATST-Clip this family
never meets the DINO-style layout:

| entry | `arch` | local filename | bytes | archive layout |
|---|---|---|---:|---|
| `atst-frame-small` | `small` | `atst_frame_small.ckpt` | 411,264,805 | Lightning |
| `atst-frame-base` | `base` | `atst_frame_base.ckpt` | 1,459,523,813 | Lightning |

Identity and source, both hosted on Google Drive (the `confirm=t` direct-download form):

```text
atst_frame_small.ckpt
    sha256 1d85b290632dd26b8725f0ae73f53f990a898888cfc2c4794c3055a8130ff5f1
    https://drive.usercontent.google.com/download
        ?id=1xZoOTuxV415icYONYbeFQzgrmJQf4a4B&export=download&confirm=t
atst_frame_base.ckpt
    sha256 9f812544983add849f45ef03c4ce10184729adee082c0bdd347764f4835bb3da
    https://drive.usercontent.google.com/download
        ?id=1bGJSZWlAIIJ6GL5Id5dW0PTB72DL-QDQ&export=download&confirm=t
```

Only about 23% of the base file is the encoder (341.7 MB of 1460 MB); the rest is optimizer state and is
kept as-is, because the fixed SHA-256 is the file's identity.

### Architecture constants

The width, head count, depth, MLP ratio, `qkv_bias`, and LayerNorm eps are fixed by the official
`FrameAST_small` / `FrameAST_base` factories and are verified against the checkpoints' tensor shapes;
`patch_h`, `patch_w`, `nprompt`, and `n_mels` are recorded in both checkpoints' own `hyper_parameters`;
`avg_blocks`, `pos_type`, and `patch_embed` are `FrameAST` constructor defaults that no released
checkpoint overrides.

| Field | `small` | `base` |
|---|---:|---:|
| embed dim `D` | 384 | 768 |
| attention heads | 6 | 12 |
| head dim | 64 | 64 |
| Transformer blocks | 12 | 12 |
| MLP ratio | 4 | 4 |
| qkv bias | `False` | `False` |
| LayerNorm eps | 1e-6 | 1e-6 |
| patch (mel bins x frames) | 64 x 4 | 64 x 4 |
| `pos_embed` | `[1, 251, 384]` | `[1, 251, 768]` |
| `n_mels` | 64 | 64 |
| `nprompt` | 0 | 0 |
| `avg_blocks` | 0 | 0 |
| `pos_type` | `"cut"` | `"cut"` |
| `patch_embed` | `"Linear"` | `"Linear"` |

The last four rows are why the local rewrite can be complete rather than partial: prompt tokens, the
data2vec block-averaging teacher, positional interpolation, and the CNN patch embedding are unreachable
under every released checkpoint, so those branches are not ported.

## Public constructor interface

```python
class AtstFrameEncoder(BaseEncoder):
    supported_granularities = frozenset(("clip", "frame"))

    def __init__(
        self,
        *,
        granularity: Granularity,
        arch: AtstArch,
        n_blocks: int = 1,
        pretrained: bool = True,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        ...
```

The constructor must call `super().__init__(granularity)` first; the remaining parameters are keyword-only.

`arch` must be `"small"` or `"base"`; anything else raises `ValueError` listing the valid values.
`n_blocks` must be a genuine `int` in `[1, 12]` — the check is `type(n_blocks) is not int`, so `True`
raises `TypeError` rather than silently acting as 1 — and an out-of-range value raises `ValueError`.
`pretrained` is type-checked the same way.

`embedding_dim` is **not** a class attribute here. The output width depends on both `arch` and `n_blocks`,
so it is assigned per instance in `__init__` (base.md row E45):

```python
self.embedding_dim = n_blocks * ATST_EMBED_DIMS[arch]
```

Unlike [`atst_clip.md`](atst_clip.md), there is no factor of two: this family has no cls token and
therefore no second pooling branch. Both granularities of one instance have the same width. Concretely:
`small` gives 384 at `n_blocks=1` and 4608 at `n_blocks=12`; `base` gives 768 and 9216.

`n_blocks` is deliberately not pinned by the registry; it reaches `create_model` through `**kwargs` and is
surfaced by the embedding-extraction CLI as `--model_kwargs` (see [`../registry.md`](../registry.md)).
`n_blocks=12` reproduces the official downstream configuration.

### `pretrained=True`

- resolves the checkpoint through `helpers.ensure_atst_checkpoint`, which downloads on first construction
  into `HF_HUB_CACHE/audioencoders/atst` (or `pretrained_dir`) and verifies the fixed SHA-256, memoized
  per process by `(path, digest)`;
- normalizes the Lightning archive through `helpers.load_atst_encoder_state`, keeping only the
  `model.teacher.encoder.*` subtree;
- loads the resulting 138 tensors with `strict=True`;
- does not automatically call `eval()`, freeze, or move devices.

### `pretrained=False`

- resolves no path, reads no file, touches no network (asserted by a monkeypatched `ensure_atst_checkpoint`
  in the default suite);
- initializes as the official `_init_weights` does: `trunc_normal_(std=0.02)` on `pos_embed`, then on
  `mask_embed`, then `trunc_normal_(std=0.02)` weights and zero biases for every `Linear`, and unit
  weight / zero bias for every `LayerNorm`;
- `arch` and `n_blocks` still fix the architecture and the output width.

## Shared weight logic (`helpers/atst.py`)

`src/timbral/models/helpers/atst.py` is shared verbatim with ATST-Clip and with the Transform: it holds the
four checkpoints' metadata, the frontend and patch geometry constants, path resolution, digest
verification with a process-level memo, the single reader that normalizes both archive layouts, and the
frame/patch count helpers. Its full description, including the `weights_only=True` allowlist and the
DINO-layout normalization that only ATST-Clip needs, is in [`atst_clip.md`](atst_clip.md).

What matters at this end: after normalization the Encoder receives a plain `dict[str, Tensor]` whose keys
already match its own module tree, so `AtstFrameEncoder` never branches on archive format, and
`atst_patch_frames` is the single definition of how many frames the frame granularity produces:

```python
atst_patch_frames(valid_feature_frames) = valid_feature_frames // 4
```

`timbral.models.helpers`'s `__init__` adds no new exports; the registry and both Encoders import directly
from `timbral.models.helpers.atst`.

## Vendored backbone (private classes inside `encoders/atst.py`)

`_AtstPatchEmbed`, `_AtstAttention`, `_AtstMlp`, and `_AtstBlock` are defined once in
`src/timbral/models/encoders/atst.py` and used by both families unchanged. Their contract — the linear
64x4 patch embedding that reproduces the official einops rearrangement without an einops dependency, the
12 pre-norm blocks, the bias-free qkv projection, and the omitted additive attention mask — is specified in
[`atst_clip.md`](atst_clip.md) and is not repeated here.

### What ATST-Frame does differently

| Aspect | ATST-Clip (`AST`) | ATST-Frame (`FrameAST`) |
|---|---|---|
| cls token | `cls_token` parameter, prepended to the patch sequence | none |
| `pos_embed` slice | `[:, : P + 1]`, slot 0 is the cls token | `[:, 1 : P + 1]`, slot 0 stays unused |
| final norm | `norm` | `norm_frame` |
| `state_dict` keys | 139 | 138 |
| clip width | `2 * n_blocks * D` (cls branch + patch mean) | `n_blocks * D` (patch mean only) |
| frame granularity | not exposed | `[T, n_blocks * D]`, one frame per patch |

`pos_embed` is still declared with the full 251 slots, because the official `FrameAST` declares and stores
it that way; slot 0 is loaded, kept in the `state_dict`, and never read. The 250 usable slots are what caps
a single forward pass at 250 patches.

`mask_embed` is declared but never read at inference; it exists solely to keep `state_dict` parity with the
official checkpoint, exactly as in the clip family.

`n_blocks` selects how many **trailing** blocks are concatenated. Every selected block output passes
through the same `norm_frame`, and the results are concatenated along the channel axis in ascending block
order, which is precisely what the official `get_intermediate_layers` does:

```python
for index, block in enumerate(self.blocks):
    tokens = block(tokens)
    if len(self.blocks) - index <= self.n_blocks:
        collected.append(self.norm_frame(tokens))
return torch.cat(collected, dim=-1)
```

### Pruned branches (training-only or unreachable under the released checkpoints)

- masking and any use of `mask_embed`;
- prompt tokens (`nprompt > 0`);
- positional interpolation (`pos_type != "cut"`);
- the CNN patch embedding;
- the data2vec `avg_blocks` teacher;
- DropPath and every dropout (all identities under eval);
- the official additive attention mask, which is uniformly zero once callers slice to the exact valid
  prefix (the alignment tests verify this equivalence bit-exactly);
- the pretraining-only `random_mask` helper; its module-scope `fairseq` import is satisfied only by a
  raising stub inside the alignment test, which asserts as a side effect that inference never reaches it.

### state_dict key inventory (138)

```text
mask_embed                                   [1, 1, D]
pos_embed                                    [1, 251, D]
patch_embed.patch_embed.{weight,bias}        [D, 256], [D]
norm_frame.{weight,bias}
blocks.{0..11}.norm1.{weight,bias}
blocks.{0..11}.attn.qkv.weight               [3D, D]   (no bias)
blocks.{0..11}.attn.proj.{weight,bias}
blocks.{0..11}.norm2.{weight,bias}
blocks.{0..11}.mlp.fc1.{weight,bias}
blocks.{0..11}.mlp.fc2.{weight,bias}
```

11 keys per block x 12 + 6 top-level keys = 138. Relative to the clip family's 139, `cls_token` is absent
and `norm.*` is named `norm_frame.*`. The key set is independent of `n_blocks`: all 12 blocks are always
constructed and loaded, and `n_blocks` only selects which outputs are read.

## Input contract

Public call:

```python
encoder(
    input_features,
    valid_seconds=valid_seconds,
    valid_feature_frames=valid_feature_frames,
)
```

### `input_features`

- shape `[B, T, 64]`, float32, **time-major** (the official code path uses `[B, 1, 64, T]`; the Transform
  emits this repository's time-major layout and the patch reshape absorbs the difference, see
  [`../transforms/atst.md`](../transforms/atst.md));
- the log-mel features normalized to `[-1, 1]` produced by `AtstMelspecTransform`;
- automatically moved to the Encoder's `device` by `BaseEncoder`.

### `valid_feature_frames`

- shape `[B]`, dtype `torch.int64`, always >= 4 (the Transform's 513-sample minimum is exactly 4 mel
  frames, i.e. the single patch this Encoder needs at minimum);
- both encoding hooks declare the parameter explicitly, with no catch-all `**kwargs`; unknown model inputs
  raise `TypeError`, and omitting this one raises `TypeError` as well.

### `valid_seconds`

- shape `[B]`, float32, used directly for geometry.

## Unique-length grouping and chunked forward pass

Both granularities follow the same three-stage shape:

1. `iter_length_groups(valid_feature_frames)` splits the batch into groups of equal valid length; every
   group is cropped to its own frame count, so zero padding never enters a forward pass and the result is
   independent of batch composition;
2. each group is split into consecutive chunks of at most `ATST_CHUNK_FRAMES = 1000` mel frames, and each
   chunk is embedded independently by `_chunk_tokens`;
3. the per-chunk results are combined according to granularity, then scattered back into a zero canvas by
   original batch index (`assemble_flat_groups` for clip, `assemble_padded_groups` for frame).

Chunk bounds come from the group's `valid_feature_frames`, never from the padded canvas width:

```text
for start in range(0, valid_feature_frames, 1000):
    end = min(start + 1000, valid_feature_frames)
    keep the chunk iff (end - start) >= 4
```

A trailing chunk shorter than one patch contributes nothing and is dropped. Every kept chunk is embedded
from positional slot 1 onward, i.e. every chunk re-uses the same slots — chunking is a hard consequence of
`pos_embed` holding only 250 patch slots, not a windowing choice.

### Chunk grid deviation from the official code

1000 mel frames is exactly 250 patches, i.e. 10.00 s, and 1000 is itself a multiple of the patch width, so:

```text
total patches == valid_feature_frames // 4        (always, for any duration)
```

and every patch stays on the global 40 ms grid. The official `embedding.py` instead chunks at 1001 frames,
which drops one frame per chunk and shifts every later chunk's patch grid by 10 ms; over 30 s the official
split yields 749 patches where the grid above yields 750. The 1000-frame grid is this repository's
deliberate deviation: a chunk length that is itself a multiple of the patch width is what makes the
identity above hold for every duration, and 1000 is the largest such length the 250 slots allow. The
alignment tests therefore assemble their multi-chunk reference from official single-chunk calls on the same
bounds (see [`../extra/atst_frame-alignment.md`](../extra/atst_frame-alignment.md)).

## Frame granularity

Per chunk:

```text
tokens = patch_embed(features[:, start:end])          # [n, P_c, D],  P_c = (end - start) // 4
tokens = tokens + pos_embed[:, 1 : P_c + 1, :]
tokens = block_11(... block_0(tokens) ...)
chunk  = cat(norm_frame(output of each of the last n_blocks blocks), dim=-1)
```

Chunks are concatenated along time:

```text
embedding_group = cat(chunk_0, chunk_1, ..., dim=1)   # [n, valid_feature_frames // 4, n_blocks * D]
```

This is the official public output of the family, not a project-derived reduction: `get_intermediate_layers`
returns exactly these per-patch vectors. There is no frequency axis to average over — one patch already
spans all 64 mel bins.

### Frame geometry

One patch spans 4 mel frames of 10 ms:

```text
step_seconds = 4 x 160 / 16000 = 0.04
num_valid    = valid_feature_frames // 4              (always >= 1)
boundaries   = arange(T_max + 1) x 0.04
start[i]     = boundaries[i]
end[i]       = min(boundaries[i + 1], valid_seconds)      (i < num_valid - 1)
end[num_valid - 1] = valid_seconds
```

`start` and `end` are sliced from the same float32 `boundaries` tensor, so adjacent valid slots satisfy
`end[i] == start[i + 1]` element-wise. The implementation shares `helpers.geometry.build_frame_geometry`
with PANNs/AST/BEATs/Wav2Vec2.

The last valid frame's end is stretched to `valid_seconds` rather than left on the grid, because
`valid_seconds` and the patch grid do not coincide in general. For any duration at or above the Transform's
513-sample floor (0.0320625 s), the mel frame count is `F = round(16000 x valid_seconds) // 160 + 1`, i.e.
`floor(100 x valid_seconds) + 1` on the 10 ms grid, and the nominal end of the last valid frame is
`(F - F % 4) / 100`, which sits within `(valid_seconds - 0.03, valid_seconds + 0.01]`: the trailing 1-3 mel
frames that produce no patch make it fall short by less than one patch, and the extra frame contributed by
`center=True` can make it overshoot by up to 10 ms, that upper end being attained when `valid_seconds` sits
on the 10 ms grid and `F` is a multiple of 4 (1.03 s gives `F = 104` and a nominal end of 1.04). Below that
floor the Transform clamps `valid_feature_frames` to 4, so there is exactly one frame and its slot is
`[0, valid_seconds]` outright. Interior boundaries are always at least 0.03 s below `valid_seconds` (in the
clamped case there are none at all), so the `min` above never fires for them; only the final assignment is
ever load-bearing. Examples:

| `valid_seconds` | mel frames `F` | patches | last slot |
|---:|---:|---:|---|
| 0.03 | 4 | 1 | `[0.00, 0.03]` |
| 1.00 | 101 | 25 | `[0.96, 1.00]` |
| 10.00 | 1001 | 250 | `[9.96, 10.00]` |
| 10.05 | 1006 | 251 | `[10.00, 10.05]` |

The 0.03 s row is that clamped minimum: 480 target samples are padded to 513, giving 4 mel frames and one
patch whose slot is shortened to the true 0.03 s. The 1.00 s row drops one remainder mel frame and still
lands exactly on `valid_seconds`. The 10.05 s row spans two chunks, 1000 frames (250 patches) plus a
6-frame tail (1 patch) with 2 remainder frames dropped, and its last slot is stretched by 10 ms; the
shortest input that reaches a second chunk is 10.03 s, at 1004 mel frames.

### Batch padding

```text
embedding  [B, T_max, n_blocks * D]      T_max = max(valid_feature_frames // 4)
geometry   [B, T_max, 2]
valid_mask [B, T_max]
```

Invalid positions have `valid_mask=False`, with embedding and geometry rows filled with exact zero values.
The zero canvas plus `index_copy` already guarantees exact zeros in the embedding, so no post-hoc
multiplication by the mask is applied; `valid_mask` is the sole source of truth for validity.

## Clip granularity

Each chunk is pooled over its own patch tokens and the chunk vectors are averaged with equal weight:

```text
chunk_vector = tokens.sum(dim=1) / (P_c + 1e-6)       # [n, n_blocks * D]
clip         = stack(chunk_vectors).mean(dim=0)
```

Three points are deliberate:

- the division by `P_c + 1e-6` reproduces the official epsilon rather than taking an exact mean;
- there is no cls branch, so the width stays `n_blocks * D`; this is the official `scene=True` path of
  `get_intermediate_layers`;
- chunks are averaged with **equal weight**, matching the official `get_scene_embedding`, and not the
  downstream `PretrainedEncoderPLModule`, which discards a trailing chunk shorter than half a chunk. Under
  equal weighting a short trailing chunk counts as much as a full one.

Output:

```text
embedding  [B, n_blocks * D]
geometry   [B, 2] = [0, valid_seconds]
valid_mask [B] all True
```

Because both the clip pooling and the frame output are linear over the same fully valid tokens,
`clip == time average of frame output` holds within numerical tolerance whenever the input fits in a single
chunk, and is used as a cross-check assertion in the tests. It does **not** hold across chunks of unequal
length, by the equal-weight rule above.

## Device, training, and serialization

- `device` is derived from `patch_embed.patch_embed.weight`;
- `BaseEncoder` automatically transfers `input_features`, `valid_seconds`, and `valid_feature_frames`;
- embedding retains the model's actual dtype (float32);
- geometry is fixed to float32; valid_mask is fixed to bool;
- does not automatically call `eval()`, does not freeze, does not wrap in `torch.no_grad()`;
- uses the plain `state_dict`;
- does not impose a maximum input duration.

## Files and exports

```text
src/timbral/models/encoders/atst.py
src/timbral/models/helpers/atst.py
src/timbral/models/helpers/geometry.py
src/timbral/models/helpers/grouping.py
```

Division of responsibilities:

- `helpers/atst.py`: checkpoint metadata and identity, download and verification, archive normalization,
  frontend/patch geometry constants, `atst_feature_frames` / `atst_patch_frames`;
- `helpers/geometry.py`: model-agnostic construction of clip/frame geometry and valid_mask;
- `helpers/grouping.py`: model-agnostic scaffolding for unique-length grouping and refilling grouped
  results back into a zero canvas;
- `encoders/atst.py`: `AtstFrameEncoder` and `AtstClipEncoder`, their granularity semantics, and the shared
  inference-only backbone (`_`-prefixed private classes, not part of the public export).

Both Encoders live in one module because they share the backbone body verbatim; only the head geometry
differs. `timbral.models.encoders` re-exports `AtstFrameEncoder`. `timbral.models` at the top level only
re-exports registry symbols (see [`../registry.md`](../registry.md)).

## Testing requirements

Weight-free, network-free tests (the default suite, `pretrained=False`) must cover at least:

- the `state_dict` key set is exactly the 138-key inventory; `cls_token` is absent, no key starts with
  `norm.`, `norm_frame.{weight,bias}` are present, and no `attn.qkv.bias` exists;
- parameter shapes: `pos_embed` `[1, 251, D]`, `mask_embed` `[1, 1, D]`, `patch_embed.patch_embed.weight`
  `[D, 256]`, `blocks.0.attn.qkv.weight` `[3D, D]`;
- `embedding_dim == n_blocks * D` for several `(arch, n_blocks)` pairs, and that it lives in
  `vars(instance)` rather than `vars(AtstFrameEncoder)`;
- `supported_granularities == {"clip", "frame"}`, and both granularities construct;
- invalid construction: unknown `arch` and unsupported `granularity` raise `ValueError`; `n_blocks=True`
  and non-bool `pretrained` raise `TypeError`; `n_blocks` of 0 or 13 raise `ValueError`;
- `pretrained=False` resolves no path and reads no file (monkeypatched `ensure_atst_checkpoint` fails the
  test if called);
- frame output contract: `valid_feature_frames` of 8/27/40 give 2/6/10 valid frames, shapes and dtypes as
  above, padded rows exactly zero in both embedding and geometry;
- frame geometry: starts on the exact 0.04 s grid, interior ends equal the next start, every interval has
  positive length, the first start is 0.0 and the last valid end equals `valid_seconds`;
- clip output contract, and `clip == mean over frames` for a single-chunk input;
- batch-composition independence at both granularities: each row of a mixed-length batch is bit-identical
  to its standalone call;
- chunking: a 2500-frame input yields 625 frames, equal to the concatenation of independent forward passes
  on the `(0, 1000)`, `(1000, 2000)`, `(2000, 2500)` bounds;
- the minimum input (`valid_feature_frames=4`) yields exactly one valid frame with geometry `[0, 0.03]`;
- unknown model inputs, and a missing `valid_feature_frames`, raise `TypeError`;
- after `.to(device)`, input and output devices are correct (when CUDA is available).

For real checkpoint loading and full-network alignment against the official `FrameAST`, see
[`../extra/atst_frame-alignment.md`](../extra/atst_frame-alignment.md): 7 tests behind
`pytest --run-alignment atst_frame`, with frame granularity bit-exact (max|delta| = 0.0) on both CPU and
CUDA for both architectures, clip granularity 0.0 on CPU and 9.5e-7 on CUDA, and multi-chunk / mixed-batch
cases 0.0 on CPU and at most 1.4e-6 on CUDA.

## Dependency boundary

The backbone private classes depend only on PyTorch; `helpers/atst.py` additionally depends on NumPy (only
for the `weights_only` allowlist) and huggingface_hub (only the `HF_HUB_CACHE` constant). No einops is
introduced despite the official patch embedding being written with it, and no fairseq is introduced: the
official module-scope `fairseq` import exists only on the pretraining path and is satisfied by a raising
stub inside the alignment test, never in the shipped package. No new third-party dependency is added.
