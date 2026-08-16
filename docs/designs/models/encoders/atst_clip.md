# `AtstClipEncoder` Design

This document freezes the design of `timbral.models.encoders.AtstClipEncoder`. The companion Transform is
[`../transforms/atst.md`](../transforms/atst.md), shared verbatim with the sibling family
[`atst_frame.md`](atst_frame.md); the official alignment contract is
[`../extra/atst_clip-alignment.md`](../extra/atst_clip-alignment.md).

This document describes the behavior the current implementation must satisfy, and follows the common
contract in [`base.md`](base.md).

## Design goals

`AtstClipEncoder` is responsible for:

- reproducing the official ATST-Clip backbone (official class `AST`, INTERSPEECH 2022, arXiv:2204.12076,
  from Audio-WestlakeU/audiossl) via an inference-only local rewrite that is operator-for-operator
  numerically equivalent, with `state_dict` keys matching the official ones one-to-one;
- supporting both released architectures, `small` (D=384, 6 heads) and `base` (D=768, 12 heads), under the
  registered names `atst-clip-small` and `atst-clip-base`;
- normalizing the two mutually incompatible archive layouts the official release ships into a single key
  set, so that the Encoder itself never branches on archive format;
- reproducing the official downstream feature extraction `get_intermediate_layers_chunks(avgpool=True)`:
  the trailing `n_blocks` block outputs, each passed through the final norm, with their cls branch and
  patch-mean branch concatenated into a `2 * n_blocks * D` vector;
- splitting inputs longer than the 250 patch slots of the learned `pos_embed` into chunks and averaging
  the per-chunk vectors with equal weight;
- grouping mixed-length batches by `valid_feature_frames`, keeping the output independent of batch
  composition;
- producing clip geometry and a valid mask conforming to `BaseEncoder`.

This component does not hold a Transform, does not accept waveforms, does not compute mel features, and
imposes no duration cap. A single forward pass is capped at 250 patches by chunking, so self-attention
memory does not grow quadratically with total duration; the number of chunks grows linearly instead.

## Clip-only granularity

The supported set is fixed to:

```python
supported_granularities = frozenset(("clip",))
```

ATST-Clip is trained and released as a clip-level model: the official `AST` exposes `forward` (the cls
token alone), `get_intermediate_layers`, and `get_intermediate_layers_chunks`, and every official
downstream head consumes the single pooled vector these produce. There is no official frame-level usage of
this family, and therefore nothing to align a frame output against. Emitting the per-patch token sequence
here would be a locally invented output with no official reference, which the repository's alignment
contract does not permit. The frame-level need is served by ATST-Frame instead, whose token sequence is
itself an official output (see [`atst_frame.md`](atst_frame.md)).

This follows the `ClapHtsatEncoder` precedent of a clip-only Encoder (see [`clap.md`](clap.md)):
`granularity="frame"` raises `ValueError` in `BaseEncoder.__init__`, before any argument validation, path
resolution, or weight loading, with the message naming `AtstClipEncoder` and its supported set.

## Supported checkpoints

Two entries, each pinned by a fixed SHA-256 that is the file's identity:

| entry | arch | filename | size (B) | archive |
|---|---|---|---:|---|
| `atst-clip-small` | small | `atst_clip_small.ckpt` | 411,267,681 | Lightning |
| `atst-clip-base` | base | `atst_clip_base.ckpt` | 1,459,530,207 | DINO-style |

```text
atst_clip_small.ckpt
    url    https://checkpointstorage.oss-cn-beijing.aliyuncs.com/atst/small.ckpt
    sha256 fcadd6411881410d27cde47f4d540ef416aa59e0197b195cf3ee7a81885a5f4a
atst_clip_base.ckpt
    url    https://checkpointstorage.oss-cn-beijing.aliyuncs.com/atst/base.ckpt
    sha256 7b20168cae0d1488a0e3334f17ca1cefb9365cbaa2401c11aa98d6ffaa668496
```

ATST-Clip is the only family of the two that ships both archive layouts, so the format normalization
described under *Shared weight logic* below is exercised in full by these two entries alone. The two
ATST-Frame checkpoints are both Lightning.

### Architecture constants

The width, head count, and depth come from the official `AST_small` / `AST_base` factories and are
verified against the checkpoints' tensor shapes. Only the DINO archive records part of the configuration
in its `["args"]` (`arch="ast_base"`, `pos_type="cut"`, `patch_height=64`, `patch_width=4`,
`use_cls=True`); the Lightning archive's `hyper_parameters` holds `arch` plus training settings only, so
the `small` entry's geometry is the factory default confirmed by the shapes it loads into. All fields are
consistent across both entries except for the width and head count:

| Field | `small` | `base` |
|---|---:|---:|
| `embed_dim` (D) | 384 | 768 |
| `num_heads` | 6 | 12 |
| `depth` | 12 | 12 |
| mlp ratio | 4 | 4 |
| `qkv_bias` | `False` | `False` |
| LayerNorm eps | 1e-6 | 1e-6 |
| `patch_h` × `patch_w` | 64 × 4 | 64 × 4 |
| `n_mels` | 64 | 64 |
| `pos_embed` slots | 251 | 251 |
| `use_cls` | `True` | `True` |
| `pos_type` | `"cut"` | `"cut"` |

`n_mels` is fixed by the shared frontend (see [`../transforms/atst.md`](../transforms/atst.md)) and equals
`patch_h`, so one patch spans the whole mel axis; `use_cls=True` is confirmed by the `cls_token` both
checkpoints carry. `pos_type="cut"` holds for every released checkpoint, so the interpolation branch is
unreachable and is not ported, and the patch embedding is always the linear `PatchEmbed_v2` (see *Pruned
branches* below). `nprompt`, `avg_blocks`, and the `patch_embed` selector are not parameters of the
official `AST` at all; they belong to the sibling `FrameAST` (see [`atst_frame.md`](atst_frame.md)).
`["args"].arch == "ast_base"` records which architecture the DINO archive holds, but the file's identity
is its SHA-256, not that field.

A single hardcoded architecture therefore covers both entries; `arch` selects only the width/head pair.

## Public constructor interface

```python
class AtstClipEncoder(BaseEncoder):
    supported_granularities = frozenset(("clip",))

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

`embedding_dim` is **not** a class attribute here. The output width depends on both `arch` and `n_blocks`,
so it is assigned per instance in `__init__`:

```python
self.embedding_dim = 2 * n_blocks * ATST_EMBED_DIMS[arch]
```

Callers must read it off the instance. `BaseEncoder.embedding_dim` is declared instance-level precisely to
admit a width like this one, which no single class-level value can express; Encoders of fixed width declare
it at class level all the same. See row E45 of [`base.md`](base.md).

Argument validation is shared with `AtstFrameEncoder` (`_validate_common_arguments`) and runs after the
granularity check:

- `arch` not in `{"small", "base"}` → `ValueError`, message listing the available archs;
- `n_blocks` whose `type` is not exactly `int` (including `bool` and `float`) → `TypeError`;
- `n_blocks` outside `[1, 12]` → `ValueError`;
- `pretrained` whose `type` is not exactly `bool` → `TypeError`.

`n_blocks` selects how many **trailing** blocks are concatenated, counted from block 11 backwards. The
default is 1; `n_blocks=12` reproduces the official downstream configuration. It is deliberately not
pinned by the registry, because it changes the output width; it reaches `create_model` through `**kwargs`
and the embedding-extraction CLI through `--model_kwargs` (see [`../registry.md`](../registry.md)).

Parameters are always initialized with the official rules before any weight loading:
`trunc_normal_(std=0.02)` on `pos_embed`, `mask_embed`, and `cls_token` in that order, then
`apply(_init_module)`, which sets `trunc_normal_(std=0.02)` / zero bias on every `Linear` and
(weight=1, bias=0) on every `LayerNorm`.

### `pretrained=True`

- resolves, downloads if absent, and verifies the checkpoint via `helpers.atst.ensure_atst_checkpoint`;
- reads the archive via `helpers.atst.load_atst_encoder_state`, using
  `torch.load(weights_only=True, map_location="cpu")` under a minimal allowlist, never falling back to
  `weights_only=False`;
- keeps only the teacher encoder subtree, discarding the student branch, the optimizer state, and (DINO
  layout) the projection head;
- performs `load_state_dict(..., strict=True)` over the resulting 139 tensors;
- does not automatically call `eval()`, freeze, or move devices.

### `pretrained=False`

- does not resolve a cache path, does not read files, does not access the network;
- keeps the official initialization described above as the final parameter state;
- `arch` and `n_blocks` still fully determine the module structure and `embedding_dim`.

## Shared weight logic (`helpers/atst.py`)

The following content is concentrated in `src/timbral/models/helpers/atst.py`, shared with
`AtstFrameEncoder`:

- `ATST_CHECKPOINTS: dict[tuple[AtstFamily, AtstArch], AtstCheckpointMetadata]`, four entries carrying
  `model_name`, `family`, `arch`, `filename`, `url`, `sha256`, `archive_format`, `embed_dim`, `num_heads`,
  plus the `ATST_CHECKPOINTS_BY_NAME` view keyed by registered name;
- the frontend and patch geometry constants shared with the Transform;
- default directory resolution, download, and SHA-256 verification;
- archive-format normalization into the single key set the Encoders load.

Path priority follows the same convention as sibling models:

```text
explicit pretrained_dir
    >
HF_HUB_CACHE/audioencoders/atst
```

`ensure_atst_checkpoint(metadata, pretrained_dir)` downloads on first construction, unlike the BEATs helper
which refuses to download (the ATST URLs are directly fetchable and need no browser session):

- if the file exists: its SHA-256 is verified and an immediate failure occurs on any mismatch;
- if the file is missing: `torch.hub.download_url_to_file` writes to a `.part` temporary file **inside the
  target directory**, the digest is verified there, and only then is `os.replace` used to move it to the
  final path, so an interrupted or corrupted download never leaves a file a later construction would
  accept;
- verification is memoized per process by `(path, digest)`: the `base` file is 1.4 GB and rehashing it on
  every construction is measurable.

### Two archive layouts

| entry | layout | encoder subtree |
|---|---|---|
| `atst_clip_small.ckpt` | Lightning | `state_dict["model.teacher.encoder.*"]` |
| `atst_clip_base.ckpt` | DINO-style | `["teacher"]["module.backbone.*"]` |

The Lightning archive carries a `pytorch-lightning_version` key and holds every tensor under `state_dict`;
the DINO archive is a plain dict with top-level `student`, `teacher`, `optimizer`, `epoch`, `args`, and
`byol_loss`. Nothing is sniffed at load time: which layout an entry uses is fixed by its
`AtstCheckpointMetadata.archive_format`, and the SHA-256 already pins which file it is, so
`load_atst_encoder_state` only checks that the expected subtree (`state_dict` or `teacher`) is present and
raises `ValueError` otherwise. The official `get_pretraied_encoder` makes the same distinction at run time,
by testing for the `pytorch-lightning_version` key.

The DINO archive stores the 7 `module.head.*` tensors of the DINO projection head next to the backbone;
they belong to the pretraining objective, not to the encoder. Normalization keeps the `module.backbone.*`
subtree only, so those keys never reach the Encoder. This deviates from the official loader, which instead
tolerates them by loading with `strict=False`; restricting the state to the backbone subtree lets this
repository keep `strict=True`, so a future archive change surfaces as a load error rather than as silently
missing weights.

After normalization both layouts yield the same 139 keys. Only about 23% of a `base` file is the encoder
(341.7 MB of 1460 MB); the remainder is optimizer and training state, kept as-is because the fixed SHA-256
is the file's identity.

Reading is done in `weights_only` safe mode under a minimal allowlist:

```text
numpy.core.multiarray.scalar     (registered via the (object, name) tuple form, since numpy.core is a
                                  compatibility shim for numpy._core under NumPy 2.x)
numpy.dtype
numpy.dtypes.Float64DType
argparse.Namespace               (DINO layout only, for the pickled training args)
```

Both archives pickle NumPy scalars for logged metrics and schedules; the allowlist is the minimum that
makes `weights_only=True` succeed, and no other global is permitted.

`timbral.models.helpers`'s `__init__` does not add new exports (following the BEATs/CLAP precedent); the
registry and Encoder import directly from `timbral.models.helpers.atst`.

## Vendored backbone (private classes inside `encoders/atst.py`)

The official `audio_transformer.py`/`transformer.py` are rewritten as inference-only, repository-style
private classes inside `encoders/atst.py` (`_AtstAttention`, `_AtstMlp`, `_AtstBlock`, `_AtstPatchEmbed`,
following the PANNs `_ConvBlock` and BEATs `_Beats*` precedent of inline private classes), requiring:

- operator-for-operator numerical equivalence with the official implementation (backed by the alignment
  contract);
- `state_dict` with exactly 139 keys, matching the official checkpoint one-to-one after normalization;
- retaining only the branches actually reached by the four released checkpoints.

The block builder, the initialization rule, the argument validator, and the chunk splitter are shared with
`AtstFrameEncoder` inside the same module.

### Retained components

- `patch_embed`: the official `PatchEmbed_v2`, a `Linear(256 -> D)` over 64×4 patches (64 mel bins × 4
  frames). The official `einops` rearrange `b c (h p1) (w p2) -> b (w h) (p1 p2 c)` is reproduced by
  reshaping the time-major features directly, `[B, W, 4, 64] -> transpose -> [B, W, 64, 4] -> [B, W, 256]`,
  which is the official mel-bin-major / frame-minor ordering; **einops is not a dependency**. The mel axis
  holds exactly one patch row and the input carries a single channel, so no further rearrangement is
  needed. A trailing remainder shorter than one patch is dropped, matching the official slice to
  `width - width % patch_width`;
- `cls_token`, a `[1, 1, D]` parameter prepended to the patch tokens;
- `pos_embed`, a `[1, 251, D]` parameter = 250 patch slots + 1 cls slot. With `pos_type="cut"`, ATST-Clip
  slices `pos_embed[:, : P + 1]`, because slot 0 belongs to its cls token (ATST-Frame slices `[:, 1 : P+1]`
  instead and leaves slot 0 unused);
- 12 pre-norm Transformer blocks: `LayerNorm(eps=1e-6)` → attention → residual → `LayerNorm(eps=1e-6)` →
  MLP → residual;
- attention: `qkv: Linear(D, 3D, bias=False)`, `scale = head_dim ** -0.5` applied to `q @ k^T`, softmax over
  the last axis, then `proj: Linear(D, D)` with bias. The absence of a qkv bias is the checkpoint's own
  configuration, so no `attn.qkv.bias` key exists on either side;
- MLP: `fc1: Linear(D, 4D)` → `GELU` → `fc2: Linear(4D, D)`;
- `norm`: the final `LayerNorm(eps=1e-6)` (ATST-Frame names the corresponding module `norm_frame`), applied
  to **every** selected block output, exactly as the official `get_intermediate_layers` does rather than
  only to the last one;
- `mask_embed`, a `[1, 1, D]` parameter that is declared but never read at inference. It exists solely to
  keep the `state_dict` in one-to-one correspondence with the official checkpoint under `strict=True`.

The official `Attention` adds an additive mask built from each sample's valid length. Callers here group
by valid length and slice to the exact prefix, so that mask is uniformly zero and is omitted; the
arithmetic is unchanged, and the equivalence is verified bit-exactly by the alignment tests.

### Pruned branches (training-only or unreachable)

- masking and any use of `mask_embed` (pretraining only; the official extraction path itself calls
  `prepare_tokens(..., mask=False)`);
- positional interpolation (`interpolate_pos_encoding`, reached only when `pos_type != "cut"`, which no
  released checkpoint sets);
- the teacher's block-averaging path (`AST.forward(avg=True)`, the mean of the last 8 block outputs; the
  sibling `FrameAST` spells the same idea as `avg_blocks > 0`), which only the pretraining objective uses;
- prompt tokens (`nprompt > 0`) and the CNN patch embedding (`patch_embed != "Linear"`), both options of
  `FrameAST` alone and never set by a released checkpoint (see [`atst_frame.md`](atst_frame.md));
- `DropPath` and every dropout (all identities under `eval`);
- the student branch, the byol/DINO loss, and the projection head, none of which are encoder state.

### state_dict key inventory (139)

```text
mask_embed
cls_token
pos_embed
patch_embed.patch_embed.{weight,bias}
norm.{weight,bias}
blocks.{0..11}.norm1.{weight,bias}
blocks.{0..11}.attn.qkv.weight
blocks.{0..11}.attn.proj.{weight,bias}
blocks.{0..11}.norm2.{weight,bias}
blocks.{0..11}.mlp.fc1.{weight,bias}
blocks.{0..11}.mlp.fc2.{weight,bias}
```

11 keys per block × 12 + 7 top-level keys = 139. `AtstFrameEncoder` has 138: it carries no `cls_token`, and
its final norm is named `norm_frame`.

## Input contract

Public call:

```python
encoder(
    input_features,
    valid_feature_frames=valid_feature_frames,
    valid_seconds=valid_seconds,
)
```

### `input_features`

- shape `[B, T, 64]`, float32, **time-major**;
- the normalized log-mel features produced by the companion Transform (see
  [`../transforms/atst.md`](../transforms/atst.md)). The official code path passes `[B, 1, 64, T]`; the
  time-major layout is this repository's uniform convention across all Encoders, and the transposition is
  absorbed by the patch reshape;
- padded positions may hold arbitrary values, since grouping slices every group to its exact prefix;
- automatically moved to the Encoder's `device` by `BaseEncoder`.

### `valid_feature_frames`

- shape `[B]`, dtype `torch.int64`, always ≥ 4 (the Transform's 513-sample zero-padding floor yields
  exactly 4 mel frames, i.e. the single patch this Encoder needs at minimum);
- `_encode_clip` explicitly declares this parameter, without a catch-all `**kwargs` that would swallow
  unknown parameters; unknown model inputs must naturally raise `TypeError`.

### `valid_seconds`

- shape `[B]`, float32, used directly for geometry.

## Unique-length grouping and forward pass

Samples with different `valid_feature_frames` must not be passed through the backbone on the same physical
canvas: self-attention is global, so padded positions would contribute to every token. The official
approach is an additive attention mask; grouping is used here instead, because it makes the padding
structurally absent rather than numerically suppressed, and it is what allows the mask to be omitted from
the vendored `Attention`. The process:

1. compute unique values of `valid_feature_frames` (`helpers.grouping.iter_length_groups`);
2. for each group, `index_select` its rows out of the batch;
3. split the group's frame count `F` into chunk bounds (see *Chunking* below);
4. for each chunk, slice `group_features[:, start:end]` and run `_chunk_features`;
5. average the per-chunk vectors, giving that group's `[B_g, 2 * n_blocks * D]` output;
6. restore batch order by original index into a zero canvas
   (`helpers.grouping.assemble_flat_groups`).

Every position inside a group is valid, so no padding enters any computation, and the forward result is
element-wise identical to calling on individual samples one at a time. The backbone holds no BatchNorm, so
grouping introduces no training-mode batch dependency either.

## Chunking

The learned `pos_embed` holds 250 patch slots, which caps a single forward pass at 250 patches. Inputs are
split into consecutive **1000-mel-frame** chunks:

```text
ATST_CHUNK_PATCHES = 250
ATST_CHUNK_FRAMES  = 250 × 4 = 1000 frames = 10.00 s
```

```text
bounds = [(start, min(start + 1000, F)) for start in range(0, F, 1000)]
         with any (end - start) < 4 dropped
```

A trailing chunk shorter than one patch would produce no token at all and is skipped; a trailing chunk of
at least one patch is kept in full.

The official chunk width is **1001** frames instead: `audiossl/methods/atstframe/embedding.py` hardcodes
`chunk_len = 1001`, and the clip family's downstream `PretrainedEncoderPLModule` reaches the same value
from a 10 s anchor (`int(10 * 16000 / 160 + 1)`). Because 1001 is not a multiple of the patch width, each
official chunk drops one frame, and every later chunk's patch grid is shifted by 10 ms relative to the
global grid; the shift accumulates (30 s yields 749 patches instead of 750). The 1000-frame grid is this
repository's deliberate deviation: the chunk length is itself a multiple of the patch width, so every
patch stays on the global 40 ms grid and

```text
total patches == valid_feature_frames // 4
```

holds exactly, for any duration and any chunk count. Because the grid is this repository's own decision,
the alignment tests assemble the reference themselves: each 1000-frame chunk is run through the official
path with `chunk_len = 1001`, one frame above the local chunk width, so that the official code treats it
as a single chunk, and the per-chunk results are then combined here.

## Clip granularity

Each chunk is encoded by `_chunk_features`:

```text
tokens  = patch_embed(features)                      # [B_g, P, D]
tokens  = cat(cls_token.expand(B_g, -1, -1), tokens) # [B_g, P + 1, D]
tokens  = tokens + pos_embed[:, : P + 1, :]
for index, block in enumerate(blocks):
    tokens = block(tokens)
    if 12 - index <= n_blocks:
        collected.append(norm(tokens))
cls_branch   = [layer[:, 0]  for layer in collected]
patch_branch = [layer[:, 1:].sum(dim=1) / (P + 1e-6) for layer in collected]
chunk_vector = cat(cls_branch + patch_branch, dim=-1)   # [B_g, 2 × n_blocks × D]
```

Three points are fixed by the official downstream default and are not configurable:

- **`avgpool=True` is fixed.** The official `get_intermediate_layers_chunks` can return the cls branch
  alone, but every official downstream configuration uses the concatenated form, so that is the only one
  reproduced. This is what makes the width `2 * n_blocks * D` rather than `n_blocks * D`.
- **The concatenation layout is `cat(cls..., avg...)`**, not an interleaving. `collected` is ordered from
  the earliest selected block to block 11, so for `n_blocks = n` the output is

  ```text
  [ cls_{12-n}, ..., cls_11, avg_{12-n}, ..., avg_11 ]
  ```

  All cls slices come first as one contiguous run, then all patch-mean slices. Callers slicing the output
  by block must respect this order.
- **The pooling denominator is `P + 1e-6`, not `P`.** The official implementation divides the token sum by
  the patch count offset by that epsilon rather than taking an exact mean; the epsilon is reproduced
  verbatim, because bit-exact CPU agreement is part of the alignment contract.

Chunks are then combined with **equal weight**:

```text
clip_embedding = stack(chunk_vectors, dim=0).mean(dim=0)
```

This matches the official `get_scene_embedding` of `audiossl/methods/atstframe/embedding.py`, the released
HEAR-style entry point of the sibling family, which averages every chunk it produces with equal weight. It
deviates from the official downstream path, where `PretrainedEncoderPLModule` calls
`get_intermediate_layers_chunks`, whose `chunk_mark` keeps a non-first chunk only while its valid length
exceeds `chunk_len // 2`; discarding audio based on where a fixed chunk boundary happens to fall would
make the embedding of a clip depend on its total length in a way the caller cannot predict, so the
full-coverage variant is used. A 10.05 s input yields 1006 mel frames and therefore contributes a
1000-frame chunk (250 patches) and a 6-frame chunk (1 patch), each weighted 1/2.

Output:

```text
embedding  [B, 2 × n_blocks × D]
geometry   [B, 2] = [0, valid_seconds]
valid_mask [B] all True
```

No L2 normalization is applied and no classification head is attached.

## Device, training, and serialization

- `device` is derived from the patch embedding weight (`patch_embed.patch_embed.weight.device`);
- `BaseEncoder` automatically transfers `input_features`, `valid_seconds`, and `valid_feature_frames`;
- embedding retains the model's actual dtype (float32);
- geometry is fixed to float32; valid_mask is fixed to bool;
- construction does not automatically call `eval()`, does not freeze parameters, and does not wrap the
  forward pass in `torch.no_grad()`; the constructor leaves the module in the default training mode;
- uses the plain `state_dict`, which is exactly the 139-key official inventory;
- does not impose a maximum input duration.

## Files and exports

```text
src/timbral/models/encoders/atst.py
src/timbral/models/helpers/atst.py
src/timbral/models/helpers/geometry.py
src/timbral/models/helpers/grouping.py
```

Division of responsibilities:

- `helpers/atst.py`: checkpoint metadata, frontend and patch geometry constants, path resolution,
  download, SHA-256 verification, safe reading, and archive-format normalization;
- `helpers/geometry.py`: model-agnostic construction of clip geometry and valid_mask;
- `helpers/grouping.py`: model-agnostic scaffolding for unique-length grouping iteration and refilling
  grouped results back into a zero canvas;
- `encoders/atst.py`: `AtstClipEncoder` and `AtstFrameEncoder`, their granularity semantics, and the
  inference-only ATST backbone (`_`-prefixed private classes, not part of the public export). Both families
  live in one module because they share the block, patch embedding, initialization, validation, and
  chunking code verbatim.

`timbral.models.encoders` re-exports `AtstClipEncoder`. `timbral.models` at the top level only re-exports
registry symbols (see [`../registry.md`](../registry.md)).

## Testing requirements

Weight-free, network-free tests (the default suite, `pretrained=False`) must cover at least:

- public export identity, `BaseEncoder` subclassing, and an entirely keyword-only signature;
- `supported_granularities == {"clip"}`, `granularity="frame"` raising `ValueError` naming the unsupported
  granularity, and an invalid granularity string raising `ValueError`;
- invalid `arch` → `ValueError`; `n_blocks` of 0 or 13 → `ValueError`; `n_blocks` of `1.5` or `"1"` →
  `TypeError`; non-bool `pretrained` → `TypeError`;
- `pretrained=False` resolves no path and reads no file (monkeypatching `ensure_atst_checkpoint` to fail
  asserts it is never triggered), and leaves the module in training mode;
- the module's own `state_dict` key set is exactly the 139-key inventory, and no `attn.qkv.bias` key exists
  on any block;
- `pos_embed` is `[1, 251, D]`, `cls_token` and `mask_embed` are `[1, 1, D]`, the patch embedding weight is
  `[D, 256]`, and there are 12 blocks;
- `embedding_dim == 2 * n_blocks * D` across archs and block counts, and `embedding_dim` is an **instance**
  attribute: absent from `vars(AtstClipEncoder)` and from `vars(BaseEncoder)`;
- clip output contract: key set, embedding shape/dtype, geometry equal to `[0, valid_seconds]`, all-True
  bool mask;
- mixed-length batches match individual calls element-wise, including a group short enough to hold one
  patch and a group long enough to be chunked;
- a 2500-frame input splits into 1000 + 1000 + 500 and equals the equal-weight mean of the three chunk
  vectors, confirming the short trailing chunk carries full weight;
- `device` follows the patch embedding weight, and after `.to(device)` input and output devices are correct
  (when CUDA is available);
- unknown model inputs raise `TypeError`.

For real checkpoint loading, both archive layouts, and full-network alignment against the official
implementation, see [`../extra/atst_clip-alignment.md`](../extra/atst_clip-alignment.md); its entry point
is `pytest --run-alignment atst_clip`. That contract also covers the shared frontend, so the ATST-Frame
alignment module does not repeat it.

## Dependency boundary

The backbone private classes depend only on PyTorch. `helpers/atst.py` depends on PyTorch (including
`torch.hub` for the download), NumPy (only for the three allowlisted pickled globals), huggingface_hub
(only the `HF_HUB_CACHE` constant), and `timbral.models.helpers.common`. No `einops`, no `audiossl`, and no
`pytorch_lightning` is introduced — the archives are read as plain nested dicts — and no new third-party
dependency is added. The official `audiossl` source is required only by the alignment tests, which
sparse-clone it at a pinned commit and never modify it.
