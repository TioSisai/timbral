# ATST-Frame Official Implementation Alignment Contract

This document freezes the official implementation alignment contract for
`AtstFrameEncoder` (the official `FrameAST`, IEEE TASLP,
arXiv:2306.04186), and records the verification summary of the completed
run.

Corresponding designs:

- [`../transforms/atst.md`](../transforms/atst.md)
- [`../encoders/atst_frame.md`](../encoders/atst_frame.md)
- [`atst_clip-alignment.md`](atst_clip-alignment.md)

The log-mel frontend is one and the same `AtstMelspecTransform` class for
both ATST families. Its alignment is owned by
[`atst_clip-alignment.md`](atst_clip-alignment.md) and by
`tests/models/test_atst_clip_alignment.py`; this document and
`tests/models/test_atst_frame_alignment.py` do not repeat it.

## Alignment Target

Verification covers the following three categories of properties:

1. Complete and accurate loading of the two official ATST-Frame
   checkpoints: both are PyTorch-Lightning archives, and the normalized
   138-key state dict is loaded into the *official* `FrameAST` with
   `strict=True`, so the key normalization itself is asserted before any
   number is compared;
2. Alignment of both granularities against the official
   `FrameAST.get_intermediate_layers`: frame granularity against
   `scene=False`, clip granularity against `scene=True`, at `n = 12`;
3. The cross-chunk assembly rule, which differs by granularity
   (concatenation along time for frame, equal-weight average for clip),
   plus equivalence between a mixed-length batch and per-sample calls.

"Official alignment" does not cover:

- The shared log-mel frontend (owned by the ATST-Clip alignment module,
  which runs the full duration × signal matrix against the official
  `MelSpectrogram` → `AmplitudeToDB` → `MinMax` chain);
- The official 1001-frame chunk grid in
  `audiossl/methods/atstframe/embedding.py`; this repository chunks at
  1000 frames on purpose, and the reason is recorded under
  [Chunk Assembly Reference](#chunk-assembly-reference);
- The downstream `PretrainedEncoderPLModule` clip rule, which discards a
  trailing chunk shorter than half a chunk; the local encoder follows
  `get_scene_embedding` instead and weights every chunk equally;
- The additive attention mask the official `Attention` builds from
  `length`: callers group by valid length and slice to the exact prefix,
  so that mask is uniformly zero and is omitted locally;
- The pretraining path: masking, `mask_embed`, `random_mask.get_mask`,
  the data2vec block-averaging teacher, and the fairseq mask sampler;
- `FrameAST_large`, `nprompt > 0`, `pos_type != "cut"`, and the CNN
  `PatchEmbed`; no released checkpoint reaches any of them;
- Training-mode behavior (DropPath, dropout sampling);
- Pointwise equivalence between torchaudio resampling or downmixing and
  the official data-loading pipeline;
- The ATST-Clip family, which
  [`atst_clip-alignment.md`](atst_clip-alignment.md) governs.

The alignment boundary starts from the `input_features` tensor produced
by `AtstMelspecTransform`: both sides consume the identical tensor, the
official side after `transpose(1, 2).unsqueeze(1)` into its
`[B, 1, 64, T]` layout.

## Pinned Official Identity

### audiossl Source Code

Official repository and pinned revision:

```text
https://github.com/Audio-WestlakeU/audiossl
ec3a14db086eaccfb69513e4a90fad89bf992e1f    (main, 2025-09-25)
```

The reference side fetches only the two subtrees it imports, via
`--filter=blob:none` + sparse-checkout:

```text
audiossl/methods/atstframe
audiossl/modules
```

into:

```text
$TMPDIR/timbral-atst-alignment/audiossl-frame/
```

This is a separate working tree from the ATST-Clip module's
`$TMPDIR/timbral-atst-alignment/audiossl/`, because the two modules
sparse-check-out different subtrees of the same commit. Neither is
cached into the project directory, the user's home directory,
`/projappl`, or `/scratch`.

Each run re-verifies `git rev-parse HEAD` against the pinned commit and
the SHA-256 of every file the reference path imports:

| File | SHA-256 |
|---|---|
| `audiossl/__init__.py` | `c3edaf9c65e45b430aa06d4617abfb1b01b24c5c09f3a7e050eef021386b0d0b` |
| `audiossl/methods/atstframe/audio_transformer.py` | `33554493407306db8efdba41082c58770d021f5c9bf3e8a3c465cf1850e719bc` |
| `audiossl/methods/atstframe/random_mask.py` | `ff6c335138e80578f2f5085c6c5066f19f8d6a592795691d68d9aa2f76bb6507` |
| `audiossl/modules/transformer.py` | `0048be36e211bfb00c177331a662b455e644a950424dc58406fd072a9f718297` |

If any pinned file is missing or its digest does not match, the tree is
removed and re-cloned; there is no partial reuse. A `HEAD` that does not
match the pinned commit fails the run outright.

`audiossl/methods` and `audiossl/methods/atstframe` ship **without**
`__init__.py` and resolve as PEP 420 namespace packages, as does
`audiossl/modules`; they carry no file to pin. The official modules use
absolute intra-package imports (`from audiossl.modules.transformer
import Block`), so the pinned checkout is placed on `sys.path` and
imported as a normal package rather than loaded file by file; `sys.path`
and every pre-existing `audiossl*` entry of `sys.modules` are restored
in a `finally` block. No package is installed, and the official source
is never modified.

Two consequences of importing the pinned source are recorded rather than
worked around:

- `audiossl/__init__.py` sets `MKL_NUM_THREADS` and `OMP_NUM_THREADS` to
  `"1"` as an import side effect; the file is pinned by digest and is
  left exactly as published;
- the pinned `audio_transformer.py` imports
  `einops.layers.torch.Rearrange` at module scope, so an explicit
  alignment run requires einops in the environment (0.8.2 in the project
  container). This is a reference-side requirement only: `pyproject.toml`
  declares no einops, and the local `_AtstPatchEmbed` reproduces the same
  rearrangement with `view`/`transpose`/`reshape`.

### The fairseq Stub

`audio_transformer.py` imports `random_mask` at module scope, and the
first line of `random_mask.py` is:

```python
from fairseq.data.data_utils import compute_mask_indices
```

So importing the official `FrameAST` pulls in fairseq before any model
object exists. fairseq is unmaintained and does not install against the
torch version this project pins, and the symbol it provides is
pretraining-only: `compute_mask_indices` samples the frame-mask pattern
for the pretraining objective and is unreachable from the inference
path.

The reference fixture therefore injects three stub modules into
`sys.modules` before importing the official module:

```text
fairseq
fairseq.data
fairseq.data.data_utils
```

`fairseq.data.data_utils.compute_mask_indices` is a function whose body
raises `AssertionError`.

A `sys.modules` stub is the correct instrument here, rather than
deleting or rewriting the offending import in the checked-out file:

- the pinned files stay **byte-identical** and therefore remain
  SHA-256 verifiable against the table above; a source patch would
  invalidate the digest that is the whole basis of the pinned identity,
  and the test could no longer prove which code produced the reference;
- the substitution is scoped and reversible: the stubs are installed in
  the module-scoped fixture and removed in the same `finally` block that
  restores `sys.path` and the saved `audiossl*` modules, so no other test
  in the session sees a fake fairseq;
- it substitutes only the *unavailable* dependency. einops, which the
  same file also imports, is installable and is therefore used for real
  rather than stubbed.

The raising body is deliberate: it turns the stub into an assertion. If
any inference-path change ever reached the mask sampler, the alignment
run would fail loudly at that call instead of silently exercising a
pretraining branch. Every passing run is evidence that the path from
`get_intermediate_layers` down never touches `compute_mask_indices`.

### Checkpoints

The two ATST-Frame entries, identical in identity to the runtime table
in `helpers/atst.py` (the tests reuse `ATST_CHECKPOINTS` value for
value):

| Entry | File | Bytes | Archive | SHA-256 |
|---|---|---:|---|---|
| `atst-frame-small` | `atst_frame_small.ckpt` | 411,264,805 | Lightning | `1d85b290632dd26b8725f0ae73f53f990a898888cfc2c4794c3055a8130ff5f1` |
| `atst-frame-base` | `atst_frame_base.ckpt` | 1,459,523,813 | Lightning | `9f812544983add849f45ef03c4ce10184729adee082c0bdd347764f4835bb3da` |

Both are published as Google Drive files (`confirm=t` direct links) and
are resolved through the same runtime path the encoder uses:

```text
explicit pretrained_dir > HF_HUB_CACHE/audioencoders/atst/<filename>
```

`ensure_atst_checkpoint` verifies the fixed digest, and downloads to a
temporary file inside the target directory that is moved into place only
after the digest matches, so an interrupted download never leaves a file
a later run would accept. An explicit alignment run therefore requires
`HF_HUB_CACHE` to point at a valid hub directory (consistent with the
practice used for the AST/PANNs/BEATs alignment runs).

Both files are read with `torch.load(weights_only=True)` under the
minimal allowlist (`numpy.core.multiarray.scalar`, `numpy.dtype`,
`numpy.dtypes.Float64DType`); unlike `atst-clip-base`, neither frame
archive needs `argparse.Namespace`, because both use the Lightning
layout. Only the teacher encoder subtree
(`state_dict["model.teacher.encoder.*"]`) is kept.

After normalization the state dict holds exactly 138 keys: 12 blocks ×
11 keys, plus `pos_embed`, `mask_embed`, `norm_frame.{weight,bias}`, and
`patch_embed.patch_embed.{weight,bias}`. There is no `cls_token`, and
the final norm is named `norm_frame` rather than the clip family's
`norm`; `cls_token` is the one extra key the clip family carries, which
is why the clip inventory is 139.

## Reference Pipeline

### Frontend

Not asserted in this module. Both families instantiate the same
`AtstMelspecTransform`, whose full matrix (6 durations × 5 signal kinds,
CPU and CUDA) runs under `--run-alignment atst_clip`. The frame module
consumes `input_features`, `valid_feature_frames`, and `valid_seconds`
unchanged and states nothing about how they were produced; repeating the
matrix here would only re-measure the same class.

### Encoder Reference

For each arch, the official model is built from the pinned factory
(`FrameAST_small` / `FrameAST_base`, i.e. `patch_h=64`, `patch_w=4`,
`depth=12`, `qkv_bias=False`, `norm_layer=LayerNorm(eps=1e-6)`) and
loaded with the state dict produced by the local
`load_atst_encoder_state`, under `strict=True`.

Loading the *local* normalization into the *official* class is
deliberate: it makes the official module the arbiter of the key set. Any
key the local helper invents, drops, or misnames fails the load before a
single number is compared, so the checkpoint contract needs no separate
key-by-key table.

One chunk of features is then run through:

```python
mel = features.transpose(1, 2).unsqueeze(1)   # [B, T, 64] -> [B, 1, 64, T]
length = torch.full((B,), T, dtype=torch.long)
model.get_intermediate_layers(mel, length, n=12, scene=scene)
```

- `scene=False` returns the frame reference `[B, T // 4, 12 * D]`: each
  selected block output passes through `norm_frame`, and the selected
  blocks are concatenated on the last axis;
- `scene=True` returns the clip reference `[B, 12 * D]`: each block's
  patch tokens are summed under the length mask and divided by
  `patch_length + 1e-6`, the official epsilon the local encoder
  reproduces verbatim. This family has no cls token, so there is no
  second branch and the width is `n_blocks * D`, not `2 * n_blocks * D`;
- `length` is set to the exact frame count of the tensor handed in, so
  the official length mask is all-true and the official additive
  attention mask is uniformly zero. That is precisely the configuration
  under which omitting the mask locally is exact, and the bit-exact CPU
  results below are the evidence;
- `n` is pinned to 12, the maximal setting, so every block output takes
  part in the comparison. A smaller `n_blocks` selects the trailing
  `n_blocks * D` columns of the same concatenation on both sides, so it
  needs no separate case.

Both the local encoder and the official model are put into `eval()`; the
whole matrix runs under `torch.inference_mode()` with TF32 disabled. The
derived reference must come from this pinned official object and the
same checkpoint; `AtstFrameEncoder` must not be used as its own
reference.

### Chunk Assembly Reference

The 1000-mel-frame chunk grid is this repository's own decision, so no
single official call produces a multi-chunk result. The reference is
assembled in the test from official single-chunk calls, over the same
bounds the encoder uses:

```python
for start in range(0, num_frames, 1000):
    end = min(start + 1000, num_frames)
    if (end - start) >= 4:
        bounds.append((start, end))
```

The assembly rule differs by granularity, and each half mirrors one
official entry point in `audiossl/methods/atstframe/embedding.py`:

- **frame**: `torch.cat(chunk_outputs, dim=1)`, as in
  `get_timestamp_embedding`, which concatenates per-chunk token
  sequences along time and stamps them at 40 ms intervals. The test
  additionally asserts the assembled length equals
  `valid_feature_frames // 4`;
- **clip**: `torch.stack(chunk_outputs, dim=0).mean(dim=0)`, as in
  `get_scene_embedding`, which averages the per-chunk vectors with equal
  weight regardless of how long the trailing chunk is.

The deliberate deviation is the chunk width alone. The official chunk is
1001 frames, of which `PatchEmbed_v2` uses 1000 and drops one; every
subsequent chunk therefore starts one frame (10 ms) later than the
previous chunk's patch grid, and the loss accumulates (30 s yields 749
patches instead of 750). The 1000-frame grid keeps every patch on the
global 40 ms grid and makes `total patches == valid_feature_frames // 4`
hold exactly, because the chunk length is itself a multiple of the patch
width. Since the deviation is in where the boundary falls, alignment is
stated per chunk plus per assembly rule, and not against `embedding.py`
end to end.

A trailing chunk shorter than one patch (fewer than 4 frames)
contributes nothing and is skipped on both sides.

## Test Matrix

### Durations

```text
single chunk : 1.0, 10.0 seconds
multi chunk  : 10.05, 21.0 seconds
mixed batch  : 0.5, 6.0, 10.0 seconds
```

Meaning:

- 1.0: 101 mel frames = 25 patches with a 1-frame remainder shorter than
  a patch, dropped by both sides;
- 10.0: 1001 frames, which fills one chunk exactly (250 patches, the
  full `pos_embed` capacity) and leaves a 1-frame tail that both sides
  drop;
- 10.05: 1006 frames, the shortest case in this matrix that crosses the
  seam: chunk two holds 6 frames, i.e. exactly one patch plus a dropped
  2-frame remainder;
- 21.0: 2101 frames = three chunks (1000 / 1000 / 101), so a middle
  chunk exists and the last one is neither full nor degenerate;
- the mixed batch spans a sub-chunk sample (51 frames), a mid-length one
  (601 frames), and an exactly-one-chunk one (1001 frames), giving three
  distinct valid lengths and therefore three distinct groups.

### Signals

Deterministic synthetic signals whose seed is derived from the case
duration:

- random Gaussian noise (scaled by 0.5) for the single-chunk and
  mixed-batch cases;
- a stationary multi-frequency sine (440 / 1237 / 3313 Hz) for the
  multi-chunk cases, so that every chunk carries comparable content and
  a boundary error cannot hide behind an amplitude difference between
  chunks.

The entry-independent signal kinds (997 Hz sine, impulse, silence) are
frontend properties and are exercised by the ATST-Clip module.

### Batch

- B=1 for the single-chunk and multi-chunk cases;
- one mixed batch `[0.5, 6.0, 10.0]` seconds: each sample is
  zero-padded on the right to the longest length, and `valid_seconds`
  declares its true extent, so the transform and the encoder each group
  by valid length and no padded frame enters any computation;
- every sample of the mixed batch must match the result of calling the
  transform and the official reference on that one sample alone.

### Device

- CPU is mandatory; CUDA is mandatory when available, and explicitly
  recorded as skipped when unavailable (`cuda_tested` in the summary
  JSON);
- TF32 is disabled for both matmul and cuDNN
  (`torch.backends.cuda.matmul.allow_tf32 = False`,
  `torch.backends.cudnn.allow_tf32 = False`) and restored on context
  exit;
- Local and official run on the same device; no gate is established
  between CPU and CUDA;
- MPS is not required.

### Coverage Allocation

| Case | Archs | Granularities | Durations | Items |
|---|---|---|---|---:|
| Single chunk | small, base | frame, clip | 1.0, 10.0 | 4 |
| Multi chunk | small | frame, clip | 10.05, 21.0 | 2 |
| Mixed batch | small | frame | 0.5 / 6.0 / 10.0 | 1 |

Seven pytest items in total; each item loops over the available devices
internally. `small` carries the chunking and batching matrix while
`base` is covered at both granularities on the single-chunk matrix:
width and head count are the only differences between the two archs, and
both checkpoints load through the identical Lightning path.

## Layered Assertions

### Frontend

Nothing is asserted in this module; see
[`atst_clip-alignment.md`](atst_clip-alignment.md).

### Checkpoint

- Both frame checkpoints are resolved and SHA-256 verified through the
  runtime `ensure_atst_checkpoint`;
- the normalized state dict loads into the official `FrameAST` with
  `strict=True`, and into `AtstFrameEncoder` with `strict=True`, for
  both archs;
- all reads use `weights_only=True`, with no fallback.

### Encoder

- Shapes: frame `[B, P, 12 * D]` with `P == valid_feature_frames // 4`;
  clip `[B, 12 * D]`; the local shape must equal the reference shape
  before values are compared;
- pointwise gates plus the float64 audit against the official output on
  the same device;
- neither side may contain NaN or Inf;
- in the mixed batch, frame `valid_mask` sums to the official token
  count for each sample, and every frame row past the valid region is
  exactly zero, asserted with `torch.equal` against zeros (no
  tolerance).

The base-class contract for this family — frame geometry, `valid_mask`,
zero-filled invalid slots, chunk bookkeeping, `n_blocks` selection — is
covered by the weight-free tests in
`tests/models/encoders/test_atst_frame.py` and is not restated here.

### Chunking and Batch

- The multi-chunk cases assert that more than one chunk actually
  participates, so a regression that silently truncates to one chunk
  cannot pass;
- frame assembly asserts the concatenated token count equals
  `valid_feature_frames // 4`;
- clip assembly is the equal-weight mean of the per-chunk references;
- each mixed-batch sample matches its own standalone call.

## Numerical Gates

Pointwise local/official comparisons on the same device
(`torch.allclose`) are split by device:

| Stage | CPU | CUDA |
|---|---|---|
| Encoder (frame and clip) | `atol=1e-4, rtol=1e-4` | `atol=2e-3, rtol=1e-4` |

float64 audit gates:

- relative-L2: CPU ≤ 1e-4, CUDA ≤ 1e-3;
- cosine ≥ 0.99999 on both devices;
- the max absolute difference is recorded per bucket in the summary
  JSON;
- NaN and positive/negative Inf are rejected on both sides.

These are the same numbers the ATST-Clip contract uses, and they are far
looser than the differences measured below. The reason for keeping the
CUDA gate at that level is the same in both families: under the tested
configuration the two sides differ only by operations that are exact
identities — an all-zero additive attention mask, an all-true length
mask applied before pooling, and a different reshape route into the
patch `Linear` (the official einops rearrange versus a local
`view`/`transpose`/`reshape`). Those identities still change kernel
selection and reduction order on the GPU, and the gate absorbs the
residual float32 difference rather than pinning a particular kernel
choice. Both sides also consume the identical `input_features` tensor
here, so no frontend cancellation can be amplified through the
backbone.

Weights and any state declared to be loaded value for value use
`strict=True` loading rather than a tolerance.

## Pytest Entry Points

Ordinary tests:

```bash
python -m pytest tests/models -v
```

By default this does not fetch the official source code, does not read
the large checkpoints, and does not run the matrix.

Explicit ATST-Frame alignment:

```bash
python -m pytest \
  tests/models --run-alignment atst_frame -v
```

Both ATST entries can be enabled in one run:

```bash
python -m pytest \
  tests/models --run-alignment atst_clip atst_frame -v
```

The allowed set in `tests/models/conftest.py` is:

```python
("panns", "ast", "clap", "beats", "wav2vec2", "atst_clip", "atst_frame")
```

The test file is `tests/models/test_atst_frame_alignment.py`, with the
module-level `pytestmark = pytest.mark.alignment("atst_frame")`, and it
collects seven tests:

```text
test_encoder_alignment_single_chunk[frame-small]
test_encoder_alignment_single_chunk[frame-base]
test_encoder_alignment_single_chunk[clip-small]
test_encoder_alignment_single_chunk[clip-base]
test_encoder_alignment_multi_chunk[frame]
test_encoder_alignment_multi_chunk[clip]
test_mixed_batch_matches_single_calls
```

An explicit run needs network access for the sparse clone (first run
only) and an `HF_HUB_CACHE` that holds, or can receive, the two frame
checkpoints.

## Temporary Files and Caching

- Official source code: `$TMPDIR/timbral-atst-alignment/audiossl-frame/`,
  reused across runs while HEAD and all four digests still match;
- Result summary:
  `$TMPDIR/timbral-atst-alignment/atst-frame-alignment-summary.json`,
  holding the pinned commit, the four file digests, the per-bucket worst
  cases, the torch version, and `cuda_tested`; not committed to Git;
- Checkpoints live in the user's hub directory
  (`HF_HUB_CACHE/audioencoders/atst/`) and are downloaded there only if
  absent; nothing else is written outside `$TMPDIR`, and the tests
  produce no report or cache inside the repository, the current
  directory, the user's home directory, or `/projappl`.

## Acceptance Criteria

The ATST-Frame migration is considered complete only once all of the
following conditions hold simultaneously:

1. All weight-free ATST-Frame model tests pass;
2. Both frame checkpoints pass SHA-256 verification and load with
   `strict=True` into both the official `FrameAST` and
   `AtstFrameEncoder`;
3. All 7 alignment items pass on CPU, and likewise on CUDA when
   available;
4. Frame granularity is aligned against `get_intermediate_layers(...,
   scene=False)` and clip granularity against `scene=True`, for both
   archs;
5. Multi-chunk assembly matches concatenation (frame) and equal-weight
   average (clip), with more than one chunk actually participating;
6. Mixed-batch results match their per-sample counterparts, and the
   invalid frame region is exactly zero;
7. The four official files are verified byte-identical before import,
   the pinned tree is left unmodified, and the fairseq stub's raising
   body is never reached;
8. No new runtime dependency (fairseq, einops) is introduced by the
   local implementation;
9. This document has a genuine, traceable results summary appended.

Do not pre-fill "passed" before the tests are actually run.

## Empirical Results (2026-08-16)

### Code and Environment

- Repository baseline commit: `21f4581` (the ATST migration is in the
  current working tree on top of this commit);
- Python: 3.12.13;
- pytest: 9.1.1;
- PyTorch: 2.11.0+cu130;
- torchaudio: 2.11.0+cu130;
- CUDA runtime: 13.0;
- cuDNN: 9.19.0;
- GPU: NVIDIA GH200 120GB;
- NumPy: 2.4.6;
- einops (reference side only): 0.8.2.

The official source was sparse-cloned at the pinned commit
`ec3a14db086eaccfb69513e4a90fad89bf992e1f`; the SHA-256 digests of the
four imported files match the "Pinned Official Identity" table value for
value. Both frame checkpoints pass `ensure_atst_checkpoint` verification
at 411,264,805 and 1,459,523,813 bytes, and both normalized state dicts
(138 keys) were accepted by the official `FrameAST` under `strict=True`.

### Execution Matrix

- Single chunk: 2 archs × 2 granularities × 2 durations = 8 audited
  comparisons per device;
- Multi chunk: 2 granularities × 2 durations = 4 per device, with 2
  chunks at 10.05 s and 3 chunks at 21.0 s;
- Mixed batch: 3 samples = 3 per device;
- 15 audited comparisons per device, 30 in total; CUDA was actually
  executed (`cuda_tested: true` in the summary JSON).

Test entry point:

```bash
python -m pytest \
  tests/models --run-alignment atst_frame -v
```

All 7 alignment items in `tests/models/test_atst_frame_alignment.py`
passed, on CPU and CUDA.

### Numerical Results

Worst-case values against the pinned official reference on the same
device (float64 audit), taken from the per-bucket summary:

| Case | Granularity | Arch | Device | Max Abs Diff | relative-L2 | Min Cosine |
|---|---|---|---|---:|---:|---:|
| Single chunk | frame | small | CPU | 0 | 0 | 0.999999999999842 |
| Single chunk | frame | small | CUDA | 0 | 0 | 0.999999999999871 |
| Single chunk | frame | base | CPU | 0 | 0 | 0.999999999999799 |
| Single chunk | frame | base | CUDA | 0 | 0 | 0.999999999999786 |
| Single chunk | clip | small | CPU | 0 | 0 | 0.999999999999997 |
| Single chunk | clip | small | CUDA | 9.54e-7 | 7.33e-8 | 0.999999999999997 |
| Single chunk | clip | base | CPU | 0 | 0 | 0.999999999999995 |
| Single chunk | clip | base | CUDA | 9.54e-7 | 6.65e-8 | 0.999999999999994 |
| Multi chunk | frame | small | CPU | 0 | 0 | 0.999999999999774 |
| Multi chunk | frame | small | CUDA | 0 | 0 | 0.999999999999771 |
| Multi chunk | clip | small | CPU | 0 | 0 | 0.999999999999996 |
| Multi chunk | clip | small | CUDA | 9.54e-7 | 7.85e-8 | 0.999999999999997 |
| Mixed batch | frame | small | CPU | 0 | 0 | 0.999999999999848 |
| Mixed batch | frame | small | CUDA | 0 | 0 | 0.999999999999874 |

Frame granularity is **bit-exact on both CPU and CUDA**: every frame
bucket reports a max absolute difference of exactly 0, for both archs,
within a single chunk, across chunk seams, and inside the mixed batch.
The cosine values below 1.0 at zero pointwise difference are float64
audit rounding, not a discrepancy.

Clip granularity is bit-exact on CPU and differs by at most 9.54e-7 on
CUDA (relative-L2 ≤ 7.9e-8). The difference is confined to the pooling
reduction: the official `scene=True` branch multiplies the normed tokens
by an all-true length mask before summing over the token axis, while the
local encoder sums them directly. The two expressions are arithmetically
identical and agree exactly on CPU; on CUDA the masked product and the
direct sum reduce in a different order, which leaves a residual of a few
float32 ULP. Every measured value sits at least three orders of
magnitude inside its gate.

### Specific Contracts

- The fairseq stub was installed for every official import, and its
  raising body was never reached in any of the 7 items, so the inference
  path demonstrably never calls `compute_mask_indices`;
- the four official files were verified byte-identical before import and
  the pinned tree was left unmodified; `sys.path` and the `audiossl*` /
  `fairseq*` entries of `sys.modules` were restored afterwards;
- the official `FrameAST` accepted the locally normalized 138-key state
  dict with `strict=True` for both archs, which is the checkpoint-key
  contract in its strongest form;
- multi-chunk assembly used 2 chunks at 10.05 s and 3 chunks at 21.0 s;
  the frame reference token count equalled `valid_feature_frames // 4`
  in both cases (251 and 525 patches), and the clip reference was the
  equal-weight mean of the same per-chunk calls;
- in the mixed batch, each sample's valid prefix matched its standalone
  official result on both devices, `valid_mask` summed to the official
  token count, and every row past the valid region was exactly zero
  under `torch.equal`;
- TF32 stayed disabled across the whole matrix and was restored on exit;
- fairseq was neither installed nor executed, einops was used only by
  the reference side, and no new runtime dependency was added; MPS was
  not tested.

The machine-readable temporary summary for this run is located at:

```text
$TMPDIR/timbral-atst-alignment/atst-frame-alignment-summary.json
```

This file is not committed to Git.
