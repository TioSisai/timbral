# ATST-Clip Official Implementation Alignment Contract

This document freezes the official implementation alignment contract for
`AtstMelspecTransform` and `AtstClipEncoder`, and records the verification
summary of the completed run. The log-mel frontend is one and the same
class for both ATST families, so it is pinned here once and is not
repeated by the ATST-Frame contract.

Corresponding designs:

- [`../transforms/atst.md`](../transforms/atst.md)
- [`../encoders/atst_clip.md`](../encoders/atst_clip.md)
- [`atst_frame-alignment.md`](atst_frame-alignment.md)

## Alignment Target

Verification covers the following three categories of properties:

1. Numerical equivalence between the local batched log-mel frontend
   (including the 513-sample zero-padding floor and the per-sample
   `top_db` floor) and the official
   `MelSpectrogram` → `AmplitudeToDB` → `MinMax` chain, driven exactly as
   the official entry point drives it;
2. Normalization of both official archive layouts into a key set the
   official `AST` module accepts with `strict=True` — `atst-clip-small` is
   a PyTorch-Lightning checkpoint, `atst-clip-base` is the earlier
   DINO-style dict, and both are exercised;
3. Alignment of the `AtstClipEncoder` clip embedding against the official
   `get_intermediate_layers_chunks(..., avgpool=True)` at chunk
   granularity, for both architectures, plus the cross-chunk combination
   and mixed-length batches.

"Official alignment" does not cover:

- A whole-clip comparison against the official `get_scene_embedding`
  beyond the first chunk: the official chunk grid is 1001 mel frames
  while this repository's is 1000, so from the second chunk onward the
  two sides embed different audio, not different arithmetic (see
  "Encoder Reference" for the resulting chunk-wise strategy, and
  [`../encoders/atst_clip.md`](../encoders/atst_clip.md) for why the
  deviation is deliberate);
- The official downstream `PretrainedEncoderPLModule` chunk weighting,
  which discards a trailing chunk shorter than half a chunk; this
  repository averages every chunk with equal weight, matching
  `get_scene_embedding` instead;
- The official additive attention mask and its padded-batch path: the
  local side groups by valid length and slices to the exact prefix, so
  that mask is uniformly zero;
- Frame granularity, which `AtstClipEncoder` does not expose (the
  official family has no frame-level usage for ATST-Clip);
- The branches the released checkpoints cannot reach and that are
  therefore not ported: masking and `mask_embed` use, positional
  interpolation (`pos_type != "cut"`), and the teacher's block-averaging
  path (`AST.forward(avg=True)`, the mean of the last 8 block outputs).
  Prompt tokens (`nprompt > 0`) and the CNN patch embedding are options
  of the sibling `FrameAST` only; the official `AST` does not take them
  as parameters at all;
- Training-mode behavior (DropPath, dropout);
- Pointwise equivalence between torchaudio downmixing/resampling and the
  official data-loading pipeline;
- The ATST-Frame family, which
  [`atst_frame-alignment.md`](atst_frame-alignment.md) governs.

The core official alignment boundary starts from a 16 kHz mono float32
waveform. Common downmixing, invalid-region clearing, and resampling are
tested separately as contract tests.

## Pinned Official Identity

### audiossl Source Code

Official repository and pinned revision:

```text
https://github.com/Audio-WestlakeU/audiossl
ec3a14db086eaccfb69513e4a90fad89bf992e1f    (main, 2025-09-25)
```

The reference side fetches only the three directories it imports
(`audiossl/models`, `audiossl/modules`, `audiossl/transforms`) via
`--filter=blob:none` + sparse-checkout (`git init`, `remote add`,
`sparse-checkout init --cone`, `sparse-checkout set`, `fetch --depth 1`,
`checkout --detach FETCH_HEAD`) into:

```text
$TMPDIR/timbral-atst-alignment/audiossl/
```

It must not be cached into the project directory, the user's home
directory, `/projappl`, or `/scratch`.

The reference side must verify the actual Git commit (`git rev-parse
HEAD` equals the pinned revision) and the SHA-256 of every source file
the reference path imports:

| File | SHA-256 |
|---|---|
| `audiossl/__init__.py` | `c3edaf9c65e45b430aa06d4617abfb1b01b24c5c09f3a7e050eef021386b0d0b` |
| `audiossl/models/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `audiossl/models/atst/__init__.py` | `a825bbc89d2806d15fecb91d43f45a82946018c7b55b74c2fa0e3e19ff0e4388` |
| `audiossl/models/atst/atst.py` | `c5e1bd0a622d7feb0cc5f6f7e8f3b7427d7033d4133e7aeec14dc9f28e186c59` |
| `audiossl/models/atst/byol.py` | `3060ccef3662f5bf02d50b3048198ceb909c9057cd3007b42561e0f19d33b423` |
| `audiossl/models/atst/audio_transformer.py` | `3ea4694fedd2d7ea1fb125440a81f5e41a55f680a2f4a1ceb0237b32542fbb16` |
| `audiossl/modules/transformer.py` | `0048be36e211bfb00c177331a662b455e644a950424dc58406fd072a9f718297` |
| `audiossl/transforms/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `audiossl/transforms/common.py` | `9f99a260fef70da529455fec4b12f0fffd297dab960ceec0e36148bc22ab49a1` |

Two of those entries are empty `__init__.py` markers (the digest of the
empty file); they are pinned all the same, because their presence is what
makes the surrounding directory a regular package. `audiossl/modules`
ships **without** an `__init__.py` and resolves as a PEP 420 namespace
package, so it carries no file to pin; the ATST-Frame contract records the
same situation for `audiossl/methods` and `audiossl/methods/atstframe`.

`atst.py` and `byol.py` are pinned although the reference never calls
them: `audiossl/models/atst/__init__.py` runs `from .atst import ATST`,
so importing anything below that package executes them.

The pinned tree is imported as a **package**, not file-by-file. The
official modules use absolute intra-package imports — `audio_transformer`
opens with `from audiossl.modules.transformer import Block` — so the
BEATs technique of loading each file through `importlib` from a
`sys.path`-injected directory cannot resolve them; the checkout root is
prepended to `sys.path` instead, and
`importlib.import_module("audiossl.models.atst.audio_transformer")` runs
the normal import machinery. Any pre-existing `audiossl*` entry in
`sys.modules` is saved and removed before the import and restored in a
`finally` block, together with the `sys.path` entry, so the pinned tree
never leaks into the rest of the session. Nothing is installed and no
byte of the official source is modified.

The checkout is reused across runs: it is rebuilt from scratch (after
`shutil.rmtree`) only when any pinned file is missing or its digest does
not match.

### Checkpoints

Only the two ATST-Clip entries are exercised here. Their identity is
governed solely by `helpers.ATST_CHECKPOINTS`, which the alignment module
reuses value-for-value instead of re-declaring digests:

| Entry | File | Bytes | Archive | SHA-256 |
|---|---|---:|---|---|
| `atst-clip-small` | `atst_clip_small.ckpt` | 411,267,681 | Lightning | `fcadd6411881410d27cde47f4d540ef416aa59e0197b195cf3ee7a81885a5f4a` |
| `atst-clip-base` | `atst_clip_base.ckpt` | 1,459,530,207 | DINO-style | `7b20168cae0d1488a0e3334f17ca1cefb9365cbaa2401c11aa98d6ffaa668496` |

Both are published under
`https://checkpointstorage.oss-cn-beijing.aliyuncs.com/atst/`
(`small.ckpt` and `base.ckpt`).

The test weight directory is the same as the runtime default resolution
result:

```text
explicit pretrained_dir > HF_HUB_CACHE/audioencoders/atst/<file>.ckpt
```

The alignment module resolves weights through the same runtime helper the
Encoders use, `ensure_atst_checkpoint(metadata, None)`: an existing file
is verified against its fixed SHA-256, and a missing file is downloaded
to a temporary file in the target directory and moved into place only
after the digest matches. An explicit alignment run therefore requires
`HF_HUB_CACHE` to point to a valid hub directory (consistent with the
existing practice used for the AST/PANNs/BEATs alignment runs).

## Reference Pipeline

### Transform Reference

The official frontend chain is rebuilt from the pinned source and the
official constants:

```python
melspec   = torchaudio.transforms.MelSpectrogram(
    16000, n_fft=1024, win_length=1024, hop_length=160,
    f_min=60, f_max=7800, n_mels=64)
to_db     = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
normalize = audiossl.transforms.common.MinMax(min=-79.6482, max=50.6842)
```

`MinMax` is imported from the pinned checkout, and its two bounds are
read from `helpers.atst`, so the local and reference sides cannot drift
apart.
The `official_frontend` fixture depends on the `official_audiossl`
fixture so that the pinned checkout is still on `sys.path` when
`audiossl.transforms.common` is imported.

- The reference is fed `[B, 1, N]`, which is the rank the official entry
  point produces. This is not cosmetic: `AmplitudeToDB`'s `top_db`
  reduction range depends on the input rank, and only the extra channel
  axis makes the floor reduce per sample instead of once for the whole
  batch. The local transform spells the dB stage out by hand for exactly
  this reason;
- Waveforms shorter than 513 samples (`n_fft // 2 + 1`, the shortest
  input `torch.stft`'s reflect padding accepts) are zero-padded to 513
  before being handed to the reference — minimum-length padding is a
  waveform-domain operation, so the official chain can recompute it
  pointwise on the same waveform, and the branch is inside the alignment
  contract;
- The official output is `[1, 64, T]`; it is transposed to `[T, 64]` and
  compared against the local time-major `input_features[0, :frames]`,
  where `frames` is the local `valid_feature_frames`. The reference
  width is asserted to equal `frames`, which pins the frame-count
  formula `valid_samples // 160 + 1` against the official chain rather
  than against a local restatement of it.

### Encoder Reference

For each architecture, the official `AST` is constructed from the pinned
factories and loaded with the locally normalized state:

```python
model = module.AST_small()          # or module.AST_base()
path  = ensure_atst_checkpoint(metadata, None)
state = load_atst_encoder_state(metadata, path)
model.load_state_dict(state, strict=True)
model.to(device).eval()
```

- The factories are called with their defaults: `use_cls=True`,
  `spec_h=64`, `spec_w=1001`, `patch_h=64`, `patch_w=4`,
  `qkv_bias=False`, `LayerNorm(eps=1e-6)`, `pos_type="cut"`. Only the
  DINO archive records part of that configuration, in its `["args"]`
  (`pos_type="cut"`, `patch_height=64`, `patch_width=4`,
  `use_cls=True`); the Lightning archive's `hyper_parameters` holds
  `arch` plus training settings only, so for `atst-clip-small` the
  defaults are asserted by the `strict=True` load itself, which the
  checkpoint's tensor shapes have to agree with. `spec_w=1001` is why
  the official `pos_embed` is `[1, 251, D]`, i.e. 250 patch slots plus
  the cls slot;
- Loading the local normalization into the official module with
  `strict=True` is the checkpoint assertion itself: it holds only if the
  normalized key set equals the official one in both directions, for
  both archive layouts;
- The input is `features.transpose(1, 2).unsqueeze(1)`, i.e. the local
  time-major `[B, T, 64]` features presented as the official
  `[B, 1, 64, T]`, and `length` is the full frame count for every sample,
  so no padding enters the official computation either;
- The reference call is
  `get_intermediate_layers_chunks(mel, length, n=12, chunk_len=1001,
  avgpool=True)`. With `avgpool=True` the official routine returns
  `cat(cls_per_block + avg_per_block, dim=-1)`, i.e. the same
  `2 * n_blocks * D` layout the local encoder produces, and the official
  pooling divides by `patch_count + 1e-6`, which the local side
  reproduces verbatim;
- Every case uses `n_blocks = 12`, the widest configuration (all twelve
  block outputs concatenated: 9216 for small, 18432 for base);
- The local `AtstClipEncoder` must never be used as the reference.

**The comparison is chunk-wise, and this is a contract requirement, not a
convenience.** `chunk_len` is set to 1001 = `ATST_CHUNK_FRAMES + 1`, one
frame above the local chunk width and exactly the official 10 s grid
(`spec_w = 1001`, the width the 251-slot `pos_embed` covers); because
every tensor handed to the official routine holds at most 1001 mel
frames — one local chunk plus at most the sub-patch remainder — it
produces exactly one non-empty chunk. So what each comparison
establishes is that **one local chunk forward pass equals one official
chunk forward pass**, and the test assembles the cross-chunk combination
itself (stack the per-chunk official results and take the equal-weight
mean, exactly what the local encoder does).
Comparing whole clips against the official `get_scene_embedding` would be
wrong beyond 10 s: the official grid advances 1001 frames per chunk and
drops one frame per chunk, shifting every later chunk's patch grid by
10 ms, whereas this repository advances 1000 frames so that every patch
stays on the global 40 ms grid and `total_patches == frames // 4` holds
exactly. Beyond the first chunk the two sides therefore embed different
audio, and any whole-clip difference would measure that deliberate
deviation instead of measuring implementation fidelity.

Both the local and official sides are put into `eval()`; the numerical
matrix runs under `torch.inference_mode()`.

## Test Matrix

### Durations

Transform matrix:

```text
0.01, 0.2, 1.0, 6.0, 10.0, 12.5 seconds
```

Meaning:

- 0.01: 160 samples, below the 513-sample floor, the minimum-length
  zero-padding path (4 mel frames, exactly one patch);
- 0.2: 3200 samples → 21 frames, a frame count that is not a multiple of
  the patch width;
- 1.0: 101 frames;
- 6.0: 601 frames, which is also the default `chunk_len` of the official
  downstream extraction;
- 10.0: 1001 frames, exactly the single-chunk ceiling (250 patches plus a
  1-frame remainder that falls short of a patch on both sides);
- 12.5: 1251 frames, past the single-chunk ceiling — the frontend itself
  never chunks, so this pins that its output is one pass over the whole
  valid region.

Encoder matrix:

```text
single chunk: 1.0, 10.0 seconds
multi chunk:  10.05 (2 chunks), 21.0 (3 chunks) seconds
mixed batch:  0.5, 6.0, 10.0 seconds
```

10.05 s yields 1006 frames, which splits into a full 250-patch chunk plus
a 6-frame chunk (1 patch, 2 frames dropped); 21.0 s yields 2101 frames,
i.e. two full chunks plus a 101-frame chunk (25 patches).

### Signals

Deterministic synthetic signals with seeds derived from the case (the
sample count for the transform matrix, the duration for the encoder
cases): random waveform (`randn * 0.5`), 997 Hz sine, unit impulse at
sample 100, multi-frequency sine (440 / 1237 / 3313 Hz), silence. The
transform matrix runs all five; the encoder single-chunk and mixed-batch
cases use the random waveform, and the multi-chunk case uses the
multi-frequency sine. Silence is included because it drives the `top_db`
floor into its degenerate branch, where every bin sits at the clamp.

### Batch

- B=1 for the transform, single-chunk, and multi-chunk matrices;
- One mixed batch `[0.5, 6.0, 10.0]`, stacked with zero padding to the
  longest sample and driven by an explicit `valid_seconds`.

Every sample in the mixed batch must match the official result of the
same sample computed on its own, which is what makes length grouping (and
therefore the omitted attention mask) auditable.

### Device

- CPU is mandatory; CUDA is mandatory when available, and explicitly
  recorded as skipped when unavailable;
- TF32 is disabled for both `torch.backends.cuda.matmul.allow_tf32` and
  `torch.backends.cudnn.allow_tf32` through a context manager that
  restores the previous values on exit;
- MPS is not required.

### Coverage Allocation

- **Transform alignment** (family-independent, weight-independent): the
  full 6 durations × 5 signals matrix, on every available device;
- **Single-chunk encoder alignment**: both architectures, which is also
  what makes both archive layouts load — `small` is the Lightning
  archive, `base` is the DINO-style archive;
- **Multi-chunk alignment** and **mixed-batch alignment**: `small` only.
  Both exercise chunk seams and length grouping, which are
  architecture-independent; `base` only widens the vector, and it is
  already covered by the single-chunk case;
- Five tests in total.

## Layered Assertions

### Frontend

- The official reference width equals the local `valid_feature_frames`
  for every case;
- The normalized features are aligned per sample against the official
  chain, over the full duration × signal matrix;
- The 0.01-second case pins the 513-sample zero-padding floor, with the
  reference recomputed on the same padded waveform;
- The reference is driven at rank 4 so that its `top_db` floor reduces
  per sample, which is the semantics the local transform implements by
  hand. Every case in this matrix is B=1, where the two reduction ranges
  coincide; batch-composition independence itself is asserted by
  `test_top_db_floor_is_taken_per_sample` in
  `tests/models/transforms/test_atst.py`.

### Checkpoint

- Both clip checkpoints resolve through `ensure_atst_checkpoint` and pass
  SHA-256 verification against `helpers.ATST_CHECKPOINTS`;
- The normalized state loads into the official `AST` with `strict=True`,
  i.e. 139 keys, no missing and no unexpected key, for both archive
  layouts;
- The DINO layout passing this assertion is what proves the seven
  `module.head.*` projection-head keys were dropped rather than tolerated
  — the official loader uses `strict=False` and would have accepted them;
- All loads use `weights_only=True` under the minimal allowlist, with no
  fallback.

### Encoder

- The local and official embeddings have identical shape
  (`2 * n_blocks * D`);
- Numerical gates plus the float64 audit on the final embedding;
- Multi-chunk: at least two chunks are actually produced, and the local
  embedding equals the equal-weight mean of the official per-chunk
  results;
- No NaN/Inf on either side;
- Mixed batch matches per-sample calls.

The base-class contract for this family — clip geometry, `valid_mask`,
zero-filled invalid slots, chunk bookkeeping, `n_blocks` selection — is
covered by the weight-free tests in
`tests/models/encoders/test_atst_clip.py` and is not restated here.

## Numerical Gates

Pointwise local/official comparisons on the same device
(`torch.allclose`) are split by device:

| Stage | CPU | CUDA |
|---|---|---|
| Transform | `atol=1e-4, rtol=1e-4` | `atol=2e-3, rtol=1e-4` |
| Encoder | `atol=1e-4, rtol=1e-4` | `atol=2e-3, rtol=1e-4` |

float64 audit gates:

- relative-L2: CPU ≤ 1e-4; CUDA ≤ 1e-3;
- cosine ≥ 0.99999 on both devices;
- the max absolute difference is recorded per stage and device;
- NaN and positive/negative Inf are rejected on both sides.

Reason for the relaxed CUDA gate: under the tested configuration the two
sides differ only by operations that are exact identities — an all-zero
additive attention mask, an all-true length mask applied before pooling,
and a different reshape route into the same patch `Linear` (the official
`einops` rearrange versus a local `view`/`transpose`/`reshape`). On CPU
the two are bitwise equal, which demonstrates the arithmetic is the same;
on CUDA those identities still change kernel selection and reduction
order, and the residual float32 difference of a few ULP is what the
relaxed gate absorbs. The frontend has no such asymmetry — both sides run
the same torchaudio operators on the same batch shape — and is bitwise
equal on CUDA as well.

CPU and CUDA are each compared against the official reference on the same
device; no direct numerical gate is established between CPU and CUDA.

## Pytest Entry Points

Ordinary tests:

```bash
python -m pytest tests/models -v
```

By default, this does not fetch the official source code, does not read
the large checkpoints, and does not run the alignment matrix.

Explicit ATST-Clip alignment:

```bash
python -m pytest \
  tests/models --run-alignment atst_clip -v
```

Both ATST entries can be enabled in one invocation
(`--run-alignment atst_clip atst_frame`). The allowed set in
`tests/models/conftest.py` is extended to:

```python
("panns", "ast", "clap", "beats", "wav2vec2", "atst_clip", "atst_frame")
```

The test file is `tests/models/test_atst_clip_alignment.py`, with the
module-level `pytestmark = pytest.mark.alignment("atst_clip")`, and it
collects five tests:

```text
test_transform_alignment
test_encoder_alignment_single_chunk[small]
test_encoder_alignment_single_chunk[base]
test_encoder_alignment_multi_chunk[small]
test_mixed_batch_matches_single_calls
```

## Temporary Files and Caching

- Official source code: `$TMPDIR/timbral-atst-alignment/audiossl/`,
  a separate working tree from the ATST-Frame module's
  `$TMPDIR/timbral-atst-alignment/audiossl-frame/`, because the two
  modules sparse-check-out different subtrees of the same commit;
- Result summary:
  `$TMPDIR/timbral-atst-alignment/atst-clip-alignment-summary.json`,
  written by a module-scoped autouse fixture that also records the torch
  version and whether CUDA was exercised; not committed to Git;
- Checkpoints live in the user's hub directory
  (`HF_HUB_CACHE/audioencoders/atst/`) and are downloaded there only if
  absent; tests write nothing to `/scratch` and do not write temporary
  content into the current directory, the user's home directory, or
  `/projappl`.

## Acceptance Criteria

The ATST-Clip migration is considered complete only once all of the
following conditions hold simultaneously:

1. All weight-free ATST model tests and registry tests pass;
2. Both clip checkpoints pass SHA-256 verification, and their normalized
   state loads into the official `AST` with `strict=True` — Lightning and
   DINO-style layouts alike;
3. The full transform matrix passes on CPU, and likewise on CUDA when
   available;
4. Single-chunk encoder alignment passes for both architectures on every
   available device;
5. Multi-chunk alignment passes against the self-assembled equal-weight
   mean of official single-chunk results, with more than one chunk
   actually produced;
6. Every mixed-batch sample matches its standalone official result;
7. The 0.01-second minimum zero-padding contract passes (the official
   reference recomputes on the same padded waveform);
8. No new runtime dependency is introduced (`einops`, `fairseq`, and
   `audiossl` itself stay outside the package); the pinned checkout stays
   in `$TMPDIR`, and `sys.path` and `sys.modules` are restored after the
   reference import;
9. This document has a genuine, traceable results summary appended.

Do not pre-fill "passed" before the tests are actually run.

## Empirical Results (2026-08-16)

### Code and Environment

- PyTorch: 2.11.0+cu130;
- GPU: NVIDIA GH200 120GB; CUDA was available, so both CPU and CUDA were
  exercised in every test;
- TF32 disabled throughout the matrix and restored afterwards.

The official source was sparse-cloned at the pinned commit
`ec3a14db086eaccfb69513e4a90fad89bf992e1f`; `git rev-parse HEAD` matched
it, and the SHA-256 digests of all nine pinned files match the "Pinned
Official Identity" table in this document value-for-value. Both clip
checkpoints passed `ensure_atst_checkpoint` verification, and both
normalized state dicts loaded into the official `AST` with `strict=True`.

### Execution Matrix

- Transform alignment: 6 durations × 5 signal types = 30 cases, actually
  executed on both CPU and CUDA;
- Single-chunk encoder alignment: 2 architectures × 2 durations, i.e. 2
  audit cases per architecture on each device, 8 in total;
- Multi-chunk alignment: 2 durations for `small`, on each device, with
  the reference assembled from 2 and 3 official single-chunk forward
  passes respectively;
- Mixed-batch alignment: 3 samples compared individually on each device.

Test entry-point result:

```text
python -m pytest tests/models --run-alignment atst_clip -v
5 passed
```

`tests/models` collects 423 tests in total on the run machine (421 when
CUDA is unavailable, because the CLAP alignment module parametrizes two
of its tests by device); without `--run-alignment atst_clip` these five
are skipped along with the other alignment entries.

### Numerical Results

Worst-case values against the pinned official reference on the same
device (float64 audit):

| Stage | Device | Max Abs Diff | relative-L2 | Min Cosine |
|---|---|---:|---:|---:|
| Transform | CPU | 0 | 0 | 0.999999999999473 |
| Transform | CUDA | 0 | 0 | 0.999999999999473 |
| Encoder small | CPU | 0 | 0 | 0.999999999999996 |
| Encoder small | CUDA | 1.91e-6 | 4.66e-8 | 0.999999999999997 |
| Encoder base | CPU | 0 | 0 | 0.999999999999992 |
| Encoder base | CUDA | 3.81e-6 | 4.30e-8 | 0.999999999999992 |
| Multi-chunk small | CPU | 0 | 0 | 0.999999999999994 |
| Multi-chunk small | CUDA | 1.43e-6 | 5.34e-8 | 0.999999999999993 |
| Mixed batch | CPU | 0 | 0 | 0.999999999999997 |
| Mixed batch | CUDA | 1.91e-6 | 4.19e-8 | 0.999999999999995 |

The frontend is bitwise equal to the official chain on both devices, over
all 30 cases: the local side reorganizes the dB stage but executes the
same operators on the same batch shape, so there is no kernel-shape
asymmetry to create a difference. The encoder is bitwise equal on CPU for
both architectures and in every case, including the chunk seams and the
mixed batch; the CUDA differences of 1e-6 to 4e-6 are the reduction-order
effect recorded in "Numerical Gates", more than two orders of magnitude
inside the `atol=2e-3` gate and more than four orders inside the
relative-L2 gate.
All cases pass the final gates.

### Specific Contracts

- Both archive layouts were exercised through the official module:
  `atst-clip-small` from the Lightning archive and `atst-clip-base` from
  the DINO-style archive both loaded with `strict=True`, which confirms
  the DINO projection-head keys are dropped and the key sets agree
  exactly;
- The 0.01-second case passes on both devices: 160 samples are padded to
  513 and the official reference recomputes on the same padded waveform,
  yielding the 4 mel frames of a single patch;
- The multi-chunk cases produced 2 chunks (10.05 s) and 3 chunks (21.0 s)
  respectively, and the local embedding matched the equal-weight mean of
  the corresponding official single-chunk results;
- Every mixed-batch sample reproduced its standalone official result, on
  both devices, which confirms that grouping by valid length makes the
  omitted attention mask an exact identity;
- The clip reference needs no `fairseq` stub (only the ATST-Frame
  reference does). einops was used by the reference side alone: the
  pinned `audio_transformer.py` imports `einops.layers.torch.Rearrange`
  at module scope, so an explicit alignment run requires einops in the
  environment (0.8.2 in the project container), while `pyproject.toml`
  declares none and the local `_AtstPatchEmbed` reproduces the same
  rearrangement with `view`/`transpose`. Nothing was installed, and the
  official source was not modified.

The machine-readable temporary summary for this run is located at:

```text
$TMPDIR/timbral-atst-alignment/atst-clip-alignment-summary.json
```

This file is not committed to Git.
