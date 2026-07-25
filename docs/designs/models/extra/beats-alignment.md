# BEATs Official Implementation Alignment Contract

This document freezes the official implementation alignment contract for
`BeatsKaldiFbankTransform` and `BeatsEncoder`, and records the verification
summary that must be committed once the implementation is complete.

Corresponding designs:

- [`../transforms/beats.md`](../transforms/beats.md)
- [`../encoders/beats.md`](../encoders/beats.md)
- [`beats-download.md`](beats-download.md)

## Alignment Target

Verification covers the following three categories of properties:

1. Numerical equivalence between the local batched kaldi fbank frontend
   (including `× 2**15` scaling, normalization, and minimum-length
   zero-padding) and the official `BEATs.preprocess`;
2. Safe, complete, and accurate loading of all 15 official checkpoints
   (strict field-by-field comparison of cfg, value-for-value equality of all
   250 keys, and fine-tuned checkpoints dropping exactly
   `predictor.{weight,bias}`);
3. Alignment of all 15 entries against the pinned official source code at
   both the clip and frame granularities.

"Official alignment" does not cover:

- The `padding_mask` path (the official implementation is an approximation
  whose result varies with batch composition; the local side instead groups
  by unique length, so there is no padding within a group);
- The `predictor` classification head, sigmoid probabilities, and
  `label_dict`;
- `Tokenizers` / `quantizer` (the acoustic tokenizer weights are not among
  the 15 entries);
- Training-mode behavior (layerdrop, `GradMultiply`, dropout sampling);
- Pointwise equivalence between torchaudio resampling and the official
  data-loading pipeline.

The core official alignment boundary starts from a 16 kHz mono float32
waveform. Common downmixing, invalid-region clearing, and resampling are
tested separately as contract tests.

## Pinned Official Identity

### BEATs Source Code

Official repository and pinned revision:

```text
https://github.com/microsoft/unilm    (beats/ subdirectory)
833df7e7832e5064a281131ee64a481afa8e5b95    (master, 2026-01-23)
```

The reference side fetches only `beats/` via `--filter=blob:none` +
sparse-checkout into:

```text
$TMPDIR/timbral-beats-alignment/unilm/
```

It must not be cached into the project directory, the user's home
directory, `/projappl`, or `/scratch`.

The reference side must verify the actual Git commit and the SHA-256 of the
following key source files (digests taken from the official repository copy
as of 2026-07-25, finalized after re-verification against a pinned clone
during the implementation phase):

| File | SHA-256 |
|---|---|
| `beats/BEATs.py` | `27f289db7c56ce26f2ceb50d3719854b91b2dec1c2830d8b1dd8de1bbee19eeb` |
| `beats/backbone.py` | `31c0378379a7e0f1d1069f9da444fb86890fe1ea078959a2dcd39640cdcadbaa` |
| `beats/modules.py` | `edeb6b6cd6a784da749f932c3e0783c0bce556fc768d0d23a4d53d4b819eb424` |

The official modules use top-level absolute imports (`from backbone import
...`); they are loaded by temporarily injecting the `beats/` directory into
`sys.path`, loading file-by-file via `importlib`, and restoring `sys.path`
afterward; no package is installed.

### Checkpoints

The entry names, official file names, and SHA-256 digests of the 15
checkpoints are governed solely by the pinned identity table in
[`beats-download.md`](beats-download.md); alignment tests reuse
`helpers.BEATS_CHECKPOINTS` value-for-value.

The test weight directory is the same as the runtime default resolution
result:

```text
explicit pretrained_dir > HF_HUB_CACHE/audioencoders/beats/<entry>.pt
```

Alignment tests **do not download weights**: if any required file is missing
or its digest does not match, they call `pytest.fail` with a message
containing the full invocation command for `scripts/extra/beats_dl.py`. An
explicit alignment run requires `HF_HUB_CACHE` to point to a valid hub
directory (consistent with the existing practice used for the AST/PANNs
alignment runs).

## Reference Pipeline

### Transform Reference

For each sample, the reference fbank is computed with the official
`BEATs.preprocess` (called at the module-function level, without
constructing the full network):

- The input is the 16 kHz mono float32 waveform produced by the local
  pipeline after grouping, resampling, normalizing to
  `target_valid_samples`, and, if necessary, zero-padding to 2800 samples —
  minimum-length zero-padding is a waveform-domain operation, and the
  official frontend can recompute it pointwise on the same waveform, so this
  branch is included in the alignment contract (unlike PANNs' spectral-domain
  `-100 dB` padding, for which the official implementation has no reference
  output);
- The official side is called sample-by-sample in a loop, while the local
  side executes in batch; per-sample slices are compared.

### Encoder Reference

For each entry, construct the official `BEATs(BEATsConfig(checkpoint['cfg']))`
and call `load_state_dict(checkpoint['model'])`:

- Pretrained entries: `extract_features(source)[0]` directly returns the
  backbone features `[1, N, 768]`;
- Fine-tuned entries: after loading, set `BEATs_model.predictor = None`, so
  the official `extract_features` code path naturally returns the backbone
  features instead — without modifying or monkeypatching any official
  computation logic;
- The official side is called per-sample, without passing `padding_mask`;
- Frame reference: the official output `view(1, T', 8, 768).mean(dim=2)`;
- Clip reference: the official output `mean(dim=1)`;
- The derived reference must come from this same pinned official object and
  the same checkpoint; the local `BeatsEncoder` must not be used as the
  reference.

Both the local and official sides are put into `eval()`; the numerical
matrix runs under `torch.inference_mode()`.

## Test Matrix

### Durations

```text
0.02, 0.175, 1.0, 4.03, 10.0, 20.0 seconds
```

Meaning:

- 0.02: minimum-length zero-padding path (320 samples → padded to 2800
  samples);
- 0.175: exactly the boundary of 16 frames / 1 patch time block;
- 1.0: frame count (98) is not an integer multiple of 16, testing patch
  floor and tail-frame semantics;
- 4.03: off-grid duration;
- 10.0 / 20.0: variable-length paths within and beyond the training
  distribution.

### Signals

Deterministic synthetic signals with a fixed seed: random waveform, 997 Hz
sine, impulse, multi-frequency sine, silence; the invalid tail of mixed
batches is filled with a non-zero constant to confirm that content outside
the valid region does not change the result. Signals must not consist
solely of all-zero input.

### Batch

- B=1; B=3 of equal length;
- mixed `[0.02, 1.0, 4.03]`;
- mixed `[1.0, 4.03, 10.0]`.

Every sample in a mixed batch must match the result of calling on the same
input one sample at a time.

### Device

- CPU is mandatory; CUDA is mandatory when available, and explicitly
  recorded as skipped when unavailable;
- TF32 is disabled on CUDA
  (`torch.backends.cuda.matmul.allow_tf32 = False`), restored in a
  `finally` block;
- MPS is not required.

### Coverage Allocation

- **Transform alignment** (entry-independent, weight-independent): the full
  duration × signal matrix;
- **Encoder alignment**: full coverage of all 15 entries, each running the
  reduced matrix `{1.0, 10.0} × {random, sine} × {clip, frame}`;
- The representative entry `beats_iter3_plus_as2m` additionally runs the
  full duration matrix and all mixed-batch combinations;
- Checkpoint-loading assertions are executed for all 15 entries.

## Layered Assertions

### Frontend

- The povey window and mel weight buffers are bitwise equal to the
  torchaudio reference;
- The normalized fbank is aligned per-sample against the official
  `preprocess`;
- `valid_feature_frames` is pinned to `1 + (max(n, 2800) - 400) // 160`;
- Mixed batch matches per-sample calls; non-zero invalid padding does not
  affect the output.

### Checkpoint

- `helpers.BEATS_CHECKPOINTS` covers all 15 entries, with SHA-256 digests
  matching the download script's table value-for-value;
- SHA-256 verification passes for all 15 files;
- cfg matches the expected tables field-by-field exactly (both the
  pretrained and fine-tuned tables);
- The pretrained state_dict has exactly 250 keys, the fine-tuned state_dict
  has exactly 252 keys, and the extra keys are exactly
  `predictor.{weight,bias}`;
- After loading, the local module parameters are `torch.equal` to the
  checkpoint value-for-value (no floating-point tolerance);
- All loads use `weights_only=True`, with no fallback.

### Encoder

- Both clip and frame reference exports;
- Numerical gates on the final embedding;
- clip/frame shape, geometry, valid_mask;
- Invalid embeddings and geometry are all zero;
- A cross-check assertion that `clip ≈ time-average of frame`;
- No NaN/Inf;
- Mixed batch matches per-sample calls.

### Common Entry Point

Downmixing and resampling do not claim pointwise equality with the official
implementation, but must verify `[B,N]` vs. `[B,C,N]`, that non-zero invalid
padding has no effect, that a 16 kHz mono input is semantically consistent
after processing through the common entry point, and the round-ties-to-even
discrete-length semantics.

## Numerical Gates

Pointwise local/official comparisons on the same device
(`torch.testing.assert_close`) are split by device:

| Stage | CPU | CUDA |
|---|---|---|
| Transform | `atol=1e-4, rtol=1e-4` | `atol=5e-3, rtol=1e-4` |
| Encoder | `atol=1e-4, rtol=1e-4` | `atol=2e-3, rtol=1e-4` |

Reason for the relaxed CUDA gates (measured on GH200, not an implementation
difference): the official reference loops sample-by-sample while the local
side runs in batch, and the two trigger cuFFT/GEMM kernels of different
shapes on the GPU with different reduction orders; for pure-tone signals,
the cancellation error in the spectral leakage floor mel bin (around
-80 dB, obtained via cancellation of large terms), amplified through the
log, produces an individual-element log-domain difference (at the
transform level) of about `2.3e-3`; after propagating through the backbone,
the worst case at the embedding level is `7.2e-4`. For random signals the
transform-level difference is only about `1e-4`, and the embedding-level
difference about `2e-6`. On CPU, the two sides are bitwise equal
(max_abs = 0), demonstrating algorithmic equivalence; this difference is a
GPU-kernel-shape effect that cannot be eliminated while keeping the batched
implementation.

float64 audit gates:

- relative-L2: CPU ≤ 1e-4 (measured 0); CUDA ≤ 1e-3 (same kernel-shape
  effect, measured worst case transform 1.1e-4, encoder 2.1e-4);
- cosine ≥ 0.99999 on both devices (measured worst case 0.99999998);
- the max absolute difference is recorded;
- NaN and positive/negative Inf are rejected.

CPU and CUDA are each compared against the official reference on the same
device; no direct numerical gate is established between CPU and CUDA.
Weights and any state declared to be loaded value-for-value use
`torch.equal`.

## Pytest Entry Points

Ordinary tests:

```bash
python -m pytest tests/models -v
```

By default, this does not fetch the official source code, does not read the
large weights, and does not run the full matrix.

Explicit BEATs alignment:

```bash
python -m pytest \
  tests/models --run-alignment beats -v
```

The allowed set in `tests/models/conftest.py` is extended to:

```python
("panns", "ast", "clap", "beats")
```

The test file is `tests/models/test_beats_alignment.py`, with the
module-level `pytestmark = pytest.mark.alignment("beats")`.

## Temporary Files and Caching

- Official source code and raw JSON: `$TMPDIR/timbral-beats-alignment/`;
- Result summary:
  `$TMPDIR/timbral-beats-alignment/beats-alignment-summary.json`, not
  committed to Git;
- Checkpoints are loaded read-only from the user's existing hub directory;
  tests write nothing to `/scratch` and do not write temporary content into
  the current directory, the user's home directory, or `/projappl`.

## Acceptance Criteria

The BEATs migration is considered complete only once all of the following
conditions hold simultaneously:

1. All weight-free model tests and download-script tests pass;
2. All 15 checkpoints pass SHA-256 verification, strict cfg comparison, and
   value-for-value weight loading;
3. The full Transform matrix passes on CPU, and likewise on CUDA when
   available;
4. The reduced Encoder matrix for all 15 entries and the full matrix for the
   representative entry pass;
5. All mixed-batch results match their per-sample counterparts;
6. The 0.02-second minimum zero-padding contract passes (the official
   reference recomputes on the same padded waveform);
7. The download script completes at least one real download in an
   environment with playwright and passes SHA-256 verification;
8. No new dependencies such as fairseq, einops, or torchlibrosa are
   introduced;
9. This document has a genuine, traceable results summary appended.

Do not pre-fill "passed" before the tests are actually run.

## Empirical Results (2026-07-25)

### Code and Environment

- Repository baseline commit: `679ec49` (the BEATs migration is in the
  current working tree on top of this commit);
- Python: 3.12.13;
- PyTorch: 2.11.0+cu130;
- torchaudio: 2.11.0+cu130;
- CUDA runtime: 13.0;
- cuDNN: 9.19.0;
- GPU: NVIDIA GH200 120GB;
- NumPy: 2.4.6.

The official source code was sparse-cloned at the pinned commit
`833df7e7832e5064a281131ee64a481afa8e5b95`; the SHA-256 digests of the three
key files match the "Pinned Official Identity" table in this document
value-for-value. All 15 checkpoints' SHA-256 digests pass
`ensure_beats_checkpoint` verification and match the pinned identity table
in [`beats-download.md`](beats-download.md) value-for-value.

### Execution Matrix

- Transform alignment: 6 durations × 5 signal types = 30 cases, actually
  executed on both CPU and CUDA;
- Encoder alignment: full coverage of all 15 entries, each with 2 durations
  × 2 signals × clip/frame; the representative entry
  `beats_iter3_plus_as2m` runs the full 6-duration matrix; 136 audit cases
  on each device;
- Batch alignment: mixed lengths `[0.02, 1.0, 4.03]`, `[1.0, 4.03, 10.0]`
  (non-zero invalid tail) and equal-length B=3 `[1.0, 1.0, 1.0]`, with
  per-sample comparison against the official reference plus all-zero
  assertions on invalid slots; 9 cases on each device;
- After loading each entry, a value-for-value `torch.equal` assertion is
  executed on the 250 keys remaining after dropping predictor, and cfg
  matches both expected tables field-by-field exactly.

Test entry-point results:

```text
python -m pytest tests -q
377 passed, 25 skipped

python -m pytest tests/models --run-alignment beats -v
226 passed, 22 skipped
```

The skips for the default command are the four alignment entries not
explicitly enabled, together with other pre-existing skips.

### Numerical Results

Worst-case values against the pinned official reference on the same device
(float64 audit):

| Stage | Device | Max Abs Diff | relative-L2 | Min Cosine |
|---|---|---:|---:|---:|
| Transform | CPU | 0 | 0 | 1.0 |
| Transform | CUDA | 2.34e-3 | 1.08e-4 | 0.9999999941 |
| Encoder (15 entries) | CPU | 0 | 0 | 1.0 |
| Encoder (15 entries) | CUDA | 7.25e-4 | 2.07e-4 | 0.9999999786 |
| Batch (B=3) | CPU | 1.67e-6 | 5.92e-7 | 1.0 |
| Batch (B=3) | CUDA | 2.15e-6 | 7.99e-7 | 1.0 |

On CPU, the single-sample matrices for both Transform and Encoder are
bitwise equal to the official reference. The B=3 batch row is on the order
of ~2e-6 on both devices: three equal-length samples form a genuine B=3
group, whose GEMM/FFT kernel shapes differ from the official per-sample B=1
execution, and the resulting reduction-order difference is amplified to
this order of magnitude through the log and backbone propagation (within
mixed-length batches, each group is a single sample, and these are bitwise
equal on CPU). The CUDA single-sample differences stem from the
batched-kernel vs. per-sample-kernel shape effect recorded in the
"Numerical Gates" section; the worst cases are all leakage-floor bins from
pure-tone signals. All cases pass the final gates.

### Specific Contracts

- For all three B=3 batches (two mixed-length groups plus one equal-length
  group), the max per-sample difference against the official reference is
  CPU 1.67e-6, CUDA 2.15e-6; invalid frame slots are exactly zero;
- The 0.02-second minimum zero-padding contract passes: after padding to
  2800 samples, the official reference recomputes on the same waveform,
  consistently on both devices;
- All 15 checkpoints are read with `weights_only=True`; cfg passes strict
  field-by-field comparison; fine-tuned checkpoints drop exactly
  `predictor.{weight,bias}`;
- Real-download verification: in the agent environment,
  `python scripts/extra/beats_dl.py --dest $TMPDIR/...` was run with
  `--entries beats_iter1` and `--entries beats_iter3_plus_as20k`
  respectively; the parsed official names and byte counts
  (`BEATs_iter1.pt`, `BEATs_iter3_plus_AS20K.pt`, each 344.75 MiB) were
  correct, and after the downloads completed, both the script's internal
  `sha256sum` and an external `sha256sum` matched the pinned values;
- fairseq, einops, and torchlibrosa were neither installed nor imported;
  MPS was not tested.

The machine-readable temporary summary for this run is located at:

```text
$TMPDIR/timbral-beats-alignment/beats-alignment-summary.json
```

This file is not committed to Git.
