# PANNs Official Implementation Alignment Contract

This document freezes the official implementation alignment contract for
`PannsLogmelTransform` and `PannsCnn14Encoder`, and records the verification
summary that must be committed once the implementation is complete.

Corresponding designs:

- [`../transforms/panns.md`](../transforms/panns.md)
- [`../encoders/panns.md`](../encoders/panns.md)

This document also records the confirmed design together with the
post-implementation empirical results completed on 2026-07-23. The frontend
equivalence pre-review conducted during the design interview and the full
post-implementation matrix are recorded separately and are not conflated as
the same evidence.

## Alignment Target

Verification covers the following three categories of properties:

1. Numerical equivalence, within the PANNs parameter domain, of the local
   `torchlibrosa`-free frontend;
2. Complete and accurate split-loading of the three official checkpoints
   into the Transform and the Encoder;
3. Alignment of the six checkpoint/granularity combinations against the
   pinned official source code, under the declared native or ultra-short
   contract semantics.

"Official alignment" does not cover:

- Pointwise equivalence between torchaudio resampling and the librosa file
  reading used in the official data-loading scripts;
- The AudioSet 527-class classification head;
- SpecAugmentation;
- mixup;
- A general-purpose torchlibrosa replacement for arbitrary `ref`, `amin`,
  `top_db` parameters.

The core official alignment boundary starts from a mono float32 waveform at
the target sample rate. Common downmixing, invalid-region clearing, and
resampling are tested separately as contract tests.

## Pinned Official Identity

### PANNs Source Code

Official repository:

```text
https://github.com/qiuqiangkong/audioset_tagging_cnn
```

Pinned revision:

```text
d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4
```

The reference side must verify the actual Git commit and the SHA-256 of
the following key source files:

| File | SHA-256 |
|---|---|
| `pytorch/models.py` | `7f9af440395ace5160bbb51d654a0dc35fb887fbf5edecb12da61ff6efb306d9` |
| `pytorch/pytorch_utils.py` | `1464fcfbfc0fe4c55f690f6b39e1c80eeed5de1e7fd1b7fd30334d304de7dbe9` |

The official source code is fetched only into:

```text
$TMPDIR/timbral-panns-alignment/
```

It must not be cached into the project directory, the user's home
directory, `/projappl`, or `/scratch`.

### Checkpoints

Official Zenodo record:

```text
https://zenodo.org/records/3987831
```

| entry | File | SHA-256 |
|---|---|---|
| `panns-32k-cnn14-max_mean` | `Cnn14_mAP=0.431.pth` | `0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31` |
| `panns-16k-cnn14-max_mean` | `Cnn14_16k_mAP=0.438.pth` | `e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6` |
| `panns-32k-cnn14-decision_level_max` | `Cnn14_DecisionLevelMax_mAP=0.385.pth` | `dd3b4043a87d4ec13df8082c0fcfee3fb5084151808e47e060987a95eabdd142` |

Test weight directory:

```text
DEBUG_ROOT/
└── panns/
    ├── panns-32k-cnn14-max_mean/Cnn14_mAP=0.431.pth
    ├── panns-16k-cnn14-max_mean/Cnn14_16k_mAP=0.438.pth
    └── panns-32k-cnn14-decision_level_max/
        └── Cnn14_DecisionLevelMax_mAP=0.385.pth
```

`DEBUG_ROOT` is obtained by `tests/models/conftest.py` using `rootutils` to
locate the project root and load `.env`. During an explicit alignment run,
a missing `DEBUG_ROOT`, a failed download, or a digest mismatch is a
failure, not a skip.

## torchlibrosa Equivalence Pre-review

An independent numerical pre-review, writing only to `$TMPDIR`, was already
completed during the design interview. This pre-review did not modify the
project and does not substitute for the subsequent tests of the new
implementation.

### Version Identity

- The official PANNs revision declares:
  - `torchlibrosa==0.0.4`
  - `librosa==0.6.3`
- The legacy full alignment harness used:
  - `torchlibrosa==0.1.0`
  - A current librosa-compatible version
- torchlibrosa 0.0.4's `stft.py` corresponds to the official repository
  commit `c2fa11bf76c7551affc1df05463a505ccf106158`;
- torchlibrosa 0.1.0's `stft.py` corresponds to
  `1c4c2361a32673738cbe6b348b8f53ea2944c435`.

0.0.4 cannot be combined as-is with a current librosa, because the
`pad_center` API changed from positional to keyword-only arguments; this is
a compatibility change, not an algorithmic change on PANNs' usage path.

### Parameters Tested

- 16 kHz: FFT/window 512, hop 160, 64 mel, 50-8000 Hz;
- 32 kHz: FFT/window 1024, hop 320, 64 mel, 50-14000 Hz;
- `center=True`, `reflect`, power=2;
- `ref=1.0`, `amin=1e-10`, `top_db=None`;
- 0.1, 1, 4.03, 10 seconds;
- B=1/B=3;
- fixed-window mixed `[1,4.03,10]`;
- non-trivial shared BN state.

### Pre-review Results

The legacy equivalent implementation relative to torchlibrosa 0.1.0 +
librosa 0.11:

- DFT, Hann, conv weights: bitwise equal;
- STFT real/imaginary parts and power spectrum: bitwise equal;
- mel weights: bitwise equal;
- PANNs power-to-dB: bitwise equal;
- The full frontend before and after `bn0`: bitwise equal.

Relative to the original official 0.0.4 + librosa 0.6.3:

- DFT, Hann, STFT, power spectrum: bitwise equal;
- Max absolute difference for mel weights: `1.862645149230957e-09`;
- Max absolute difference for log-mel: no more than `3.8146973e-06`;
- Max absolute difference for post-BN: no more than `1.1444092e-05`;
- All pass `torch.allclose` at both `atol=rtol=1e-6` and `1e-4`.

Once a real PANNs checkpoint is loaded, the STFT, mel, and `bn0` states are
all overwritten by that same checkpoint, so the differences caused by the
librosa version at initialization time disappear.

Conclusion: the runtime and automated tests do not need `torchlibrosa`
installed. Subsequent tests must still retain `torch.stft`, independent
mel/dB computation, and value-for-value checkpoint-buffer assertions; the
local frontend must not be compared only against itself.

## Native and Ultra-short Contract

### Normal Length

When:

```text
valid_feature_frames >= 32
```

The local frontend executes the official reflect semantics at the true
length, and the Encoder uses the official floor pooling:

```text
num_embedding_frames = valid_feature_frames // 32
```

All native features enter the official backbone. A remainder that is
shorter than one full 32-feature block does not produce an extra embedding
frame, but it may still affect existing outputs through the convolutional
receptive field; it must not be pre-truncated to a multiple of 32.

### Ultra-short Input

When:

```text
valid_feature_frames < 32
```

The official frontend can still produce a log-mel as long as the reflect
condition is satisfied, but Cnn14's five rounds of temporal pooling cannot
be executed. The project's contract:

1. Run the official native log-mel;
2. Append `-100 dB` before `bn0`;
3. Pad only up to 32 features;
4. Run the official `bn0` and backbone;
5. Return one valid embedding frame.

0.02 seconds falls under this contract. The reference side must use this
same official frontend output with the minimum silence padding; a shape
error from the official native Encoder must not be treated as an alignment
failure.

At roughly 0.016 seconds and below (`target_valid_samples <= n_fft / 2`),
the official reflect padding condition is violated. In that range, the
local implementation degrades the padding mode to zero-padding to keep the
computation well-defined (see "Padding Mode for Ultra-short Input" in
[`../transforms/panns.md`](../transforms/panns.md)). This branch is a local
extension; the official frontend has no reference output here, so the
alignment tests do not include this region in the success contract.

## Six Alignment Combinations

| checkpoint | granularity | Reference Export |
|---|---|---|
| 16 kHz max_mean | clip | official Cnn14 `embedding` |
| 16 kHz max_mean | frame | per-position `fc1 + ReLU + dropout` after the official backbone's frequency mean |
| 32 kHz max_mean | clip | official Cnn14 `embedding` |
| 32 kHz max_mean | frame | per-position `fc1 + ReLU + dropout` after the official backbone's frequency mean |
| 32 kHz DecisionLevelMax | frame | 2048-dim segment hidden, before interpolation and before the classification head |
| 32 kHz DecisionLevelMax | clip | temporal `amax` of the above valid segment hidden |

The derived export must come from this same pinned official object and the
same checkpoint; the local `PannsCnn14Encoder` must not be used as the
reference.

## Official Reference Adapter

Directly importing the pinned official source code references
`torchlibrosa`. The test side provides a minimal compatibility shim:

- `Spectrogram`;
- `LogmelFilterBank`;
- The `SpecAugmentation` interface, needed at construction time but not
  executed under eval.

The shim uses the already-audited legacy equivalent algorithm and does not
install the `torchlibrosa` package.

To avoid the frontend becoming its own sole proof of correctness:

- STFT is additionally checked against `torch.stft`;
- mel is additionally checked against an independent librosa / hand-built
  matrix projection;
- dB is additionally checked against a frozen input vector;
- the STFT/mel/`bn0` buffers in the checkpoint are checked value-for-value;
- the backbone and final export are judged by the pinned official
  source-code object.

## Test Matrix

### Durations

```text
0.02, 0.32, 1, 4.03, 10, 20 seconds
```

Meaning:

- 0.02: the special case of fewer than one Encoder output block;
- 0.32: close to one output stride;
- 4.03: off-grid duration;
- 1/10/20: short, medium, and long variable-length paths.

Weight-free unit tests may additionally cover exact frame-count boundaries
such as 0.63 seconds, but this is not required in the full large-weight
matrix.

### Batch

- B=1;
- B=3 of equal length;
- mixed `[0.02,1,4.03]`;
- mixed `[1,4.03,10]`.

Every sample in a mixed batch must match the result of calling on the same
input one sample at a time. This assertion verifies both the unique-length
grouping and batch-order restoration.

### Device

- CPU is mandatory;
- CUDA is mandatory when available;
- CUDA is explicitly recorded as skipped when unavailable;
- MPS is not required.

### Signals

Fixed-seed, deterministic synthetic signals are used, including at least:

- random waveform;
- single-/multi-frequency sine;
- impulse;
- silence;
- non-zero invalid padding, used to confirm that content outside the valid
  region does not change the result.

Signals must not consist solely of all-zero input, since that would not
sufficiently expose errors in STFT, mel, convolution, or weight mapping.

## Layered Assertions

### Frontend

- DFT/Hann/conv buffers;
- STFT real part, imaginary part, and power;
- mel weights and projection;
- dB;
- before and after `bn0`;
- feature shape;
- `valid_feature_frames`;
- the 0.02-second minimum silence padding;
- no completion padding at normal lengths;
- mixed batch matches per-sample calls.

### Checkpoint

- The single checkpoint metadata table in `helpers/panns.py` covers all
  three official identities;
- The Transform and the Encoder use the same helper metadata and the same
  safe-loading entry point;
- The SHA digests of the three files are correct;
- The Transform's frontend keys are loaded value-for-value;
- The Encoder's conv blocks and `fc1` are loaded value-for-value;
- `fc_audioset` does not enter the local state;
- No missing keys, no unexpected keys;
- All three checkpoints use `weights_only=True`;
- Only the 16 kHz checkpoint uses a minimal NumPy allowlist;
- No checkpoint falls back to `weights_only=False`.

### Encoder

- All six reference exports;
- Intermediate backbone hidden state;
- Final embedding;
- clip/frame shape;
- frame geometry;
- valid_mask;
- Invalid embeddings and geometry are all zero;
- No NaN/Inf;
- Mixed batch matches per-sample calls.

### Common Entry Point

Downmixing and resampling do not claim pointwise equality with official
file loading, but must verify:

- `[B,N]` vs. `[B,C,N]`;
- non-zero invalid padding does not affect the output;
- a mono input at the target sample rate is semantically consistent after
  processing through the common entry point;
- the source/target discrete-length semantics of `torch.round`
  ties-to-even.

## Numerical Gates

Same-device local/official comparison:

```text
atol = 1e-4
rtol = 1e-4
```

The local Transform, local Encoder, and official reference must all be put
into `eval()`; the numerical matrix runs under `torch.inference_mode()`.
CPU and CUDA are each compared against the official reference on the same
device; no direct numerical gate is established between CPU and CUDA.

Also recorded:

- max/mean/p99 absolute difference;
- max/mean/p99 relative difference;
- relative-L2;
- cosine min/mean;
- full or common-prefix all-close;
- NaN and positive/negative Inf patterns.

Weights and buffers declared to be loaded value-for-value use
`torch.equal`, with no floating-point tolerance.

Normal outputs must not contain non-finite values. Even if the reference
side and the local side both produce non-finite values, this must not pass
merely because the patterns match, unless the contract explicitly declares
that case expected to fail.

## Pytest Entry Points

Ordinary tests:

```bash
python -m pytest tests/models -v
```

By default, this does not fetch the official source code, does not
download the large weights, and does not run the full matrix.

Explicit PANNs alignment:

```bash
python -m pytest \
  tests/models --run-alignment panns -v
```

`--run-alignment` accepts one or more entries:

```bash
python -m pytest \
  tests/models --run-alignment panns ced -v
```

The currently allowed set is only:

```python
["panns"]
```

An unknown entry produces a clear error during pytest argument parsing or
session initialization. When new models are added in the future, the
test-side allowed set is extended; no production registry is established.

Since `pyproject.toml` is not being restored for this task:

- `tests/models/conftest.py` registers the CLI option;
- The same conftest registers the alignment marker;
- `rootutils` locates the root directory;
- `rootutils` loads `.env`;
- Ordinary tests do not require `DEBUG_ROOT`;
- Explicit alignment requires a valid `DEBUG_ROOT`.

The test dependency boundary is pytest, rootutils, and Git. See the
Transform design for the runtime dependency boundary; neither the runtime
nor the tests install or import `torchlibrosa`.

## Temporary Files and Caching

- Official source code, shim build artifacts, and raw JSON: `$TMPDIR`;
- PANNs checkpoints: `DEBUG_ROOT/panns/{model_name}`;
- Temporary content is not written into the current directory, the user's
  home directory, `/projappl`, or `/scratch`;
- Large raw alignment JSON is not committed.

`DEBUG_ROOT` is a weight/debug root directory explicitly specified by the
user, and is an exception to this temporary-caching rule.

## Results Summary Format

Once the implementation is complete and the matrix has actually been run,
an "Empirical Results" section is appended at the end of this document,
recording at least:

- The current Git commit;
- Date;
- Versions of Python, PyTorch, CUDA, cuDNN, NumPy, librosa;
- The official source-code revision and key-file SHA digests;
- The SHA digests of the three checkpoints;
- The actual CPU/CUDA execution status;
- Pass count / total count for each checkpoint/granularity;
- Worst-case error for transform, backbone, and embedding;
- Mixed-batch consistency;
- The 0.02-second contract result;
- All skips or known limitations.

Do not pre-fill "passed" before the tests are actually run.

## Acceptance Criteria

The PANNs migration is considered complete only once all of the following
conditions hold simultaneously:

1. All weight-free model tests pass;
2. All three checkpoints pass safe loading and value-for-value weight
   mapping;
3. The CPU matrix passes for all six checkpoint/granularity combinations;
4. The CUDA matrix passes when CUDA is available;
5. Both groups of mixed-batch results match their per-sample counterparts;
6. The 0.02-second minimum-padding contract passes;
7. Normal lengths preserve the official floor pooling;
8. Neither the runtime nor the tests depend on `torchlibrosa`;
9. This document has a genuine, traceable results summary appended;
10. No registry, factory, YAML, top-level export, or dependency
    configuration unrelated to this task is introduced.

## Empirical Results (2026-07-23)

### Code and Environment

- Repository baseline commit:
  `9df72a874873bee88f574007f377bf6b007038d9` (the PANNs migration is in the
  current working tree on top of this commit);
- Python: 3.12.13;
- PyTorch: 2.11.0+cu130;
- torchaudio: 2.11.0+cu130;
- CUDA runtime: 13.0;
- cuDNN: 9.19.0;
- GPU: NVIDIA GH200 120GB;
- NumPy: 2.4.6;
- librosa: 0.11.0.

Official source-code revision and digests:

```text
d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4
pytorch/models.py
7f9af440395ace5160bbb51d654a0dc35fb887fbf5edecb12da61ff6efb306d9
pytorch/pytorch_utils.py
1464fcfbfc0fe4c55f690f6b39e1c80eeed5de1e7fd1b7fd30334d304de7dbe9
```

The SHA-256 digests of the three checkpoints all match the "Pinned
Official Identity" table in this document value-for-value.

### Execution Matrix

Each checkpoint runs 15 cases on each device:

- B=1 at 0.02, 0.32, 1, 4.03, 10, 20 seconds;
- B=3 at the same set of durations;
- mixed `[0.02,1,4.03]`;
- mixed `[1,4.03,10]`;
- 1-second silence.

Each case verifies both clip and frame. Results:

| checkpoint / granularity | CPU | CUDA |
|---|---:|---:|
| 16 kHz max_mean / clip | 15/15 | 15/15 |
| 16 kHz max_mean / frame | 15/15 | 15/15 |
| 32 kHz max_mean / clip | 15/15 | 15/15 |
| 32 kHz max_mean / frame | 15/15 | 15/15 |
| 32 kHz DecisionLevelMax / clip | 15/15 | 15/15 |
| 32 kHz DecisionLevelMax / frame | 15/15 | 15/15 |

Both CPU and CUDA were actually executed, with no device skips.

Test entry-point results:

```text
python -m pytest tests/models -q
65 passed, 1 skipped

python -m pytest tests/models --run-alignment panns -v
66 passed
```

The only skip in the default command is the PANNs alignment, which was not
explicitly enabled.

### Numerical Results

Relative to the pinned official reference on the same device:

| Stage | CPU Worst Abs Error | CUDA Worst Abs Error |
|---|---:|---:|
| post-`bn0` Transform | 0 | 0 |
| backbone hidden after frequency mean | 0 | 0 |
| frame embedding | 0 | 0 |
| clip embedding | 0 | 0 |

All max/mean/p99 absolute errors, relative errors, and relative-L2 are 0.
The cosine values, affected by floating-point reduction, are all no lower
than 0.9999685; the tensors being compared are themselves element-wise
equal. All cases pass the `atol=rtol=1e-4` gate.

### Specific Contracts

- Both mixed-batch groups match the official reference, grouped by true
  length, element-wise;
- At 0.02 seconds, both sample rates produce 3 native features, correctly
  generating one embedding frame after the `-100 dB` minimum padding;
- Normal lengths all retain the remainder input, with the output count
  following the official floor pooling;
- The checkpoint mapping is equal value-for-value;
- All three checkpoints use `weights_only=True`, and the 16 kHz
  minimal-allowlist path passes;
- Normal outputs are all finite values;
- The real `torchlibrosa` package was neither installed nor imported;
- At roughly 0.016 seconds and below, the local zero-padding branch is
  taken; the official implementation has no reference output there, so
  this region is not included in this alignment, and MPS was not tested.

The machine-readable temporary summary for this run is located at:

```text
$TMPDIR/timbral-panns-alignment/panns-alignment-summary.json
```

This file is not committed to Git.
