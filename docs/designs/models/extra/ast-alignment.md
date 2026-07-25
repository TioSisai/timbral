# AST Official Implementation Alignment Contract

This document freezes the official implementation alignment contract for
`AstKaldiFbankTransform` and `AstEncoder`, and records the fixed verification
matrix and measured summary once the implementation is complete.

Corresponding designs:

- [`../transforms/ast.md`](../transforms/ast.md)
- [`../encoders/ast.md`](../encoders/ast.md)

## Alignment Target

The mandatory primary reference is the pinned Hugging Face implementation:

```text
ASTFeatureExtractor + ASTModel
MIT/ast-finetuned-audioset-10-10-0.4593
```

Verification covers:

1. Numerical alignment between the local batched Torch fbank and the official
   per-sample `ASTFeatureExtractor`;
2. Complete and accurate loading of the 199 backbone tensors from the pinned
   checkpoint;
3. Alignment between the clip embedding and the official
   `ASTModel.pooler_output`;
4. Alignment between the project's frame export and the fixed derivation
   formula applied to the official `last_hidden_state`;
5. Compliance of geometry, valid_mask, and zero padding with the project's
   base-class contract;
6. Independent testing of the project's length extension and intentional
   differences.

"Official alignment" does not cover:

- Standalone execution of the original MIT/timm legacy environment;
- The AudioSet 527-class classification logits;
- Officially published frame embeddings, since no such upstream API exists;
- Pointwise upstream equivalence for the project's multi-channel downmixing
  and arbitrary sample-rate resampling;
- Project extensions for fewer than 400 target samples;
- Hugging Face's silent feature truncation beyond 10.255 seconds;
- fp16, bf16, MPS;
- SpecAugment or other training-time augmentation.

## Pinned Identity

### Model Snapshot

```text
repo_id:
  MIT/ast-finetuned-audioset-10-10-0.4593
revision:
  f826b80d28226b62986cc218e5cec390b1096902
```

Pinned files:

| File | SHA-256 |
|---|---|
| `config.json` | `a93d525511d77e8ecc933d09674b85099815bbbb417c228a4edd655e252fb9ff` |
| `preprocessor_config.json` | `8d04ba5a9c6fca5d39d0de2b1fd05ecf79deb589fbba279728bbebac39934231` |
| `model.safetensors` | `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d` |

Hub `main`, other revisions, pickle checkpoints, or local files lacking
digest verification are not accepted.

### Transformers Implementation

```text
version: 5.13.1
tag commit: 4626421dc6b741a329300682a6408246ee465490
```

Key source files:

| File | SHA-256 |
|---|---|
| `feature_extraction_audio_spectrogram_transformer.py` | `ab4957749b5113067413dcd662dc212952b9a610d297e8b4515e2cab1ff1fce4` |
| `modeling_audio_spectrogram_transformer.py` | `5ef9fe1c7847400453095c158c76191913226788eaa1f4ba6afbb378b9e70547` |

An explicit alignment run must verify both the version and the key
source-file digests. If they do not match, the implementation closure
mismatch must be reported, and no numeric result that could be mistaken for a
pinned official result may be generated.

Ordinary offline unit tests do not enforce a global Transformers version, nor
do they restore `pyproject.toml` for this task.

## Entry Points

Ordinary model tests:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest \
  tests/models -v -p no:cacheprovider
```

Explicit AST official alignment:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest \
  tests/models --run-alignment ast -v -p no:cacheprovider
```

`tests/models/conftest.py` registers the `ast` alignment entry. Without
passing `--run-alignment ast`, tests against real weights must be skipped and
must not trigger any download.

Temporary reference source code, intermediate download artifacts, and
dynamic run outputs must be placed only under `$TMPDIR`. Test runs must not
write reports or caches into the repository.

## Reference Pipeline

### Transform

The official reference is executed independently on each sample's real,
valid waveform:

```python
ASTFeatureExtractor(
    sampling_rate=16000,
    num_mel_bins=128,
    max_length=1024,
    do_normalize=True,
    mean=-4.2677393,
    std=4.5689974,
)
```

The local side uses a unified 10.255-second canvas plus `valid_seconds`. The
reference side uses per-sample cropped waveforms. Both sides are compared
over the full `[B,1024,128]`, including the spectral-domain padding rows.

Official processing order:

```text
real waveform
→ Kaldi fbank
→ zero-pad or truncate to 1024 rows at the fbank level
→ (fbank - mean) / (2 × std)
```

This project reproduces this semantics whenever the valid input does not
exceed 10.255 seconds; behavior beyond that limit is separately tested as
raising `ValueError`.

### Encoder

The reference and the local side use the same pinned backbone weights, but
their inputs come from the official Transform and the local Transform
respectively. Legacy parameter keys in the checkpoint are first converted
according to Transformers 5.13.1's official ViT-style conversion rule; tests
must confirm that the converted 199 local keys and values match the official
loader's `ASTModel` `state_dict` exactly.

clip:

```text
reference = ASTModel(reference_features).pooler_output
actual    = AstEncoder(local_features)["embedding"]
```

frame:

```text
reference_hidden = ASTModel(reference_features).last_hidden_state
reference_frame = mean(
    reshape(reference_hidden[:,2:,:], [B,12,101,768]),
    frequency_axis,
)
```

The reference frame is then subjected to the same valid_mask and zero
padding as the project contract. This comparison verifies that the local
derived implementation is faithful to the frozen formula; it does not claim
this to be an officially published frame API.

## Test Matrix

### Signals

At minimum:

- fixed-seed random waveform;
- multi-frequency sine;
- impulse;
- silence;
- non-zero invalid padding.

All random inputs use a fixed seed.

### Durations

Mandatory durations:

```text
0.025s
0.9999375s
4.03s
10.0s
10.245s
10.255s
```

Additional discrete boundaries:

```text
0, 1, 399, 400, 559, 560, 164079, 164080 target samples
```

Where:

- 0 target samples uses a positive-duration input at a different sample rate
  that rounds to 0, validating only the project extension;
- `1..399` validates only the project extension;
- `400..164080` validates the full official Transform alignment;
- beyond 10.255 seconds validates the project's `ValueError`;
- Hugging Face's silent truncation beyond this length is not treated as a
  successful project output.

### Batch

Also covers:

- `B=1`;
- mixed-length batch;
- mixed batch vs. per-sample cropping;
- short valid prefixes inside physically long tensors;
- changes to invalid-tail content must not affect the output.

### Output Levels

Verified layer by layer:

1. full fbank;
2. checkpoint/config/source identity;
3. patch token grid;
4. `last_hidden_state`;
5. clip embedding;
6. frame embedding;
7. geometry;
8. valid_mask;
9. exact-zero invalid frames.

### Device

- CPU is mandatory;
- CUDA is mandatory when available;
- CUDA is explicitly skipped when unavailable;
- TF32 is disabled, to avoid gate drift from hardware defaults;
- MPS is not required.

## Numerical Gates

Transform:

```text
atol = 1e-4
rtol = 0
cosine_similarity >= 0.99999
relative_L2 <= 1e-4
```

Encoder:

```text
atol = 1e-4
rtol = 1e-4
cosine_similarity >= 0.99999
relative_L2 <= 1e-4
```

The pointwise Encoder `atol/rtol` gate applies to the public clip/frame
embeddings. The full `last_hidden_state`, as a non-public intermediate audit
quantity, is recorded via its maximum absolute difference and is subject to
the cosine and relative L2 gates; after front-end micro-errors propagate
through 12 layers, a few near-zero hidden elements are not held to the
pointwise gate used for public outputs.

The following use exact assertions:

- shape;
- dtype;
- geometry;
- valid_mask;
- exact-zero invalid frames;
- checkpoint/source SHA;
- key config fields;
- state-key set;
- exception type for out-of-range inputs.

Absolute L2 is recorded as an audit metric but does not itself carry a
cross-shape threshold; relative L2 is the unified gate across Transform,
clip, and frame.

## Intentional Differences

### Overlong Input

Hugging Face truncates waveforms producing more than 1024 native fbank
frames to the first 1024 rows; this project raises `ValueError` for inputs
whose valid duration exceeds 10.255 seconds. This is a deliberate difference
made to keep `valid_seconds`, geometry, and the encoded content consistent.

### Ultra-short Input

Hugging Face's torchaudio path rejects waveforms shorter than 400 samples;
this project returns a feature that is fully spectral-domain padding while
retaining one valid ownership slot. Under a different sample rate, even a
positive duration that rounds to 0 target samples follows the same ownership
semantics. This is a project extension.

### Frame

The official implementation has no frame embedding API. The project's frame
is a deterministic derivation from the pinned official `last_hidden_state`.

### Final 80 ms

After patch convolution (16×16, stride 10) over the 1024 fbank rows, only
101 time patches are produced; the last one uses rows `1000..1015`. Rows
`1016..1023` never enter the model, so the tail of roughly 10.175-10.255
seconds does not affect the embedding.

Alignment tests must include a tail-perturbation case demonstrating that:

- tail perturbation can change the last 8 fbank rows;
- patch tokens, clip, and frame embeddings remain unchanged.

## Pre-implementation Review Measurements

During the design interview, a migrated reference implementation was
pre-reviewed using the pinned revision, Transformers 5.13.1, Torch
`2.11.0+cu130`, and an NVIDIA GH200. It covered the six durations above
together with random, multi-frequency sine, impulse, and non-zero
invalid-padding signals.

Worst-case results:

| Path | Output | Max Abs Diff | Min Cosine Similarity | Max Relative L2 |
|---|---|---:|---:|---:|
| CPU | Transform | 1.43e-6 | 0.999999999999938 | 4.31e-8 |
| CPU | clip | 1.26e-5 | 0.999999999998674 | 1.63e-6 |
| CPU | frame | 2.29e-5 | 0.999999999999461 | 1.04e-6 |
| CUDA | Transform | 7.74e-5 | 0.999999999996055 | 2.79e-6 |
| CUDA | clip | 7.99e-6 | 0.999999999998774 | 1.57e-6 |
| CUDA | frame | 7.25e-5 | 0.999999999996704 | 2.55e-6 |

These results demonstrate that the chosen batched frontend can be aligned
with the pinned HF pipeline within the proposed gates, but they cannot
substitute for a formal re-run against the new class and new output contract
once the migration is complete.

## Post-implementation Formal Verification

### Implementation and Environment Identity

This verification corresponds to a working tree based on repository commit
`d493201f384e1832d5b2a765b91329e2409a59a4`, containing the AST migration
changes described in this design. No additional commit had yet been created
when the implementation was completed.

| Component | Version or Device |
|---|---|
| Python | 3.12.13 |
| pytest | 9.1.1 |
| Torch | 2.11.0+cu130 |
| torchaudio | 2.11.0+cu130 |
| Transformers | 5.13.1 |
| safetensors | 0.8.0 |
| CUDA runtime | 13.0 |
| CUDA device | NVIDIA GH200 120GB |

Both CPU and CUDA are available; CUDA matmul and cuDNN TF32 were disabled
during the alignment tests.

### Commands and Results

Full AST alignment:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest \
  tests/models --run-alignment ast -q -p no:cacheprovider
```

Result:

```text
117 passed, 1 skipped in 42.12s
```

Of these, the AST-specific alignment file accounts for `10 passed`; the only
skip is the PANNs alignment, which was not enabled. This result covers the
pinned identity, the 199 backbone tensors, the CPU/CUDA primary matrix, five
additional official discrete boundaries, the 0/1/399-sample project
extensions, and the out-of-range error. Results for the ordinary offline
model tests and the full-repository tests appear at the end of this section.

### Formal Numerical Summary

Each column in the table below is taken from the worst case across the full
formal matrix, so different columns in the same row are not necessarily from
the same case. Absolute L2 is for audit only; the gates use the pointwise
difference, cosine, and relative L2.

| Device | Output | Max Abs Diff | Min Cosine Similarity | Max Absolute L2 | Max Relative L2 |
|---|---|---:|---:|---:|---:|
| CPU | Transform | 1.10e-5 | 0.999999999998311 | 2.05e-5 | 9.88e-8 |
| CPU | hidden audit | 1.54e-4 | 0.999999999995171 | 3.04e-3 | 3.06e-6 |
| CPU | clip | 1.60e-5 | 0.999999999997051 | 8.30e-5 | 2.44e-6 |
| CPU | frame | 2.29e-5 | 0.999999999999418 | 2.90e-4 | 1.07e-6 |
| CUDA | Transform | 7.80e-5 | 0.999999999997181 | 3.97e-4 | 1.91e-6 |
| CUDA | hidden audit | 7.92e-4 | 0.999999999986911 | 5.09e-3 | 5.09e-6 |
| CUDA | clip | 2.53e-5 | 0.999999999992136 | 1.42e-4 | 3.97e-6 |
| CUDA | frame | 6.87e-5 | 0.999999999992078 | 4.39e-4 | 3.98e-6 |

Worst-case cases:

- CPU Transform: `sine:0.9999375`;
- CPU clip: `impulse:4.03`;
- CPU frame: `multisine:10.245`;
- CUDA Transform, hidden audit, and frame: `multisine:10.245`;
- CUDA clip: `sine:0.9999375`.

The public Transform, clip, and frame outputs all pass the established
pointwise, cosine, and relative L2 gates. A few near-zero elements in the
hidden audit amplify small front-end differences, so only the global gates
specified in this section apply to it; its worst-case relative L2 is
`5.09e-6`.

### Identity and Semantic Checks

- All three snapshot-file SHA-256 digests match the pinned table;
- Both Transformers key source-file SHA-256 digests match the pinned table;
- The safetensors file contains exactly 199 backbone tensors and 4
  classifier tensors;
- After legacy checkpoint keys are converted via the official ViT-style
  rule, all 199 `state_dict` keys and values between the local and official
  loaders are exactly equal item by item;
- geometry, valid_mask, and invalid-frame zero values pass exact assertions;
- AST's adjacent valid-frame geometry boundaries are element-wise equal;
- when the waveform is perturbed starting at 10.175 seconds, only the last 8
  fbank rows change, while patch tokens, hidden states, clip, and frame
  embeddings remain exactly unchanged;
- when a positive duration at a different sample rate rounds to 0 target
  samples, one valid ownership slot is still retained; the 1- and
  399-sample project extensions succeed, and a 164081-sample valid input
  explicitly raises `ValueError`;
- this formal-run environment had no CUDA skips, failures, or other
  environment limitations.

Ordinary offline model tests:

```text
107 passed, 11 skipped in 27.57s
```

Full-repository offline tests:

```text
222 passed, 11 skipped, 82 warnings in 32.94s
```

All 11 skips are official alignment tests not explicitly enabled; the
warnings come from deprecation notices in existing dataset tests and
`fork()` notices in multi-threaded processes.

The test run itself did not modify the repository; this summary was
reviewed and committed by the implementer.
