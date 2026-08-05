# Wav2Vec2 Official Implementation Alignment Contract

This document freezes the official implementation alignment contract for
`Wav2Vec2WaveformTransform` and `Wav2Vec2Encoder`, and records the fixed
verification matrix and measured summary once the formal pinned run is
complete.

Corresponding designs:

- [`../transforms/wav2vec2.md`](../transforms/wav2vec2.md)
- [`../encoders/wav2vec2.md`](../encoders/wav2vec2.md)

## Alignment Target

The mandatory primary reference is the pinned Hugging Face implementation:

```text
Wav2Vec2FeatureExtractor + Wav2Vec2Model
facebook/wav2vec2-base
```

Verification covers:

1. Numerical alignment between the local batched waveform frontend and the
   official per-sample `Wav2Vec2FeatureExtractor`;
2. Complete and accurate extraction of the 211 backbone tensors from the
   pinned `Wav2Vec2ForPreTraining` checkpoint (prefix stripping, pos_conv
   weight-norm renaming, strict loading), verified value by value against
   the official `from_pretrained` loader;
3. Alignment between the local frame embedding and the official
   `last_hidden_state` on identical exact-length inputs;
4. Alignment between the project-derived clip embedding and the valid-frame
   mean of the official `last_hidden_state`;
5. The conv frame-count formula against the actual official model output
   lengths at the discrete boundaries;
6. Compliance of geometry, valid_mask, and zero padding with the project's
   base-class contract;
7. Independent testing of the project's length restriction and intentional
   differences.

"Official alignment" does not cover:

- Standalone execution of the original fairseq implementation;
- The CTC/ASR heads and the pretraining quantizer heads;
- An officially published clip embedding, since no such upstream API exists;
- The official padded-batch normalization path
  (`return_attention_mask=False` leaks padding into valid samples upstream;
  this behavior is deliberately not replicated);
- Pointwise upstream equivalence for the project's multi-channel downmixing
  and arbitrary sample-rate resampling;
- fp16, bf16, MPS;
- SpecAugment or other training-time behavior.

## Pinned Identity

### Model Snapshot

```text
repo_id:
  facebook/wav2vec2-base
revision:
  0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8
```

Pinned files:

| File | SHA-256 |
|---|---|
| `config.json` | `4937977e24d12d1bba70cdce8709c3c04807a8e4ae8ddac4229c48c436ae99ae` |
| `preprocessor_config.json` | `b225d617c025463b9e157e06afea8b90dc7078fc70b013c533328423e0486b4a` |
| `pytorch_model.bin` | `3249fe98bfc62fcbc26067f724716a6ec49d12c4728a2af1df659013905dff21` |

Hub `main`, other revisions, or local files lacking digest verification are
not accepted. The checkpoint is a pickle file; it is read exclusively with
`torch.load(weights_only=True)`.

### Transformers Implementation

```text
transformers: 5.14.1
torch: 2.11.0
```

Key source files:

| File | SHA-256 |
|---|---|
| `feature_extraction_wav2vec2.py` | `e5e9a0baf70716fee503f4f66a7a61312a132be989b2d7e2649e057ccbefa2cc` |
| `modeling_wav2vec2.py` | `c6ee256c01c9c640f7e00dabe6bb480d6e6f8aae532671f46acbbe95c198cd2d` |

An explicit alignment run must verify both the version and the key
source-file digests, plus every architecture field in
`WAV2VEC2_CONFIG_FIELDS` and the `do_normalize` / `sampling_rate` /
`return_attention_mask` fields of the pinned preprocessor config. If they do
not match, the implementation closure mismatch must be reported, and no
numeric result that could be mistaken for a pinned official result may be
generated.

Ordinary offline unit tests do not enforce a global Transformers version.

## Entry Points

Ordinary model tests:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest \
  tests/models -v -p no:cacheprovider
```

Explicit wav2vec2 official alignment:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest \
  tests/models --run-alignment wav2vec2 -v -p no:cacheprovider
```

`tests/models/conftest.py` registers the `wav2vec2` alignment entry. Without
passing `--run-alignment wav2vec2`, tests against real weights must be
skipped and must not trigger any download.

The snapshot directory can be pinned explicitly through the environment:

```bash
TIMBRAL_WAV2VEC2_SNAPSHOT=/path/to/snapshot
```

When set, the fixture verifies (or downloads into) that directory via
`ensure_wav2vec2_checkpoint`; otherwise the snapshot lives under
`$TMPDIR/timbral-wav2vec2-alignment/snapshot`. Intermediate download
artifacts and dynamic run outputs must be placed only under `$TMPDIR`. Test
runs must not write reports or caches into the repository.

## Reference Pipeline

### Transform

The official reference is executed independently on each sample's real,
valid waveform, always at the exact valid length:

```python
Wav2Vec2FeatureExtractor.from_pretrained(
    snapshot_directory,
    local_files_only=True,
)
```

The local side uses a unified `[B, max(valid_samples)]` canvas plus
`valid_seconds`; the reference side receives per-sample cropped waveforms.
Comparison runs over the exact valid prefix of each row, and the canvas tail
beyond `valid_samples` is asserted to be exactly 0. Feeding the official
extractor exact lengths is deliberate: with the base checkpoint's
`return_attention_mask=False`, the official padded-batch path normalizes
over the padded length, which the project intentionally does not replicate.

For foreign sample rates, the reference is constructed as: torchaudio
resampling of the exact valid prefix to 16 kHz (the same resampler the
Transform uses), crop or right-pad to `round(valid_seconds × 16000)`, then
the official extractor.

### Encoder

```python
Wav2Vec2Model.from_pretrained(
    snapshot_directory,
    local_files_only=True,
)
```

Loading the pretraining checkpoint through `from_pretrained` prints a load
report with 7 UNEXPECTED keys (`quantizer.*`, `project_q.*`,
`project_hid.*`); that is the expected pretraining-head remainder, not an
error. The local Encoder instead performs the project's strict extraction
(prefix stripping, pos_conv renaming, 211-tensor count, `strict=True`), and
the two state dicts must be exactly equal item by item.

At the encoder gate, both sides consume the same exact-length input (the
local Transform's valid prefix), so encoder alignment is isolated from
transform micro-errors:

```text
reference_hidden = Wav2Vec2Model(exact_input).last_hidden_state
clip reference   = mean(reference_hidden, time axis)
frame reference  = reference_hidden scattered onto the project canvas
```

The 44.1 kHz case additionally audits the compounded end-to-end error, with
each side consuming its own transform output. The clip embedding is a
project-derived quantity (valid-frame mean); the frame embedding is the
official `last_hidden_state` itself.

## Test Matrix

### Signals

At minimum:

- fixed-seed random waveform;
- single-frequency sine;
- impulse;
- silence;
- multi-frequency sine plus noise (multisine);
- non-zero invalid padding.

All random inputs use a fixed seed.

### Durations

Mandatory 16 kHz durations, with their exact sample and frame counts:

```text
0.025 s     ->    400 samples ->   1 frame
0.0449375 s ->    719 samples ->   1 frame
0.045625 s  ->    730 samples ->   2 frames
1.0 s       ->  16000 samples ->  49 frames
2.0 s       ->  32000 samples ->  99 frames
10.0 s      -> 160000 samples -> 499 frames
```

Additional discrete boundaries:

```text
400, 719, 720, 730, 16000 target samples
```

Where:

- 400 is the one-receptive-field minimum and must succeed;
- 719/720 is the exact 1-frame/2-frame conv boundary, checked against the
  actual official model output length;
- 399 target samples (0.0249375 s) must raise `ValueError`, both natively
  and via a foreign sample rate that rounds to 399;
- wav2vec2 has no upper duration bound, so no overlength error exists.

### Sample Rates

- 16 kHz native;
- 44.1 kHz foreign (0.9 s and 2.5 s), aligned per sample against the
  official extractor applied after identical resampling, plus an
  end-to-end hidden/clip audit.

### Batch

Also covers:

- `B=1`;
- mixed-length batch;
- mixed batch vs. per-sample cropping bit-identical, at transform and
  frame-encoder level (all matrix lengths are unique, so every encoder
  group holds a single sample; see Intentional Differences);
- short valid prefixes inside physically long tensors;
- rewriting non-zero invalid-tail content must not change any output bit.

### Output Levels

Verified layer by layer:

1. checkpoint/config/preprocessor/source identity;
2. exact backbone state equality against the official loader;
3. full waveform features;
4. `last_hidden_state` (audit);
5. clip embedding;
6. frame embedding;
7. frame counts vs. the official output length;
8. geometry;
9. valid_mask;
10. exact-zero invalid frames.

### Device

- CPU is mandatory;
- CUDA is mandatory when available;
- CUDA is explicitly skipped when unavailable;
- TF32 (matmul and cuDNN) is disabled during the run and restored after;
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
quantity, is recorded via its maximum absolute difference and is subject
only to the cosine and relative L2 gates.

Silence maps to exactly zero features on both sides; after the pointwise
gate passes, its degenerate direction metrics are recorded as perfect rather
than dividing by a zero norm.

The following use exact assertions:

- shape;
- dtype;
- `valid_samples` counts;
- frame counts vs. the official output length;
- geometry;
- valid_mask;
- exact-zero invalid frames and canvas tails;
- checkpoint/source SHA-256;
- key config and preprocessor fields;
- state-key set and every tensor value;
- exception type for sub-minimum inputs.

Absolute L2 is recorded as an audit metric but does not itself carry a
cross-shape threshold; relative L2 is the unified gate across Transform,
clip, and frame.

## Intentional Differences

### Padded-batch Normalization

With `return_attention_mask=False` (the `wav2vec2-base` default), the
official extractor normalizes a padded batch over the padded length, leaking
padding into valid samples. The project always normalizes each sample over
its exact valid region. Alignment therefore feeds the official extractor
per-sample exact-length waveforms; the padded-batch path is out of scope.

### Ultra-short Input

The official extractor accepts any length, but below 400 samples the conv
stack produces zero output frames and the official model itself fails. The
project raises `ValueError` for fewer than 400 target samples instead of
silently padding, turning the inherent model limit into an explicit error.

### Clip

The official implementation has no clip embedding API. The project's clip is
the arithmetic mean of the valid last-layer frames, a deterministic
derivation from the pinned official `last_hidden_state`.

### Resampling and Downmixing

Mean downmixing and arbitrary-rate torchaudio resampling are project
extensions; the foreign-rate reference is constructed with the same
resampler before the official extractor, not claimed as an upstream API.

### No Attention Mask

The project never builds an attention mask; the grouped exact-prefix forward
makes one unnecessary, matching how the base checkpoint was trained.

### Gradient Checkpointing

The official `config.json` carries the legacy `gradient_checkpointing: true`
field, and Transformers 5.14.1 still honors it: `from_pretrained` (and a
config built via `from_json_file`) constructs the model with gradient
checkpointing enabled. The project's fixed config deliberately omits this
training-runtime toggle — the constructor makes no training-policy
decisions, consistent with not calling `eval()` or freezing — so the local
backbone reports `is_gradient_checkpointing=False`. Evaluation outputs and
this alignment are unaffected; callers that want the official training
memory behavior call `encoder.backbone.gradient_checkpointing_enable()`.

### Multi-sample Groups

The bit-identity statements in this contract cover the transform output at
any batch composition, and encoder outputs whose length group holds a single
sample — the formal matrix uses unique lengths, so every encoder comparison
runs in that regime. A group holding several same-length samples matches
per-sample calls only to tolerance (floating-point kernel batching inside
the Transformer layers; see the encoder design document).

## Pre-run Smoke Reference

During the design interview, a 44.1 kHz per-sample smoke comparison against
the pinned identity measured: transform max abs diff ~6e-8; last hidden
max abs diff <= 1e-4; cosine >= 0.999999999. These measurements motivate the
starting gates above but cannot substitute for the formal pinned run against
the final test matrix.

## Post-implementation Formal Verification

### Implementation and Environment Identity

The formal run used the project container on the pinned identity.

| Component | Version or Device |
|---|---|
| Python | 3.12.13 |
| pytest | 9.1.1 |
| Torch | 2.11.0+cu130 |
| torchaudio | 2.11.0+cu130 |
| Transformers | 5.14.1 |
| CUDA runtime | 13.0 |
| CUDA device | NVIDIA GH200 120GB |

### Commands and Results

Full wav2vec2 alignment (file-scoped, as actually executed for this
record; the suite-wide
`tests/models --run-alignment wav2vec2` form runs the same 11 alignment
items alongside the offline suite):

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest \
  tests/models/test_wav2vec2_alignment.py \
  --run-alignment wav2vec2 -v -p no:cacheprovider
```

Result:

```text
11 passed in 42.23s (CPU cases plus CUDA cases on a GH200, TF32 disabled)
```

### Formal Numerical Summary

Each column in the table below is taken from the worst case across the full
formal matrix, so different columns in the same row are not necessarily from
the same case. Absolute L2 is for audit only; the gates use the pointwise
difference, cosine, and relative L2. The numbers come from the per-device
worst-case buckets printed by the alignment tests. All `TBD` cells are
backfilled after the pinned run.

| Device | Output | Max Abs Diff | Min Cosine Similarity | Max Absolute L2 | Max Relative L2 |
|---|---|---:|---:|---:|---:|
| CPU | Transform | 1.91e-06 | 0.99999999999995 | 1.91e-06 | 7.07e-08 |
| CPU | hidden audit | 0.0 | 0.99999999999990 | 0.0 | 0.0 |
| CPU | clip | 0.0 | 1.0 | 0.0 | 0.0 |
| CPU | frame | 0.0 | 0.99999999999990 | 0.0 | 0.0 |
| CUDA | Transform | 1.91e-06 | 0.99999999999995 | 1.91e-06 | 7.07e-08 |
| CUDA | hidden audit | 0.0 | 0.99999999999991 | 0.0 | 0.0 |
| CUDA | clip | 0.0 | 1.0 | 0.0 | 0.0 |
| CUDA | frame | 0.0 | 0.99999999999991 | 0.0 | 0.0 |
| CPU 44.1 kHz | Transform | 4.77e-07 | 0.99999999999997 | 1.47e-05 | 1.23e-07 |
| CPU 44.1 kHz | hidden audit | 4.61e-05 | 0.99999999993199 | 1.11e-03 | 1.17e-05 |
| CPU 44.1 kHz | clip | 2.38e-05 | 0.99999999993701 | 9.34e-05 | 1.12e-05 |

Encoder-level comparisons feed the identical exact-length input to the local
backbone and the official `Wav2Vec2Model`, so the native-rate hidden, clip,
and frame outputs are bit-identical (max abs diff exactly 0.0); the sub-1.0
cosine entries at zero pointwise difference are float64 audit rounding. The
44.1 kHz rows measure the end-to-end composite (project resampling and
normalization feeding both stacks), where the transform's ~5e-7 input
difference is amplified through the 12 Transformer layers.

Worst-case cases:

```text
transform (16 kHz):        impulse:0.045625 (max abs / L2),
                           random_invalid_tail:10.0 (min cosine)
hidden/clip/frame (16 kHz): random:0.025 (max abs, exactly 0),
                           random_invalid_tail:10.0 (min cosine audit)
transform (44.1 kHz):      multisine:0.9 (max abs / L2),
                           random:2.5 (min cosine)
hidden/clip (44.1 kHz):    random:2.5 (all buckets)
```

### Identity and Semantic Checks

All exact (non-tolerance) checks passed in the pinned run:

- three snapshot files verified against the pinned SHA-256 set, and the
  module constants cross-checked against `WAV2VEC2_CHECKPOINT`;
- Transformers 5.14.1 version assertion plus SHA-256 source locks on
  `modeling_wav2vec2.py` and `feature_extraction_wav2vec2.py`;
- `config.json` architecture fields and `preprocessor_config.json`
  (`do_normalize=true`, `sampling_rate=16000`,
  `return_attention_mask=false`) matched the fixed contract;
- the 211 filtered backbone tensors are `torch.equal` to the official
  `Wav2Vec2Model.from_pretrained` state, including the pos_conv
  parametrize key names;
- valid frame counts matched the official model's actual output lengths at
  the 400/719/720/730/16000-sample boundaries;
- canvas tails beyond `valid_samples`, invalid frame embeddings, and
  invalid geometry rows are exactly 0; rewriting invalid tails left every
  output bit-identical;
- 399 target samples raised `ValueError` at native and foreign rates;
- mixed batches are bit-identical to per-sample cropped calls on both
  devices (unique matrix lengths: every encoder group held one sample).
