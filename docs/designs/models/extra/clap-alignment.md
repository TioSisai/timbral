# CLAP Official Alignment Design

This document defines the official identity, project routing contract, test
gates, and reproduction boundaries for `ClapLogmelTransform` and
`ClapHtsatEncoder`.

## Scope of the Alignment Conclusion

The pinned reference implementation is `ClapFeatureExtractor + ClapModel`
from Hugging Face Transformers 5.13.1, with the pinned model
`laion/clap-htsat-fused`.

The project uses per-sample length-based routing:

```python
fusion_mask = round(valid_seconds * 48000) > 480479
```

Short audio always takes the global-only path; long audio uses the official
global path plus three deterministic-anchor local crops. Within an all-short
batch, Hugging Face randomly marks one short sample as `is_longer=True`, but
this does not change that sample's four-channel front-end features. The
project's reference side applies the length routing above before invoking
the official audio tower, thereby verifying the official component's
numerics under the project's contract.

The project's long-audio crop starting points are given by a
deterministic-anchor formula (see
[`../transforms/clap.md`](../transforms/clap.md)), which intentionally
deviates from the official random-crop protocol: generating a
representation requires the calls to be reproducible and batch-independent,
whereas the upstream random protocol has no fixed output that could be
aligned against in the first place. Official alignment is achieved by
forcing the official reference side's crop selection to the project's
deterministic starting points, thereby verifying the feature and model
computation given the same starting point; the crop starting point itself
is defined by the project's formula and is not compared against the
official RNG sequence.

"Passing official alignment" therefore specifically means:

1. The numerics from waveform to four-channel input features match the
   pinned official frontend;
2. Given the same crop starting point, the long-audio global/local fusion
   features match, and the project's forward pass does not consume the
   global RNG;
3. The project's routing mask matches the pinned discrete-length formula;
4. The audio tower, audio projection, and L2 normalization match the full
   official model;
5. Routing for an all-short batch is determined solely by per-sample
   length.

## Pinned Identity

### Model Snapshot

```text
repo_id:  laion/clap-htsat-fused
revision: 365dea6ef167def6676140ed93bbc43f84dabb28
```

| File | SHA-256 |
|---|---|
| `config.json` | `b1d63489dc5061da229c23d2b11e9ca731639574449f82319fabb01da7fcf480` |
| `preprocessor_config.json` | `072bdd9ba771b6d213c56f15c0f765e33192b92e481581b52271cf16c9013684` |
| `model.safetensors` | `3f648de6d030e17494be455d323b8d191233fbae0c7ce0ba745fd21a926a63a6` |

### Official Implementation

The formal closure:

| Component | Version or SHA-256 |
|---|---|
| Transformers | `5.13.1` |
| `feature_extraction_clap.py` | `a2bc74b2f7e3d11bb704b9e7699705e2d5bfe62400375f18020dda6f7382db45` |
| `modeling_clap.py` | `2e1739468cd53541dcb53a985a66b5858ac2be8047cc75fc5a1dcc2fd268f1c8` |

At alignment-session initialization, the version, source-code digests,
snapshot revision, and the three file digests are all strictly checked.
Production construction does not perform a hard version-string check; other
Transformers versions may join the supported closure only after completing
the same alignment.

## Test Entry Points

Ordinary model tests:

```bash
python -m pytest tests/models -v
```

Real CLAP alignment:

```bash
python -m pytest \
  tests/models/test_clap_alignment.py \
  --run-alignment clap -v
```

`clap` is added to the `--run-alignment` choices in
`tests/models/conftest.py`. Ordinary tests skip real-weight alignment by
default and do not download the checkpoint.

## Identity and Weight Gates

The tests first confirm:

- All key fields of config and preprocessor;
- The top-level key set of safetensors;
- The project-selected `audio_model.*` amounts to exactly 266 tensors;
- The project-selected `audio_projection.*` amounts to exactly 4 tensors;
- The excluded set contains only the pinned text tower, text projection,
  and logit scale;
- The project's audio-only state matches the corresponding state of the
  full official model using `torch.equal`;
- After the project model is loaded with `strict=True`, all 270 tensors are
  equal value-for-value.

Any unknown, missing, duplicated, or shape-mismatched key causes the
alignment to fail.

## Transform Alignment

### Official Per-sample Features

The pinned `ClapFeatureExtractor` is invoked independently on each valid
prefix, comparing `input_features [4,1001,64]`.

Target sample counts covered:

```text
1
479999
480000
480001
480479
480480
480481
typical long audio
long audio beyond the range where seconds can round-trip exactly in float32
```

Where:

- 1-479999 verifies repeatpad;
- 480000 verifies the identity window;
- 480001-480479 verifies the 1001-frame global-only boundary;
- 480480 verifies the first fusion boundary;
- longer inputs verify the deterministic-anchor global/local fusion.
- Long inputs with `valid_seconds=None` verify that the physical sample
  count directly determines the mel grid, without going through a
  float32-seconds back-calculation.

Signal coverage:

- silence;
- impulse;
- sine;
- multisine;
- fixed-seed random waveform.

### Deterministic Crop Alignment

Inside `_random_mel_fusion`, the official reference side uses the global
`np.random.choice` to successively pick the front, middle, and back crop
starting points. For each long-audio comparison:

1. Compute the three deterministic-anchor starting points using the
   project's formula;
2. Monkeypatch `np.random.choice` to return these three starting points in
   order from a queue, while the rest of the official code path executes
   unchanged;
3. Run the project's Transform and compare the four-channel features;
4. Assert that the project's forward pass does not alter the global NumPy
   RNG state.

For an all-short batch, no crop RNG is consumed, and the routing mask is
entirely False. The test separately records HF
`ClapFeatureExtractor.__call__`'s random forced-flag behavior and confirms
that it does not enter the project's routing contract.

### Project's Common Input Layer

The official Hugging Face frontend only accepts mono 48 kHz arrays. The
project's extensions are verified in layers:

- Mean-downmixed multi-channel input matches an explicit mono input;
- The project's resampling result for non-48 kHz input matches resampling
  by the same torchaudio rule followed by the official frontend;
- Mixed `valid_seconds` matches the result of processing each valid prefix
  individually;
- Filling the invalid tail with non-zero values does not change the
  output;
- When the physical tensor is longer than the valid prefix, only the valid
  prefix is consumed.

## Encoder and End-to-End Alignment

The official reference uses the full `ClapModel`, while the project side
uses `ClapAudioModelWithProjection`. Both sides:

- Load the same pinned snapshot;
- Enter eval mode;
- Use inference mode;
- Receive value-aligned `input_features`;
- Produce the same routing mask using the project's length formula;
- Have their audio backbone pooler compared;
- Have their audio projection compared;
- Have their final L2-normalized `[B,512]` embedding compared.

The end-to-end matrix covers:

- batch size 1 and greater than 1;
- an all-short batch;
- a short/long mixed batch;
- the 480479/480480 discrete boundary;
- mixed `valid_seconds`;
- CPU;
- CUDA when available, with TF32 disabled.

CLAP does not support frame; the alignment does not include frame
embeddings or frame geometry.

## Numerical Gates

| Object | Gate |
|---|---|
| File digest | Exact string equality |
| Weights and fixed buffers | `torch.equal` |
| routing mask, shape, dtype | Exact equality |
| float64 internal frontend output | `torch.testing.assert_close(atol=1e-4, rtol=1e-5)` |
| backbone/projection/final embedding | `torch.testing.assert_close(atol=1e-5, rtol=1e-5)` |
| Final embedding cosine | `>= 0.999999` per sample |

The max absolute error, relative L2, and cosine are recorded as well;
failing any primary gate fails the pytest run. CPU and CUDA are each
compared against the official reference on the same device.

## Environment and Reproduction

The formal alignment report records:

- Git commit;
- Versions of Python, PyTorch, CUDA, cuDNN, NumPy, Transformers,
  torchaudio, huggingface_hub, and safetensors;
- Device name;
- Snapshot revision and file digests;
- Official source-code digests;
- Random seed;
- Error metrics and pass/fail status for each matrix case.

Downloads and temporary caches use explicit test directories; development
and audit runs prefer `$TMPDIR`. Tests must not create new temporary caches
in the project directory, the user's home directory, `/projappl`, or
`/scratch`.
