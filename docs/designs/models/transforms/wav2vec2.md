# `Wav2Vec2WaveformTransform` Design

This document finalizes the design of
`timbral.models.transforms.Wav2Vec2WaveformTransform`. The companion Encoder is
described in [`../encoders/wav2vec2.md`](../encoders/wav2vec2.md); the official
alignment contract is in
[`../extra/wav2vec2-alignment.md`](../extra/wav2vec2-alignment.md).

This document describes the behavior the current implementation must satisfy.
Where this document conflicts with the `BaseTransform` common contract,
[`base.md`](base.md) takes precedence.

## Design Goals

`Wav2Vec2WaveformTransform` is responsible for:

- accepting the waveform input mandated by `BaseTransform`;
- producing the raw 16 kHz waveform input expected by the fixed Hugging Face
  wav2vec2 frontend — this is the project's first waveform-passthrough
  Transform, and `input_features` is a `[B,N]` waveform rather than a
  `[B,T,F]` spectrogram;
- replicating the official `Wav2Vec2FeatureExtractor` per-sample zero-mean
  unit-variance normalization on the valid region;
- supporting mean downmixing and arbitrary positive-integer input sample
  rates;
- isolating any non-zero padding outside `valid_seconds`;
- rejecting valid audio shorter than one conv receptive field;
- maintaining the device, gradient, and serialization semantics of an
  ordinary `nn.Module`.

This component does not hold an Encoder, does not load model weights, does not
generate an attention mask, and does not run any part of the conv frontend
(the conv feature extractor belongs to the Encoder's backbone weights).

## Supported Model Identity

At this stage, only the following is supported:

```text
facebook/wav2vec2-base
```

The public constructor exposes a single switch:

```python
class Wav2Vec2WaveformTransform(BaseTransform):
    def __init__(self, *, do_normalize: bool = True) -> None:
        ...
```

`do_normalize` mirrors the field of the same name in the official
`preprocessor_config.json` and must be a strict Python `bool`. The fixed
`facebook/wav2vec2-base` preprocessor enables it, and the registry entry
pins `do_normalize=True` through `fixed_kwargs`, so the registered name
cannot be constructed with a non-official frontend. The switch exists because the transform body is otherwise
identical across the wav2vec2 family, and future variants fix their own value
at registration time. No sample-rate, hop, or receptive-field configuration is
exposed; those are inseparable from the fixed checkpoint identity.

## Fixed Frontend Parameters

| Parameter | Value |
|---|---:|
| Target sample rate | 16,000 Hz |
| Conv receptive field | 400 samples / 25 ms |
| Conv hop | 320 samples / 20 ms |
| Normalization epsilon | 1e-7 |
| Minimum valid target samples | 400 |
| Maximum valid duration | none |

The target sample rate is also set as the public instance attribute
`target_sample_rate = 16000` at construction time, fulfilling the
`BaseTransform` attribute contract. The receptive field and hop are derived
from the fixed conv stack (`kernel=(10,3,3,3,3,2,2)`,
`stride=(5,2,2,2,2,2,2)`) declared in
`helpers/wav2vec2.py`; the Transform only consumes the 400-sample minimum.

wav2vec2 has no absolute positional embedding and therefore no inherent upper
duration bound; the Transform imposes none. Memory for the downstream
Transformer grows quadratically with duration, and running out of memory
surfaces as the underlying operator failure, not as a Transform-level check.

## Public Input

Calling convention:

```python
transform(
    waveform,
    sample_rate=sample_rate,
    valid_seconds=valid_seconds,
)
```

### `waveform`

- must be a floating-point Tensor;
- shape is `[B,N]` or `[B,C,N]`;
- multi-channel input is downmixed by arithmetic mean;
- computation is uniformly cast to `float32`;
- automatically moved to the Transform's `device`.

Integer waveforms are not implicitly scaled or converted; they raise
`TypeError` directly.

### `sample_rate`

- must be a positive Python `int`;
- shared across a batch;
- when not 16 kHz, torchaudio resampling to 16 kHz is used.

### `valid_seconds`

- `None` means the entire physical Tensor is valid, i.e. `N / sample_rate`;
- when not `None`, must be a Tensor of shape `[B]`;
- each element must satisfy `0 < valid_seconds <= N / sample_rate`;
- automatically moved to the Transform's device and cast to `float32`.

The valid sample count on the model's sampling grid is:

```text
source_valid_samples = round(valid_seconds × sample_rate)
target_valid_samples = round(valid_seconds × 16000)
```

When `valid_seconds=None`, the source count is the physical `N` and the
target count is converted directly via the integer ratio
`round(N × 16000 / sample_rate)` (ties to even), avoiding dropped samples
from float32-second round-trips on very long waveforms (> 2^24 samples).

### Minimum length

Every sample must satisfy:

```text
target_valid_samples >= 400
```

i.e. `valid_seconds` of approximately 0.025 s after rounding to the 16 kHz
grid. Below this the conv frontend produces zero output frames, and the
official model itself cannot process the input; the Transform raises
`ValueError` instead of silently padding the audio up to one receptive field.
This mirrors the project rule that turns inherent model limits into explicit
errors.

At extremely low source sample rates, `source_valid_samples` may round to 0
while `target_valid_samples` is still >= 400; the group is then materialized
as an all-zero target-length waveform (same handling as BEATs and CLAP).

## Waveform Preprocessing

The forward-pass order is fixed as:

1. validate input type, shape, sample rate, and `valid_seconds`;
2. move to the Transform's device and cast to `float32`;
3. mean-downmix `[B,C,N]`;
4. group the batch by unique `(source_valid_samples, target_valid_samples)`
   pairs;
5. per group, slice the exact valid prefix `[:, :source_valid_samples]` and
   resample it to 16 kHz;
6. crop or right-pad the group result to exactly `target_valid_samples`;
7. when `do_normalize=True`, normalize each group row to zero mean and
   unit variance — the rows are exact valid lengths, so plain row
   statistics equal valid-region statistics;
8. scatter all groups back onto a `[B, max(target_valid_samples)]` zero
   canvas by batch index.

Although downmix and row selection may access the full physical tensor, each
group forwards only its exact valid prefix, so padding outside the valid
region never influences the output. The batch output is bit-identical to
per-sample calls on cropped inputs; no `masked_fill` sweep is needed. This is
the same unique-length grouping used by the PANNs and BEATs frontends
(`helpers/grouping.py`).

## Normalization

For each sample, with `n = target_valid_samples`:

```text
mean = sum(x[:n]) / n
var  = sum((x[:n] - mean)^2) / n
x[:n] = (x[:n] - mean) / sqrt(var + 1e-7)
```

This replicates `Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm` applied to
an exact-length input: population variance, epsilon 1e-7, float32
computation. The statistics are computed on the exact-length group rows
before canvas assembly — float reductions depend on the reduced width, so
computing them on the padded canvas would break the bit-identity between
mixed batches and per-sample calls. Canvas positions at and beyond `n`
remain exactly 0.

The official extractor, when handed a padded batch with
`return_attention_mask=False` (the `wav2vec2-base` default), normalizes over
the padded length instead — that behavior leaks padding into valid samples
and is deliberately not replicated. The project semantics are always "as if
each sample were processed alone".

An all-zero valid region (silence) yields `mean=0`, `var=0`, and output 0
after the epsilon-guarded division, matching the official extractor.

## Output

```python
{
    "input_features": input_features,
    "valid_samples": valid_samples,
    "valid_seconds": valid_seconds,
}
```

| Key | Shape | Dtype | Semantics |
|---|---|---|---|
| `input_features` | `[B,N16k]` | `float32` | 16 kHz waveform canvas, normalized when `do_normalize=True` |
| `valid_samples` | `[B]` | `int64` | Valid 16 kHz sample count per sample |
| `valid_seconds` | `[B]` | `float32` | Valid duration of the input |

`N16k` is the largest `target_valid_samples` in the batch. `valid_samples`
carries the exact discrete lengths to the companion Encoder for its own
grouping, playing the same role as BEATs' `valid_feature_frames`; deriving it
again from float32 `valid_seconds` inside the Encoder would risk off-by-one
sample counts on long inputs. No attention mask is returned.

## Device, Training, and Serialization

- `device` is derived from a non-persistent empty anchor buffer (the
  Transform owns no numeric state);
- the Transform's `state_dict` is empty;
- it does not automatically call `eval()`;
- it does not freeze parameters;
- it does not wrap `torch.no_grad()` or `torch.inference_mode()`;
- resampling and normalization are continuous Torch operators, so waveform
  gradients propagate naturally;
- discrete control flow such as rounding, grouping, and length checks is not
  promised to be differentiable.

The official alignment contract only covers `float32`. The Transform does not
promise any particular behavior after an explicit conversion to half
precision.

## Files and Exports

Implementation location:

```text
src/timbral/models/transforms/wav2vec2.py
```

`timbral.models.transforms` re-exports `Wav2Vec2WaveformTransform`;
`timbral.models` at the top level only re-exports registry symbols (see
[`../registry.md`](../registry.md)).

## Testing Requirements

The ordinary offline tests must cover at least:

- public export and the keyword-only constructor;
- `do_normalize` strict-bool validation and the `False` path;
- `[B,N]`, `[B,C,N]`, and mean downmixing;
- float64 input normalized to float32, integer input rejected;
- `sample_rate` and `valid_seconds` validation;
- a physically long padded input with a short valid region being legal;
- the 400-target-sample minimum: 400 succeeding, 399 raising `ValueError`,
  including via a foreign sample rate;
- isolation of invalid regions: mixed-length batches bit-identical to
  per-sample cropping, at both native and foreign sample rates;
- normalization statistics computed only over the valid region, with the
  padded tail exactly 0;
- an all-zero valid region producing all-zero output;
- the `valid_samples` integer fast path for `valid_seconds=None`;
- CPU/CUDA device transfer;
- an empty `state_dict`;
- waveform gradients propagating;
- unknown parameters naturally raising `TypeError`.

The full numerical verification against the official implementation is in
[`../extra/wav2vec2-alignment.md`](../extra/wav2vec2-alignment.md).
