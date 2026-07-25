# `BeatsKaldiFbankTransform` Design

This document finalizes the design of
`timbral.models.transforms.BeatsKaldiFbankTransform`. The corresponding Encoder design
is in [`../encoders/beats.md`](../encoders/beats.md), the weight-acquisition procedure is
in [`../extra/beats-download.md`](../extra/beats-download.md), and the official
alignment contract is in [`../extra/beats-alignment.md`](../extra/beats-alignment.md).

This document describes the behavior the current implementation must satisfy. Where this
document conflicts with the `BaseTransform` design, the common contract in
[`base.md`](base.md) takes precedence.

## Design Goals

`BeatsKaldiFbankTransform` is responsible for:

- accepting mono or multi-channel floating-point waveforms;
- clearing invalid regions, mean-downmixing, and resampling to 16 kHz;
- grouping by true valid length so that the result is independent of batch
  composition;
- reproducing the official `torchaudio.compliance.kaldi.fbank` frontend with a
  batched, equivalent implementation;
- applying the official BEATs normalization;
- outputting the exact `valid_feature_frames` so the Encoder can regroup by length.

The official BEATs frontend has no fixed canvas and no duration cap; this Transform
likewise imposes no duration cap. The memory cost of long audio is determined by the
Encoder's self-attention (see [`../encoders/beats.md`](../encoders/beats.md)), and is
the caller's responsibility.

This component does not hold weights (the full state of all 15 checkpoints lives in
the Encoder), does not hold an Encoder, and does not implement a registry, factory, or
YAML configuration layer.

## The Official Frontend

The official implementation (`BEATs.preprocess`) runs sample-by-sample:

```python
waveform = waveform.unsqueeze(0) * 2 ** 15
fbank = ta_kaldi.fbank(
    waveform, num_mel_bins=128, sample_frequency=16000,
    frame_length=25, frame_shift=10,
)
fbank = (fbank - 15.41663) / (2 * 6.55582)
```

Explicit parameters and the torchaudio kaldi defaults that are fixed by this design:

| Parameter | Value | Note |
|---|---|---|
| `sample_frequency` | 16000 | Target sample rate |
| `num_mel_bins` | 128 | Mel dimension |
| `frame_length` | 25 ms = 400 samples | Window length |
| `frame_shift` | 10 ms = 160 samples | Hop |
| `snip_edges` | `True` (default) | No center padding; frame count floors |
| `dither` | `0.0` (default) | No dithering, deterministic result |
| `remove_dc_offset` | `True` (default) | Subtract the mean per frame |
| `preemphasis_coefficient` | `0.97` (default) | Replicate semantics for the first sample |
| `window_type` | `"povey"` (default) | `hann^0.85`, periodic=False grid |
| `round_to_power_of_two` | `True` (default) | 400-point window right-padded to `n_fft=512` |
| `use_power` | `True` (default) | Power spectrum `abs().pow(2)` |
| `low_freq` / `high_freq` | 20.0 / 0.0 (default) | Mel band 20 Hz–Nyquist (8 kHz) |
| `use_log_fbank` | `True` (default) | Natural log, floored at the float32 eps |
| `subtract_mean` / `use_energy` | `False` (default) | Not enabled |

Fixed normalization constants (official hardcoded defaults):

```text
fbank_mean = 15.41663
fbank_std  = 6.55582
normalized = (fbank - fbank_mean) / (2 * fbank_std)
```

Note the `2 *` in the denominator. The `× 2**15` scaling must be applied in the
waveform domain exactly as in the official implementation, and must not be rewritten
as adding a constant in the log domain: the two are mathematically equivalent but
follow different floating-point computation paths.

## Public Constructor Interface

```python
class BeatsKaldiFbankTransform(BaseTransform):
    target_sample_rate = 16000

    def __init__(self) -> None:
        ...
```

No constructor parameters (consistent with `AstKaldiFbankTransform`). The constructor
first calls `super().__init__()`, then registers the frontend buffers (the povey
window and the mel weight matrix), all with `persistent=False`: this Transform holds no
checkpoint state, so its `state_dict` is empty.

No `pretrained` / `pretrained_dir` parameters are provided; weight identity belongs
entirely to the Encoder.

## Input Contract

The calling interface inherits from `BaseTransform`:

```python
transform(waveform, sample_rate=sample_rate, valid_seconds=valid_seconds)
```

### `waveform`

- must be a floating-point Tensor;
- shape is `[B, N]` or `[B, C, N]`;
- `[B, C, N]` is downmixed to mono via arithmetic channel averaging;
- normalized to `float32` before entering the frontend;
- automatically moved to the Transform's `device`.

Integer waveforms, ragged lists, and other ranks must raise errors; no implicit
acceptance is allowed.

### `sample_rate`

- must be a positive Python `int`;
- shared across a batch;
- when different from 16000, `torchaudio.functional.resample` is used.

### `valid_seconds`

- `None` means the entire physical Tensor is valid;
- when not empty, must be `[B]`, satisfying `0 < valid_seconds <= N / sample_rate`;
- moved to the Transform's device and normalized to `float32`.

Discrete valid lengths are rounded per sampling grid using `torch.round`'s
ties-to-even rule (same as PANNs):

```text
source_valid_samples = round(valid_seconds × sample_rate)
target_valid_samples = round(valid_seconds × 16000)
```

When `valid_seconds=None`, no float32-seconds round trip is used:
`source_valid_samples` takes the physical sample count `N` directly, and
`target_valid_samples` is converted using the same ties-to-even rule via integer
rational arithmetic (`helpers.common.round_positive_ratio`), which guarantees no sample
loss for very long waveforms (> 2^24 samples) (same as PANNs/CLAP).

## Unique-Length Grouping

The official frontend runs sample-by-sample and has no padding concept; kaldi
snip_edges framing is sensitive to physical length. The whole batch must not simply be
padded to the longest sample and run through the frontend once. The pipeline is the
same as PANNs:

1. validate the input and compute the source/target valid sample counts;
   `target_valid_samples` must all be greater than 0 (same guard as CLAP), otherwise
   an error is raised;
2. arithmetic-mean downmix the channels;
3. group by the joint key `(source_valid_samples, target_valid_samples)`;
4. each group keeps only the true source valid prefix (content outside the valid
   region never participates in downstream computation, so no batch-wide zeroing is
   needed);
5. groups with `source_valid_samples = 0` skip resampling and directly construct a
   zero waveform of the target length (same handling as CLAP; empty input cannot go
   through resample);
6. resample that group to 16 kHz;
7. normalize to that group's exact target valid sample count (crop or right-pad with
   zeros);
8. apply minimum-input zero-padding (see the section below);
9. run `× 2**15`, the equivalent kaldi fbank, and normalization in a batch within the
   group;
10. `index_copy` back into a zero-filled canvas in original batch order.

Samples sharing the same discrete length stay batch-vectorized; samples of different
lengths never affect each other's physical frontend boundary.

## Minimum-Input Zero-Padding

The companion Encoder's patch convolution (kernel = stride = 16) needs at least 16
fbank frames to produce one patch time block; producing 16 frames under kaldi
snip_edges requires:

```text
400 + 15 × 160 = 2800 samples = 0.175 seconds @ 16 kHz
```

When `target_valid_samples < 2800`, the group's valid waveform is right-padded with
zeros to 2800 samples. This padding is a waveform-domain operation: the official
frontend can recompute the same result point-by-point on the same zero-padded
waveform, so this branch is part of the official alignment contract (unlike PANNs'
`-100 dB` spectral-domain padding). `valid_seconds` keeps its true value unchanged.

## `valid_feature_frames`

```text
n_physical = max(target_valid_samples, 2800)
valid_feature_frames = 1 + (n_physical - 400) // 160
```

- dtype `torch.int64`, shape `[B]`;
- always satisfies `valid_feature_frames >= 16`;
- its semantics are "the number of frames the official frontend actually produces
  for the (zero-padded if needed) valid waveform." This differs from PANNs, which
  records the frame count before padding: BEATs' zero-padding happens in the waveform
  domain, ahead of the official frontend, so every frame after padding is one the
  official frontend can genuinely reproduce.

## Equivalent Batched Kaldi fbank

Implemented independently using the batching techniques validated during the AST
migration (not shared code with `AstKaldiFbankTransform`, since the two differ in
window function, scaling, and normalization):

- `unfold(kernel=400, stride=160)` implements snip_edges framing;
- subtract the mean per frame (`remove_dc_offset`);
- replicate the first sample, then apply pre-emphasis via `x[i] - 0.97 × x[i-1]`;
- multiply by the povey window: `(0.5 - 0.5 × cos(2πn/399))^0.85`;
- right-pad with zeros to 512 points, then `torch.fft.rfft`, power spectrum
  `abs().pow(2)` (same operators as ta_kaldi, floating-point path matches bit-for-bit);
- project with `torchaudio.compliance.kaldi.get_mel_banks(128, 512, 16000, 20.0, 0.0)`
  weights (padding the trailing column, consistent with ta_kaldi);
- `clamp(min=float32 eps)` followed by the natural log;
- normalize with `(x - 15.41663) / (2 × 6.55582)`.

All of this runs in float32. Sample-by-sample Python loops calling `ta_kaldi.fbank`
are prohibited; that official entry point only appears on the test reference side.

## Output Contract

Returns:

```python
{
    "input_features": input_features,
    "valid_feature_frames": valid_feature_frames,
    "valid_seconds": valid_seconds,
}
```

| Key | Shape | Dtype | Semantics |
|---|---|---|---|
| `input_features` | `[B, T, 128]` | `float32` | Normalized kaldi fbank |
| `valid_feature_frames` | `[B]` | `int64` | Actual official fbank frame count (≥ 16) |
| `valid_seconds` | `[B]` | `float32` | Valid audio duration |

`T` is the maximum `valid_feature_frames` within the batch; the outer padding filling
the gaps between groups is 0. The companion Encoder regroups by `valid_feature_frames`
and drops the outer padding, so this 0 value never enters model computation.

## Device, Training, and Serialization

- `device` is derived from the registered buffers;
- waveform and valid_seconds are moved automatically;
- the frontend is fixed to float32;
- no trainable parameters, no persistent buffers, `state_dict` is empty;
- does not automatically call `eval()`, does not freeze, does not wrap
  `torch.no_grad()`;
- unknown `**kwargs` are not accepted or silently ignored.

## Files and Exports

```text
src/timbral/models/transforms/beats.py
src/timbral/models/helpers/common.py     # round_positive_ratio
src/timbral/models/helpers/grouping.py   # unique-length grouping and zero-canvas scatter-back
```

`timbral.models.transforms` re-exports `BeatsKaldiFbankTransform`. `timbral.models` at
the top level only re-exports registry symbols (see [`../registry.md`](../registry.md)).

## Testing Requirements

The no-weight, no-network test suite (default suite) must cover at least:

- `[B,N]` and `[B,C,N]`; integer waveforms and wrong ranks raising errors;
- mean downmixing, resampling, and float32 normalization;
- `sample_rate` and `valid_seconds` validation and the round ties-to-even semantics;
- non-zero invalid padding not affecting any valid output (each group only takes the
  source-domain valid prefix);
- batch/per-sample consistency after unique-length grouping;
- minimum-padding behavior and the frame-count formula at the 2799/2800/2801-sample
  boundaries;
- `valid_feature_frames` locking to `1 + (max(n, 2800) - 400) // 160`;
- the batched fbank matching a per-sample `ta_kaldi.fbank` loop reference within
  `atol=1e-5` across multiple signals and durations (including `× 2**15` and
  normalization);
- the povey window and mel-weight buffers matching the torchaudio reference
  bit-for-bit;
- output keys, shapes, and dtypes; the outer padding between groups being 0;
- unknown parameters raising errors;
- correct input/output devices after `.to(device)` (when CUDA is available).

## Dependency Boundaries

At runtime only PyTorch and torchaudio (`functional.resample`,
`compliance.kaldi.get_mel_banks`) are used. No new third-party dependency is
introduced.

The real-weight, official full-network alignment is in
[`../extra/beats-alignment.md`](../extra/beats-alignment.md).
