# `ClapLogmelTransform` Design

This document finalizes the design of `timbral.models.transforms.ClapLogmelTransform`.
The companion Encoder is described in [`../encoders/clap.md`](../encoders/clap.md); the
official alignment contract is in
[`../extra/clap-alignment.md`](../extra/clap-alignment.md).

## Design Goals

`ClapLogmelTransform` produces the Hugging Face CLAP audio-tower input for the fixed
`laion/clap-htsat-fused` checkpoint. It is responsible for:

- accepting padded mono or multi-channel waveforms;
- isolating each sample's true valid prefix according to `valid_seconds`;
- mean-downmixing and resampling to 48 kHz;
- reproducing `ClapFeatureExtractor`'s log-mel, repeatpad, and long-audio fusion
  features;
- deciding, per sample, between the global-only path or the fused path based on
  per-sample valid length;
- returning a Tensor dict that can be unpacked directly into `ClapHtsatEncoder`.

This implementation uses per-sample length routing. Once a sample's valid audio
reaches 480480 target samples at 48 kHz, it enters fusion; when the whole batch is
shorter than that, every sample stays global-only.

## Public Interface

```python
class ClapLogmelTransform(BaseTransform):
    def __init__(self) -> None:
        ...

    def forward(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
        valid_seconds: Tensor | None = None,
    ) -> dict[str, Tensor]:
        ...
```

The constructor has no public parameters. The target sample rate is set as the public
instance attribute `target_sample_rate = 48000` at construction time, fulfilling the
`BaseTransform` attribute contract. The frontend parameters are fixed and tied to the
single supported checkpoint:

| Parameter | Value |
|---|---:|
| Target sample rate | 48000 Hz |
| Fixed short-audio canvas | 480000 samples |
| FFT/window size | 1024 |
| Hop length | 480 |
| Mel bins | 64 |
| Mel scale | HTK |
| Mel norm | None |
| Minimum frequency | 50 Hz |
| Maximum frequency | 14000 Hz |
| Output time frames | 1001 |
| Output channels | 4 |

The frontend does not expose acoustic parameters that could produce a
checkpoint-mismatched combination.

## Input and Valid Length

Input follows `BaseTransform`:

- `waveform` is a floating-point Tensor with shape `[B,N]` or `[B,C,N]`;
- `sample_rate` is a positive Python `int`;
- `valid_seconds` is `None` or a Tensor of shape `[B]`;
- `valid_seconds=None` means the entire physical Tensor of every sample is valid;
- every valid duration satisfies `0 < valid_seconds <= N / sample_rate`.

After grouping samples by valid length, the Transform moves each group's waveform
valid prefix to its own device and converts it to `float64`; invalid padding
undergoes no device transfer or dtype conversion. `valid_seconds` is moved to the
Transform's device and converted to `float32`. The discrete valid length for an
explicit `valid_seconds` uses:

```python
source_valid_samples = round(valid_seconds * sample_rate)
target_valid_samples = round(valid_seconds * 48000)
```

When `valid_seconds=None`, `source_valid_samples` takes the physical sample count `N`
directly, and `target_valid_samples` is computed via integer rational rounding as
`round(N * 48000 / sample_rate)`. This path does not round-trip through float32
seconds, so the discrete length of a full physical Tensor stays exact.

When `target_valid_samples == 0`, meaning the audio is shorter than CLAP's minimum
target sampling grid, the forward pass immediately raises `ValueError`.

The processing order is fixed as:

1. group samples by `(source_valid_samples, target_valid_samples)`;
2. crop out the source valid prefix for each group;
3. arithmetic-mean multi-channel input along the channel dimension;
4. resample to 48 kHz with `torchaudio.functional.resample` when needed;
5. crop or pad to `target_valid_samples`;
6. batch-compute log-mel for samples within the same group.

Different valid lengths cannot run STFT directly on a uniform, longest zero canvas,
because reflect padding must act on each sample's true end. Cropping happens before
downmixing and resampling, so the invalid tail never participates in computation and
never affects the output.

## Short Audio and Boundary Intervals

### 1 to 479999 Samples

Uses the official `repeatpad`:

```text
n_repeat = 480000 // valid_samples
The valid prefix is fully repeated n_repeat times
Remaining positions are zero-padded
```

After repeatpad, the waveform is fixed at 480000 samples.

### 480000 Samples

The waveform is left unchanged.

### 480001 to 480479 Samples

The centered STFT of the complete waveform still produces only 1001 frames, so it
stays global-only. All four output channels are the same copy of the `[1001,64]`
log-mel.

### 480480 Samples and Above

The centered STFT of the complete waveform produces at least 1002 frames, entering
long-audio fusion feature construction.

## Log-Mel

Each waveform to be processed uses the following chain:

1. reflect-pad 512 samples on both sides;
2. periodic Hann window;
3. `torch.stft(..., center=False, return_complex=True)`;
4. sum of the squared real part and squared imaginary part to get power;
5. HTK mel filter-bank projection;
6. clamp to `1e-10`;
7. `10 * log10`;
8. transpose to `[T,64]`.

For short audio and boundary intervals, the single-channel mel is duplicated as:

```text
[global, global, global, global]
```

## Long-Audio Fusion Features

Let the complete log-mel be `[T,64]`, where `T >= 1002`.

Channel 0 is the global feature:

- convert the complete `[T,64]` to `float32`;
- compress to `[1001,64]` using PyTorch bilinear interpolation.

Channels 1–3 are local features, with crop start points given by a deterministic
anchor formula:

- the three anchors are `(2k + 1) * T / 6` on the frame axis (`k = 0, 1, 2`),
  corresponding to the 1/6, 1/2, and 5/6 positions of the audio, rounded using
  integer rational rounding (round-half-even, the same rule used for length rounding
  on the `valid_seconds=None` path);
- each crop is centered on its anchor frame, with start point `center - 500`;
- the start point is clamped to the legal interval `[0, T - 1001]`: when the first
  segment's left side goes out of bounds it is shifted right to align with the start
  of the audio, and when the third segment's right side goes out of bounds it is
  shifted left to align with the last valid frame; the middle segment never goes out
  of bounds when `T >= 1002`;
- each local crop takes 1001 consecutive frames.

```python
center_k = round_half_even((2 * k + 1) * T / 6)
start_k = clamp(center_k - 500, 0, T - 1001)
```

The crop start points depend only on the discrete target length, so samples within the
same valid-length group share the same set of start points; for shorter long audio
(e.g. `T = 1002`), the clamped three segments may overlap heavily or even duplicate
each other. Waveform, mel, interpolation, and crop Tensors all stay on the Transform's
device. Local crops are batch-assembled using shared frame indices.

Deterministic local crops are a formal part of long-audio fusion. The forward pass
consumes no global RNG state, is consistent across repeated calls, and is independent
of batch composition. This design deliberately departs from the official random-crop
protocol: this project needs deterministic representations, whereas the upstream
random protocol has no alignable fixed output of its own.

## Fusion Routing

The Transform does not output a public `is_longer`. Per-sample routing is defined by
the single formula:

```python
fusion_mask = target_valid_samples > 480479
```

The companion Encoder uses the same formula to produce the `is_longer` argument
required internally by the Hugging Face audio tower. An all-short batch does not
change the routing mask of any sample.

## Precision, Device, and State

The Hugging Face frontend computes the waveform/window/spectrum in float64 and
outputs float32 log-mel. This implementation fixes:

- the transfer, downmixing, and resampling of the waveform valid prefix use
  `float64`;
- STFT, power, mel, and log all use `float64`;
- log-mel is converted to `float32` before fusion interpolation;
- the final `input_features` is `float32`.

The mel filter bank and Hann window are `persistent=False` float64 buffers. The
Transform's `_apply` keeps these two buffers at float64 precision:

- `.to(device)` migrates the device normally;
- `.float()`, `.half()`, `.bfloat16()`, and `.to(dtype=...)` do not change the
  internal computation precision.

The Transform has no persistent parameters or buffers, so `state_dict()` is empty.
The first version supports CPU and CUDA; the operators required for float64 STFT
mean that MPS is out of scope.

The Transform keeps the ordinary `nn.Module` lifecycle: it does not automatically
switch to eval, does not freeze, and does not wrap `no_grad` or `inference_mode`.

## Output

```python
{
    "input_features": input_features,
    "valid_seconds": valid_seconds,
}
```

- `input_features`: `[B,4,1001,64]`, `float32`;
- `valid_seconds`: `[B]`, `float32`;
- both located on the Transform's device.

## Files and Exports

- implemented at `src/timbral/models/transforms/clap.py`;
- `timbral.models.transforms` re-exports `ClapLogmelTransform`;
- `timbral.models` at the top level only re-exports registry symbols (see
  [`../registry.md`](../registry.md));
- this component does not hold an Encoder, nor does it implement YAML or a
  composition Pipeline.

## Testing Requirements

The ordinary tests stay offline and must cover at least:

- parameterless construction, public export, empty `state_dict`;
- CPU/CUDA device transfer and the fixed float64 buffers;
- `[B,N]`, `[B,C,N]`, downmixing, and non-48 kHz resampling;
- `valid_seconds=None`, mixed lengths, and non-zero invalid tails;
- long physical input with `valid_seconds=None` not going through a float32-seconds
  round trip;
- float64 waveform not losing precision before entering the frontend;
- a clear error when the target grid maps to 0;
- the 1, 479999, 480000, 480001, 480479, 480480-sample boundaries;
- the full repetition and tail zero-padding of repeatpad;
- neither long nor short audio forward passes consuming global RNG, and an
  all-short batch not changing routing;
- the global and three deterministic-anchor local channels for long audio;
- consistency across repeated calls and independence from batch composition;
- the exact values of the anchor start-point formula at boundary and general
  lengths;
- output shape, dtype, and device;
- gradients propagating;
- unknown parameters raising `TypeError`.

The real official numerical tests are in
[`../extra/clap-alignment.md`](../extra/clap-alignment.md).
