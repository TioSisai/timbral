# `AtstMelspecTransform` Design

This document finalizes the design of
`timbral.models.transforms.AtstMelspecTransform`, the log-mel frontend that the two
official ATST families share verbatim. The corresponding Encoder designs are in
[`../encoders/atst_clip.md`](../encoders/atst_clip.md) and
[`../encoders/atst_frame.md`](../encoders/atst_frame.md); the official alignment
contract for this frontend (and for ATST-Clip) is in
[`../extra/atst_clip-alignment.md`](../extra/atst_clip-alignment.md).

This document describes the behavior the current implementation must satisfy. Where
this document conflicts with the `BaseTransform` design, the common contract in
[`base.md`](base.md) takes precedence.

One parameterless class serves all four registered models — `atst-clip-small`,
`atst-clip-base`, `atst-frame-small`, `atst-frame-base`. ATST-Clip (official class
`AST`, INTERSPEECH 2022) and ATST-Frame (official class `FrameAST`, IEEE TASLP)
publish different backbones but the same frontend, and all four official checkpoints
record the same 64-mel, 64×4-patch geometry: the two ATST-Frame archives carry
`n_mels=64`, `patch_h=64`, `patch_w=4` in their `hyper_parameters`, the DINO-style
`atst-clip-base` archive carries `patch_height=64`, `patch_width=4` in its `args`, and
every one of the four stores a `[D, 256]` patch-embedding weight.

## Design Goals

`AtstMelspecTransform` is responsible for:

- accepting mono or multi-channel floating-point waveforms;
- mean-downmixing, keeping only the valid prefix, and resampling to 16 kHz;
- grouping by true valid length so that the result is independent of batch
  composition;
- reproducing the official `MelSpectrogram` → `AmplitudeToDB(top_db=80)` →
  `MinMax(-79.6482, 50.6842)` chain with a batched, equivalent implementation;
- taking the `top_db` floor per sample, never per batch;
- emitting time-major features together with the exact `valid_feature_frames`, so the
  Encoder can regroup by length and cut its own 250-patch chunks.

The official frontend has no fixed canvas and no duration cap; this Transform likewise
imposes no duration cap. Unlike BEATs, a long file does not cost quadratic attention
downstream: both Encoders split their input into 1000-mel-frame chunks (see
[`../encoders/atst_frame.md`](../encoders/atst_frame.md)), so the model cost grows
linearly with duration. The frontend's own cost is one STFT over the whole valid
region.

This component does not hold weights (the full state of all four checkpoints lives in
the Encoder), does not hold an Encoder, and does not implement a registry, factory, or
YAML configuration layer.

## The Official Frontend

The official chain is three stages, constructed identically by both families:

```python
melspec = torchaudio.transforms.MelSpectrogram(
    16000, f_min=60, f_max=7800, hop_length=160,
    win_length=1024, n_fft=1024, n_mels=64,
)
to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
normalize = MinMax(min=-79.6482, max=50.6842)
```

Explicit parameters and the torchaudio defaults that are fixed by this design:

| Parameter | Value | Note |
|---|---|---|
| `sample_rate` | 16000 | Target sample rate |
| `n_fft` / `win_length` | 1024 / 1024 | 64 ms window, no zero-padded FFT |
| `hop_length` | 160 | 10 ms frame shift |
| `n_mels` | 64 | Mel dimension; also the patch height |
| `f_min` / `f_max` | 60.0 / 7800.0 | Mel band, narrower than Nyquist on both ends |
| `window_fn` | `torch.hann_window` (default) | Periodic Hann |
| `power` | `2.0` (default) | Power spectrogram |
| `center` | `True` (default) | Symmetric padding of `n_fft // 2` |
| `pad_mode` | `"reflect"` (default) | Reflect padding; source of the 513-sample floor |
| `norm` / `mel_scale` | `None` / `"htk"` (default) | Unnormalized HTK mel filters |
| `normalized` | `False` (default) | No STFT normalization |

The dB stage is written out instead of calling `AmplitudeToDB` (see
[Per-Sample `top_db` Reduction](#per-sample-top_db-reduction)):

```text
decibel = 10 × log10(clamp(power, min=1e-10))
decibel = maximum(decibel, per_sample_peak(decibel) - 80)
```

`AmplitudeToDB(stype="power")` means `multiplier=10`, `amin=1e-10`, `ref=1.0`, hence
`db_multiplier = log10(max(amin, ref)) = 0`; the subtraction the module performs is a
no-op, and the two lines above are the whole module minus that no-op.

Fixed normalization constants (the official hardcoded AudioSet statistics for
`n_mels=64`):

```text
norm_min = -79.6482
norm_max =  50.6842
normalized = (x - norm_min) / (norm_max - norm_min) × 2 - 1
```

The official `MinMax` is affine and does not clip, and neither does this Transform.
A digitally silent sample sits at the `10 × log10(1e-10) = -100` dB floor in every
bin, so it leaves the frontend as the constant `-1.3123`, outside `[-1, 1]`. Clamping
the output into `[-1, 1]` would be a deviation from the official frontend and is not
performed.

The constants the two components share live in `helpers/atst.py`
(`ATST_TARGET_SAMPLE_RATE`, `ATST_HOP_LENGTH`, `ATST_NUM_MELS`, `ATST_TOP_DB`,
`ATST_NORM_MIN`, `ATST_NORM_MAX`); the Encoders derive their 40 ms patch grid from the
same hop. The purely frontend-local ones (`n_fft`, `win_length`, `f_min`, `f_max`, the
`1e-10` amplitude floor, the multiplier 10, and the 513-sample minimum) stay
module-private in `transforms/atst.py`, since no other component can read them.

## Per-Sample `top_db` Reduction

This is the subtlest requirement in the component, and the only reason the dB stage is
hand-written rather than delegated. `torchaudio.functional.amplitude_to_DB` picks its
reduction range from the RANK of its input:

```python
shape = x_db.size()
packed_channels = shape[-3] if x_db.dim() > 2 else 1
x_db = x_db.reshape(-1, packed_channels, shape[-2], shape[-1])
x_db = torch.max(x_db, (x_db.amax(dim=(-3, -2, -1)) - top_db).view(-1, 1, 1, 1))
```

- for a rank-3 input `[B, 64, T]`, `packed_channels = shape[-3] = B`, the tensor
  reshapes to `[1, B, 64, T]`, and the `amax` collapses to ONE scalar for the whole
  batch: every sample is floored at the loudest sample's peak minus 80 dB;
- for a rank-4 input `[B, 1, 64, T]`, `packed_channels = 1`, the reshape is a no-op,
  and the `amax` yields one peak per sample.

torchaudio documents this itself ("if the input shape is 2D or 3D, a single cutoff
value is used"), but the trap is that both ranks accept the same batch and return the
same shape, so the wrong one fails silently.

Only the per-sample reduction is admissible here. The repository contract requires a
sample's features to be independent of the batch it happens to travel in, and
unique-length grouping cannot rescue the rank-3 form: samples of equal valid length
share a group and would then share a peak. The failure is not marginal once
triggered: a quiet sample batched with a loud one is floored in every bin, and its
whole feature map collapses to a single constant value.

The Transform therefore reduces explicitly over that sample's own mel and time axes:

```python
peak = decibel.amax(dim=(-2, -1), keepdim=True)
decibel = torch.maximum(decibel, peak - ATST_TOP_DB)
```

This reproduces official behavior rather than deviating from it: the official
`embedding.py` entry point feeds `[B, 1, 64, T]` (a single channel axis), so the
official path also reduces per sample. What is rejected is only the convenient rank-3
call, which the official code path never takes.

The floor is one scalar per sample applied after the log, not a per-frame or per-bin
operation. That is also why a sample's features must be produced in a single pass over
its whole valid region (see [Whole-Region Extraction](#whole-region-extraction)).

## Public Constructor Interface

```python
class AtstMelspecTransform(BaseTransform):
    def __init__(self) -> None:
        ...
```

No constructor parameters (consistent with `BeatsKaldiFbankTransform` and
`AstKaldiFbankTransform`). The constructor first calls `super().__init__()`, then sets
`target_sample_rate = 16000`, then builds a single
`torchaudio.transforms.MelSpectrogram` submodule with the official parameters above.
`device` is read off `self.melspec.mel_scale.fb`.

No `pretrained` / `pretrained_dir` parameters are provided; weight identity belongs
entirely to the Encoder, which owns checkpoint download, SHA-256 verification, and
archive-format normalization (see
[`../encoders/atst_clip.md`](../encoders/atst_clip.md)).

Reusing torchaudio's module instead of registering hand-built window and filterbank
buffers is deliberate: the official frontend IS this module, and the alignment test's
reference side is a second instance of it. The consequence is that
`melspec.spectrogram.window` and `melspec.mel_scale.fb` are persistent buffers and do
appear in `state_dict` (see
[Device, Training, and Serialization](#device-training-and-serialization)).

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
- when different from 16000, `torchaudio.functional.resample` is used, per group and
  never across the whole padded canvas.

### `valid_seconds`

- `None` means the entire physical Tensor is valid;
- when not empty, must be `[B]`, satisfying `0 < valid_seconds <= N / sample_rate`;
- moved to the Transform's device and normalized to `float32`.

Discrete valid lengths are rounded per sampling grid using `torch.round`'s
ties-to-even rule (same as PANNs and BEATs):

```text
source_valid_samples = round(valid_seconds × sample_rate)
target_valid_samples = round(valid_seconds × 16000)
```

When `valid_seconds=None`, no float32-seconds round trip is used:
`source_valid_samples` takes the physical sample count `N` directly, and
`target_valid_samples` is converted using the same ties-to-even rule via integer
rational arithmetic (`helpers.common.round_positive_ratio`), which guarantees no
sample loss for very long waveforms (> 2^24 samples).

After rounding, every `target_valid_samples` must be greater than 0, otherwise an
error is raised (same guard as CLAP and BEATs). `source_valid_samples` may still round
to 0 when upsampling from a very low rate; that case is handled inside the grouping
pipeline rather than rejected.

## Unique-Length Grouping

The whole batch must not simply be padded to the longest sample and run through the
frontend once. Two independent mechanisms make the result depend on the physical
canvas: `center=True` reflect padding reads the two ends of the physical waveform, and
the `top_db` floor reads the peak of the whole physical waveform. The pipeline is the
same as PANNs and BEATs:

1. validate the input and compute the source/target valid sample counts;
   `target_valid_samples` must all be greater than 0, otherwise an error is raised;
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
9. run the mel, dB, and MinMax stages for the whole group in one batched pass;
10. `index_copy` back into a zero-filled canvas in original batch order
    (`helpers.grouping.assemble_padded_groups`).

Samples sharing the same discrete lengths stay batch-vectorized; samples of different
lengths never affect each other's physical frontend boundary or dB floor. Grouping by
source length alone is not allowed, because the same source rounding result can map to
different target rounding results under upsampling.

## Whole-Region Extraction

The Encoders chunk; this Transform does not. The learned `pos_embed` holds 250 patch
slots, which caps one Encoder forward pass at 250 patches, so a long input is cut into
consecutive 1000-mel-frame chunks. That cut happens after this Transform and never
inside it: a sample's features are always computed over its entire valid region in a
single mel/dB/MinMax pass, for two reasons.

- The `top_db` floor is a reduction over the whole region. Computing chunk by chunk
  would floor each chunk at its own peak, so a quiet passage inside a loud file would
  be rescaled differently depending only on where the boundaries happen to fall, and a
  10.0 s file and a 10.1 s file would disagree on the features of their common first
  10.0 s.
- `center=True` reflect padding at an internal chunk boundary would invent content
  that the official frontend never sees at that position. Slicing an
  already-computed feature map instead leaves every frame bit-identical to the
  un-chunked one.

Because the mel grid produced here is global and unbroken, and the Encoder's chunk
length (1000 frames) is itself a multiple of the patch width (4 frames), the identity
`total patches == valid_feature_frames // 4` holds exactly at every duration. See
[`../encoders/atst_frame.md`](../encoders/atst_frame.md) for why the repository chunks
at 1000 frames where the official `embedding.py` chunks at 1001.

## Minimum-Input Zero-Padding

`torch.stft` with `center=True` reflect-pads by `n_fft // 2 = 512` samples on each
side, and reflect padding requires the input to be strictly longer than the padding.
512 samples or fewer therefore raise inside torchaudio, and

```text
513 = n_fft // 2 + 1
```

is the shortest waveform the official frontend can process at all. That floor
coincides exactly with what the Encoder needs: 513 samples produce
`513 // 160 + 1 = 4` mel frames, i.e. exactly one 64×4 patch, which is the minimum a
forward pass requires (a trailing remainder shorter than one patch is dropped by the
patch embedding).

When `target_valid_samples < 513`, the group's valid waveform is right-padded with
zeros:

```text
physical_length = max(target_valid_samples, 513)
```

This padding is a waveform-domain operation applied ahead of the frontend, so the
official chain recomputes the same result point-by-point on the same zero-padded
waveform; the alignment test pads its reference input identically. It is therefore
part of the official alignment contract (unlike PANNs' `-100 dB` spectral-domain
padding). `valid_seconds` keeps its true value unchanged, so a 0.01 s file still
reports 0.01 s.

## `valid_feature_frames`

```text
n_physical = max(target_valid_samples, 513)
valid_feature_frames = n_physical // 160 + 1
```

- dtype `torch.int64`, shape `[B]`;
- `N // hop + 1` is exactly the frame count `MelSpectrogram` produces with
  `center=True`;
- always satisfies `valid_feature_frames >= 4`, i.e. at least one patch;
- its semantics are "the number of mel frames the official frontend actually produces
  for the (zero-padded if needed) valid waveform." This follows BEATs rather than
  PANNs, which records the frame count before padding: ATST's zero-padding happens in
  the waveform domain, ahead of the official frontend, so every frame after padding is
  one the official frontend can genuinely reproduce.

| `target_valid_samples` | 160 | 512 | 513 | 640 | 1600 | 16000 | 160000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `valid_feature_frames` | 4 | 4 | 4 | 5 | 11 | 101 | 1001 |

The computation is `helpers.atst.atst_feature_frames(clamp(target_valid_samples,
min=513))`; the Encoder's frame geometry starts from this same value and maps it to its
patch count with `helpers.atst.atst_patch_frames` (`frames // 4`). Both sides read the
geometry from that one module, so they cannot drift apart.

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
| `input_features` | `[B, T, 64]` | `float32` | Normalized log-mel, time-major |
| `valid_feature_frames` | `[B]` | `int64` | Actual official mel frame count (≥ 4) |
| `valid_seconds` | `[B]` | `float32` | Valid audio duration |

`T` is the maximum `valid_feature_frames` within the batch; the outer padding filling
the gaps between groups is 0. The companion Encoders regroup by
`valid_feature_frames` and slice each group to its exact prefix, so this 0 never
enters model computation — which is also what makes the official additive attention
mask uniformly zero and therefore omissible (see
[`../encoders/atst_clip.md`](../encoders/atst_clip.md)).

### Time-Major Layout

The official frontend hands the model `[B, 1, 64, T]`: one channel, mel-major, time
last. This Transform emits `[B, T, 64]` instead, applying a single `transpose(1, 2)`
at the end of feature extraction.

- Every spectrogram-like Transform in this repository is time-major (`[B, 1024, 128]`
  for AST, `[B, T, 128]` for BEATs, `[B, T, n_mels]` for PANNs). Keeping ATST
  consistent means `valid_feature_frames` always indexes dimension 1, and the shared
  grouping and scatter-back helpers (`assemble_padded_groups` pads dimension 1 of a
  `[n, T, D]` tensor) work unchanged.
- The singleton channel axis carries no information, and the mel axis is exactly one
  patch tall (`patch_h = 64 = n_mels`), so the official `einops` rearrange
  `b c (h p1) (w p2) -> b (w h) (p1 p2 c)` degenerates to grouping four consecutive
  frames of all 64 bins.

The Encoder transposes back internally: `_AtstPatchEmbed` views the time-major
features as `[B, W, 4, 64]`, transposes to `[B, W, 64, 4]`, and reshapes to
`[B, W, 256]`, which is the official `(p1 p2 c)` ordering — mel bin major, frame
minor. The two transposes cancel exactly; no value is reordered or reinterpreted, and
reproducing the official layout needs no `einops` dependency. The alignment tests feed
the official model `features.transpose(1, 2).unsqueeze(1)`, the exact inverse of this
layout choice, and obtain bit-identical results on CPU.

## Device, Training, and Serialization

- `device` is derived from the mel filterbank buffer `melspec.mel_scale.fb`;
- `waveform` and `valid_seconds` are moved automatically, and all three outputs are
  returned on that device;
- the frontend is fixed to `float32`; the waveform is cast on entry;
- there are no trainable parameters and no buffer requires grad;
- the `state_dict` is not empty: torchaudio registers `spectrogram.window` and
  `mel_scale.fb` as persistent buffers, so it holds exactly
  `{"melspec.spectrogram.window", "melspec.mel_scale.fb"}`. Both are pure functions of
  the fixed constructor arguments and carry no checkpoint identity: every construction
  rebuilds them identically, and no weight provenance leaks into the Transform. AST
  and BEATs register their hand-built equivalents with
  `persistent=False` and therefore serialize nothing; this Transform reuses the
  official module verbatim rather than reproducing it, and accepts torchaudio's own
  registration instead of rewriting it;
- does not automatically call `eval()`, does not freeze, does not wrap
  `torch.no_grad()`;
- unknown `**kwargs` are not accepted or silently ignored.

The official alignment contract only covers `float32`. The Transform does not promise
any particular STFT or mel behavior after an explicit conversion to half precision.

## Files and Exports

```text
src/timbral/models/transforms/atst.py
src/timbral/models/helpers/atst.py       # frontend constants, atst_feature_frames
src/timbral/models/helpers/common.py     # round_positive_ratio
src/timbral/models/helpers/grouping.py   # unique-length grouping and zero-canvas scatter-back
```

`timbral.models.transforms` re-exports `AtstMelspecTransform`. All four registry
entries name it as their `transform_cls`. Registry arguments are routed by constructor
signature, so each entry's fixed `arch`, and any `n_blocks` passed through
`--model_kwargs`, reach the Encoder only; this Transform declares no constructor
parameters and receives nothing (see [`../registry.md`](../registry.md)).
`timbral.models` at the top level only re-exports registry symbols.

## Testing Requirements

The no-weight, no-network test suite (default suite,
`tests/models/transforms/test_atst.py`) must cover at least:

- the public export identity and the parameterless signature;
  `target_sample_rate == 16000`; no parameters; `state_dict` holding exactly the two
  torchaudio buffers, none of which requires grad;
- output keys, shapes, dtypes, and devices;
- `valid_feature_frames` locking to `max(n, 513) // 160 + 1` at 160, 512, 513, 640,
  1600, 16000, and 160000 samples, with `input_features` sharing that frame count;
- features matching a directly constructed official `MelSpectrogram` +
  `AmplitudeToDB(stype="power", top_db=80)` + MinMax reference exactly
  (`torch.equal`) across several durations;
- the 513-sample floor: the bare `MelSpectrogram` raising below it, and the Transform
  instead zero-padding and matching the reference computed on the same padded
  waveform;
- digital silence landing on the single constant `-1.3123` the dB and MinMax constants
  predict, and a valid region that rounds to 0 source samples materializing as that
  same silence instead of reaching `resample`;
- resampled frame counts and features (8 kHz and 44.1 kHz inputs) matching the
  resample-then-reference path;
- `[B, C, N]` mean downmixing matching the pre-averaged `[B, N]` input;
- non-zero invalid padding, both quiet and loud, not affecting any valid output (each
  group only takes the source-domain valid prefix);
- mixed-length batches matching single-sample calls bit-for-bit, with the outer
  padding between groups exactly 0;
- the `top_db` floor being per sample: equal to the rank-4 reference, measurably
  different from the rank-3 shared-peak reference, and the shared-peak variant
  demonstrably collapsing a quiet sample to a constant map;
- the `top_db` width being 80 dB: a loud burst followed by digital silence spans more
  than that, so the minimum of its feature map must be the rescaled `peak - 80` and
  must form a plateau over whole frames rather than one stray bin;
- every invalid input raising: non-floating-point waveform, wrong rank, float
  `sample_rate`, non-positive `sample_rate`, non-Tensor `valid_seconds`, wrong
  `valid_seconds` shape, out-of-range or non-positive `valid_seconds`, a
  `valid_seconds` that rounds to 0 target samples, and unknown keyword arguments;
- correct input/output devices after `.to(device)` (when CUDA is available).

Official alignment of the frontend is exercised by
`pytest --run-alignment atst_clip` (`test_transform_alignment`), which rebuilds the
official chain from the pinned `audiossl` source over 6 durations × 5 signal kinds
(random, sine, impulse, multisine, silence) on CPU and CUDA; the measured
`max|delta|` is 0.0 on both devices. The ATST-Frame alignment module does not repeat
it, because the frontend is one and the same class for both families.

## Dependency Boundaries

At runtime only PyTorch and torchaudio (`transforms.MelSpectrogram`,
`functional.resample`) are used. No new third-party dependency is introduced; in
particular no `einops`, no `librosa`, and no `torchlibrosa`.
`torchaudio.transforms.AmplitudeToDB` is deliberately not used, for the reason given
in [Per-Sample `top_db` Reduction](#per-sample-top_db-reduction).

The official `audiossl` package is never imported by the runtime path; it appears only
on the test reference side, sparse-cloned at a pinned commit and verified per file.

The real-weight, official full-network alignment is in
[`../extra/atst_clip-alignment.md`](../extra/atst_clip-alignment.md) (ATST-Clip and
this shared frontend) and
[`../extra/atst_frame-alignment.md`](../extra/atst_frame-alignment.md) (ATST-Frame).
