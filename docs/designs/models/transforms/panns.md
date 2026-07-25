# `PannsLogmelTransform` Design

This document finalizes the design of `timbral.models.transforms.PannsLogmelTransform`.
The corresponding Encoder design is in [`../encoders/panns.md`](../encoders/panns.md);
the official alignment contract is in
[`../extra/panns-alignment.md`](../extra/panns-alignment.md).

This document describes the behavior the current implementation must satisfy. Where
this document conflicts with the legacy repository implementation, this document takes
precedence; where this document conflicts with the `BaseTransform` design, the common
contract in [`base.md`](base.md) takes precedence.

## Design Goals

`PannsLogmelTransform` is responsible for:

- accepting mono or multi-channel floating-point waveforms;
- clearing invalid regions, mean-downmixing, and resampling;
- grouping by true valid length while preserving the official reflect semantics at
  the end of short audio;
- using an equivalent STFT, power spectrum, mel, and dB implementation that does not
  depend on `torchlibrosa`;
- loading the STFT, mel, and `bn0` state of the corresponding checkpoint;
- producing the post-`bn0` log-mel features required by the PANNs Cnn14;
- outputting the exact `valid_feature_frames` so the Encoder can regroup by length.

This component does not hold an Encoder, does not create a Transform–Encoder
composition object, and does not implement a registry, factory, or YAML configuration
layer.

## Supported Official Variants

The shared type definitions live in `timbral.models.helpers.panns`:

```python
PannsVariant: TypeAlias = Literal["max_mean", "decision_level_max"]
PannsTargetSampleRate: TypeAlias = Literal[16000, 32000]
```

`timbral.models.transforms` and `timbral.models.encoders` both re-export the same
`PannsVariant` object; it is not redefined per component file.

Only the following three official checkpoint identities are supported:

| `target_sample_rate` | `variant` | Official model | Checkpoint |
|---:|---|---|---|
| 16,000 | `max_mean` | `Cnn14_16k` | `Cnn14_16k_mAP=0.438.pth` |
| 32,000 | `max_mean` | `Cnn14` | `Cnn14_mAP=0.431.pth` |
| 32,000 | `decision_level_max` | `Cnn14_DecisionLevelMax` | `Cnn14_DecisionLevelMax_mAP=0.385.pth` |

`16,000 + decision_level_max` has no corresponding checkpoint for this migration and
is not a valid pretrained combination.

Official frontend parameters:

| Parameter | 16 kHz | 32 kHz |
|---|---:|---:|
| `n_fft` | 512 | 1024 |
| `win_length` | 512 | 1024 |
| `hop_length` | 160 | 320 |
| `n_mels` | 64 | 64 |
| `f_min` | 50 Hz | 50 Hz |
| `f_max` | 8,000 Hz | 14,000 Hz |
| Frame shift | 10 ms | 10 ms |

The 32 kHz max_mean and DecisionLevelMax checkpoints have identical STFT convolution
kernels and mel matrices value-for-value, but all 5 of the `bn0` states differ.
Therefore the Transform must know the `variant`; it cannot select the pretrained
frontend from the sample rate alone.

## Public Constructor Interface

Expected interface:

```python
class PannsLogmelTransform(BaseTransform):
    def __init__(
        self,
        *,
        target_sample_rate: PannsTargetSampleRate,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        variant: PannsVariant,
        pretrained: bool = True,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        ...
```

All parameters are keyword-only, to avoid several adjacent int or float parameters
being passed positionally by mistake. The constructor must call `super().__init__()`
first, then register the STFT and mel buffers plus `bn0`.

`target_sample_rate` is preferred over `required_sample_rate`: the Transform accepts
any input sample rate and actively resamples it, and the constructor parameter
describes the frontend's target sample rate; the `sample_rate` in `forward` continues
to represent the original input sample rate.

### `pretrained=True`

- only the three official combinations in the table above are allowed;
- the explicit frontend parameters must exactly match the configuration of the
  selected official checkpoint;
- the checkpoint is downloaded if it does not exist, and read directly if it does;
- newly downloaded and existing files are both SHA-256-verified before every load;
  within the same process, a full hash is computed only once for the same
  `(path, digest)` pair, and the same checkpoint is deserialized only once
  (process-level memoization; in `create_model`, Transform and Encoder share the same
  loaded result, and the caller must not modify the state dict in place);
- the STFT real/imaginary convolution kernels, mel matrix, and all `bn0` state are
  loaded;
- `eval()` is not called automatically, parameters are not frozen, and no device
  transfer occurs.

Explicit parameters that do not match the official configuration must raise
`ValueError`. This is an internal checkpoint-identity check, not a runtime
Transform–Encoder pairing check.

### `pretrained=False`

- no cache directory is resolved, and no network access occurs;
- the STFT and mel buffers are built via an equivalent NumPy/librosa algorithm;
- `bn0` uses PyTorch's default initialization;
- experimental frontend parameters are allowed;
- `16,000 + decision_level_max` can serve as a randomly initialized experimental
  combination that has no official checkpoint;
- whether it can be paired with a given Encoder is the caller's responsibility.

Experimental parameters do not change `PannsCnn14Encoder`'s temporal downsampling
ratio of 32. For the output to pair correctly with the current `PannsCnn14Encoder` and
produce geometry consistent with the time grid, the following must hold:

```text
hop_length / target_sample_rate = 0.01 seconds
n_mels >= 32
```

`n_mels >= 32` is the minimum requirement for the frequency dimension to survive five
rounds of 2× floor pooling. `n_fft` must be even, to guarantee that the actual frame
count after symmetric center padding matches
`floor(target_valid_samples / hop_length) + 1`. Under this constraint, `n_fft`,
`win_length`, `n_mels`, `f_min`, and `f_max` may be experimentally varied, and the
output frequency dimension correspondingly becomes `n_mels`.

## Pretrained Directory and Download

Path priority:

```text
explicit pretrained_dir
    >
HF_HUB_CACHE/audioencoders/{model_name}
```

`HF_HUB_CACHE` is resolved by `huggingface_hub` according to its own environment
variable rules. The PANNs weights actually come from a fixed Zenodo URL; the Hugging
Face logic is only used to determine the default cache root.

The fixed identities of the three checkpoints:

| Model name | URL | SHA-256 |
|---|---|---|
| `panns-32k-cnn14-max_mean` | `https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth` | `0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31` |
| `panns-16k-cnn14-max_mean` | `https://zenodo.org/records/3987831/files/Cnn14_16k_mAP=0.438.pth` | `e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6` |
| `panns-32k-cnn14-decision_level_max` | `https://zenodo.org/records/3987831/files/Cnn14_DecisionLevelMax_mAP=0.385.pth` | `dd3b4043a87d4ec13df8082c0fcfee3fb5084151808e47e060987a95eabdd142` |

A digest mismatch fails immediately; it is never silently used, nor does it
automatically overwrite an existing suspicious file.

The following shared content is centralized in `src/timbral/models/helpers/panns.py`:

- `PannsVariant` and the set of valid variant/target-sample-rate combinations;
- the official frontend parameter table `PANNS_OFFICIAL_FRONTENDS` (used by the
  Transform for `pretrained=True` validation, and by the registry for construction);
- the model name, URL, filename, SHA-256, and safe-loading marker for the three
  checkpoints;
- default HF-cache path resolution and Zenodo download;
- SHA-256 verification;
- `weights_only=True` safe loading and the 16 kHz NumPy allowlist.

Both the Transform and the Encoder import directly from this single source, and do
not import from each other. The Transform file only retains the checkpoint frontend
key mapping and the acoustic frontend implementation.

All three checkpoints use `torch.load(weights_only=True, map_location="cpu")`; none
of them may fall back to `weights_only=False`. The 16 kHz checkpoint additionally
contains training-time NumPy sampler state, so a minimal `safe_globals` allowlist is
enabled only for it. After filtering and mapping the frontend state, strict loading is
used; the STFT, mel, and `bn0` must not have missing or unexpected keys.

## Input Contract

The calling interface inherits from `BaseTransform`:

```python
transform(
    waveform,
    sample_rate=sample_rate,
    valid_seconds=valid_seconds,
)
```

### `waveform`

- must be a floating-point Tensor;
- shape is `[B, N]` or `[B, C, N]`;
- `[B, C, N]` is downmixed to mono via arithmetic channel averaging;
- normalized to `float32` before entering the acoustic frontend;
- automatically moved to the Transform's `device`.

Integer waveforms, ragged lists, and other ranks must raise errors; no implicit
acceptance is allowed.

### `sample_rate`

- must be a positive Python `int`;
- shared across a batch;
- represents the original sample rate of the input waveform;
- when different from `target_sample_rate`, `torchaudio.functional.resample` is used.

### `valid_seconds`

- `None` means the entire physical Tensor is valid;
- when not empty, must be `[B]`;
- must satisfy `0 < valid_seconds <= N / sample_rate`;
- moved to the Transform's device and normalized to `float32`.

Discrete valid lengths are rounded per sampling grid using `torch.round`'s
ties-to-even rule:

```text
source_valid_samples = round(valid_seconds × sample_rate)
target_valid_samples = round(valid_seconds × target_sample_rate)
```

When `valid_seconds=None`, no float32-seconds round trip is used:
`source_valid_samples` takes the physical sample count `N` directly, and
`target_valid_samples` is converted using the same ties-to-even rule via integer
rational arithmetic (`helpers.common.round_positive_ratio`), which guarantees no
sample loss for very long waveforms (> 2^24 samples).

The resampling result is cropped or, when necessary, right-padded with zeros to
`target_valid_samples`, to eliminate ambiguity from length-rounding rules across
different rational sample rates.

## Unique-Length Grouping

The whole batch must not simply be padded to the current longest sample and run
through the frontend once. The official PANNs has no `valid_length` or attention
mask; a short sample's end-of-sequence STFT context changes with the physical
canvas.

The specific pipeline:

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
6. resample that group to the target sample rate;
7. normalize to that group's exact target valid sample count;
8. run the official reflect frontend at the target valid length;
9. restore the original batch order.

Samples sharing the same discrete source/target length stay batch-vectorized;
samples of different lengths never affect each other's physical frontend boundary.
Grouping by source length alone is not allowed, because the same source rounding
result can map to different target rounding results under upsampling.

## Equivalent Log-Mel Frontend

### STFT

Uses the legacy repository's equivalent implementation:

- construct the DFT matrix with NumPy;
- construct the periodic Hann window with librosa;
- center-pad the window function to `n_fft`;
- register the real and imaginary parts separately as `conv1d` buffers;
- `stride=hop_length`;
- `center=True`;
- `pad_mode="reflect"`, degrading to `"constant"` for very short input (see
  [Padding Mode for Very Short Input](#padding-mode-for-very-short-input) below).

Power spectrum:

```text
power = real² + imag²
```

### Mel and dB

```text
mel = clamp(power @ mel_weight, min=1e-10)
logmel = 10 × log10(mel)
```

This is equivalent to the official PANNs parameters:

```text
ref=1.0
amin=1e-10
top_db=None
```

This implementation must not be claimed as a general replacement for
`torchlibrosa.power_to_db` under arbitrary `ref` or `top_db`.

### Padding Mode for Very Short Input

PyTorch's reflect padding requires:

```text
n_fft / 2 < waveform_samples
```

which corresponds to:

- 16 kHz: at least 257 samples, about 0.01606 seconds;
- 32 kHz: at least 513 samples, about 0.01603 seconds.

When this condition holds, the official reflect semantics are used. When
`waveform_samples <= n_fft / 2`, the official frontend leaves the case undefined, and
locally the padding mode degrades to zero-padding under the semantics of "everything
outside the valid audio is silence":

```text
pad_mode = "reflect" if waveform_samples > n_fft // 2 else "constant"
```

Both modes use the same padding width, `n_fft // 2`, so the frame-count formula stays
unified across the whole length domain as
`floor(target_valid_samples / hop_length) + 1`; `valid_feature_frames` and the
`-100 dB` minimum-padding logic are unaffected by this branch. `target_valid_samples
== 0` also falls into this branch, producing 1 official frame.

This branch is a local extension, not official PANNs behavior, and is not part of the
official alignment contract (see
[`../extra/panns-alignment.md`](../extra/panns-alignment.md)).

## `valid_feature_frames` and Minimum Encoder Padding

The official center-STFT produces, for the target valid waveform:

```text
valid_feature_frames = floor(target_valid_samples / hop_length) + 1
```

This field represents the number of log-mel frames actually computed by the official
frontend that can be fed to the Encoder, with dtype `torch.int64` and shape `[B]`.

### Normal Length

When `valid_feature_frames >= 32`:

- it is not padded to a multiple of 32;
- all officially produced features are kept;
- the Encoder's five rounds of floor pooling do not generate an extra embedding
  frame for a remainder that falls short of a complete 32-feature block;
- the remainder can still influence existing output through the convolutional
  receptive field before pooling, so the input must not be pre-cropped to a
  multiple of 32.

### Very Short Special Case

When `valid_feature_frames < 32`, the official Cnn14 cannot complete five rounds of
temporal pooling. To satisfy the project contract of "always return at least one
embedding frame":

1. keep all the official native log-mel;
2. append `-100 dB` before `bn0`;
3. pad only up to 32 frames;
4. then run that checkpoint's `bn0`.

`-100 dB` comes from:

```text
10 × log10(1e-10) = -100
```

It represents the value of a completely silent power spectrum after the PANNs dB
floor. It must not be replaced by the numeric value 0 after `bn0`.

`valid_feature_frames` still records the official frame count before padding. For
example, 0.02 seconds produces 3 frames; the output metadata still reads 3, while the
physical `input_features` has 32 frames.

## `bn0`

`bn0` belongs to the Transform:

- its channel count is `n_mels`;
- it corresponds to the official BatchNorm applied after log-mel and before the
  convolutional backbone;
- it is loaded from the selected checkpoint when `pretrained=True`;
- the `bn0` of the two 32 kHz variants is not interchangeable.

In training mode, different unique-length groups update the `bn0` statistics
separately. Inference alignment requires an explicit call to `eval()`; the
constructor does not automatically change the lifecycle.

## Output Contract

Returns:

```python
{
    "input_features": input_features,
    "valid_feature_frames": valid_feature_frames,
    "valid_seconds": valid_seconds,
}
```

Meaning:

| Key | Shape | Dtype | Semantics |
|---|---|---|---|
| `input_features` | `[B, T, n_mels]` | `float32` | Post-`bn0` log-mel |
| `valid_feature_frames` | `[B]` | `int64` | Actual official feature frame count before padding |
| `valid_seconds` | `[B]` | `float32` | Valid audio duration |

When different-length groups are merged back into a batch Tensor, the outer padding
between groups is filled with 0. The companion Encoder must regroup by
`valid_feature_frames` and drop the outer padding before the backbone, so this value
never enters model computation.

## Device, Training, and Serialization

- `device` is derived from the registered parameters or buffers;
- waveform and valid_seconds are moved automatically;
- the frontend always uses float32;
- does not automatically call `eval()`;
- does not automatically freeze;
- does not wrap `torch.no_grad()`;
- STFT, mel, and `bn0` follow ordinary `state_dict` semantics;
- unknown `**kwargs` are not accepted or silently ignored.

## Files and Exports

Implementation location:

```text
src/timbral/models/transforms/panns.py
```

Division of responsibility:

- `src/timbral/models/helpers/panns.py`: shared types, official frontend parameter
  table, checkpoint identities, download, verification, and safe checkpoint reading;
- `src/timbral/models/helpers/common.py`: integer-rational ties-to-even conversion
  (`round_positive_ratio`);
- `src/timbral/models/helpers/grouping.py`: model-agnostic scaffolding for
  unique-length grouping iteration and scattering grouped results back into a zero
  canvas;
- `src/timbral/models/transforms/panns.py`: checkpoint frontend key mapping, the
  equivalent STFT/log-mel, and `PannsLogmelTransform`.

`timbral.models.helpers`'s `__init__` performs no re-export; the registry, Encoder,
and Transform import directly from `timbral.models.helpers.panns`.
`timbral.models.transforms` re-exports the same `PannsVariant` and
`PannsLogmelTransform`. `timbral.models` at the top level only re-exports registry
symbols (see [`../registry.md`](../registry.md)).

## Testing Requirements

The no-weight tests must cover at least:

- the three valid official configurations and invalid combinations;
- experimental `n_fft` must be even;
- `pretrained=False` resolving no path and making no network calls;
- `[B,N]` and `[B,C,N]`;
- mean downmixing, resampling, and float32;
- valid_seconds validation and round semantics;
- batch/per-sample consistency after unique-length grouping;
- non-zero invalid padding not affecting any valid output;
- STFT against `torch.stft`;
- independent mel and dB references;
- 0.02 seconds producing fewer than 32 native features, then padded to 32;
- input at or below `n_fft // 2` taking the zero-padding branch and returning
  normally, with `valid_feature_frames` still equal to
  `floor(target_valid_samples / hop_length) + 1`;
- normal length not being padded to a multiple of 32;
- `valid_feature_frames` using the official `floor(N/hop)+1`;
- the outer padding between groups being 0;
- unknown parameters raising errors;
- correct input/output devices after `.to(device)`.

## Dependency Boundaries

At runtime only PyTorch, torchaudio, NumPy, librosa, and huggingface_hub are used.
`torchlibrosa` is not introduced, and the project's dependency declarations are not
restored or modified.

The real-weight, torchlibrosa-equivalence, and official full-network tests are in
[`../extra/panns-alignment.md`](../extra/panns-alignment.md).
