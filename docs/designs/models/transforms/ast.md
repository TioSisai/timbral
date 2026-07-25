# `AstKaldiFbankTransform` Design

This document finalizes the design of `timbral.models.transforms.AstKaldiFbankTransform`.
The companion Encoder is described in [`../encoders/ast.md`](../encoders/ast.md); the
official alignment contract is in
[`../extra/ast-alignment.md`](../extra/ast-alignment.md).

This document describes the behavior the current implementation must satisfy. Where
this document conflicts with the legacy repository implementation, this document takes
precedence; where this document conflicts with the `BaseTransform` common contract,
[`base.md`](base.md) takes precedence.

## Design Goals

`AstKaldiFbankTransform` is responsible for:

- accepting the waveform input mandated by the new `BaseTransform`;
- producing a `[B,1024,128]` fbank that is numerically aligned with the fixed Hugging
  Face AST frontend;
- using a batched, vectorized Torch implementation to avoid calling the official CPU
  frontend sample-by-sample on the production path;
- supporting mean downmixing and arbitrary positive-integer input sample rates;
- isolating any non-zero padding outside `valid_seconds`;
- enforcing the project's hard `10.255s` cap on valid audio;
- preserving the project extension for positive durations shorter than one Kaldi
  window;
- maintaining the device, gradient, and serialization semantics of an ordinary
  `nn.Module`.

This component does not hold an Encoder, does not load model weights, does not
generate an attention mask, and is not responsible for automatic chunking or
over-window truncation.

## Supported Model Identity

At this stage, only the following is supported:

```text
MIT/ast-finetuned-audioset-10-10-0.4593
```

The public constructor has no parameters:

```python
class AstKaldiFbankTransform(BaseTransform):
    def __init__(self) -> None:
        ...
```

No mel count, window, hop, maximum frame count, or normalization configuration is
exposed. These parameters, together with the fixed checkpoint's positional encoding and
patch grid, form an inseparable model identity; if another AST variant is needed, a new
explicit design should be added rather than expanding this class into a semi-generic,
unverifiable frontend.

## Fixed Frontend Parameters

| Parameter | Value |
|---|---:|
| Target sample rate | 16,000 Hz |
| Mel bands | 128 |
| Kaldi window length | 400 samples / 25 ms |
| Kaldi hop | 160 samples / 10 ms |
| FFT length | 512 |
| Minimum frequency | 20 Hz |
| Pre-emphasis coefficient | 0.97 |
| Maximum fbank frame count | 1024 |
| AudioSet mean | -4.2677393 |
| AudioSet std | 4.5689974 |
| Maximum target samples | 164080 |
| Maximum valid duration | 10.255 s |

The target sample rate is also set as the public instance attribute
`target_sample_rate = 16000` at construction time, fulfilling the `BaseTransform`
attribute contract.

The maximum target sample count is derived from:

```text
(max_length - 1) × hop_length + win_length
= (1024 - 1) × 160 + 400
= 164080
```

This is the minimum waveform length needed to produce the 1024th complete Kaldi frame.
This value is a project contract, not the official Hugging Face waveform rejection
boundary; the official frontend silently truncates longer waveforms at the fbank layer,
whereas this project deliberately turns that case into an explicit error.

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

Integer waveforms are not implicitly scaled or converted; they raise `TypeError`
directly.

### `sample_rate`

- must be a positive Python `int`;
- shared across a batch;
- when not 16 kHz, torchaudio resampling to 16 kHz is used.

### `valid_seconds`

- `None` means the entire physical Tensor is valid, i.e. `N / sample_rate`;
- when not empty, must be a Tensor of shape `[B]`;
- each element must satisfy `0 < valid_seconds <= N / sample_rate`;
- automatically moved to the Transform's device and cast to `float32`;
- the maximum-length check targets the valid audio, not the physical padded Tensor;
- each element must satisfy `valid_seconds <= 10.255`, otherwise `ValueError` is
  raised.

Consequently, a padded Tensor with a physical length of 20 seconds but
`valid_seconds=5` seconds is legal; the same Tensor with `valid_seconds=None`,
representing 20 seconds of valid audio, must raise an error.

The valid sample count on the model's sampling grid is:

```text
source_valid_samples = round(valid_seconds × sample_rate)
target_valid_samples = round(valid_seconds × 16000)
```

For a positive-duration input at a foreign sample rate that is shorter than half a
target sampling period, `target_valid_samples` may be 0. This does not change the fact
that the original `valid_seconds>0`; the Transform returns fully spectral-domain
padding, and the companion Encoder still reserves an ownership slot for this positive
duration.

## Waveform Preprocessing

The forward-pass order is fixed as:

1. validate input type, shape, sample rate, and `valid_seconds`;
2. move to the Transform's device and cast to `float32`;
3. clear the invalid tail according to `source_valid_samples`;
4. mean-downmix `[B,C,N]`;
5. crop away physical padding that is impossibly beyond the valid region;
6. batch-vectorized resampling to 16 kHz for the whole batch;
7. crop or right-pad with zeros to 164080 samples;
8. clear the invalid tail again according to `target_valid_samples`.

Step 8 cannot be omitted. The resampling filter produces ringing past the true endpoint;
re-zeroing after resampling is what keeps a mixed-length batch consistent with per-sample
cropping of the valid prefix.

AST uses a uniform fixed canvas, so it does not need to replicate PANNs' Python loop
that groups samples by unique length.

## Batched Kaldi fbank

For a `[B,164080]` waveform, execute:

1. `unfold(win_length=400, hop_length=160)` to obtain 1024 frames;
2. subtract the mean per frame;
3. apply 0.97 pre-emphasis using in-frame replicate boundary handling;
4. multiply by `torch.hann_window(400, periodic=False)`;
5. right-pad with zeros to 512 points;
6. compute `rfft` and then the power spectrum;
7. project onto the fixed Kaldi 128-bin mel filter;
8. compute `log(max(mel, finfo(float32).eps))`.

For each sample, the true number of complete fbank frames is:

```text
num_valid_fbank_frames =
    max(floor((target_valid_samples - 400) / 160) + 1, 0)
```

Starting from row `num_valid_fbank_frames` in the raw fbank domain, all rows are
replaced with 0, and then the following is applied uniformly:

```text
input_features = (fbank - mean) / (2 × std)
```

This is the spectral-domain padding semantics of Hugging Face's `ASTFeatureExtractor`.
The raw spectral-domain 0, after normalization, is approximately `0.4670324`, and must
not be replaced by the log-mel value produced from a silent waveform.

### Very Short Input

`0..399` target samples do not produce a single complete Kaldi frame:

- `num_valid_fbank_frames=0`;
- all 1024 rows use the spectral-domain 0;
- after normalization this yields a fixed padding constant;
- the Transform still returns normally.

The current Hugging Face torchaudio frontend rejects this kind of input, so this is a
project extension and not part of the official native numerical-alignment scope.

## Maximum Length

This project's ceiling for valid input is:

```text
target_valid_samples <= 164080
valid_seconds <= 10.255
```

Exceeding this must raise `ValueError` in the Transform; it must not:

- silently crop;
- automatically chunk;
- return features that correspond only to a prefix of the input while still keeping
  the original `valid_seconds`.

This guarantees that the downstream embedding, geometry, and `valid_seconds` all
describe the same segment of audio.

## Output

The Transform returns only the fields required by the base class:

```python
{
    "input_features": input_features,
    "valid_seconds": valid_seconds,
}
```

| Key | Shape | Dtype | Semantics |
|---|---|---|---|
| `input_features` | `[B,1024,128]` | `float32` | Normalized AST fbank |
| `valid_seconds` | `[B]` | `float32` | Valid duration of the input |

It does not return the legacy `valid_samples`, nor does it return an attention mask.
The companion Encoder derives the discrete length it needs from
`round(valid_seconds × 16000)`.

## Device, Training, and Serialization

- `device` is derived from the fixed mel-filter buffer;
- the mel filter and Hann window are registered as `persistent=False` buffers;
- both move with `.to(device)`, but do not enter the `state_dict`;
- the Transform's `state_dict` is empty;
- it does not automatically call `eval()`;
- it does not freeze parameters;
- it does not wrap `torch.no_grad()` or `torch.inference_mode()`;
- the continuous Torch frontend allows waveform gradients to propagate naturally;
- discrete control flow such as rounding, masking, and length checks is not promised
  to be differentiable.

The official alignment contract only covers `float32`. The Transform does not promise
any particular FFT/fbank behavior after an explicit conversion to half precision.

## Files and Exports

Implementation location:

```text
src/timbral/models/transforms/ast_transform.py
```

`timbral.models.transforms` re-exports `AstKaldiFbankTransform`;
`timbral.models` at the top level only re-exports registry symbols (see
[`../registry.md`](../registry.md)). A bare `ast.py` must not be created, to avoid
confusion with the Python standard library's `ast` module.

## Testing Requirements

The ordinary offline tests must cover at least:

- public export and the parameterless constructor;
- `[B,N]`, `[B,C,N]`, and mean downmixing;
- float64 input normalized to float32, integer input rejected;
- `sample_rate` and `valid_seconds` validation;
- a physically long padded input with a short valid region being legal;
- overlength behavior with `valid_seconds=None`;
- isolation of invalid regions before and after resampling;
- mixed-length batches versus per-sample cropping;
- a positive duration whose target sample count rounds to 0, plus the 1, 399, 400,
  559, 560, 164079, and 164080-sample boundaries;
- 164080 succeeding and beyond that raising `ValueError`;
- the spectral-domain padding constant differing from a silent-waveform fbank;
- CPU/CUDA device transfer;
- an empty `state_dict`;
- waveform gradients propagating;
- unknown parameters naturally raising `TypeError`.

The full numerical verification against the official implementation is in
[`../extra/ast-alignment.md`](../extra/ast-alignment.md).
