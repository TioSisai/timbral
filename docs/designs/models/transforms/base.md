# `BaseTransform` Design

This document finalizes the design of `timbral.models.transforms.BaseTransform`. The
design conclusions come from an implementation audit of the 7 legacy model families,
together with multiple rounds of interface interviews; where an early conclusion from
those interviews conflicts with this document, the final conclusion recorded here takes
precedence.

This document only describes the abstract interface and its behavioral contract. It does
not migrate any legacy Transform, nor does it implement common preprocessing helpers, a
registry, a factory, or a Transform–Encoder composition layer.

## Design Goals

`BaseTransform` constrains the calling convention, input semantics, and minimal output
shared by all audio Transforms, while still letting each model decide for itself:

- whether to resample, and to what sample rate;
- how to handle mono, stereo, or a larger number of channels;
- how to produce waveforms, spectrograms, fbank, or other model inputs;
- whether to accept arbitrary-length input;
- whether repeat-pad, spectral-domain padding, cropping, or other model-specific
  processing is required;
- whether extra forward-pass arguments and output Tensors are needed.

The common interface only fixes the semantics that are genuinely shared. It does not
bake the legacy fixed 10-second window, uniform downmixing, uniform resampling, or
fixed-length zero-padding pipeline into the base class.

## Relationship to the Encoder

Transform and Encoder are two mutually independent components:

- `BaseTransform` does not hold an Encoder;
- `BaseEncoder` does not hold a Transform either;
- the output dict of a Transform can be unpacked and passed directly to an Encoder;
- pairing a Transform with an Encoder is the responsibility of a future entry point,
  factory, or composition layer.

The base class does not introduce `FeatureSpec`, `family`, class-name-prefix validation,
or a runtime compatibility check. When a user chooses an incorrect combination, the
resulting shape, argument, or operator error surfacing from the concrete model's forward
pass is allowed to be the mechanism that reveals the problem.

## Abstract Interface

The expected interface is:

```python
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn


class BaseTransform(nn.Module, ABC):
    """Abstract base class for all audio Transforms."""

    target_sample_rate: int

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device on which this Transform expects its input."""

    @abstractmethod
    def forward(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
        valid_seconds: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Convert a batch of waveforms into the input required by the Encoder."""
```

`BaseTransform` inherits from both `torch.nn.Module` and `abc.ABC`. `device` and
`forward` are both genuinely abstract members, so the base class cannot be instantiated
directly.

### Constructor Interface

Transform has no public constructor parameters. The base class declares, via a type
annotation, a single public instance attribute, `target_sample_rate` (the target sample
rate used for feature extraction), which each concrete Transform sets during
construction; callers use it to determine whether the input sample rate is below the
model's native sample rate. The base class does not uniformly require the following
state:

- maximum input length;
- target number of samples;
- fixed number of output frames;
- model name or family;
- output feature spec.

Concrete Transforms should explicitly declare their own constructor parameters. When a
model genuinely needs extension parameters of its own, the concrete constructor may keep
a trailing `**kwargs`, but it must not silently ignore unknown parameters; unknown
parameters must raise `TypeError`.

## Input Contract

### `waveform`

`waveform` must be a floating-point Tensor and supports two shapes:

```text
[B, N]
[B, C, N]
```

where:

- `B` is the batch size;
- `C` is the number of channels;
- `N` is the number of physical samples for that batch.

Different samples may have different valid durations, but they must already be padded
into a single Tensor. The interface does not accept a ragged Tensor list.

The Transform must automatically move `waveform` to its own `device`, but it is not
required to uniformly convert the waveform dtype. A concrete implementation may choose
whatever compute dtype its algorithm needs.

### `sample_rate`

`sample_rate` represents the original sample rate of the input `waveform` and must
satisfy:

- it is a positive Python `int`;
- all samples within a batch share the same sample rate;
- it is passed as a keyword-only argument;
- it does not accept a float, a per-sample Tensor, or a list of sample rates.

Samples with different sample rates should be split into separate batches by the
caller.

### `valid_seconds`

`valid_seconds` represents each sample's valid audio duration, in seconds, with shape
`[B]`.

- `None` means the entire physical Tensor of every sample is valid, i.e.
  `N / sample_rate`;
- when not `None`, every element must satisfy:

  ```text
  0 < valid_seconds[i] <= N / sample_rate
  ```

- the concrete Transform must move it to its own `device` and normalize it to
  `float32`;
- padding content beyond the valid duration must not affect the output features;
- if a concrete model needs a sample count, it should round the number of seconds to
  the corresponding sample grid using the sample rate that model requires.

Seconds are used as the common temporal coordinate because seconds do not change with
resampling rate or feature time resolution. Sample counts are only used as a derived
quantity internal to a concrete model.

### Multi-Channel Semantics

Every Transform must accept both `[B, N]` and `[B, C, N]`, but the base class does not
mandate mean downmixing.

- mono models may downmix in their concrete implementation;
- natively multi-channel models may preserve or reorganize channels;
- the final channel semantics must be documented by the concrete Transform.

### Automatic Device Transfer

The concrete Transform's `forward` must automatically move:

- `waveform`;
- `valid_seconds`;
- any top-level Tensor found in `**kwargs`.

The base class does not require recursively traversing lists, tuples, dicts, or other
nested containers. Tensors inside nested structures are the responsibility of the
concrete implementation.

Because `forward` is fully abstract, the device transfer described above is a contract
that the concrete implementation must honor, not a shared implementation provided by the
base class.

## Output Contract

Transform returns a plain `dict[str, Tensor]` that must contain at least:

```python
{
    "input_features": input_features,
    "valid_seconds": valid_seconds,
}
```

### Required Fields

`input_features`
: The Encoder's primary input Tensor. Its layout and shape besides the batch dimension
  are defined by the concrete model.

`valid_seconds`
: A `float32` Tensor of shape `[B]`, located on the Transform's `device`, representing
  each sample's valid duration.

### Extension Fields

A concrete Transform may add arbitrary model-specific Tensors, such as an attention
mask, `is_longer`, or other auxiliary inputs. This allows calling directly:

```python
transform_output = transform(
    waveform,
    sample_rate=sample_rate,
    valid_seconds=valid_seconds,
)
encoder_output = encoder(**transform_output)
```

Model-specific fields are received by the corresponding Encoder's `**kwargs`. The common
interface does not use a dataclass, TypedDict, or a nested `extras` container.

## Variable Length and Maximum Length

The base class does not declare `max_input_seconds`, `max_input_samples`, or any uniform
maximum length.

- architectures that natively support variable length, such as PANNs or CRNN, should
  not be artificially constrained;
- Transformer variants that support variable length should likewise not be constrained
  by the legacy fixed canvas;
- models such as AST, which do have an official or architectural length ceiling,
  should check for it and raise a clear error inside their own `forward`;
- the base class does not automatically crop, chunk, or silently truncate input.

Automatic chunking would change clip embeddings, cross-chunk context, and frame
geometry, so it is not part of the default responsibility of the Transform abstract
interface.

## Error and Validation Responsibilities

Even though the base class does not implement a common `forward`, every concrete
Transform must ensure the following errors are never silently swallowed:

- waveform is not a floating-point Tensor;
- waveform rank does not match `[B, N]` or `[B, C, N]`;
- `sample_rate` is not a positive Python `int`;
- `valid_seconds` does not have shape `[B]`;
- `valid_seconds` exceeds the bounds of the physical waveform;
- input exceeds that concrete model's own inherent limit;
- unsupported `**kwargs` are received.

The base class is not required to provide a debug-validation switch.

## Training and Serialization

`BaseTransform` is an ordinary trainable `nn.Module`:

- it does not automatically enter `eval()`;
- it does not automatically freeze parameters;
- it does not wrap `torch.no_grad()`;
- it does not override PyTorch's default `state_dict` semantics.

`device` is an abstract property, and how it is implemented is up to the concrete
Transform. The base class does not register a uniform device buffer, nor does it assume
a concrete implementation necessarily has parameters or buffers.

## Files and Exports

File layout:

```text
src/timbral/models/
├── __init__.py
└── transforms/
    ├── __init__.py
    └── base.py
```

- `BaseTransform` is defined in `timbral.models.transforms.base`;
- `timbral.models.transforms` re-exports `BaseTransform`;
- the `timbral.models` top level only re-exports registry symbols (see
  [`../registry.md`](../registry.md)) and does not include `BaseTransform`;
- the top-level public API is owned by the registry.

## Testing Requirements

At this stage, only a minimal dummy Transform is used to test the abstract interface;
no legacy model is migrated. Tests must cover at least:

- `BaseTransform` cannot be instantiated directly;
- a dummy subclass must implement `device` and `forward`;
- `sample_rate` and `valid_seconds` are keyword-only;
- the `[B, N]` and `[B, C, N]` input contract;
- the semantics of `valid_seconds=None`;
- the range, dtype, and device of `valid_seconds`;
- automatic transfer of common Tensors and top-level kwargs Tensors;
- padding regions do not affect the dummy output;
- the output contains at least `input_features` and `valid_seconds`;
- the output is allowed to contain extra model-specific Tensors;
- unknown kwargs raise `TypeError`.

Interface docstrings, test descriptions, comments, and error messages all use Simplified
Chinese and follow the Google style.

## Final Decision Audit

The table below re-audits every design branch related to Transform that arose during
the interviews. Early designs that were superseded by later answers have not been
retained.

| No. | Topic | Final Conclusion |
|---:|---|---|
| T01 | Scope of this stage's implementation | Only the interface and its tests are implemented; no concrete model is migrated |
| T02 | Degree of unification | Unify the calling signature and input/output semantics and types; do not force all models to share the same feature shape |
| T03 | Legacy API compatibility | Not compatible with the legacy `timbral.encoders.*` |
| T04 | API stability tier | Component classes are exported from `timbral.models.transforms`; the top-level public API is owned by the registry |
| T05 | Abstraction mechanism | Use `nn.Module + ABC` |
| T06 | Common `forward` implementation | Not provided; `forward` is an abstract method |
| T07 | Common preprocessing template | Legacy validation, downmixing, resampling, and fixed-length templates are not baked in |
| T08 | Common helpers | Not implemented at this stage; to be extracted once the first concrete Transform exists |
| T09 | Composition with Encoder | Fully decoupled; neither holds the other |
| T10 | Pairing check | No `FeatureSpec` or runtime check is used |
| T11 | Pairing naming | The base class does not require a matching family or class-name prefix |
| T12 | Pairing responsibility | Delegated to a future entry point, factory, or composition layer |
| T13 | Common constructor parameters | None |
| T14 | Model-specific constructor parameters | Declared explicitly by the concrete class, with a `**kwargs` fallback where necessary |
| T15 | Unknown kwargs | Must raise `TypeError`; must not be silently ignored |
| T16 | Waveform carrier | A padded Tensor is used; a ragged list is not accepted |
| T17 | Waveform rank | Both `[B,N]` and `[B,C,N]` are supported |
| T18 | Waveform dtype | A floating-point Tensor is required; float32 is not uniformly enforced |
| T19 | Batch sample rate | A single positive Python `int` shared within a batch |
| T20 | Multi-sample-rate batch | Split by the caller |
| T21 | Common length coordinate | Seconds are used, not sample counts |
| T22 | Default valid length | `valid_seconds=None` means the full physical Tensor |
| T23 | Valid-length dtype | Normalized to float32 by the Transform |
| T24 | Valid-length constraint | Must satisfy `0 < valid_seconds <= N / sample_rate` |
| T25 | Padding semantics | Content beyond the valid duration must not affect the output |
| T26 | Channel strategy | Decided by the concrete Transform; mono is not enforced |
| T27 | Automatic device transfer | The Transform automatically moves common Tensors and top-level kwargs Tensors |
| T28 | Nested Tensor transfer | Not uniformly recursive; delegated to the concrete implementation |
| T29 | Source of `device` | An abstract property; no uniform device buffer is registered |
| T30 | Output carrier | A plain `dict[str, Tensor]` |
| T31 | Required output keys | `input_features`, `valid_seconds` |
| T32 | Model-specific output | Arbitrary extra Tensor keys are allowed |
| T33 | Common maximum length | Not declared |
| T34 | Inherent maximum length | Checked by the constrained model in its own `forward` |
| T35 | Default behavior on overlength | No automatic cropping, chunking, or silent truncation |
| T36 | Lifecycle | eval, freezing, or no-grad are not enforced |
| T37 | State serialization | Uses PyTorch's default `state_dict` semantics |
| T38 | Argument position | `sample_rate` and `valid_seconds` are both keyword-only |
| T39 | Output-to-Encoder handoff | Supports `encoder(**transform_output)` |
| T40 | Interface validation method | Relies on ABC, docstrings, concrete implementations, and unit tests |
