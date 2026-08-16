# `BaseEncoder` Design

This document freezes the final design of `timbral.models.encoders.BaseEncoder`, describing the abstract
interface, capability declarations, granularity dispatch, and output contract. Each concrete model defines
its own supported granularities and their implementation semantics in a separate document.

The corresponding Transform design is
[`../transforms/base.md`](../transforms/base.md).

## Design goals

`BaseEncoder` is used to constrain what all audio Encoders share:

- the embedding granularity at instantiation time;
- the way Transform feature inputs are consumed;
- the clip/frame dispatch mechanism;
- the output semantics for embedding, geometry, and the valid-position mask;
- automatic device-transfer behavior;
- the granularity capabilities each concrete model expresses through intrinsic class attributes.

Concrete models are free to decide their own backbone network, pooling method, frame time grid,
model-specific inputs, embedding dimension, intrinsic maximum length, and additional outputs.

## Relationship with Transform

Encoder does not hold a Transform, and does not accept raw waveforms. The standard call is split into two
steps:

```python
transform_output = transform(
    waveform,
    sample_rate=sample_rate,
    valid_seconds=valid_seconds,
)
encoder_output = encoder(**transform_output)
```

The base class does not provide a one-shot `encoder(waveform, ...)`, nor does it provide an
`encoder.transform` property.

Transform-Encoder pairing is entirely the responsibility of the caller or a future construction layer:

- no `FeatureSpec` is introduced;
- no `family` attribute is required;
- class-name prefixes are not compared;
- no compatibility check is performed at initialization or in the forward pass.

If the user pairs the wrong concrete Transform with the wrong Encoder, it is acceptable to rely on shape,
parameter, or operator errors surfacing during the forward pass.

## Abstract interface

The expected interface is as follows:

```python
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, TypeAlias

import torch
from torch import Tensor, nn


Granularity: TypeAlias = Literal["clip", "frame"]


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all audio Encoders."""

    supported_granularities: ClassVar[frozenset[Granularity]] = frozenset()
    embedding_dim: int

    def __init__(self, granularity: Granularity) -> None:
        """Initializes the Encoder and fixes the output granularity."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Returns the device on which the current Encoder receives its input."""

    def forward(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Moves the input and dispatches to the corresponding encoding hook based on the instance's granularity."""

    def _encode_clip(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Produces clip-granularity embedding, geometry, and mask."""

    def _encode_frame(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Produces frame-granularity embedding, geometry, and mask."""
```

`BaseEncoder` inherits from `torch.nn.Module` and `abc.ABC`. `device` is an abstract property.
`_encode_clip` and `_encode_frame` are optional extension hooks; the base class's default implementation
raises `NotImplementedError`, and concrete classes override the hook corresponding to their supported
granularity.

Granularity capability is an intrinsic class-level attribute:

```python
supported_granularities: ClassVar[
    frozenset[Granularity]
] = frozenset()
```

The default set is empty; subclasses override it according to their own semantics. For example: AST and
PANNs use `{"clip", "frame"}`, and CLAP uses `{"clip"}`. The attribute follows ordinary Python
class-attribute inheritance rules and can be queried before instantiation and before weight preparation.
The base class performs no class-definition-time consistency check; the capability set and hook
implementations of each concrete Encoder are guaranteed by its own tests.

The embedding dimension is an intrinsic attribute of the concrete Encoder:

```python
embedding_dim: int
```

`embedding_dim` is the last dimension `D` of the output `embedding`; the base class provides no default.
Callers (such as an embedding-extraction orchestration layer) use it to build a well-shaped output schema
before the forward pass, and must read it **off the instance**.

A concrete Encoder whose width is fixed by its architecture declares it as a class attribute, which also
makes it queryable before instantiation (AST 768, CLAP 512, PANNs 2048, BEATs 768, wav2vec2 768). A concrete
Encoder whose width depends on its constructor arguments assigns it per instance: the ATST families derive
`D` from both `arch` and `n_blocks` (`2 * n_blocks * D` for ATST-Clip, `n_blocks * D` for ATST-Frame), so no
single class-level value exists for them. The declaration is therefore instance-level, which admits both
cases without requiring one Encoder class per width combination.

## Constructor interface

The only public constructor parameter is:

```python
granularity: Literal["clip", "frame"]
```

At initialization, the base class first checks whether the string belongs to the public granularity set,
and then checks whether that granularity is contained in the concrete class's `supported_granularities`.
Both kinds of construction-parameter errors immediately raise `ValueError`; the error message for a
model-unsupported granularity must include the concrete class name and its supported set. The capability
check happens before the concrete Encoder's weight preparation. Granularity is not a dynamic per-`forward`
parameter.

The following are not public constructor parameters or public base-class state:

- `embedding_dim`;
- model name;
- Transform instance;
- input feature spec;
- family or entry;
- maximum input length;
- whether to return geometry.

Geometry is a core output of whichever granularities a model supports, so there is no `return_geometry`
parameter.

Concrete Encoders should explicitly declare model-specific constructor parameters. When extension is
genuinely needed, `**kwargs` may be retained at the end of the concrete constructor, but unknown parameters
must not be silently ignored; unknown parameters must raise `TypeError`.

## Input contract

### `input_features`

`input_features` is the primary feature Tensor produced by the Transform. The public interface only
constrains the batch dimension; the remaining layout and shape are agreed upon between the paired
concrete Transform and Encoder.

### `valid_seconds`

`valid_seconds`:

- is a `float32` Tensor of shape `[B]`;
- is in units of seconds;
- is passed via a keyword-only parameter;
- represents each sample's valid audio duration;
- typically comes directly from the Transform output.

`BaseEncoder` automatically moves it to `device`, but does not convert its dtype. Users who call the
Encoder directly, bypassing the Transform, should ensure themselves that it is already `float32`.

When a concrete model needs sample counts, it should round the number of seconds to its own sample-rate
grid.

### Model-specific inputs

Additional keys in the Transform output are passed via `**kwargs`, e.g. an attention mask. A concrete
Encoder only consumes the parameters it supports; unknown parameters must raise `TypeError`.

## Automatic device transfer

Before granularity dispatch, `BaseEncoder.forward` moves the following to `self.device`:

- `input_features`;
- `valid_seconds`;
- top-level Tensors within `**kwargs`.

The base class does not recursively traverse lists, tuples, dicts, or other nested structures. Tensors
within nested containers are handled by the concrete Encoder.

`device` is an abstract property, and its implementation is left to the concrete Encoder. The base class
does not register a unified device buffer, nor does it assume a model has exactly one parameter or buffer.

## Granularity dispatch

Concrete Encoders provide a corresponding hook for every semantically valid granularity, and declare it in
`supported_granularities`. The model design document must state the supported set; dropping a granularity
requires confirmation at the model design stage, with its semantic rationale recorded.

`BaseEncoder.forward` performs only two pieces of common work:

1. automatically moving the input to the device;
2. calling the corresponding extension hook based on the instance's `granularity`.

The dispatched hook returns the complete output dict directly. The base class does not:

- default to deriving clip from frame mean-pooling;
- derive the number of valid frames;
- generate geometry;
- generate or apply `valid_mask`;
- validate output keys, rank, batch size, or shape.

These tasks are the responsibility of the concrete Encoder, guaranteed correct via unit tests.

## Output contract

A clip or frame hook declared as supported must return a plain `dict[str, Tensor]`, containing at least:

```python
{
    "embedding": embedding,
    "geometry": geometry,
    "valid_mask": valid_mask,
}
```

Concrete Encoders may add model-specific output keys.

### Clip granularity

```text
embedding   [B, D]
geometry    [B, 2]
valid_mask  [B]
```

Semantics:

- `geometry[i] = [0, valid_seconds[i]]`;
- `geometry` is in units of seconds;
- `valid_mask` has dtype `bool`, with all elements `True`;
- `embedding` retains the model's original output dtype.

### Frame granularity

```text
embedding   [B, T, D]
geometry    [B, T, 2]
valid_mask  [B, T]
```

where `T` is the output time length for the padded batch, and `D` is the concrete model's embedding
dimension.

Each sample's valid frames must satisfy:

- geometry is arranged in frame order;
- time intervals are non-overlapping;
- the first valid interval starts at 0 seconds;
- the last valid interval ends at `valid_seconds`;
- all valid intervals together cover the entire valid audio;
- intervals are generated according to the concrete model's frame step or time grid, not by unconditional
  equal division;
- geometry represents time ownership, not a claim about the actual neural-network receptive field.

Invalid padding frames must satisfy:

- `valid_mask=False`;
- the corresponding embedding row filled entirely with 0;
- the corresponding geometry row filled entirely with 0.

Invalid frames use exact zero values, avoiding NaN contamination of loss, normalization, or downstream
aggregation.

### dtype and device

- `geometry` fixed to `float32`;
- `valid_mask` fixed to `bool`;
- `embedding` retains the model's output dtype;
- output Tensors should reside on the concrete Encoder's output device.

## Variable length and maximum length

`BaseEncoder` does not declare a unified maximum input length, nor does it expose
`max_input_seconds`/`max_input_samples`.

- models such as PANNs and CRNN that natively support variable length must not be forced onto an
  artificially fixed canvas;
- Transformer variants that support variable length likewise keep variable length;
- Encoders such as AST that have genuine intrinsic limits, or their companion Transforms, should
  explicitly check for and raise errors in their own forward pass;
- the base class does not automatically crop, chunk, or silently truncate.

## Error and validation responsibility

The base class checks at initialization whether `granularity` belongs to the public value set and the
model's supported set. The base class does not check consistency between capability attributes and hook
implementations, nor does it validate the complete output contract after every forward pass. The following
issues are surfaced by the concrete implementation, the underlying operators, and tests:

- mismatched Transform and Encoder;
- incorrect `input_features` layout or shape;
- missing model-specific kwargs;
- unknown kwargs present;
- model input exceeding its own intrinsic limit;
- geometry or mask implementations violating the contract.

This design avoids extra shape, key, or compatibility checks on every forward pass.

## Training and serialization

`BaseEncoder` is an ordinary trainable `nn.Module`:

- does not automatically enter `eval()`;
- does not automatically freeze parameters;
- does not wrap in `torch.no_grad()`;
- does not override PyTorch's default `state_dict` semantics;
- does not restrict concrete models from using AMP or other training strategies.

Whether the model is frozen at inference time, which device it uses, and whether it enters eval mode, is
decided by a future factory or the caller.

## Files and exports

File structure:

```text
src/timbral/models/
├── __init__.py
└── encoders/
    ├── __init__.py
    └── base.py
```

- `BaseEncoder` and `Granularity` are defined in `timbral.models.encoders.base`;
- `timbral.models.encoders` re-exports `BaseEncoder` and `Granularity`;
- `timbral.models` at the top level only re-exports registry symbols (see
  [`../registry.md`](../registry.md)), and does not include them;
- the top-level public API is handled by the registry.

## Testing requirements

Use a minimal dummy Encoder to test the abstract interface and capability declarations. Tests must cover
at least:

- `BaseEncoder` cannot be instantiated directly;
- the default capability set is empty;
- declared capabilities can be queried before instantiation;
- a clip-only dummy can implement only `_encode_clip`;
- a clip-only dummy initialized with `granularity="frame"` immediately raises `ValueError`;
- correct dispatch for `granularity="clip"` and `"frame"`;
- an invalid granularity raises `ValueError` at initialization;
- `valid_seconds` is keyword-only;
- automatic transfer of `input_features`, `valid_seconds`, and top-level kwargs Tensors;
- nested kwargs are not recursively transferred by the base class;
- clip output shape, geometry, and an all-True mask;
- frame output shape, ownership geometry, mask, and zero padding;
- both granularities allow additional model-specific outputs;
- unknown kwargs raise `TypeError`.

Interface docstrings, test descriptions, comments, and error messages are all written in Simplified
Chinese, following the Google style.

## Final decision audit

The table below records the final design of the Encoder public interface.

| No. | Topic | Final conclusion |
|---:|---|---|
| E01 | Document scope | Defines the interface and capability protocol shared by all concrete Encoders |
| E02 | Degree of unification | Unifies construction, forward, output semantics, and types; does not force identical embedding dimensions |
| E03 | Legacy API compatibility | Not compatible with the legacy `timbral.encoders.*` |
| E04 | API stability tier | Component classes are exported from `timbral.models.encoders`; the top-level public API is handled by the registry |
| E05 | Abstraction mechanism | Uses `nn.Module + ABC` |
| E06 | Training/inference positioning | An ordinary trainable module, not restricted to being an inference component |
| E07 | Whether it holds a Transform | Does not hold one; the Encoder only consumes Transform outputs |
| E08 | One-shot waveform forward | Not provided |
| E09 | Transform-Encoder pairing | Handled by a future construction layer |
| E10 | Compatibility metadata | Does not use `FeatureSpec`, family, or prefix protocols |
| E11 | Transform pairing check | Not checked at initialization or in forward |
| E12 | Sole public constructor parameter | `granularity` |
| E13 | granularity type | `Literal["clip", "frame"]` |
| E14 | granularity selection timing | Selected and fixed at instantiation |
| E15 | granularity validation | Checked at initialization against the public value set and the model's supported set; errors raise `ValueError` |
| E16 | Model granularity capability | Intrinsic class attribute, default empty set, subclasses override per their own semantics |
| E17 | Granularity extension hooks | Base class provides a default failing implementation; subclasses override the hook for their supported granularity |
| E18 | Default clip pooling | Not provided, to avoid assuming clip equals frame mean |
| E19 | Hook return content | A hook declared as supported returns the complete output dict |
| E20 | Base class forward responsibility | Only handles device transfer and granularity dispatch |
| E21 | forward output validation | Does not check keys, rank, or shape |
| E22 | Public inputs | `input_features`, keyword-only `valid_seconds`, `**kwargs` |
| E23 | Public time coordinate | Uses seconds, not sample counts |
| E24 | valid_seconds dtype | Required to be float32; the Encoder transfers it but does not convert it |
| E25 | Model sample-count conversion | Concrete implementation rounds according to its own sample rate |
| E26 | Automatic device transfer | Transfers public Tensors and top-level kwargs Tensors |
| E27 | Nested Tensor transfer | Not recursive; left to the concrete implementation |
| E28 | device source | Abstract property; no unified device buffer registered |
| E29 | Output carrier | Plain `dict[str, Tensor]` |
| E30 | Required output keys | `embedding`, `geometry`, `valid_mask` |
| E31 | geometry optionality | Not optional; a core output of clip/frame |
| E32 | clip geometry | `[0, valid_seconds]` |
| E33 | frame geometry | Non-overlapping ownership intervals on the model's time grid |
| E34 | geometry coverage | Covers from 0 to the full valid duration |
| E35 | geometry unit/dtype | Seconds, float32 |
| E36 | clip valid_mask | `[B]`, all True |
| E37 | frame valid_mask | `[B,T]` bool |
| E38 | Invalid frame embedding | Filled with 0 |
| E39 | Invalid frame geometry | Filled with 0 |
| E40 | Invalid frame values | embedding and geometry use exact zero values |
| E41 | embedding dtype | Retains the model's output dtype |
| E42 | Model-specific inputs | Via `**kwargs` |
| E43 | Model-specific outputs | Additional Tensor keys allowed |
| E44 | Unknown kwargs | Must raise `TypeError` |
| E45 | embedding_dim | Not a public constructor parameter or base-class state; read off the instance, declared as a class attribute only by Encoders of fixed width |
| E46 | return_geometry | Removed; geometry is always returned |
| E47 | Public maximum length | Not declared |
| E48 | Intrinsic length limit | Checked by the concrete model's forward pass |
| E49 | Default behavior on overlength | No automatic cropping, chunking, or silent truncation |
| E50 | Lifecycle | Does not force eval, freezing, or no-grad |
| E51 | State serialization | Uses PyTorch's default `state_dict` semantics |
| E52 | Interface correctness assurance | Relies on ABC, docstrings, concrete implementations, and unit tests |
