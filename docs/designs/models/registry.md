# Model Registry Design

This document freezes the design of `timbral.models.registry`. Each
component's own behavioral contract is documented in the corresponding
files under [`transforms/`](transforms/) and [`encoders/`](encoders/).

This document describes the behavior that the current implementation must
satisfy. Where this document conflicts with the historical statements in
the component documents that "no registry or factory is implemented," this
document takes precedence; each component's own construction and
forward-pass contract still follows its own document.

## Design goals

`timbral.models.registry` is responsible for:

- Managing Transform-Encoder pairing knowledge under a unique registration
  name, so callers no longer pair components themselves;
- Constructing a paired `(transform, encoder)` in one call via
  `create_model`;
- Listing all available registration names via `list_models`;
- Supporting extension and test injection via `register_model`;
- Pinning all deterministic configuration for each pairing under its
  registration name. Adding a new model only requires adding one
  `ModelSpec` declaration to the central table, without maintaining a
  per-model factory function.

This component does not implement a YAML configuration layer, does not
perform device migration, does not automatically call `eval()` or freeze
parameters, and does not duplicate validation that each component's
constructor already performs.

## Registration units and naming

Each registration name corresponds to one paired Transform + Encoder, at a
granularity as fine as each individually available pretrained-weight
combination. For models with an official Hugging Face entry, the repo_id
itself serves as the registration name; PANNS weights come from Zenodo, so
the registration name is the local cache model name from
`timbral.models.helpers`; BEATs weights come from a public OneDrive share
(see [`extra/beats-download.md`](extra/beats-download.md)), so the
registration name is the entry name from helpers (a naming conversion of
the cells in the official README table). ATST weights are published as
direct downloads outside any model hub (Aliyun OSS for ATST-Clip, Google
Drive for ATST-Frame), so the registration name is likewise the local
`model_name` from helpers, composed as `atst-{family}-{arch}`. The
registry's keys uniformly reference helpers constants
(`AST_CHECKPOINT.repo_id`, `CLAP_CHECKPOINT.repo_id`,
`WAV2VEC2_CHECKPOINT.repo_id`, `PANNS_CHECKPOINTS[...].model_name`,
`ATST_CHECKPOINTS[...].model_name`, and the keys of `BEATS_CHECKPOINTS`),
rather than literal strings, sharing a single source of truth with the
checkpoint identities:

| Registration name | Transform | Encoder | Supported granularities |
|---|---|---|---|
| `MIT/ast-finetuned-audioset-10-10-0.4593` | `AstKaldiFbankTransform` | `AstEncoder` | clip, frame |
| `laion/clap-htsat-fused` | `ClapLogmelTransform` | `ClapHtsatEncoder` | clip |
| `facebook/wav2vec2-base` | `Wav2Vec2WaveformTransform` | `Wav2Vec2Encoder` | clip, frame |
| `atst-clip-small` | `AtstMelspecTransform` | `AtstClipEncoder` | clip |
| `atst-clip-base` | `AtstMelspecTransform` | `AtstClipEncoder` | clip |
| `atst-frame-small` | `AtstMelspecTransform` | `AtstFrameEncoder` | clip, frame |
| `atst-frame-base` | `AtstMelspecTransform` | `AtstFrameEncoder` | clip, frame |
| `panns-16k-cnn14-max_mean` | `PannsLogmelTransform` | `PannsCnn14Encoder` | clip, frame |
| `panns-32k-cnn14-max_mean` | `PannsLogmelTransform` | `PannsCnn14Encoder` | clip, frame |
| `panns-32k-cnn14-decision_level_max` | `PannsLogmelTransform` | `PannsCnn14Encoder` | clip, frame |
| `beats_iter1` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `fine_tuned_beats_iter1_cpt1` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `fine_tuned_beats_iter1_cpt2` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `beats_iter2` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `fine_tuned_beats_iter2_cpt1` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `fine_tuned_beats_iter2_cpt2` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `beats_iter3` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `fine_tuned_beats_iter3_cpt1` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `fine_tuned_beats_iter3_cpt2` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `beats_iter3_plus_as20k` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `fine_tuned_beats_iter3_plus_as20k_cpt1` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `fine_tuned_beats_iter3_plus_as20k_cpt2` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `beats_iter3_plus_as2m` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `fine_tuned_beats_iter3_plus_as2m_cpt1` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |
| `fine_tuned_beats_iter3_plus_as2m_cpt2` | `BeatsKaldiFbankTransform` | `BeatsEncoder` | clip, frame |

`16k + decision_level_max` has no official checkpoint, so no registration
name is set for it; when this randomly-initialized experimental
combination is needed, the component classes must be constructed directly.

## Public interface

Implemented at `src/timbral/models/registry.py`:

```python
class ModelPair(NamedTuple):
    """A paired Transform and Encoder."""

    transform: BaseTransform
    encoder: BaseEncoder


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """The pairing declaration for one registration name."""

    transform_cls: type[BaseTransform]
    encoder_cls: type[BaseEncoder]
    fixed_kwargs: Mapping[str, Any] = field(default_factory=dict)


def create_model(
    name: str,
    *,
    granularity: Granularity,
    pretrained: bool = True,
    pretrained_dir: str | Path | None = None,
    **kwargs: Any,
) -> ModelPair:
    """Construct a paired Transform and Encoder by registration name."""


def list_models() -> list[str]:
    """Return all registration names, sorted alphabetically."""


def register_model(name: str, spec: ModelSpec) -> None:
    """Register a new model pairing declaration (for new-model extension or test injection)."""
```

This follows the same registration mechanism as
`timbral.datasets.adapters`: a single central, explicit dict `MODELS`
within the module, keyed by registration name with `ModelSpec` values;
components are eagerly imported at module top level. The three PANNS
entries are generated at module top level by iterating over
`PANNS_CHECKPOINTS`, so they appear automatically whenever a new official
PANNS checkpoint is added; the 15 BEATs entries are likewise generated by
iterating over `BEATS_CHECKPOINTS`. `register_model` assigns directly, so a
repeated registration simply overwrites the previous entry.

### Parameter routing

`create_model` does not maintain a factory function for each registration
name. At construction time, the common parameters, `fixed_kwargs`, and the
caller's `**kwargs` are merged and routed according to the signatures of
the two component constructors:

- A component whose constructor signature declares a given parameter name
  receives that parameter; shared parameters declared by both constructors
  (such as PANNS's `target_sample_rate`, `variant`, `pretrained`,
  `pretrained_dir`) are passed with the same value to both;
- `granularity`, `pretrained`, and `pretrained_dir` are common parameters
  and are not passed to a component constructor that does not declare them
  (e.g., the parameter-free Transforms of AST/CLAP);
- A parameter in `fixed_kwargs` or `**kwargs` that neither constructor
  declares: `TypeError`;
- `**kwargs` sharing a name with `fixed_kwargs`: `TypeError`. The
  registration name is itself the pinned identity, so it may not be
  overridden at the call site;
- `fixed_kwargs` may not contain `granularity`, `pretrained`, or
  `pretrained_dir`, to prevent pinned configuration from overriding
  `create_model`'s explicit common parameters.

Routing relies on the existing contract of the component constructors: all
parameters are explicitly declared keyword-only, and none accept
`**kwargs`. When both components declare a parameter with the same name,
that name is treated as a shared parameter and passed with the same value
to both; component design must guarantee that identical names carry
identical meaning (existing components satisfy this). Callers only need to
know "which parameters this registration name accepts," not
component-level details; `create_model` does not provide a structural
split such as `transform_kwargs` / `encoder_kwargs`.

## Pinned configuration per registration name

The specs for `MIT/ast-finetuned-audioset-10-10-0.4593` and
`laion/clap-htsat-fused` have no pinned parameters: their Transform
constructors take no parameters, and the common parameters are routed to
reach only the Encoder.

The spec for `facebook/wav2vec2-base` pins
`fixed_kwargs={"do_normalize": True}`, matching the official
preprocessor's `do_normalize=true`, so the registered name always
reproduces the official frontend and the call site cannot override it
(same policy as the PANNS DSP parameters). The switch remains available
when constructing `Wav2Vec2WaveformTransform` directly; future wav2vec2
variants pin their own value at registration time.

For the three PANNS registration names, `fixed_kwargs` consists of
`target_sample_rate`, `variant`, and the official frontend DSP parameters.
The DSP parameters directly reference
`timbral.models.helpers.panns.PANNS_OFFICIAL_FRONTENDS` — the same table
used by `PannsLogmelTransform`'s `pretrained=True` validation; the
registry does not keep its own copy. The values are as follows (consistent
with the official frontend parameter table in
[`transforms/panns.md`](transforms/panns.md)):

| Parameter | `panns-16k-cnn14-max_mean` | `panns-32k-cnn14-max_mean` | `panns-32k-cnn14-decision_level_max` |
|---|---:|---:|---:|
| `target_sample_rate` | 16000 | 32000 | 32000 |
| `variant` | `"max_mean"` | `"max_mean"` | `"decision_level_max"` |
| `n_fft` | 512 | 1024 | 1024 |
| `win_length` | 512 | 1024 | 1024 |
| `hop_length` | 160 | 320 | 320 |
| `n_mels` | 64 | 64 | 64 |
| `f_min` | 50.0 | 50.0 | 50.0 |
| `f_max` | 8000.0 | 14000.0 | 14000.0 |

Among these, `target_sample_rate` and `variant` are declared by both
constructors and passed with the same value to both; the DSP parameters
are declared only by the Transform; `granularity` is declared only by the
Encoder.

For the four ATST registration names, `fixed_kwargs` has only one entry:
`arch` (`"small"` or `"base"`), taken from
`ATST_CHECKPOINTS[...].arch`. `arch` is declared only by the Encoder, and
`AtstMelspecTransform` has no constructor parameters (both families share
one frontend), so the common parameters are routed to reach only the
Encoder. The family itself is carried by the Encoder class rather than by
a pinned parameter: `AtstClipEncoder` for the two `atst-clip-*` names and
`AtstFrameEncoder` for the two `atst-frame-*` names.

`n_blocks` is deliberately **not** pinned. It selects how many trailing
Transformer blocks are concatenated and therefore changes the output
width, so it stays a call-site parameter with a default of 1; passing
`n_blocks=12` reproduces the official downstream configuration. Because
it is a model-specific parameter rather than a public one, it reaches
`create_model` through `**kwargs`, and the embedding-extraction CLI
surfaces it through `--model_kwargs` (see
[`../../../README.md`](../../../README.md)).

For the 15 BEATs registration names, `fixed_kwargs` has only one entry:
`checkpoint`, whose value is the registration name itself (i.e., the key
from `BEATS_CHECKPOINTS`). `checkpoint` is declared only by the Encoder;
`BeatsKaldiFbankTransform` has no constructor parameters, so the common
parameters are routed to reach only the Encoder.

## Error behavior

- An unregistered name: `KeyError`, with the message listing all
  registered names, in a style consistent with
  `timbral.datasets.adapters.load_annotation`;
- An invalid `granularity`, or one outside that Encoder's supported range
  (e.g., passing `"frame"` for `laion/clap-htsat-fused`): surfaced via
  `BaseEncoder.__init__`'s existing `ValueError`; the registry does not add
  redundant defensive checks;
- Overriding a pinned parameter, or a parameter not declared by either
  component constructor: an explicit `TypeError` from the registry;
- The validity of `pretrained` / `pretrained_dir`: the responsibility of
  each component constructor.

## Files and exports

```text
src/timbral/models/registry.py
```

`timbral.models.registry` exports `ModelPair`, `ModelSpec`, `create_model`,
`list_models`, and `register_model`.

`timbral/models/__init__.py` re-exports the same five symbols. The
top-level exports are limited to these five; component classes are still
imported from `timbral.models.transforms` and `timbral.models.encoders`.

## Testing requirements

`tests/models/test_registry.py`, entirely using `pretrained=False`, with no
network access:

- `list_models` returns twenty-one built-in registration names (6 single
  entries + 15 BEATs), sorted alphabetically;
- For each registration name, `create_model` returns a `ModelPair` whose
  `transform` / `encoder` are the concrete classes for that name, with
  `encoder.granularity` matching what was passed in;
- The components constructed for the three PANNS registration names hold
  their respective pinned sample rate, variant, and frontend DSP
  parameters;
- The Encoders constructed for the 15 BEATs registration names hold a
  `checkpoint` consistent with their registration name;
- The `create_model` return value supports both unpacking and access by
  name;
- Passing `granularity="frame"` for `laion/clap-htsat-fused` raises
  `ValueError`;
- An unregistered name raises `KeyError` with a message containing the
  registered names;
- Passing a parameter not declared by either constructor raises
  `TypeError`;
- Passing a parameter with the same name as a pinned parameter raises
  `TypeError`;
- `fixed_kwargs` containing any common parameter raises `ValueError`;
- A `ModelSpec` injected via `register_model` is routed according to its
  signature (shared parameters passed with the same value to both, common
  parameters ignored when no component declares them, `**kwargs`
  delivered to whichever component declares it), and a repeated
  registration overwrites the old entry;
- The five exported symbols can be imported from the `timbral.models` top
  level.

## Dependency boundary

The registry depends only on the standard library (`inspect`,
`dataclasses`) and the existing component modules of `timbral.models`; it
introduces no new third-party dependencies and does not modify any
component's construction signature or behavior.
