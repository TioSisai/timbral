# `Wav2Vec2Encoder` Design

This document freezes the design of
`timbral.models.encoders.Wav2Vec2Encoder`. The companion Transform is
[`../transforms/wav2vec2.md`](../transforms/wav2vec2.md); the official
alignment contract is
[`../extra/wav2vec2-alignment.md`](../extra/wav2vec2-alignment.md).

This document describes the behavior the current implementation must satisfy,
and follows the common contract in [`base.md`](base.md).

## Design goals

`Wav2Vec2Encoder` is responsible for:

- inheriting `BaseEncoder` while keeping Transform-Encoder decoupling;
- wrapping the fixed Hugging Face `Wav2Vec2Model` backbone;
- safely loading the 211 backbone tensors of the only supported checkpoint;
- providing a project-derived clip embedding;
- providing the official last-layer frame embedding;
- guaranteeing padding isolation via unique-length grouped forwards, with
  batch outputs equal to per-sample calls (bit-identical for
  single-sample groups, tolerance-level within multi-sample groups);
- generating geometry, valid_mask, and zero padding conforming to the base
  class;
- behaving as an ordinary trainable `nn.Module`.

This component does not accept an attention mask, does not hold a Transform,
does not resample or normalize waveforms, and does not expose intermediate
Transformer layers or the pretraining quantizer.

## Supported model

Sole model identity:

```text
facebook/wav2vec2-base
```

Fixed backbone architecture (self-supervised base, no ASR head):

| Parameter | Value |
|---|---:|
| hidden size | 768 |
| number of Transformer layers | 12 |
| attention heads | 12 |
| intermediate size | 3072 |
| hidden activation | GELU |
| feature extractor norm | group |
| conv dims | 512 × 7 |
| conv kernels | 10,3,3,3,3,2,2 |
| conv strides | 5,2,2,2,2,2,2 |
| conv bias | False |
| conv positional embedding | kernel 128, 16 groups |
| stable layer norm | False |
| LayerNorm epsilon | 1e-5 |
| hidden dropout | 0.1 |
| attention dropout | 0.1 |
| feature projection dropout | 0.1 |
| SpecAugment | enabled, mask_time_prob 0.05 |

No arbitrary `Wav2Vec2Config` or backbone injection parameters are exposed.
The wav2vec2-large family differs in `embedding_dim` (1024), which is a
ClassVar and therefore a class-level identity; large variants must be
introduced as a separate Encoder class, not as parameters of this one.

## Public constructor interface

```python
class Wav2Vec2Encoder(BaseEncoder):
    supported_granularities = frozenset(("clip", "frame"))
    embedding_dim = 768

    def __init__(
        self,
        *,
        granularity: Granularity,
        pretrained: bool = True,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        ...
```

All parameters are keyword-only, and the constructor first calls
`super().__init__(granularity)`.

### `pretrained=True`

- prepares the fixed-revision `config.json`, `preprocessor_config.json`, and
  `pytorch_model.bin`;
- files that already exist are still individually verified against SHA-256;
- verifies that the key architecture fields of `config.json` match the local
  contract;
- reads the checkpoint with `torch.load(weights_only=True)` — the official
  repository ships no safetensors, so the weights-only loader is the safe
  reading path (same as BEATs);
- precisely confirms that the excluded tensors are exactly the 7
  `Wav2Vec2ForPreTraining` head tensors (`quantizer.*`, `project_q.*`,
  `project_hid.*`);
- selects all `wav2vec2.*` tensors and strips the prefix;
- renames the two legacy pos_conv weight-norm tensors
  (`weight_g`/`weight_v`) to the parametrize convention used by the fixed
  `Wav2Vec2Model`;
- confirms the resulting state holds exactly 211 tensors, including
  `masked_spec_embed`;
- performs `strict=True` loading on `Wav2Vec2Model`;
- does not automatically call `eval()`, freeze, or move the device.

Must not rely on `Wav2Vec2Model.from_pretrained()`'s lenient handling of the
pretraining-head keys, nor on the torch weight-norm state-dict hook silently
mapping the legacy names.

### `pretrained=False`

- does not resolve a cache directory;
- does not read config files;
- does not access the network;
- constructs `Wav2Vec2Config + Wav2Vec2Model` using architecture fields
  fixed in code;
- uses Transformers' official initialization;
- produces a random backbone structurally identical to the pretrained model.

## Inputs

Canonical call:

```python
encoder(
    input_features,
    valid_seconds=valid_seconds,
    valid_samples=valid_samples,
)
```

### `input_features`

- shape is `[B,N16k]`: the 16 kHz waveform canvas produced by the companion
  Transform, normalized when its `do_normalize=True`;
- automatically moved to the Encoder's `device` by `BaseEncoder`;
- the official contract is `float32`.

The concrete implementation reads the batch dimension for cardinality checks
but adds no full `input_features` shape check; incorrect feature shapes surface
through the underlying `Wav2Vec2Model` operators.

### `valid_seconds`

- shape is `[B]`, dtype `float32`;
- moved to device by BaseEncoder but not dtype-converted;
- used only for geometry.

### `valid_samples`

- shape is `[B]`, integer dtype;
- the exact valid 16 kHz sample count per sample, produced by the companion
  Transform;
- declared explicitly as a keyword-only hook parameter, per the base-class
  rule that model-specific inputs are not swallowed by a catch-all;
- used for grouping and for the valid frame count.

When calling the Encoder directly, bypassing the Transform, the caller must
respect the `valid_samples >= 400` contract and keep `valid_samples`
consistent with the physical width of `input_features`. Fewer samples than
one conv receptive field produce zero frames and surface as an operator
failure; an overlong `valid_samples` is silently truncated by the prefix
slice and yields unspecified outputs. The Encoder checks only that
`valid_seconds` and `valid_samples` each have one entry per batch item; these
metadata-only checks do not synchronize the device. Per the base-class rule,
it adds no runtime value or physical-width pairing checks (same behavior as
BEATs' `valid_feature_frames`).

## Grouped backbone forward

The conv frontend, its GroupNorm (`feat_extract_norm="group"` normalizes
statistics over the full time axis), and the 128-wide conv positional
embedding all leak padding into valid positions, and the base checkpoint was
trained without an attention mask (`return_attention_mask=False`). A padded
whole-batch forward therefore cannot satisfy the project's isolation
contract. Instead:

1. group the batch by unique `valid_samples`;
2. per group, slice the exact valid prefix `[:, :valid_samples]`;
3. run `Wav2Vec2Model(input_values=...)` on the group with no attention
   mask, taking `last_hidden_state`;
4. scatter group results back onto a zero canvas by batch index
   (`helpers/grouping.py`).

Padding therefore never influences any output, and invalid positions are
exactly 0 without a final mask multiplication. A group holding a single
sample reproduces the per-sample call bit-identically; a group holding
several same-length samples matches per-sample calls only up to
floating-point kernel batching inside the Transformer layers (observed
around 1e-5 max absolute difference), because matmul accumulation order
depends on the batch dimension. The backbone call is encapsulated in the
single private method `_backbone_hidden`, the seam used by the alignment
tests.

## Frame grid

The conv stack downsamples by a factor of 320 with a 400-sample receptive
field. The number of output frames follows the official per-layer formula
(`Wav2Vec2Model._get_feat_extract_output_lengths`):

```text
length_{l+1} = floor((length_l - kernel_l) / stride_l) + 1
```

implemented once in `helpers/wav2vec2.py::wav2vec2_feature_frames` and shared
by tests. It is monotonically non-decreasing in `valid_samples`, satisfying
the embeddings builder's frame-count probing contract. The Transform's
400-sample minimum guarantees at least one valid frame.

## Clip output

Clip is a project-derived API: wav2vec2 is a pure self-supervised model with
no official clip-level embedding. The project defines it as the arithmetic
mean of the valid last-layer frames (same convention as BEATs' token mean):

```text
embedding = mean(last_hidden_state[:, :valid_frames, :], time axis)
```

Because the backbone runs per group on exact prefixes, every frame in a group
forward is valid and the mean needs no mask. No L2 normalization or
additional pooling is added.

Output:

```python
{
    "embedding": Tensor[B,768],
    "geometry": Tensor[B,2],       # [0, valid_seconds]
    "valid_mask": Tensor[B],       # all True
}
```

## Frame output

The frame embedding is the official `last_hidden_state` of the fixed
`Wav2Vec2Model`, i.e. the final Transformer layer output. It must not be
substituted with an intermediate layer, the conv `extract_features`, or
additional pooling. (SUPERB-style layer selection is a known research
extension; introducing it requires a design revision, not an ad-hoc
parameter.)

### Number of valid frames

```text
valid_frames = wav2vec2_feature_frames(valid_samples)
```

For the batch canvas, `T = max(valid_frames)`.

### Geometry

Nominal ownership step:

```text
frame_step = 320 / 16000 = 0.02 s
```

Geometry is built by the shared `helpers/geometry.py::build_frame_geometry`
on the fixed 0.02 s grid: slot `t` owns `[t × 0.02, (t+1) × 0.02)`, the end
of the last valid slot is absorbed into `valid_seconds`, and invalid slots
are zeroed. Because the conv formula yields
`valid_frames ≈ floor((n - 400) / 320) + 1`, the nominal grid ends before
`valid_seconds` and the last slot always stretches to cover the remainder
— between about 25 ms and just under 45 ms (hop plus receptive field), e.g.
1.0 s → 49 frames, last slot `[0.96, 1.0]` spanning 40 ms. Geometry is a non-overlapping ownership partition covering
`[0, valid_seconds]`, not the conv receptive field.

### Batch padding

Returns:

```python
{
    "embedding": Tensor[B,T,768],
    "geometry": Tensor[B,T,2],
    "valid_mask": Tensor[B,T],
}
```

Valid positions retain the model output and geometry. Invalid positions have
`embedding=0`, `geometry=0`, `valid_mask=False`, with exact zero values
guaranteed by the grouped zero-canvas assembly.

## SpecAugment and training mode

The fixed config keeps the official `apply_spec_augment=True` with
`mask_time_prob=0.05`. Under `eval()` the forward is deterministic; under
`train()` the official backbone applies random time masking using
`masked_spec_embed` and consumes the global RNG, matching an upstream
per-sample call. The Encoder does not alter this lifecycle; determinism
contracts (alignment, batch-composition independence) are stated for
evaluation mode.

Training mode inherits an upstream input floor: SpecAugment's default
`mask_time_length=10` requires every sample's frame count to reach at
least 10 frames, so a train-mode forward with `valid_samples` in
`[400, 3279]` (0.025 s to about 0.205 s, fewer than 10 frames) raises the
upstream `ValueError` from `_compute_mask_indices` — the same error an upstream
per-sample call on the exact-length input produces. A padded whole-batch
upstream forward would mask on the padded length instead and not raise;
this project's grouped exact-prefix forwards deliberately keep the
per-sample semantics, so one short sample fails the whole train-mode
batch. Evaluation mode accepts every length the Transform admits.

## Weights and shared helpers

The shared logic is located at:

```text
src/timbral/models/helpers/wav2vec2.py
```

It is responsible only for:

- fixed checkpoint identity (repo id, revision, filenames, SHA-256);
- fixed-revision download via `helpers.common.ensure_hf_snapshot`;
- verification of existing files;
- config reading and architecture-field validation;
- weights-only checkpoint reading and backbone state extraction;
- the conv frame-count formula and the hop/receptive-field constants shared
  with the Transform.

Transform does not import Encoder; Encoder does not import Transform. The two
only import the shared helper.

## Device, dtype, training, and serialization

- `device` is derived from the feature projection weights;
- BaseEncoder moves the common inputs;
- the formal official alignment contract is float32;
- embedding retains the backbone's actual output dtype;
- geometry is fixed as float32; valid_mask is fixed as bool;
- does not automatically call `eval()`, freeze, or wrap in no-grad;
- does not enable gradient checkpointing: the legacy
  `gradient_checkpointing: true` in the official `config.json` is a
  training-runtime toggle (which `from_pretrained` still honors in
  Transformers 5.14.1); the fixed config omits it, and callers wanting the
  official training memory behavior call
  `backbone.gradient_checkpointing_enable()` — see the alignment
  contract's Intentional Differences;
- the Encoder can be trained normally and propagates gradients through both
  parameters and inputs;
- callers are allowed to use PyTorch autocast; a bare `.half()` without
  autocast is not guaranteed;
- uses the plain PyTorch `state_dict`, saving the complete `Wav2Vec2Model`
  backbone.

## Files and exports

Implementation location:

```text
src/timbral/models/encoders/wav2vec2.py
```

`timbral.models.encoders` re-exports `Wav2Vec2Encoder`; `timbral.models` at
the top level only re-exports registry symbols (see
[`../registry.md`](../registry.md)).

## Testing requirements

Ordinary offline tests must cover at least:

- public exports and the keyword-only constructor;
- `pretrained=False` reads no files and accesses no network;
- fixed `Wav2Vec2Config` architecture fields;
- clip `[B,768]` output equal to the masked mean of the frame output;
- frame `[B,T,768]` output and the conv frame-count formula, including the
  400/719/720-sample boundaries;
- geometry on the 0.02 s grid with adjacent valid boundaries element-wise
  equal and the last valid end equal to `valid_seconds`;
- batch cardinality validation for `valid_seconds` and `valid_samples`;
- zero padding for mixed-length batches, equal to per-sample calls
  (bit-identical for single-sample groups);
- the train-mode SpecAugment input floor: fewer than 10 frames raises the
  upstream `ValueError`, and `eval()` accepts the same input;
- unknown kwargs raise `TypeError`;
- BaseEncoder device transfer;
- gradient propagation through parameters and inputs;
- the training/eval lifecycle is not altered by the constructor;
- strict loading after checkpoint state filtering and pos_conv renaming.

For the full real-weight verification, see
[`../extra/wav2vec2-alignment.md`](../extra/wav2vec2-alignment.md).
