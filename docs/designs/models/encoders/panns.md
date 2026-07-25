# `PannsCnn14Encoder` Design

This document freezes the design of `timbral.models.encoders.PannsCnn14Encoder`. The companion Transform
is [`../transforms/panns.md`](../transforms/panns.md); the official alignment contract is
[`../extra/panns-alignment.md`](../extra/panns-alignment.md).

This document describes the behavior the current implementation must satisfy, and follows the common
contract in [`base.md`](base.md).

## Design goals

`PannsCnn14Encoder` is responsible for:

- implementing the official Cnn14's six convolutional blocks and the `fc1` embedding layer;
- retaining the official training-mode dropout;
- supporting the two variants max_mean and DecisionLevelMax;
- making all three checkpoints support both clip and frame;
- grouping mixed-length batches by `valid_feature_frames`;
- keeping the official floor pooling for normal-length inputs;
- padding out to one embedding only for extremely short inputs of fewer than 32 features;
- producing geometry, valid_mask, and zero padding conforming to `BaseEncoder`.

This component does not hold a Transform, does not accept waveforms, and is not responsible for
downmixing, resampling, STFT, or log-mel.

## Supported variants

Shared type definitions in `timbral.models.helpers.panns`:

```python
PannsVariant: TypeAlias = Literal["max_mean", "decision_level_max"]
PannsTargetSampleRate: TypeAlias = Literal[16000, 32000]
```

`timbral.models.transforms` and `timbral.models.encoders` both re-export the same `PannsVariant` object.

Supported checkpoint identities:

| `target_sample_rate` | `variant` | checkpoint |
|---:|---|---|
| 16,000 | `max_mean` | `Cnn14_16k_mAP=0.438.pth` |
| 32,000 | `max_mean` | `Cnn14_mAP=0.431.pth` |
| 32,000 | `decision_level_max` | `Cnn14_DecisionLevelMax_mAP=0.385.pth` |

The three checkpoints' backbone keys and shapes are structurally identical, but weight values differ. A
loaded backbone must not be reused across variants.

## Public constructor interface

Expected interface:

```python
class PannsCnn14Encoder(BaseEncoder):
    supported_granularities = frozenset(("clip", "frame"))
    embedding_dim = 2048

    def __init__(
        self,
        *,
        granularity: Literal["clip", "frame"],
        target_sample_rate: PannsTargetSampleRate,
        variant: PannsVariant,
        pretrained: bool = True,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        ...
```

The constructor must call:

```python
super().__init__(granularity)
```

All parameters are keyword-only. `supported_granularities` explicitly declares at class-definition time
that PANNs supports both clip and frame. `embedding_dim` declares as a ClassVar that the last dimension of
the output embedding is 2048 (i.e. the channel count of Cnn14 Block 6), so that callers can build the
output schema before the forward pass.

`target_sample_rate` does not indicate that the Encoder resamples; it identifies the companion PANNs
front end and the official checkpoint identity. Incorrect Transform-Encoder pairing remains the caller's
responsibility.

### `pretrained=True`

- allows only the three official checkpoint combinations;
- downloads if the file does not exist; existing files are still verified against SHA-256; within the
  same process, for the same `(path, checksum)` a full hash is computed only once, and the same
  checkpoint is deserialized only once (process-level memoization, sharing the same loaded result with
  the Transform);
- loads the six convolutional blocks and `fc1`;
- explicitly discards STFT, mel, `bn0`, and `fc_audioset`;
- all three checkpoints use `torch.load(weights_only=True, map_location="cpu")`; none may fall back to
  `weights_only=False`;
- only the 16 kHz checkpoint enables a minimal NumPy `safe_globals` allowlist;
- performs strict loading on the filtered Encoder state;
- does not automatically call `eval()`, freeze, or move devices.

### `pretrained=False`

- does not resolve a cache path, does not access the network;
- the convolutional blocks and `fc1` use the official initialization rules;
- variant still determines the clip/frame computation semantics;
- `16,000 + decision_level_max` can serve as a randomly initialized experimental combination with no
  official checkpoint.

## Shared weight logic

The following shared content is concentrated in `src/timbral/models/helpers/panns.py`:

- `PannsVariant` and the set of valid variant/target-sample-rate combinations;
- the three checkpoints' model names, URLs, filenames, and SHA-256;
- HF cache default path resolution;
- Zenodo download;
- checksum verification;
- the safe-reading allowlist for the 16 kHz checkpoint.

Transform and Encoder both import directly from this single source, without importing from each other.
The Encoder file retains only the Cnn14 layer definitions, initialization rules, weight filtering, and
granularity semantics.

Path rules, checksums, and failure semantics are identical to those in
[`../transforms/panns.md`](../transforms/panns.md).

## Input contract

Public call:

```python
encoder(
    input_features,
    valid_feature_frames=valid_feature_frames,
    valid_seconds=valid_seconds,
)
```

### `input_features`

- shape `[B, T, n_mels]`;
- the post-`bn0` log-mel produced by the companion Transform;
- automatically moved to the Encoder's `device` by BaseEncoder;
- embedding retains the actual model computation dtype.

The official pretrained combinations use `n_mels=64`. Randomly initialized experimental combinations
allow `n_mels>=32`, so that the frequency dimension can pass through five rounds of 2× floor pooling.

### `valid_feature_frames`

- shape `[B]`;
- dtype `torch.int64`;
- represents the actual official center-STFT frame count before padding;
- generated by the Transform as `floor(valid_samples/hop)+1`;
- a top-level Tensor, automatically moved by BaseEncoder.

The concrete encoding hook explicitly accepts `valid_feature_frames`, without a catch-all `**kwargs` that
swallows unknown parameters. Unknown model inputs must naturally raise `TypeError`.

Logical length and physical length must satisfy:

```text
encoder_physical_frames =
    32                           if valid_feature_frames < 32
    valid_feature_frames         otherwise
```

The 32-frame physical features for extremely short inputs must already have been generated by the
Transform with padding applied before `bn0`. The Encoder cannot and must not fabricate this padding
itself in post-`bn0` space. When calling the Encoder directly, bypassing the Transform, the caller must
also supply the above physical layout.

### `valid_seconds`

- shape `[B]`;
- dtype `float32`;
- keyword-only;
- used directly for geometry;
- BaseEncoder only transfers the device, without converting dtype.

## Unique-length grouping

Samples with different `valid_feature_frames` must not be passed through the backbone directly within the
same physical time canvas. Concrete process:

1. compute unique values of `valid_feature_frames`;
2. select the corresponding batch indices for each length;
3. crop normal lengths to that length;
4. for extremely short lengths, use the physical features already padded to 32 by the Transform;
5. run the convolutional backbone and granularity branch separately for each length group;
6. restore batch order by original index;
7. pad the output to the batch's maximum embedding frame count.

This way, every sample in a mixed-length batch has the same front-end boundary and backbone canvas as it
would when called individually.

In training mode, each unique-length group updates the convolutional BatchNorm statistics separately.
Since standard BatchNorm itself depends on batch composition, training mode does not guarantee per-sample
batch independence; official numerical alignment and embedding production are performed under `eval()`.

## Cnn14 backbone

Six `_ConvBlock`s:

```text
Block 1:   1 ->   64, avg pool (2, 2)
Block 2:  64 ->  128, avg pool (2, 2)
Block 3: 128 ->  256, avg pool (2, 2)
Block 4: 256 ->  512, avg pool (2, 2)
Block 5: 512 -> 1024, avg pool (2, 2)
Block 6:1024 -> 2048, no spatial downsampling
```

Each block:

```text
conv3x3 -> BatchNorm -> ReLU
conv3x3 -> BatchNorm -> ReLU
avg pool
dropout(p=0.2)
```

The first five temporal poolings together produce a fixed downsampling ratio:

```text
2⁵ = 32
```

This value is determined by the architecture and is not exposed as a public `interpolate_ratio`
parameter. The implementation may either derive it from the structure or use a private constant, but
tests must lock it at 32.

After the six blocks, the mean is taken over the frequency dimension:

```text
[B, 2048, T_feature] -> [B, 2048, T_embedding]
```

Output count for normal lengths:

```text
T_embedding = valid_feature_frames // 32
```

The backbone still receives the entire `valid_feature_frames`. The remainder short of a complete
32-feature block does not produce an additional embedding frame, but can still affect existing outputs
through the convolution's receptive field; cropping to a multiple of 32 before entering the backbone is
prohibited.

Extremely short lengths:

```text
valid_feature_frames < 32 -> padded to 32 by Transform -> T_embedding = 1
```

Unified formula:

```text
num_valid_embeddings = max(1, valid_feature_frames // 32)
```

## Official dropout

Retains the following training-mode paths:

- `dropout(p=0.2)` after each convolutional block;
- `dropout(p=0.5)` before `fc1`;
- `dropout(p=0.5)` after `fc1 + ReLU`.

The exported embedding is the Tensor after the last dropout, consistent with the official Cnn14
`embedding` export point. Under `eval()`, all dropouts become identity.

Not ported:

- SpecAugmentation;
- waveform or spectral-domain mixup;
- the `fc_audioset` classification head.

The module remains an ordinary trainable `nn.Module`, and is not automatically frozen due to the absence
of these training augmentations.

## max_mean variant

### Clip

Over the time dimension of the backbone output:

```text
pooled = amax(x, time) + mean(x, time)
pooled = dropout(p=0.5)
embedding = fc1 + ReLU
embedding = dropout(p=0.5)
```

Output:

```text
embedding  [B, 2048]
geometry   [B, 2]
valid_mask [B]
```

geometry:

```text
[0, valid_seconds]
```

valid_mask is all `True`.

This is consistent with the official Cnn14 embedding branch for the two max_mean checkpoints.

### Frame

No max+mean temporal pooling is applied. After the frequency mean, each time position shares the
following execution:

```text
x = dropout(x, p=0.5)
x = transpose_to_[B,T,2048]
embedding = fc1(x) + ReLU
embedding = dropout(embedding, p=0.5)
```

Output `[B,T,2048]`. This is a derived frame embedding based on the official Cnn14 backbone and `fc1`,
not an official public output.

## DecisionLevelMax variant

### Frame

Over the time dimension:

```text
x = max_pool1d(x, kernel=3, stride=1, padding=1)
  + avg_pool1d(x, kernel=3, stride=1, padding=1)
x = dropout(x, p=0.5)
x = transpose_to_[B,T,2048]
embedding = fc1(x) + ReLU
embedding = dropout(embedding, p=0.5)
```

The output is the official `Cnn14_DecisionLevelMax`:

- before interpolation;
- before `fc_audioset`;
- the 2048-dimensional segment hidden state.

It is not the official public 527-class interpolated framewise probability.

### Clip

The above frame embedding is computed in full first, and then, only over the valid time positions:

```text
clip_embedding = amax(frame_embedding, time)
```

No additional average pooling or new `fc1` is added. This is the derived clip semantics this project
defines for DecisionLevelMax.

## Frame geometry

Both variants use the same ownership grid:

```text
frame_step = 32 × hop_length / target_sample_rate = 0.32 seconds
```

The current Encoder constructor interface does not accept `hop_length`, so a correctly paired Transform
must use a 10 ms feature hop. `pretrained=False` can change other front-end parameters, but any
non-10-ms hop does not constitute a correct pairing for the current Encoder.

For each sample:

```text
num_valid = max(1, valid_feature_frames // 32)
boundaries = arange(T_max + 1) × 0.32
start[i] = boundaries[i]
end[i] = min(boundaries[i + 1], valid_seconds)
end[num_valid - 1] = valid_seconds
```

start and end must both be sliced from the same float32 `boundaries` Tensor, so that any adjacent valid
slots strictly satisfy `end[i] == start[i + 1]`; the same boundary must not be computed twice separately
via multiplication and addition.

Since normal lengths use floor for the output count, the last ownership interval absorbs any trailing
duration that does not correspond to an additional embedding frame. For example, 1 second yields:

```text
[0.00, 0.32]
[0.32, 0.64]
[0.64, 1.00]
```

An extremely short 0.02-second input yields:

```text
[0.00, 0.02]
```

0.32 seconds is the output step, not the neural network's actual receptive field:

- the max_mean frame's convolutional backbone receptive field is approximately 284 log-mel frames;
- DecisionLevelMax's final `kernel=3` smoothing expands the receptive field to approximately 348 log-mel
  frames;
- geometry only represents the time ownership for downstream labels or aggregation.

## Frame batch padding

After merging the mixed batch:

```text
embedding  [B, T_max, 2048]
geometry   [B, T_max, 2]
valid_mask [B, T_max]
```

Valid positions:

- `valid_mask=True`;
- embedding retains the model output;
- geometry generated according to the ownership rule above.

Invalid positions:

- `valid_mask=False`;
- embedding row filled with 0;
- geometry row filled with 0.

Invalid frame embeddings and geometry use exact zero values.

## Clip batch semantics

Each sample's clip pooling occurs only over the officially valid embedding sequence produced by its own
unique-length group. Samples of different lengths must not first be placed into a shared canvas and
masked only at final pooling; the CNN's finite receptive field and boundary effects would make such an
implementation batch-composition dependent.

## Device, training, and serialization

- `device` is derived from the backbone parameters;
- BaseEncoder automatically transfers `input_features`, `valid_seconds`, and `valid_feature_frames`;
- embedding retains the model's actual dtype;
- geometry is fixed to float32;
- valid_mask is fixed to bool;
- does not automatically call `eval()`;
- does not automatically freeze;
- does not wrap in `torch.no_grad()`;
- uses the plain `state_dict`;
- does not retain `embedding_dim` as public base-class state;
- does not set a maximum input duration.

## Files and exports

Implementation location:

```text
src/timbral/models/encoders/panns.py
```

Division of responsibilities:

- `src/timbral/models/helpers/panns.py`: shared types, checkpoint identity, download, verification, and
  safe checkpoint reading;
- `src/timbral/models/helpers/geometry.py`: model-agnostic construction of clip/frame geometry and
  valid_mask;
- `src/timbral/models/helpers/grouping.py`: model-agnostic scaffolding for unique-length grouping
  iteration and refilling grouped results back into a zero canvas;
- `src/timbral/models/encoders/panns.py`: initialization rules, `_ConvBlock`, `PannsCnn14Encoder`, and
  Encoder weight filtering.

`timbral.models.helpers`'s `__init__` does not re-export anything; registry, Encoder, and Transform
import directly from `timbral.models.helpers.panns`. `timbral.models.encoders` re-exports the same
`PannsVariant` and `PannsCnn14Encoder`. `timbral.models` at the top level only re-exports registry
symbols (see [`../registry.md`](../registry.md)).

## Testing requirements

Weight-free tests must cover at least:

- the three valid checkpoint identities and invalid combinations;
- `pretrained=False` accesses no network;
- 32 as the immutable architectural downsampling ratio;
- shapes of the six convolutional blocks;
- normal lengths use floor pooling;
- an extremely short input with metadata of 1-31 features, whose physical input has been padded to 32,
  produces one embedding;
- max_mean clip's official pooling;
- max_mean frame performs no temporal pooling;
- DecisionLevelMax frame's smoothing and export point;
- DecisionLevelMax clip's `amax` over valid frames;
- the official dropout locations for both granularities;
- unique-length batches match individual calls;
- clip geometry and an all-True mask;
- frame geometry, zero padding, and bool mask;
- adjacent valid frame geometry boundaries are element-wise equal;
- unknown model inputs raise `TypeError`;
- the helper's checkpoint identities fully cover the three official combinations;
- the `PannsVariant` re-exported by Transform/Encoder is the same object as in the helper;
- after `.to(device)`, input and output devices are correct.

For real checkpoints and independent official-source-code tests, see
[`../extra/panns-alignment.md`](../extra/panns-alignment.md).
