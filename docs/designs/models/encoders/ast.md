# `AstEncoder` Design

This document freezes the design of `timbral.models.encoders.AstEncoder`. The companion Transform is
[`../transforms/ast.md`](../transforms/ast.md); the official alignment contract is
[`../extra/ast-alignment.md`](../extra/ast-alignment.md).

This document describes the behavior the current implementation must satisfy, and follows the common contract in
[`base.md`](base.md).

## Design goals

`AstEncoder` is responsible for:

- inheriting `BaseEncoder` while keeping Transform-Encoder decoupling;
- wrapping the fixed Hugging Face `ASTModel` backbone;
- safely loading the 199 backbone tensors of the only supported checkpoint;
- providing the official clip embedding;
- providing a project-derived frame embedding;
- generating geometry, valid_mask, and zero padding conforming to the new base class;
- behaving as an ordinary trainable `nn.Module`.

This component does not accept waveforms, does not hold a Transform, does not compute fbank, does not return
AudioSet classification logits, and does not implement an attention mask.

## Supported model

Sole model identity:

```text
MIT/ast-finetuned-audioset-10-10-0.4593
```

Fixed backbone architecture:

| Parameter | Value |
|---|---:|
| hidden size | 768 |
| number of Transformer layers | 12 |
| attention heads | 12 |
| intermediate size | 3072 |
| hidden activation | GELU |
| hidden dropout | 0 |
| attention dropout | 0 |
| initializer range | 0.02 |
| LayerNorm epsilon | 1e-12 |
| patch size | 16 |
| frequency stride | 10 |
| time stride | 10 |
| input mel bins | 128 |
| input fbank frames | 1024 |
| qkv bias | True |

No arbitrary `ASTConfig` or backbone injection parameters are exposed. Other AST configurations must be
introduced through a separate design.

## Public constructor interface

```python
class AstEncoder(BaseEncoder):
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

All parameters are keyword-only, and the constructor first calls:

```python
super().__init__(granularity)
```

`supported_granularities` explicitly declares at class-definition time that AST supports both clip and
frame. `embedding_dim` declares as a ClassVar that the last dimension of the output embedding is 768 (i.e. the
fixed config's hidden size), so that callers can build the output schema before the forward pass.

### `pretrained=True`

- prepares the fixed-revision `config.json`, `preprocessor_config.json`, and
  `model.safetensors`;
- files that already exist are still individually verified against SHA-256;
- verifies that the key architecture fields of `config.json` match the local contract;
- performs a safe read of the safetensors;
- selects all `audio_spectrogram_transformer.*` tensors and strips the prefix;
- converts `encoder.layer.*`, legacy attention, and MLP key names to the current `ASTModel` key names
  following the official ViT-style checkpoint conversion rules of Transformers 5.13.1;
- precisely confirms that only 4 `classifier.*` tensors are excluded;
- performs `strict=True` loading on `ASTModel`;
- does not automatically call `eval()`, freeze, or move the device.

Must not rely on `ASTModel.from_pretrained()`'s lenient handling of unexpected classifier keys.

### `pretrained=False`

- does not resolve a cache directory;
- does not read config files;
- does not access the network;
- constructs `ASTConfig + ASTModel` using architecture fields fixed in code;
- uses Transformers' official initialization;
- produces a random backbone structurally identical to the pretrained model.

## Inputs

Canonical call:

```python
encoder(
    input_features,
    valid_seconds=valid_seconds,
)
```

### `input_features`

- shape is fixed at `[B,1024,128]`;
- is the normalized fbank produced by the companion Transform;
- automatically moved to the Encoder's `device` by `BaseEncoder`;
- the first official contract is `float32`.

The concrete implementation adds no redundant per-forward-pass shape-pairing checks; incorrect shapes are
allowed to surface through the underlying `ASTModel` operators.

### `valid_seconds`

- shape is `[B]`;
- dtype is `float32`;
- moved to device by BaseEncoder but not dtype-converted;
- the Encoder internally computes the target number of valid samples as `round(valid_seconds × 16000)`;
- used for the number of valid frames and geometry.

The Transform is already responsible for max-duration validation. When calling the Encoder directly,
bypassing the Transform, the caller must respect the `0 < valid_seconds <= 10.255` contract.

## Backbone and token grid

The fixed input `[B,1024,128]` is converted to `[B,1,128,1024]` inside Hugging Face AST, and then passed
through:

```text
Conv2d(kernel_size=(16,16), stride=(10,10), padding=0)
```

Patch grid:

```text
frequency_out = floor((128 - 16) / 10) + 1 = 12
time_out      = floor((1024 - 16) / 10) + 1 = 101
patch_tokens  = 12 × 101 = 1212
```

Plus the CLS and distillation tokens, the total sequence length is 1214.

Token order is frequency-major, time-minor, i.e.:

```text
index = frequency_index × 101 + time_index
```

No attention mask is passed to `ASTModel`. All spectral-domain padding patches participate in the 12-layer
global self-attention; the output `valid_mask` only expresses project ownership, not whether a token
participated in computation.

## Clip output

The clip embedding directly uses:

```python
backbone(input_values=input_features).pooler_output
```

which is the mean of the CLS and distillation tokens after the final LayerNorm, with shape `[B,768]`.
No L2 normalization, additional pooling, or classification head is added.

Output:

```python
{
    "embedding": Tensor[B,768],
    "geometry": Tensor[B,2],
    "valid_mask": Tensor[B],
}
```

where:

```text
geometry = [0, valid_seconds]
valid_mask = True
```

## Frame output

Frame is a project-derived API; it is not an official Hugging Face or MIT public output.

Computation:

```text
last_hidden_state[:, 2:, :]
→ reshape [B,12,101,768]
→ mean(frequency axis)
→ [B,101,768]
```

`last_hidden_state` has already passed through the official final LayerNorm. It must not be substituted
with an intermediate layer, the pre-LayerNorm state, the CLS token, or additional pooling.

### Number of valid frames

```text
target_valid_samples = round(valid_seconds × 16000)
num_valid = min(max(ceil(target_valid_samples / 1600), 1), 101)
```

Any positive duration produces at least one valid ownership slot. At non-native sample rates, an
extremely short positive duration may round to 0 target samples; in that case the first
`[0, valid_seconds]` slot is still retained. For project inputs with fewer than 400 target samples,
the first slot is still valid even though its fbank input is entirely spectral-domain padding.

### Geometry

Nominal ownership step:

```text
frame_step = time_stride × fbank_hop / sample_rate
           = 10 × 160 / 16000
           = 0.1s
```

For each valid slot:

```text
boundaries = arange(102) × 0.1
start[i] = boundaries[i]
end[i] = min(boundaries[i + 1], valid_seconds)
end[num_valid - 1] = valid_seconds
```

start and end must both be sliced from the same float32 `boundaries` Tensor, so that any adjacent valid
slots strictly satisfy `end[i] == start[i + 1]`; the same boundary must not be computed twice separately
via multiplication and addition.

Therefore:

- 0.025 seconds: `[0,0.025]`;
- 10 seconds: last slot `[9.9,10.0]`;
- 10.255 seconds: last slot `[10.0,10.255]`.

Geometry is a non-overlapping ownership partition covering `[0,valid_seconds]`, not a local patch
support interval or Transformer receptive field.

### Unused input tail

The last of the 101 time patches covers fbank rows `1000..1015`, so rows
`1016..1023` never enter the model at all. Consequently:

- the last 8 fbank rows do not affect any embedding;
- roughly the last 80 ms within the 10.175-10.255 second range is not consumed by the patch backbone;
- the 10.255-second project input contract and the complete ownership geometry are still preserved;
- documentation and tests must record this accurately, and must not describe geometry as the actual
  support region.

### Batch padding

Returns:

```python
{
    "embedding": Tensor[B,101,768],
    "geometry": Tensor[B,101,2],
    "valid_mask": Tensor[B,101],
}
```

Valid positions retain the model output and geometry. Invalid positions:

- `embedding=0`;
- `geometry=0`;
- `valid_mask=False`.

Invalid frame embeddings and geometry use exact zero values.

## Weights and shared helpers

The shared logic is located at:

```text
src/timbral/models/helpers/ast_helpers.py
```

It is responsible only for:

- fixed checkpoint identity;
- revision, filenames, and SHA-256;
- default cache directory;
- fixed-revision download (via `helpers.common.ensure_hf_snapshot`: downloads first to a
  temporary directory within the snapshot directory; once the checksum passes, atomically moves it into
  the final path, so a corrupt or interrupted package never ends up at the final path);
- verification of existing files;
- config reading and architecture-field validation;
- safe reading of safetensors and backbone state extraction.

The fixed checkpoint uses the legacy Transformers parameter naming, whereas 5.13.1's `ASTModel`
internally uses `layers.*`, `q_proj/k_proj/v_proj/o_proj`, and `mlp.fc1/fc2`. The helper explicitly
hardcodes the same ordered renaming as the official loader, and then performs a local `strict=True`
load; it cannot simply strip the backbone prefix and load directly.

Transform does not import Encoder; Encoder does not import Transform. The two only import the shared
helper when genuinely necessary.

## Device, dtype, training, and serialization

- `device` is derived from the patch projection weights;
- BaseEncoder moves the common inputs;
- the formal official alignment contract is float32;
- embedding retains the backbone's actual output dtype;
- geometry is fixed as float32;
- valid_mask is fixed as bool;
- does not automatically call `eval()`;
- does not freeze;
- does not wrap in no-grad;
- the Encoder can be trained normally and propagates gradients through both parameters and inputs;
- callers are allowed to use PyTorch autocast;
- does not guarantee correct forward behavior after a bare `.half()` or `.bfloat16()` call without
  autocast;
- uses the plain PyTorch `state_dict`, saving the complete ASTModel backbone.

## Files and exports

Implementation location:

```text
src/timbral/models/encoders/ast_encoder.py
src/timbral/models/helpers/geometry.py   # clip/frame geometry and valid_mask
```

`timbral.models.encoders` re-exports `AstEncoder`; `timbral.models` at the top level only re-exports
registry symbols (see [`../registry.md`](../registry.md)). A bare
`ast.py` must not be created.

## Testing requirements

Ordinary offline tests must cover at least:

- public exports and the keyword-only constructor;
- `pretrained=False` reads no files and accesses no network;
- fixed ASTConfig architecture fields;
- clip `[B,768]` output, geometry, and mask;
- frame `[B,101,768]` output points and reshape order;
- the ceil formula for the number of valid entries;
- a positive duration that rounds to 0 target samples still retains one valid slot;
- geometry at 0.025, 10, 10.245, and 10.255 seconds;
- adjacent valid geometry boundaries are element-wise equal;
- zero padding for mixed-length batches;
- the last 8 fbank rows do not affect the embedding;
- unknown kwargs raise `TypeError`;
- BaseEncoder device transfer;
- gradient propagation through parameters and inputs;
- the training/eval lifecycle is not altered by the constructor;
- strict loading after checkpoint state filtering.

For the full real-weight verification, see
[`../extra/ast-alignment.md`](../extra/ast-alignment.md).
