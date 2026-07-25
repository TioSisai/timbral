# `BeatsEncoder` Design

This document freezes the design of `timbral.models.encoders.BeatsEncoder`. The companion Transform is
[`../transforms/beats.md`](../transforms/beats.md); weight acquisition is
[`../extra/beats-download.md`](../extra/beats-download.md); the official alignment contract is
[`../extra/beats-alignment.md`](../extra/beats-alignment.md).

This document describes the behavior the current implementation must satisfy, and follows the common
contract in [`base.md`](base.md).

## Design goals

`BeatsEncoder` is responsible for:

- reproducing the official BEATs backbone (patch embedding + TransformerEncoder) via an inference-only
  local rewrite that is operator-for-operator numerically equivalent, with `state_dict` keys matching the
  official ones one-to-one (only the two pos_conv weight_norm keys use parametrize naming, mapped at load
  time);
- supporting all 15 official checkpoints (5 pretrained + 10 AudioSet fine-tuned);
- discarding the fine-tuned checkpoints' `predictor` classification head and `label_dict`, uniformly
  outputting 768-dimensional backbone features;
- grouping mixed-length batches by `valid_feature_frames`, keeping the output independent of batch
  composition;
- supporting both clip and frame granularities, producing geometry, valid_mask, and zero padding
  conforming to `BaseEncoder`.

This component does not hold a Transform, does not accept waveforms, does not compute fbank, and imposes
no duration cap. Self-attention memory grows quadratically with the number of tokens (roughly
50 tokens/second); resource constraints for long audio are the caller's responsibility.

## Supported checkpoints

The 15 entry names are converted from the cell text of the last three columns of the official README
table: lowercased; hyphens, spaces, and parentheses converted to underscores; `+` converted to `_plus`;
consecutive underscores collapsed and leading/trailing underscores stripped.

| entry | official table cell | type |
|---|---|---|
| `beats_iter1` | `BEATs_iter1` | pretrained |
| `fine_tuned_beats_iter1_cpt1` | `Fine-tuned BEATs_iter1 (cpt1)` | fine-tuned |
| `fine_tuned_beats_iter1_cpt2` | `Fine-tuned BEATs_iter1 (cpt2)` | fine-tuned |
| `beats_iter2` | `BEATs_iter2` | pretrained |
| `fine_tuned_beats_iter2_cpt1` | `Fine-tuned BEATs_iter2 (cpt1)` | fine-tuned |
| `fine_tuned_beats_iter2_cpt2` | `Fine-tuned BEATs_iter2 (cpt2)` | fine-tuned |
| `beats_iter3` | `BEATs_iter3` | pretrained |
| `fine_tuned_beats_iter3_cpt1` | `Fine-tuned BEATs_iter3 (cpt1)` | fine-tuned |
| `fine_tuned_beats_iter3_cpt2` | `Fine-tuned BEATs_iter3 (cpt2)` | fine-tuned |
| `beats_iter3_plus_as20k` | `BEATs_iter3+ (AS20K)` | pretrained |
| `fine_tuned_beats_iter3_plus_as20k_cpt1` | `Fine-tuned BEATs_iter3+ (AS20K) (cpt1)` | fine-tuned |
| `fine_tuned_beats_iter3_plus_as20k_cpt2` | `Fine-tuned BEATs_iter3+ (AS20K) (cpt2)` | fine-tuned |
| `beats_iter3_plus_as2m` | `BEATs_iter3+ (AS2M)` | pretrained |
| `fine_tuned_beats_iter3_plus_as2m_cpt1` | `Fine-tuned BEATs_iter3+ (AS2M) (cpt1)` | fine-tuned |
| `fine_tuned_beats_iter3_plus_as2m_cpt2` | `Fine-tuned BEATs_iter3+ (AS2M) (cpt2)` | fine-tuned |

Local filenames are uniformly `<entry>.pt`. Note that for the `iter3+ (AS20K)` row, the two fine-tuned
checkpoints' official filenames are `BEATs_iter3_plus_AS20K_finetuned_on_AS2M_cpt*.pt` — AS20K refers to
their pretrained base, and the fine-tuning data is likewise AS2M; the entry naming follows the table cell.
Download links and SHA-256 are in
[`../extra/beats-download.md`](../extra/beats-download.md).

### Architecture constants (verified consistent across all 15 checkpoints)

The following cfg fields have been checked value-by-value against all 15 checkpoints, and the
inference-relevant fields are entirely consistent:

| Field | Value |
|---|---:|
| `input_patch_size` | 16 |
| `embed_dim` | 512 |
| `conv_bias` | `False` |
| `encoder_layers` | 12 |
| `encoder_embed_dim` | 768 |
| `encoder_ffn_embed_dim` | 3072 |
| `encoder_attention_heads` | 12 |
| `activation_fn` | `"gelu"` |
| `layer_norm_first` | `False` |
| `deep_norm` | `True` |
| `conv_pos` | 128 |
| `conv_pos_groups` | 16 |
| `relative_position_embedding` | `True` |
| `num_buckets` | 320 |
| `max_distance` | 800 |
| `gru_rel_pos` | `True` |

`max_distance=800` differs from the official code's default value of 1280; the checkpoint value must take
precedence.

Differences are limited to training-time fields, split into two groups by checkpoint type:

| Field | Pretrained | Fine-tuned |
|---|---:|---:|
| `dropout` | 0.1 | 0.0 |
| `attention_dropout` | 0.1 | 0.0 |
| `dropout_input` | 0.1 | 0.0 |
| `activation_dropout` | 0.0 | 0.0 |
| `encoder_layerdrop` | 0.05 | 0.05 |
| `layer_wise_gradient_decay_ratio` | 1.0 | 0.6 |
| `finetuned_model` | field absent | `True` |
| `predictor_dropout` | field absent | 0.0 |
| `predictor_class` | field absent | 527 |

Thus a single hardcoded architecture covers all 15 entries; dropout p is taken from the table above
according to entry type (all become identity under eval).

## Public constructor interface

```python
class BeatsEncoder(BaseEncoder):
    supported_granularities = frozenset(("clip", "frame"))
    embedding_dim = 768

    def __init__(
        self,
        *,
        granularity: Literal["clip", "frame"],
        checkpoint: str,
        pretrained: bool = True,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        ...
```

The constructor must call `super().__init__(granularity)`; the remaining parameters are keyword-only.

`checkpoint` must be one of the 15 entry names, validated even when `pretrained=False`: entry type
determines the dropout p group, which is part of the architecture identity. An invalid name raises
`ValueError`, with the message listing all valid entries.

### `pretrained=True`

- resolves and validates the checkpoint file via `helpers.ensure_beats_checkpoint` (no download, see
  below);
- `torch.load(weights_only=True, map_location="cpu")`, must not fall back to `weights_only=False` (all 15
  checkpoints have been verified readable with `weights_only=True`);
- the checkpoint's `cfg` is strictly compared against the expected table, field by field;
- for fine-tuned checkpoints, discards the `predictor.weight` and `predictor.bias` keys and only those
  two, ignoring `label_dict`;
- performs strict loading over the remaining 250 tensors;
- does not automatically call `eval()`, freeze, or move devices.

### `pretrained=False`

- does not resolve a cache path, does not read files;
- uses the official initialization rules (`init_bert_params`, deep_norm xavier gain, pos_conv normal
  init);
- `checkpoint` still determines the dropout p group.

## Shared weight logic (`helpers/beats.py`)

The following content is concentrated in `src/timbral/models/helpers/beats.py`:

- `BEATS_CHECKPOINTS: dict[str, BeatsCheckpointMetadata]`, 15 entries mapping
  `entry -> (filename, sha256, finetuned)`;
- two expected-cfg tables (22 fields for pretrained / 25 fields for fine-tuned, each value taken from
  actual measurements);
- default directory resolution and SHA-256 verification;
- `weights_only=True` safe reading, strict cfg validation, predictor key discarding, and pos_conv
  weight_norm key-name mapping.

Path priority follows the same convention as sibling models:

```text
explicit pretrained_dir
    >
HF_HUB_CACHE/audioencoders/beats
```

`ensure_beats_checkpoint(entry, pretrained_dir)` performs **no download whatsoever**:

- if the file exists: SHA-256 is verified every time, and an immediate failure occurs on any checksum
  mismatch, never silently used;
- if the file is missing: raises `FileNotFoundError`, with a message giving a directly executable command
  (paths taken from `timbral.paths.project_root()` and the actually resolved directory, not hardcoded):

```text
python {project_root}/scripts/extra/beats_dl.py \
    --dest {resolved_dir} --entries {entry}
(this script requires playwright and requests)
```

The download script's own design is in
[`../extra/beats-download.md`](../extra/beats-download.md). The script and helpers each maintain their
own entry/SHA-256 tables; consistency between the two tables is asserted by the default test suite.

`timbral.models.helpers`'s `__init__` does not add new exports (following the CLAP precedent); the
registry and Encoder import directly from `timbral.models.helpers.beats`.

## Vendored backbone (private classes inside `beats.py`)

The official `backbone.py`/`modules.py` are rewritten as inference-only, repository-style private classes
inside `encoders/beats.py` (`_BeatsSelfAttention`, `_BeatsEncoderLayer`, `_BeatsTransformerEncoder`,
following the PANNs `_ConvBlock` precedent of inline private classes), requiring:

- operator-for-operator numerical equivalence with the official implementation (backed by the alignment
  contract);
- `state_dict` with exactly 250 keys, matching the official checkpoint one-to-one (only the two pos_conv
  weight_norm keys use parametrize naming, mapped by the helpers when loading);
- retaining only the branches actually reached by the cfg of the 15 checkpoints.

### Retained components

- `patch_embedding`: `Conv2d(1, 512, kernel_size=16, stride=16, bias=False)`;
- post-patch `LayerNorm(512)`, `post_extract_proj: Linear(512, 768)`, `dropout_input`;
- convolutional positional encoding: `Conv1d(768, 768, kernel=128, padding=64, groups=16)` +
  `nn.utils.parametrizations.weight_norm(dim=2)` (forward pass is the same `torch._weight_norm` operator
  as the official legacy API; the checkpoint's `weight_g`/`weight_v` are mapped at load time to
  `parametrizations.weight.original{0,1}`) + SamePad (trims the trailing column) + GELU, added as a
  residual;
- `LayerNorm(768)` after the addition (the post-norm configuration sits before the layer stack, with no
  post-stack LayerNorm);
- 12-layer post-LN Transformer layers, deep_norm residual `residual × (2×12)^0.25 + x`;
- self-attention: 12 heads, the official numerical-stability trick of dividing by `alpha=32` after q
  scaling and applying `(attn - max) × alpha` before softmax;
- T5-style bidirectional relative position bias: `Embedding(320, 12)`, `max_distance=800`, only layer 0
  holds an instance, layers 1-11 share the same module (in the state_dict, all 12 layers'
  `relative_attention_bias.weight` keys exist and refer to the same tensor); pos_bias is passed through
  between layers and computed only once;
- gru_rel_pos gating: `grep_linear: Linear(64, 8)`, `grep_a`, scaling pos_bias per the official formula
  `gate_a_1 = gate_a × (gate_b × grep_a - 1.0) + 2.0`;
- FFN: `Linear(768, 3072)` + gelu + `Linear(3072, 768)`;
- all official dropout locations are retained per the entry type's p (identity under eval).

### Pruned branches (unreachable at inference or training-only)

- `encoder_layerdrop` (only randomly drops layers during training; not ported locally, the training-mode
  behavioral difference is recorded here);
- `GradMultiply` / `layer_wise_gradient_decay_ratio` (affects only backward gradients);
- `incremental_state` (fairseq legacy dead code);
- `quant_noise` (p=0, identity);
- non-gelu activations (including `GLU_Linear`);
- the `relative_position_embedding=False`, `gru_rel_pos=False`, `deep_norm=False`,
  `layer_norm_first=True` branches;
- the `padding_mask` path (locally, groups have no intra-group padding after grouping, see below);
- `tgt_layer` / per-layer `layer_results` export;
- the `predictor` classification head, `predictor_dropout`, and `label_dict`.

### state_dict key inventory (250)

```text
patch_embedding.weight
layer_norm.{weight,bias}
post_extract_proj.{weight,bias}
encoder.pos_conv.0.bias
encoder.pos_conv.0.parametrizations.weight.{original0,original1}
    (weight_g/weight_v in the checkpoint, mapped at load time)
encoder.layer_norm.{weight,bias}
encoder.layers.{0..11}.self_attn.{k,v,q,out}_proj.{weight,bias}
encoder.layers.{0..11}.self_attn.grep_linear.{weight,bias}
encoder.layers.{0..11}.self_attn.grep_a
encoder.layers.{0..11}.self_attn.relative_attention_bias.weight
encoder.layers.{0..11}.self_attn_layer_norm.{weight,bias}
encoder.layers.{0..11}.fc1.{weight,bias}
encoder.layers.{0..11}.fc2.{weight,bias}
encoder.layers.{0..11}.final_layer_norm.{weight,bias}
```

20 keys per layer × 12 + 10 top-level keys = 250. The fine-tuned checkpoint's physical key count is 252,
with the extra `predictor.{weight,bias}` discarded before loading.

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

- shape `[B, T, 128]`, float32;
- the normalized kaldi fbank produced by the companion Transform;
- automatically moved to the Encoder's `device` by BaseEncoder.

### `valid_feature_frames`

- shape `[B]`, dtype `torch.int64`, always ≥ 16 (guaranteed by the Transform's minimum zero-padding);
- the concrete encoding hook explicitly accepts this parameter, without a catch-all `**kwargs` that would
  swallow unknown parameters; unknown model inputs must naturally raise `TypeError`.

### `valid_seconds`

- shape `[B]`, float32, used directly for geometry.

## Unique-length grouping and forward pass

Samples with different `valid_feature_frames` must not be passed through the backbone directly within the
same physical canvas (the convolutional positional encoding's kernel=128 would leak padding; the official
padding_mask path is itself an approximate implementation whose results vary with batch composition, and
is not ported). The process:

1. compute unique values of `valid_feature_frames`;
2. for each group, crop to the actual frame count `F`: `input_features[:, :F]`;
3. `unsqueeze(1)` and pass through the patch convolution, yielding `[B_g, 512, F//16, 8]`;
4. `reshape(B_g, 512, -1).transpose(1, 2)` flattens into a token sequence `[B_g, (F//16)×8, 512]` — time
   as the outer dimension, frequency as the inner;
5. `LayerNorm(512)` → `post_extract_proj` → `dropout_input` → TransformerEncoder, yielding
   `[B_g, N, 768]`;
6. produce that group's output according to the granularity branch;
7. restore batch order by original index, zero-padding to the batch's maximum embedding frame count.

The trailing remainder `F % 16` produces no token; the patch convolution has kernel=stride with no
overlap, so the remainder has no receptive-field effect on existing tokens either (unlike PANNs' remainder
semantics). No padding occurs within a group, and the forward result is element-wise identical to calling
on individual samples one at a time.

In training mode, the backbone has no BatchNorm, so grouping itself introduces no training-mode batch
dependency; however, the official layerdrop is not ported, so training-mode behavior is not guaranteed to
match the official implementation. Official numerical alignment and embedding production are performed
under `eval()`.

## Frame granularity

For each patch time block, the 8 frequency tokens are arithmetically averaged:

```text
[B_g, T'×8, 768] -> view(B_g, T', 8, 768) -> mean(dim=2) -> [B_g, T', 768]
T' = valid_feature_frames // 16
```

This is a derived frame embedding based on the official backbone output (following the AST frame
frequency-mean precedent), not an official public output.

Frame step:

```text
frame_step = 16 × 10 ms = 0.16 seconds
```

Geometry (PANNs precedent, start/end sliced from the same float32 `boundaries`):

```text
num_valid = valid_feature_frames // 16    (always ≥ 1)
boundaries = arange(T'_max + 1) × 0.16
start[i] = boundaries[i]
end[i] = boundaries[i + 1]          (i < num_valid - 1)
end[num_valid - 1] = valid_seconds
```

The time grid guarantees intermediate boundaries are strictly less than `valid_seconds`; the last valid
frame absorbs any trailing duration for which no token was produced (including the degenerate case of
`valid_seconds < 0.16` under minimum zero-padding). The implementation shares
`helpers.geometry.build_frame_geometry` with PANNs/AST: that construction uniformly clamps intermediate
boundaries to `valid_seconds`, which for BEATs is always a no-op given the guarantee above.

After merging the mixed batch:

```text
embedding  [B, T'_max, 768]
geometry   [B, T'_max, 2]
valid_mask [B, T'_max]
```

Invalid positions have `valid_mask=False`, with embedding and geometry rows filled with exact zero values.

## Clip granularity

The arithmetic mean is taken over all tokens in the group:

```text
clip_embedding = mean(x, dim=1)    # [B_g, 768]
```

All tokens within a group are valid, and this pooling is consistent with the time-averaging semantics of
the official fine-tuning path before the predictor (officially, in the no-padding case this is exactly
the pooling point of `logits.mean(dim=1)`). No L2 normalization is applied and no classification head is
attached.

Output:

```text
embedding  [B, 768]
geometry   [B, 2] = [0, valid_seconds]
valid_mask [B] all True
```

Since both the frame frequency mean and the clip token mean are linear averages over a fully valid group,
`clip == time average of frame output` holds within numerical tolerance, and can serve as a cross-check
assertion in tests.

## Device, training, and serialization

- `device` is derived from the backbone parameters;
- BaseEncoder automatically transfers `input_features`, `valid_seconds`, and `valid_feature_frames`;
- embedding retains the model's actual dtype (float32);
- geometry is fixed to float32; valid_mask is fixed to bool;
- does not automatically call `eval()`, does not freeze, does not wrap in `torch.no_grad()`;
- uses the plain `state_dict`;
- does not impose a maximum input duration.

## Files and exports

```text
src/timbral/models/encoders/beats.py
src/timbral/models/helpers/beats.py
src/timbral/models/helpers/geometry.py
src/timbral/models/helpers/grouping.py
```

Division of responsibilities:

- `helpers/beats.py`: entry metadata, expected cfg tables, path resolution, SHA-256 verification, safe
  reading, predictor key discarding, and weight_norm key-name mapping;
- `helpers/geometry.py`: model-agnostic construction of clip/frame geometry and valid_mask;
- `helpers/grouping.py`: model-agnostic scaffolding for unique-length grouping iteration and refilling
  grouped results back into a zero canvas;
- `encoders/beats.py`: `BeatsEncoder`, grouped forward pass and granularity semantics, plus the
  inference-only BEATs backbone (`_`-prefixed private classes, not part of the public export).

`timbral.models.encoders` re-exports `BeatsEncoder`. `timbral.models` at the top level only re-exports
registry symbols (see [`../registry.md`](../registry.md)).

## Testing requirements

Weight-free, network-free tests (the default suite, `pretrained=False`) must cover at least:

- 15 valid entries and `ValueError` for an invalid name;
- `pretrained=False` resolves no path, reads no file (monkeypatch asserts ensure is never triggered);
- the module's own `state_dict` key set is exactly the 250-key inventory, with the 12 layers'
  `relative_attention_bias.weight` being the same tensor;
- patch convolution flooring: 98 frames → 6 time blocks × 8 tokens;
- token flattening order is time-outer, frequency-inner (assert via a constructed distinguishable input);
- frame = frequency mean, clip = token mean (matches a manual reference);
- `clip ≈ time average of frame` cross-check assertion;
- mixed-length batches match individual calls element-wise;
- frame geometry, zero padding, bool mask; adjacent valid boundaries element-wise equal; the last frame's
  end equals `valid_seconds`;
- the minimum input (`valid_feature_frames=16`) produces exactly one valid frame;
- unknown model inputs raise `TypeError`;
- helpers: `BEATS_CHECKPOINTS` has all 15 entries complete, filenames are `<entry>.pt`, finetuned flags
  are correct; expected cfg table fields match this document;
- ensure raises `FileNotFoundError` for a missing file, with the message containing the script's absolute
  path and the actually resolved directory;
- after `.to(device)`, input and output devices are correct (when CUDA is available).

For real checkpoint loading and full-network alignment with the official implementation, see
[`../extra/beats-alignment.md`](../extra/beats-alignment.md).

## Dependency boundary

The backbone private classes depend only on PyTorch; `helpers/beats.py` depends on PyTorch,
huggingface_hub (only the `HF_HUB_CACHE` constant), and `timbral.paths`. No fairseq or einops is
introduced, and no new third-party dependency is added.
