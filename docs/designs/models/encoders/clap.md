# `ClapHtsatEncoder` Design

This document freezes the design of `timbral.models.encoders.ClapHtsatEncoder`. The companion Transform is
[`../transforms/clap.md`](../transforms/clap.md); the official alignment contract is
[`../extra/clap-alignment.md`](../extra/clap-alignment.md).

## Design goals

`ClapHtsatEncoder` wraps the audio tower and audio projection of the fixed `laion/clap-htsat-fused`
checkpoint, outputting a clip embedding in the CLAP contrastive space. It is responsible for:

- inheriting `BaseEncoder` and declaring support for clip only;
- building only the audio tower and audio projection;
- strictly loading the 270 relevant tensors of the fixed checkpoint;
- computing per-sample fusion routing based on `valid_seconds`;
- producing an L2-normalized 512-dimensional embedding;
- producing clip geometry and a valid mask conforming to the base class.

CLAP fused HTSAT has no frame representation that can be mapped onto the full original time axis. For long
audio, the three deterministic anchor local crops may overlap or leave gaps, and are fused with the global
channel in AFF and global attention. Therefore this model's supported set is fixed to:

```python
supported_granularities = frozenset(("clip",))
```

## Public interface

```python
class ClapHtsatEncoder(BaseEncoder):
    supported_granularities = frozenset(("clip",))
    embedding_dim = 512

    def __init__(
        self,
        *,
        granularity: Granularity,
        pretrained: bool = True,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        ...
```

All parameters are keyword-only. The constructor first calls:

```python
super().__init__(granularity)
```

`granularity="frame"` raises `ValueError` before any weight preparation or model construction, with the
error message including `ClapHtsatEncoder` and its supported clip granularity.

`embedding_dim` declares as a ClassVar that the last dimension of the output embedding is 512 (i.e. the
projection dim), so that callers can build the output schema before the forward pass.

`pretrained` must be a Python `bool`.

### `pretrained=True`

- prepares the fixed-revision `config.json`, `preprocessor_config.json`, and `model.safetensors`;
- each file is verified against a fixed SHA-256;
- validates the key fields of config and preprocessor;
- reads the weights safely via safetensors;
- precisely selects 266 `audio_model.*` tensors and 4 `audio_projection.*` tensors;
- performs `strict=True` loading on `ClapAudioModelWithProjection`;
- does not read or build the text tower, text projection, or logit scale.

### `pretrained=False`

- does not resolve a cache directory;
- does not read files;
- does not access the network;
- builds `ClapAudioModelWithProjection` using a `ClapAudioConfig` fixed in code;
- uses Transformers' official random initialization;
- the model structure is identical to the pretrained audio tower.

## Fixed model identity

```text
repo:      laion/clap-htsat-fused
revision:  365dea6ef167def6676140ed93bbc43f84dabb28
```

Core architecture:

| Parameter | Value |
|---|---:|
| hidden size | 768 |
| projection dim | 512 |
| enable fusion | True |
| spec size | 256 |
| mel bins | 64 |
| patch size | 4 |
| patch stride | 4 × 4 |
| patch embed dim | 96 |
| stage depths | 2, 2, 6, 2 |
| window size | 8 |

Checkpoint identity, config fields, download, checksum verification, and strict state filtering are
concentrated in `src/timbral/models/helpers/clap.py`; download goes through
`helpers.common.ensure_hf_snapshot` (downloads first to a temporary directory within the snapshot
directory; once the checksum passes, atomically moves it into the final path, so a corrupt or interrupted
package never ends up at the final path, same as AST).

## Input contract

Canonical call:

```python
encoder(
    input_features,
    valid_seconds=valid_seconds,
)
```

`_encode_clip` performs metadata-only fixed-contract checks:

- `input_features` shape is strictly `[B,4,1001,64]`;
- `input_features` dtype is strictly `float32`;
- `valid_seconds` shape is strictly `[B]`;
- the two have consistent batch size.

The fixed HTSAT silently interpolates some incorrect time dimensions, and also degrades to a non-official
path when the channel count is wrong, so the shape check is part of the model's input semantics. The
check only reads shape and dtype, and performs no device synchronization.

`BaseEncoder` moves both Tensors to the Encoder's device without converting dtype.

## Fusion routing

The Encoder computes, on the same target sample-rate grid as the Transform:

```python
target_valid_samples = torch.round(valid_seconds * 48000).to(torch.long)
fusion_mask = target_valid_samples > 480479
```

It is converted to the internal parameter only when calling the Hugging Face audio tower:

```python
outputs = self.backbone(
    input_features=input_features,
    is_longer=fusion_mask.unsqueeze(1),
)
```

`is_longer` is not a public input, Transform output, or Encoder state. It is merely an internal routing
Tensor required by the fixed upstream model's forward-pass signature.

## Clip embedding

`ClapAudioModelWithProjection.audio_embeds` is the 512-dimensional output after the audio projection. The
final embedding uses:

```python
embedding = torch.nn.functional.normalize(
    outputs.audio_embeds,
    dim=-1,
)
```

This result is element-wise identical to the official L2-normalized output of the full
`ClapModel.get_audio_features`.

Output:

```python
{
    "embedding": Tensor[B,512],
    "geometry": Tensor[B,2],
    "valid_mask": Tensor[B],
}
```

- `geometry[i] = [0, valid_seconds[i]]`, dtype `float32`;
- `valid_mask` is an all-True bool Tensor of shape `[B]`;
- embedding retains the model's actual computation dtype;
- all outputs reside on the Encoder's device.

Geometry represents the time ownership of the original clip. Repeatpad, global compression, local crops,
and global attention all participate in the embedding computation.

## Lifecycle, device, and serialization

- `device` is determined by the audio projection's weight device;
- construction does not automatically call `eval()`;
- parameters are not automatically frozen;
- the forward pass is not wrapped in `no_grad` or `inference_mode`;
- `state_dict` saves the complete audio tower and projection;
- alignment and inference callers explicitly choose eval, inference mode, and device.

## Files and exports

- the Encoder is located at `src/timbral/models/encoders/clap.py`;
- the checkpoint helper is located at `src/timbral/models/helpers/clap.py`;
- clip geometry and valid_mask construction are located at `src/timbral/models/helpers/geometry.py`;
- `timbral.models.encoders` re-exports `ClapHtsatEncoder`;
- the helper is not exported from the `timbral.models` top level;
- this component does not hold a Transform, nor does it implement YAML or a Pipeline.

## Testing requirements

Ordinary offline tests must cover at least:

- public exports and `supported_granularities == {"clip"}`;
- `granularity="frame"` raises `ValueError` before any weight side effect;
- `pretrained` is strictly bool;
- `pretrained=False` is fully offline;
- model structure of the fixed config;
- `input_features` shape/dtype and batch consistency;
- internal routing corresponding to 480479/480480 samples;
- embedding shape, L2 norm, geometry, and mask;
- gradients in training mode;
- CPU/CUDA device behavior;
- unknown model parameters naturally raise `TypeError`;
- strict mapping and loading of the checkpoint's 270 tensors.

For real-weight and official numerical tests, see
[`../extra/clap-alignment.md`](../extra/clap-alignment.md).
