<p align="right">
  🌐 <a href="README.md">English</a> | <a href="README-zh.md">简体中文</a>
</p>

# 🎶 TIMBRAL

Toolkit for Ingestion and Model-Based Representation Archival and Loading (TIMBRAL) —
a model-agnostic, cache-driven pipeline that turns audio datasets into reusable embeddings.

The top-level Python package is `timbral`, currently containing dataset preparation and embedding extraction components:

- **`timbral.datasets`** — Slices, resamples, and converts channels for raw audio classification datasets by split, then builds and saves a Hugging Face `DatasetDict` cache. See [docs/datasets.md](docs/datasets.md) for details.
- **`timbral.embeddings`** — Batch-extracts clip/frame embeddings from pretrained audio encoders using the raw cache produced by `timbral.datasets`, and saves them as a reusable Hugging Face `DatasetDict` cache.

## Repository Layout

```
.project-root        # rootutils root-locating marker (unique across the whole repo)
pyproject.toml       # single source of project/dependency/pytest configuration
assets/              # asset directory (data only, no code), organized as a subtree per component
  datasets/splits/   #   per-dataset split JSON ({dataset_name}/default.json, auto-generated
                     #   when missing; a local build artifact, not version-controlled)
src/timbral/
  paths.py           # repository-root locating (the single mechanism); each component derives its asset dir from this
  storage.py         # S3 storage parameters, cache target resolution, and map temp-dir management
  datasets/          # dataset-building components
    adapters/        #   one annotation adapter per dataset (ADAPTERS registry)
    split_generators/#   one default split generator per dataset (GENERATORS registry, mirrors adapters)
  embeddings/        # embedding extraction config, label conversion, and build orchestration
  models/            # registered audio Transforms/Encoders
scripts/             # CLI entry points (thin dispatch layer)
  raw_prep.py        #   build the raw cache
  emb_prep.py        #   extract embeddings from the raw cache
  gen_default_split.py
tests/
  datasets/          # tests for the datasets component
  embeddings/        # tests for the embeddings component
  models/            # tests for the models component
  scripts/           # tests for CLI entry points
```

## Environment Setup

### Dependencies

Requires Python >= 3.12. All runtime dependencies and their minimum versions are listed in
[pyproject.toml](pyproject.toml); the core ones are `datasets`, `torch`/`torchaudio`,
`transformers`, `librosa`/`soundfile`, `huggingface_hub`, and `rootutils`. `datasets` is
pinned to the 5.x line because this repository relies on its `map(num_proc=...)` process
semantics, which change silently across major versions.

If a specific CUDA/ROCm build is required, install the matching `torch` and `torchaudio`
first, following the [official PyTorch guide](https://pytorch.org/get-started/locally/), and
only then install this package — otherwise the default wheels pulled in by this package would
overwrite them.

### Installation

```bash
git clone <this-repo> && cd timbral
python -m venv .venv && source .venv/bin/activate   # or: conda create -n timbral python=3.12
pip install -e .
```

Install this package in editable mode: at import time, rootutils locates the repository root
via its `.project-root` marker, and since that marker is not shipped with the package, root
locating fails after a regular (non-editable) install.

`scripts/*.py` locate the repository root and inject the `src` import path on their own, so
`PYTHONPATH` never needs to be set; the editable install above is only required when doing
`import timbral` from outside the repository.

The BEATs weight-download script has its own optional dependency group (independent of the
runtime, not part of the pipeline):

```bash
pip install -e ".[beats-dl]" && playwright install chromium
```

### Environment Variables

All optional, set as needed:

| Variable | Purpose | Default |
|---|---|---|
| `HF_HOME` / `HF_HUB_CACHE` | Root of the Hugging Face cache; this repository's encoder weights are stored under `$HF_HUB_CACHE/audioencoders/` (PANNs by model name, AST/CLAP by repo_id, BEATs all under `beats/`) | `~/.cache/huggingface`, `$HF_HOME/hub` |
| `TMPDIR` | Where `map`'s temporary Arrow files are written when the output goes to `s3://` | system temp directory |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_S3_ENDPOINT` | Credentials and endpoint for writing to `s3://` (S3-compatible storage other than AWS must specify the endpoint) | none |

## Usage Guide

The pipeline runs in two stages: `raw_prep.py` slices, resamples, and converts channels for the
raw audio by split, producing a raw cache; `emb_prep.py` batch-forwards the raw cache through a
pretrained encoder, producing an embedding cache. Both stages name their output directory after
a config hash, so re-running with identical parameters naturally hits the existing artifact and
skips the work.

A minimal end-to-end chain:

```bash
python scripts/raw_prep.py \
    --dataset_name ESC-50 --dataset_dir /path/to/ESC-50 \
    --cache_dir /path/to/raw_cache/ESC-50

python scripts/emb_prep.py \
    --cache_dir /path/to/raw_cache/ESC-50 \
    --model_name panns-32k-cnn14-max_mean \
    --granularity clip \
    --output_dir /path/to/emb_cache
```

### 1. Prepare the Dataset Source Files

Each dataset must be obtained and extracted from its official source on your own;
`--dataset_dir` points to the root of the extracted dataset. Each adapter parses the directory
layout and annotation files as published officially — see the expected format in the
module of the same name under
[src/timbral/datasets/adapters/](src/timbral/datasets/adapters/).

Datasets with a registered adapter (valid values for `--dataset_name`):

`AudioSetStrong`, `AudioSetWeak`, `BSD10K`, `BSD35K`, `BirdVox-14SD`, `DB3V`,
`DCASE-2024-Task-5`, `DESED`, `DataSED`, `ESC-50`, `FSD50K`, `FSDnoisy18k`,
`HyenaSET`, `RealDESED`, `SONYC-UST`, `UrbanSound8K`

### 2. Build the Raw Cache

```bash
python scripts/raw_prep.py \
    --dataset_name ESC-50 \
    --dataset_dir /path/to/ESC-50 \
    --cache_dir /path/to/raw_cache/ESC-50 \
    --sr 32000 --seg_sec 10.0 --hop_sec 10.0 \
    --num_proc 4 --batch_size 16
```

| Argument | Default | Description |
|---|---|---|
| `--dataset_name` | required | Dataset name; see the table above for valid values |
| `--dataset_dir` | required | Root directory of the dataset source files |
| `--cache_dir` | `{cwd}/{dataset_name}/{config_hash}` | Output directory; supports both local paths and `s3://` |
| `--split_json` | `assets/datasets/splits/{dataset_name}/default.json` | Split file; auto-generated when the default file is missing, raises an error when an explicit path is missing |
| `--sr` | `16000` | Target sample rate; recommended to match the downstream model's native sample rate |
| `--mono` / `--no-mono` | `--mono` | Whether to mix down to mono |
| `--seg_sec` / `--hop_sec` | `10.0` / `10.0` | Segment length and hop size (seconds) |
| `--tol_sec` | `0.0` | Minimum length to keep for a short trailing segment; only applies to `segment_id > 0` |
| `--label_type` | `weak` | `weak` aggregates labels per segment; `strong` keeps the in-segment event list |
| `--num_proc` / `--batch_size` | `4` / `16` | Number of `map` processes and batch size; `--num_proc 0` runs in the main process (in datasets 5.x, `1` actually spawns one worker process) |
| `--overwrite` | off | Force a rebuild when the directory already exists (default is to skip, after verifying that `config_hash` matches) |

`config_hash` is computed jointly from `dataset_name`, the split-file hash, `sr`, `mono`,
`seg_sec`, `hop_sec`, `tol_sec`, and `label_type`, so changing any of these always changes the
output directory. Besides the `DatasetDict`, the output directory also contains
`prep_config.json` and `label_index.json` (a mapping from class name to index).

The split file defaults to `assets/datasets/splits/{dataset_name}/default.json`, generated on
the fly by the registered generator when it doesn't exist. It can also be generated separately
or to a specific location:

```bash
python scripts/gen_default_split.py \
    --dataset_name ESC-50 --dataset_dir /path/to/ESC-50 [--output /path/to/split.json]
```

See [docs/datasets.md](docs/datasets.md) for the split JSON format convention, how k-fold
datasets are organized, and the segmenting trade-offs for variable-length (ragged) datasets.

### 3. Extract Embeddings

```bash
python scripts/emb_prep.py \
    --cache_dir /path/to/raw_cache/ESC-50 \
    --model_name panns-32k-cnn14-max_mean \
    --granularity clip \
    --output_dir /path/to/emb_cache \
    --device auto --batch_size 32
```

| Argument | Default | Description |
|---|---|---|
| `--cache_dir` | required | The cache directory produced by `raw_prep.py` (local path) |
| `--model_name` | required | A registered model name; see the table below |
| `--granularity` | required | `clip` (one `[D]` vector per clip) or `frame` (per-frame `[T, D]`) |
| `--output_dir` | current working directory | Output root prefix; supports both local paths and `s3://` |
| `--device` | `auto` | `auto` selects cuda > mps > cpu; can also be set explicitly to `cpu`/`cuda:0`/`mps` |
| `--batch_size` | `32` | Number of segments per forward pass |
| `--pretrained_dir` | none | Custom weights directory; defaults to the project-specific directory under the Hugging Face cache |
| `--overwrite` | off | Delete and rebuild if the output already exists |

The output path always follows a three-level structure
`{output_dir}/{dataset_name}/{model_name}/{emb_hash}`, with `/` in the model name replaced by
`--`. `emb_hash` is computed from the raw cache's `config_hash`, `model_name`, and
`granularity` (execution parameters such as device and batch size are not included). The
directory contains the `DatasetDict`, `label_index.json` copied from the raw cache, and
`emb_config.json` written last (a full parameter snapshot that also serves as a completion
marker).

The data shape is inferred automatically from the cache, with no extra declaration needed:
weak-cache labels pass through unchanged; strong + clip aggregates per class into a `[C]`
three-state multi-hot vector; strong + frame produces `[T, C]` frame labels. The frame
granularity additionally stores a temporal geometry array `geometry [T, 2]` and a valid-frame
mask `valid_mask [T]`; embeddings/labels/geometry for invalid frames are always zero-padded,
with validity determined by `valid_mask`.

Registered `--model_name` values:

| Family | Registered name |
|---|---|
| PANNs | `panns-16k-cnn14-max_mean`, `panns-32k-cnn14-max_mean`, `panns-32k-cnn14-decision_level_max` |
| AST | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| CLAP | `laion/clap-htsat-fused` (clip granularity only) |
| BEATs | `beats_iter1`/`beats_iter2`/`beats_iter3`/`beats_iter3_plus_as20k`/`beats_iter3_plus_as2m`, plus the corresponding `fine_tuned_*_cpt1`/`cpt2`, 15 in total |

Get the full list programmatically: `from timbral.models import list_models; list_models()`.

### 4. Pretrained Weights

PANNs, AST, and CLAP weights are downloaded automatically to `$HF_HUB_CACHE/audioencoders/`
the first time the model is constructed, and verified against a fixed SHA-256 — no manual
preparation is needed.

BEATs officially distributes weights only via OneDrive share links, and the runtime code
contains no download logic; a standalone script must be run first (requires playwright, see
the optional dependency above):

```bash
python scripts/extra/beats_dl.py \
    --dest ~/.cache/huggingface/hub/audioencoders/beats \
    --entries beats_iter3_plus_as2m fine_tuned_beats_iter3_cpt1 \
    --workers 3
```

When `--dest` points at the default weights directory (`$HF_HUB_CACHE/audioencoders/beats`),
`emb_prep.py` picks it up with no extra argument needed; if placed elsewhere, point to it with
`--pretrained_dir`. Omitting `--entries` downloads all 15 checkpoints. See
[docs/designs/models/extra/beats-download.md](docs/designs/models/extra/beats-download.md)
for the mechanism in detail.

### 5. Reading the Outputs

Both cache stages are standard Hugging Face `DatasetDict`s and can be read independently of
this repository:

```python
from datasets import load_from_disk

emb = load_from_disk("/path/to/emb_cache/ESC-50/panns-32k-cnn14-max_mean/<emb_hash>")
train = emb["train"].with_format("numpy")
print(train.column_names)
# ['audio_path', 'audio_id', 'segment_id', 'start', 'end', 'valid_sec',
#  'embedding', 'label']  (frame granularity additionally has 'geometry' and 'valid_mask')
print(train[0]["embedding"].shape)
```

`with_format("numpy")` is noticeably faster than `"torch"`; it's recommended to convert to
tensors on the DataLoader side instead. The mapping between class names and indices is in
`label_index.json` in the same directory.

### 6. Writing Output to Object Storage

When `--cache_dir` / `--output_dir` is given as `s3://bucket/prefix`, `map`'s temporary Arrow
files are written to `$TMPDIR`, while the final artifact is written directly to object storage.
Set up the credentials and `AWS_S3_ENDPOINT` beforehand (see the environment variables table).
The S3 path must include a directory within the bucket — the bucket root alone is not allowed.

## Citation

If TIMBRAL is helpful to your research, please consider citing this repository:

```bibtex
@misc{zhang2026timbral,
  author       = {Shiqi Zhang},
  title        = {{TIMBRAL}: Toolkit for Ingestion and Model-Based Representation Archival and Loading},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/TioSisai/timbral}
}
```
