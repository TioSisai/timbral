# datasets component (`timbral.datasets`)

Slices, resamples, and converts the channel layout of raw audio classification datasets according to a specified split, then builds and saves a Hugging Face `DatasetDict`.

## Usage

`dataset_dir` must be passed explicitly:

```bash
python scripts/raw_prep.py \
    --dataset_name ESC-50 \
    --dataset_dir /path/to/ESC-50
```

When `--split_json` is not passed, the program uses
`assets/datasets/splits/{dataset_name}/default.json` under the repository root. If the default file does not exist, the program calls
the corresponding generator registered in `timbral.datasets.split_generators` (a GENERATORS registry
structurally identical to adapters) to generate it; if an explicitly passed split file does not exist, it raises an error directly.

The default split for a given dataset can also be regenerated independently:

```bash
python scripts/gen_default_split.py \
    --dataset_name ESC-50 \
    --dataset_dir /path/to/ESC-50
```

When `--cache_dir` is not passed, the output directory is:

```text
{current working directory}/{dataset_name}/{config_hash}
```

where `config_hash` is computed jointly from `dataset_name`, `split_json_hash`, `sr`, `mono`,
`seg_sec`, `hop_sec`, `tol_sec`, and `label_type`. A local directory or an `s3://` path can also be
explicitly specified via `--cache_dir`.

`DatasetDict.save_to_disk` writes directly to the final `cache_dir`; it does not first create an
intermediate cache and then rename it.

## Testing

```bash
python -m pytest tests/datasets -v
```

## Notes

Some datasets are ragged, i.e. the audio length varies across items, e.g. UrbanSound8K, FSD50K, etc. For such datasets, depending on the user's needs, one can choose:
  1. Slice into fixed-length segments during the `raw_prep` stage.
  2. Set `seg_sec` to the longest audio length, or even slightly longer than the longest audio length (for the sake of downstream throughput), during the `raw_prep` stage, though this incurs larger storage usage.
The two options above require the user to weigh trade-offs according to their own needs.
