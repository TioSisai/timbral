"""
Batch-extract embeddings from a raw cache using the given parameters: build a huggingface datasets.DatasetDict and save_to_disk it to a three-level output path, so that training and evaluation can reuse a pretrained Encoder's features without an on-the-fly forward pass.

Args:
    - cache_dir: str, required, the DatasetDict cache directory produced by raw_prep (local path); it must contain prep_config.json and label_index.json. The data shape (weak/strong and multiclass/multilabel) is inferred automatically from the cached prep_config and features, so no redundant declaration parameter is needed.
    - model_name: str, required, a model name already registered in timbral.models (see timbral.models.list_models() for all available values), e.g. `panns-32k-cnn14-max_mean`, `MIT/ast-finetuned-audioset-10-10-0.4593`.
    - granularity: str, required, `clip` or `frame`: clip produces one vector [D] per clip, frame produces a per-frame vector [T, D] (with time geometry and a valid-frame mask). strong caches support both granularities; weak cache + frame outputs frame embeddings with the clip label passed through unchanged (weakly-labeled SED task scenario); models that don't support frame granularity (e.g. CLAP) are rejected by their natural construction-time error.
    - output_dir: str, output root directory prefix, supports local paths and s3:// paths, default None meaning the current working directory. When using s3://, the S3 environment must be loaded and configured beforehand (module load allas on CSC); map's temporary Arrow files are written to $TMPDIR, and the final artifact is written directly to the S3 target path. The final output path is always the three-level structure {output_dir}/{dataset_name}/{model_name with "/" replaced by "--"}/{emb_hash} (joined as posix paths after normalizing a trailing slash on an s3 root); this cannot be bypassed, so artifacts from multiple datasets x multiple models naturally never collide; emb_hash = datasets.fingerprint.Hasher.hash({the raw cache's config_hash, model_name, granularity}) — execution parameters such as device/batch size do not participate in the hash, so rerunning with the same parameters naturally hits the same directory and is skipped.
    - device: str, target device, default auto (auto-selects cuda > mps > cpu); the Transform and Encoder share this device.
    - batch_size: int, the batch_size parameter of the map function, default 32, i.e. 32 segments per forward pass; reduce it if GPU memory/RAM is insufficient. Under frame granularity, writer_batch_size follows this value to control memory usage; under clip granularity, writer_batch_size is fixed at 1000 (each row is only ~D×4B, decoupled from the forward batch to reduce small-batch Arrow overhead).
    - pretrained_dir: str, custom weights directory, default None meaning the local Hugging Face cache directory is used; passed through to timbral.models.create_model.
    - overwrite: bool, whether to forcibly delete and rebuild the output directory when it already exists, default False: if the directory exists and contains emb_config.json (completion marker), it is skipped directly; if the directory exists but lacks the completion marker, an error is raised suggesting --overwrite, to prevent a half-finished cache from being mistaken for a valid artifact; if overwrite is specified, the directory is deleted and rebuilt.

Auxiliary files and provenance: besides the DatasetDict, the output directory also contains label_index.json copied verbatim from the raw cache, and emb_config.json written last (containing all semantic parameters of this run, emb_hash, and a full snapshot of the raw cache's prep_config, which doubles as the completion marker); the artifact is therefore self-explanatory and traceable even apart from the original cache.

Label and valid-frame conventions (labels take one of three states: 1.0 = confirmed present, NaN = uncertain, 0.0 = unlabeled): weak-cache labels pass through unchanged — multiclass passes through as a ClassLabel int, multilabel passes through as a float32 multi-hot, neither is transformed at either granularity; strong+clip aggregates in-segment events per class by 1 > NaN > 0 into a [C] float32 tri-state multi-hot; strong+frame produces [T, C] float32 frame labels (events and time-geometry slots are matched by positive-length intersection). The tri-state values retain NaN throughout without zeroing it out; zeroing is left to downstream consumers. At frame granularity, the Encoder contract's time geometry geometry [T, 2] and valid-frame mask valid_mask [T] (bool) are also written: embedding/label/geometry for invalid frames are always zero-filled, validity has valid_mask as its sole source of truth, and downstream consumers select valid frames via the mask; T is constant across the whole dataset — at the start of the run, a dummy forward pass over the full window (seg_sec) probes the maximum frame count, and every batch is padded to that value uniformly, so the artifact does not depend on batch_size. The row order of each output split corresponds one-to-one with the input; each row contains provenance fields (audio_path/audio_id/segment_id/start/end/valid_sec) + embedding (clip [D] / frame [T, D] float32) + label (+ geometry and valid_mask for frame), while the waveform (raw) and sample-rate (sr) columns are not included in the output.

Execution model: single-stage in the main process — timbral.models.create_model constructs a matched Transform and Encoder by registered name (both .to(device) and .eval(), with the forward pass wrapped in torch.inference_mode()); for each split, HF map (batched=True, num_proc not passed; in datasets 5.0.0, num_proc=1 spawns a real subprocess, only None runs in the main process) batches the forward pass and writes to disk — map's temporary Arrow files are written to a temp directory alongside the output directory (or to $TMPDIR for s3:// output) and cleaned up afterward; each sample's valid length is passed to the Transform as float32 valid_seconds via valid_sec in seconds unchanged, the Transform's remaining output keys are exactly the Encoder forward's kwargs, and zeroing of padding is guaranteed by the models components' existing mechanism; if the cache's sample rate is lower than the model's native sample rate (transform.target_sample_rate), a notice line is printed (non-blocking; silently upsampling cannot recover already-lost high frequencies); windows longer than expected rely on the Transform's natural error, with no pre-check.

The actual logic is split by responsibility under src/timbral/embeddings (config/labels/builder); this script is only the command-line entry point for embedding extraction, handling argument parsing and dispatch.
"""

import argparse
import sys

import rootutils

# Root setup: locate the project root (.project-root) and inject the src import path, without loading the project .env
ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=False,
                            dotenv=False, cwd=False)
sys.path.insert(0, str(ROOT / "src"))

from timbral.embeddings.builder import prepare_embeddings
from timbral.embeddings.config import resolve_config


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: List of arguments to parse; reads ``sys.argv`` when ``None``.

    Returns:
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Batch-extract embeddings from a raw cache and build a "
                    "DatasetDict cache (see the module docstring at the top "
                    "of this script for details)")
    parser.add_argument("--cache_dir", required=True,
                        help="DatasetDict cache directory produced by raw_prep (local path)")
    parser.add_argument("--model_name", required=True,
                        help="Model name registered in timbral.models "
                             "(see timbral.models.list_models())")
    parser.add_argument("--granularity", required=True,
                        choices=("clip", "frame"), help="Embedding output granularity")
    parser.add_argument("--output_dir", default=None,
                        help="Output root directory prefix, defaults to the current working directory")
    parser.add_argument("--device", default="auto",
                        help="Target device: auto/cpu/cuda/cuda:x/mps")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch_size and writer_batch_size for map")
    parser.add_argument("--pretrained_dir", default=None,
                        help="Custom weights directory, defaults to the local Hugging Face cache")
    parser.add_argument("--overwrite", action="store_true", help="Force rebuild of the output")
    return parser.parse_args(args)


def main() -> None:
    """Thin dispatch: arguments → config → execution."""
    prepare_embeddings(resolve_config(**vars(parse_args())))


if __name__ == "__main__":
    main()
