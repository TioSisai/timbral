"""
Perform raw audio preparation from the given parameters: build a huggingface datasets.DatasetDict and save_to_disk it to the specified cache path, for use in subsequent training and evaluation.

Args:
    - dataset_name: str, the dataset name, e.g. `BSD35K`, `ESC-50`, `FSD50K`, etc.
    - dataset_dir: str, the dataset's source-file root directory; must be passed explicitly.
    - cache_dir: str, the save_to_disk output path for the resulting DatasetDict; supports local paths and s3:// paths. When using s3://, the S3 environment must be loaded and configured beforehand (module load allas on CSC); map's temporary Arrow files are written to $TMPDIR, and the final artifact is written directly to the S3 target path. Defaults to current working directory/{dataset_name}/{config_hash: datasets.fingerprint.Hasher.hash({dataset_name, split_json_hash, sr, mono, seg_sec, hop_sec, tol_sec, label_type})}; a label_index.json file is also saved in cache_dir, recording the mapping between class_name (human-readable label names, numbered in lexicographic order) and index, for use in subsequent training and evaluation.
    - split_json: str, the dataset split; defaults to `{repo root}/assets/datasets/splits/{dataset_name}/default.json`. If the default file does not exist, the corresponding generator registered in timbral.datasets.split_generators is invoked with dataset_dir to generate it. A custom split file can also be specified via `/path/to/custom/split.json`; if an explicitly given path does not exist, an error is raised immediately.
        The default default.json file must contain train/validation/test splits (for any split missing validation or test, the existing one is copied directly to fill in the missing one — e.g. if only validation exists, it is copied to test; a custom json may use different splits). Each split contains audio files' relative paths and the start/end time at which each audio falls within that split (0/inf for the whole clip). Across different splits, the start/end times of the same audio must not overlap (except when copied due to a missing validation or test, where overlap is judged as left-closed, right-open), for example:
        {
            "train": [
                {
                    "audio_path": "train/audio1.wav",
                    "start": 0.0,
                    "end": inf,
                },
                {
                    "audio_path": "train/audio2.wav",
                    "start": 0.0,
                    "end": 20.0,
                },
                ...
            ],
            "validation": [
                {
                    "audio_path": "validation/audio2.wav",
                    "start": 20.0,
                    "end": 40.0,
                },
                ...
            ],
            "test": [
                {
                    "audio_path": "test/audio3.wav",
                    "start": 0.0,
                    "end": inf,
                },
                ...
            ]
        }
        For k-fold splits like ESC-50 and UrbanSound8K, multiple json files are needed, with each fold's split managed as an independent cache; although this wastes storage space, it keeps the semantics consistent overall and is also convenient for subsequent training and evaluation.
    - sr: int, the sample rate; audio is resampled to this rate when read, default 16000. (If a downstream model needs a higher sample rate, e.g. 32000, it is best to set it correctly here — otherwise the model's own transform may apply a default resample that silently downsamples and cuts high-frequency information, hurting model performance.)
    - mono: bool, whether to convert audio to mono, default True, i.e. converted to mono.
    - seg_sec: float, the length of each audio segment, in seconds, default 10.0.
    - hop_sec: float, the hop length between audio segments, in seconds, default 10.0.
    - tol_sec: float, the minimum segment length, used to filter out short trailing segments after audio is split, in seconds, default 0.0, i.e. no filtering. This only applies to segments with segment_id > 0; the 0th segment of each entry is always kept unconditionally, guaranteeing that every audio produces at least one segment.
    - label_type: str, the label type, either `weak` or `strong`, default `weak`, meaning each segment corresponds to one aggregated label (which can be multi-class, e.g. ESC-50, or multi-label, e.g. FSD50K — multi-label is unified into a float32 multi-hot; if the dataset itself has strong annotations, e.g. DataSED, events intersecting the segment are aggregated per class by the tri-state label into a float32 multi-hot, with same-class conflicts resolved by 1 > NaN > 0, keeping NaN throughout without zeroing it). If `strong`, each segment corresponds to a different label form instead: the label is a list, each element a dict containing fields such as target (ClassLabel, always a valid label index) + start (start within the segment) + end (end within the segment) + value (float32 tri-state label value, 1.0 = confirmed present, NaN = uncertain; confirmed-absent and unlabeled are not persisted, and downstream consumers infer 0 from the absence of an event) — e.g. AudioSetStrong. If the dataset itself only has weak annotations but label_type=strong is specified, an error is raised immediately.
    - num_proc: int, the number of processes for reading and processing audio, default 4, i.e. processed with 4 processes (note that the map function's batched parameter must be True to process audio entry by entry; since audio of different lengths can be split into different numbers of segments, batched=True is required to support ragged batches; a segment longer than tol_sec but shorter than seg_sec is still kept, but must be zero-padded to seg_sec for use in subsequent training and evaluation).
    - batch_size: int, the batch_size parameter of the map function (also used as map's writer_batch_size), default 16, i.e. each process handles 16 audio files at a time, to make full use of each process's CPU resources; watch memory usage though, and reduce batch_size if memory is insufficient.
    - overwrite: bool, whether to force a rebuild when cache_dir already exists, default False, meaning it is skipped directly if it already exists (before skipping, the config_hash in the cache's prep_config.json is checked against the current parameters; a mismatch raises an error immediately, preventing an explicitly specified cache_dir with changed parameters from silently hitting a stale artifact).
First, pandas is used to preprocess all relevant metadata (starting from the split.json file, obtaining audio_path, split, label, and other related information); then, within map(batched=True), librosa.load is called per recording and per segment (passing the audio's absolute path, mono=mono, sr=sr, offset={current_offset}, duration={valid_sec}) to read the audio — segmentation is accomplished purely through each audio's {current_offset} and {valid_sec}.
The final output DatasetDict contains train/validation/test and other splits; each row contains fields such as audio_path, audio_id, segment_id, raw, sr, start, end, valid_sec, label, where audio_path is the audio file's relative path, audio_id is the audio file's unique int id (0-based), segment_id is the unique int id (0-based) of the segment within that audio_id, raw is the numpy array of the current segment's audio, sr is the sample rate, start and end are the current segment's start/end time within the original audio, valid_sec is the current segment's valid length (may be shorter than seg_sec), and label can be int (multi-class weak), float32 multi-hot (multi-label weak, or strong aggregated down, possibly containing NaN per the tri-state labeling), or a list of dicts (strong, each dict containing target/start/end/value, in which case the label field is necessarily ragged)

The actual logic is split by responsibility under src/timbral/datasets (config/split_io/segmentation/labels/audio_io/adapters/builder); this script is only the command-line entry point for raw audio preparation, handling argument parsing and dispatch.
"""

import argparse
import sys

import rootutils

# Root setup: locate the project root (.project-root) and inject the src import path, without loading the project .env
ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=False,
                            dotenv=False, cwd=False)
sys.path.insert(0, str(ROOT / "src"))

from timbral.datasets.builder import prepare_dataset
from timbral.datasets.config import resolve_config


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: List of arguments to parse; reads ``sys.argv`` when ``None``.

    Returns:
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare raw audio and build a segmented huggingface DatasetDict "
                    "cache (see the module docstring at the top of this script for details)")
    parser.add_argument("--dataset_name", required=True, help="Dataset name, e.g. ESC-50")
    parser.add_argument("--dataset_dir", required=True, help="Dataset source-file root directory")
    parser.add_argument("--cache_dir", default=None,
                        help="DatasetDict output directory (local path or s3:// path)")
    parser.add_argument("--split_json", default=None, help="Path to the split json file")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--mono", action=argparse.BooleanOptionalAction,
                        default=True, help="Whether to convert to mono")
    parser.add_argument("--seg_sec", type=float, default=10.0, help="Segment length (seconds)")
    parser.add_argument("--hop_sec", type=float, default=10.0, help="Segment hop length (seconds)")
    parser.add_argument("--tol_sec", type=float, default=0.0,
                        help="Minimum length to keep for trailing segments (seconds); "
                             "0 means no filtering, segment 0 is always kept")
    parser.add_argument("--label_type", choices=("weak", "strong"), default="weak",
                        help="Label form")
    parser.add_argument("--num_proc", type=int, default=4,
                        help="Number of processes for audio processing; 0 for the main "
                             "process, >=1 for worker processes")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch_size and writer_batch_size for map")
    parser.add_argument("--overwrite", action="store_true", help="Force rebuild of the cache")
    return parser.parse_args(args)


def main() -> None:
    """Thin dispatch: arguments → config → execution."""
    prepare_dataset(resolve_config(**vars(parse_args())))


if __name__ == "__main__":
    main()
