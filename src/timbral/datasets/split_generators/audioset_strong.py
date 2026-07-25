"""AudioSetStrong default split generator.

Strategy (see SPEC section 5):
  - train = all existing files under audio/train/*.wav;
  - test = all existing files under audio/eval/*.wav;
  - no official validation set; use fill_missing_splits to copy test into validation;
  - audio_path is a POSIX path relative to the dataset root directory.
"""

from pathlib import Path

from ..adapters.audioset_strong import EVAL_AUDIO_DIR, TRAIN_AUDIO_DIR
from . import base as u


def list_wavs(dataset_dir, relative_dir):
    """List the existing WAV files under the given directory.

    Args:
        dataset_dir: AudioSetStrong dataset root directory.
        relative_dir: Audio directory relative to the dataset root.

    Returns:
        list[str]: Sorted POSIX paths relative to the dataset root.
    """
    audio_dir = dataset_dir / relative_dir
    filenames = sorted(
        path.name
        for path in audio_dir.iterdir()
        if path.name.endswith(".wav") and path.is_file()
    )
    return [f"{relative_dir}/{filename}" for filename in filenames]


def generate(dataset_dir):
    """Build the AudioSetStrong default split.

    Args:
        dataset_dir: AudioSetStrong dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = Path(dataset_dir).resolve()

    train_paths = list_wavs(dataset_dir, TRAIN_AUDIO_DIR)
    eval_paths = list_wavs(dataset_dir, EVAL_AUDIO_DIR)
    print("train wav present:", len(train_paths))
    print("eval  wav present:", len(eval_paths))

    splits = {
        "train": [u.make_entry(path) for path in train_paths],
        "test": [u.make_entry(path) for path in eval_paths],
    }
    splits, note = u.fill_missing_splits(splits)
    print("copy note:", note)
    return splits
