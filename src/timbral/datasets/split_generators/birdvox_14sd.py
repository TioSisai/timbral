"""BirdVox-14SD default split generator.

Pipeline:
  1. Open the 14 target-species HDF5 files and enumerate all keys under the
     waveforms group;
  2. Use the HDF5-internal path as audio_path and the taxonomy code as the label;
  3. Perform a stratified random split by class into 80%/10%/10%.
"""

from pathlib import Path

import h5py

from ..adapters.birdvox_14sd import TAXONOMY_CODE_TO_NAME, h5_filename
from . import base as u

# Layout constants are defined once in adapters.birdvox_14sd; only referenced here
TAXONOMY_CODES = [code.replace(".", "-") for code in TAXONOMY_CODE_TO_NAME]


def generate(dataset_dir):
    """Build the BirdVox-14SD default split.

    Args:
        dataset_dir: BirdVox-14SD dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = Path(dataset_dir).resolve()

    audio_paths = []
    labels = []
    per_code_count = {}
    for taxonomy_code in TAXONOMY_CODES:
        filename = h5_filename(taxonomy_code.replace("-", "."))
        h5_path = dataset_dir / filename
        assert h5_path.exists(), "missing h5: {}".format(h5_path)
        with h5py.File(h5_path, "r") as h5_file:
            waveform_keys = list(h5_file["waveforms"].keys())
        per_code_count[taxonomy_code] = len(waveform_keys)
        for waveform_key in waveform_keys:
            audio_paths.append(
                "{}::waveforms/{}".format(filename, waveform_key)
            )
            labels.append(taxonomy_code)

    print("samples per class:", per_code_count)
    print("total samples:", len(audio_paths))

    split_ids = u.stratified_split_single_label(
        audio_paths, labels, ratios=(0.8, 0.1, 0.1)
    )
    splits = {
        split_name: [
            u.make_entry(audio_path, start=0.0, end=u.INF)
            for audio_path in split_ids[split_name]
        ]
        for split_name in u.SPLIT_NAMES
    }

    splits, copy_note = u.fill_missing_splits(splits)
    print("copy_note:", repr(copy_note))
    return splits
