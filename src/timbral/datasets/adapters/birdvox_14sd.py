"""BirdVox-14SD annotation adapter: target-species HDF5 waveform groups -> multiclass weak.

The default split uses only the 14 target species. Each HDF5 file corresponds to
one species, and every dataset inside the file's ``waveforms`` group is an
independent audio sample. ``audio_path`` follows the HDF5 virtual path form of
``timbral.datasets.audio_io``, and reading is dispatched uniformly by audio_io.
"""

import os

import h5py

from .base import DatasetAnnotation

# Official taxonomy code -> readable species name. Filenames separate the code
# with ``-``, while the HDF5 internal taxonomy_code and official documentation
# use ``.``.
TAXONOMY_CODE_TO_NAME = {
    "1.1.1": "American tree sparrow",
    "1.1.2": "Chipping sparrow",
    "1.1.3": "Savannah sparrow",
    "1.1.4": "White-throated sparrow",
    "1.2.1": "Rose-breasted grosbeak",
    "1.3.1": "Gray-cheeked thrush",
    "1.3.2": "Swainson's thrush",
    "1.4.1": "American redstart",
    "1.4.2": "Bay-breasted warbler",
    "1.4.3": "Black-throated blue warbler",
    "1.4.4": "Canada warbler",
    "1.4.5": "Common yellowthroat",
    "1.4.6": "Mourning warbler",
    "1.4.7": "Ovenbird",
}


def h5_filename(taxonomy_code: str) -> str:
    """The official HDF5 filename corresponding to a taxonomy code (shared with the split generator).

    Args:
        taxonomy_code: Dot-separated taxonomy code, e.g. ``"1.1.1"``.

    Returns:
        The official HDF5 filename, e.g. ``"BirdVox-14SD_1-1-1_original.h5"``.
    """
    return f"BirdVox-14SD_{taxonomy_code.replace('.', '-')}_original.h5"


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load BirdVox-14SD target-species annotations.

    Args:
        dataset_dir: BirdVox-14SD dataset root directory.

    Returns:
        DatasetAnnotation: ``audio_path`` looks like
        ``BirdVox-14SD_1-1-1_original.h5::waveforms/{key}``.
    """
    weak_labels = {}
    for taxonomy_code, class_name in TAXONOMY_CODE_TO_NAME.items():
        filename = h5_filename(taxonomy_code)
        with h5py.File(os.path.join(dataset_dir, filename), "r") as h5_file:
            weak_labels.update({
                f"{filename}::waveforms/{key}": class_name
                for key in h5_file["waveforms"]
            })

    return DatasetAnnotation(
        label_kind="multiclass",
        annotation_kind="weak",
        classes=sorted(TAXONOMY_CODE_TO_NAME.values()),
        weak_labels=weak_labels,
    )
