"""HyenaSET annotation adapter: call-type events from same-named HDF5 files -> multilabel strong.

Each raw event also carries a ``foc`` flag, but the current task only uses the
ten call-type labels and does not encode whether the vocalizing individual is a
collared focal individual as a class.

There are a large number of label files, so ``strong_events`` reads and caches
lazily by ``audio_path``, avoiding a scan of all HDF5 files just to load the
adapter.
"""

import os

import h5py

from .base import DatasetAnnotation


_CALL_CLASSES = (
    "groan",
    "oth",
    "whoop",
    "alarm_rumble",
    "squitter",
    "squeal",
    "feeding",
    "giggle",
    "growl",
    "snore",
)


class _MissingAnnotationError(KeyError):
    """Indicates the audio path has no corresponding label, without swallowing HDF5 internal structural errors."""


def _label_path(dataset_dir: str, audio_path: str) -> str:
    """Convert a split's WAV relative path into the same-named HDF5 label path.

    Args:
        dataset_dir: HyenaSET dataset root directory.
        audio_path: The audio relative path in the split, starting with ``wav/``.

    Returns:
        str: The absolute label path under the dataset's ``lbl/`` subtree.

    Raises:
        _MissingAnnotationError: ``audio_path`` does not match HyenaSET's WAV path format.
    """
    stem, extension = os.path.splitext(audio_path)
    if not stem.startswith("wav/") or extension.lower() != ".wav":
        raise _MissingAnnotationError(audio_path)
    label_relpath = f"lbl/{stem.removeprefix('wav/')}.h5"
    return os.path.join(dataset_dir, *label_relpath.split("/"))


def _read_strong_events(label_path: str) -> list:
    """Read the call-type events from one HyenaSET HDF5 label file.

    Args:
        label_path: Absolute path of the HDF5 label file.

    Returns:
        list: ``[(class_name, start_sec, end_sec, 1.0), ...]``, all events have
        value=1.0 (confirmed present). An empty annotation file returns an
        empty list; the ``foc`` field in the HDF5 file does not participate in
        label construction.
    """
    with h5py.File(label_path, "r") as h5_file:
        label_dataset = h5_file["lbl"]
        if len(label_dataset) == 0:
            return []
        call_types = label_dataset.asstr()[...].tolist()
        starts = h5_file["start_time_lbl"][...]
        ends = h5_file["end_time_lbl"][...]

    return [(call_type, float(start), float(end), 1.0)
            for call_type, start, end in zip(call_types, starts, ends)]


class _LazyStrongEvents:
    """Lazily read and cache HyenaSET strong annotations by audio relative path."""

    def __init__(self, dataset_dir: str):
        """Initialize the lazy annotation mapping.

        Args:
            dataset_dir: HyenaSET dataset root directory.
        """
        self._dataset_dir = dataset_dir
        self._cache = {}

    def __getitem__(self, audio_path: str) -> list:
        """Read the events for a given audio; raises ``KeyError`` if the label file does not exist."""
        if audio_path in self._cache:
            return self._cache[audio_path]

        label_path = _label_path(self._dataset_dir, audio_path)
        try:
            events = _read_strong_events(label_path)
        except FileNotFoundError:
            raise _MissingAnnotationError(audio_path) from None
        self._cache[audio_path] = events
        return events

    def get(self, audio_path: str, default=None):
        """Return the events for a given audio, or ``default`` if no label exists."""
        try:
            return self[audio_path]
        except _MissingAnnotationError:
            return default

    def __contains__(self, audio_path: str) -> bool:
        """Check whether the audio has a corresponding label file (the read result is cached)."""
        return self.get(audio_path) is not None


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load HyenaSET event-level annotations.

    Args:
        dataset_dir: HyenaSET dataset root directory.

    Returns:
        DatasetAnnotation: ``audio_path`` looks like
        ``wav/24000Hz/{bucket}/{random_name}.wav``, with classes being the ten
        call types.
    """
    return DatasetAnnotation(
        label_kind="multilabel",
        annotation_kind="strong",
        classes=sorted(_CALL_CLASSES),
        strong_events=_LazyStrongEvents(dataset_dir),
    )
