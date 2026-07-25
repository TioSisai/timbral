"""DCASE-2024-Task-5 annotation adapter: per-recording csv files -> multilabel strong.

The training-set CSV's class columns use the ``class_code`` within each
sub-dataset, while the validation set uniformly uses ``Q``, and class names must
be recovered by combining the ``(dataset, recording)`` pair from the validation
class table. A ``POS`` cell becomes an event with value=1.0 for the corresponding
class; a ``UNK`` cell becomes, per class, an event with value=NaN for the
corresponding class (multiple UNK cells on the same row each produce their own
class-identified event, without collapsing); ``NEG`` and unannotated cells
produce no event.
"""

import os

import numpy as np
import pandas as pd

from .base import DatasetAnnotation

_TRAIN_CLASS_CSV = "DCASE2024_task5_training_set_classes.csv"
_VALIDATION_CLASS_CSV = "DCASE2024_task5_validation_set_classes.csv"
PARTITION_SUBSETS = (
    ("Training_Set", ("BV", "HT", "JD", "MT", "WMW")),
    ("Validation_Set", ("HB", "ME", "PB", "PB24", "PW", "RD")),
)


def _build_training_class_names(class_table: pd.DataFrame) -> dict:
    """Build the ``(dataset, class_code) -> official readable class name`` mapping.

    Class identity follows the user-confirmed official ``class_name``;
    consequently WMW/c_4 and WMW/c_5 map to the same
    ``Tachybaptus ruficollis-song`` class.

    Args:
        class_table: The training class table, containing the
            dataset/class_code/class_name columns.

    Returns:
        The official class-name mapping keyed by ``(dataset, class_code)``.
    """
    rows = class_table[["dataset", "class_code", "class_name"]].itertuples(
        index=False, name=None)
    return {(dataset, code): name for dataset, code, name in rows}


def _extract_events(annotation: pd.DataFrame,
                    column_to_class: dict) -> list:
    """Extract ordered tri-state strong events per class from one recording's POS/NEG/UNK matrix.

    Args:
        annotation: The per-event annotation table, whose first three columns
            are the filename, start time, and end time.
        column_to_class: A mapping from class column names to output class names.

    Returns:
        ``[(class_name, start_sec, end_sec, value), ...]``. A POS cell produces
        an event with value=1.0 for the corresponding class; a UNK cell
        produces, per class, an event with value=NaN for the corresponding
        class (multiple UNK cells on the same row each produce their own
        event, without collapsing); NEG and unannotated cells produce no
        event. Events are ordered by CSV row order, and within a row by class
        column order.
    """
    class_columns = list(annotation.columns[3:])
    status = annotation[class_columns].to_numpy(copy=False)

    # The user specified merging same-named class_code entries into one class;
    # after merging, each (row, class) keeps at most one event: all POS
    # entries are written first, then UNK entries via setdefault, so on key
    # collision POS (value=1.0) wins, consistent with the pipeline-wide
    # convention 1 > NaN > 0; when duplicates share the same state (both POS
    # or both UNK), the first-occurring column is kept.
    events = {}
    for state, value in (("POS", 1.0), ("UNK", float("nan"))):
        state_rows, state_columns = np.nonzero(status == state)
        for row, column in zip(state_rows.tolist(), state_columns.tolist()):
            events.setdefault(
                (row, column_to_class[class_columns[column]]),
                (column, value))

    # Stable sort by (row, retained column) to restore CSV row order, with
    # class column order within each row.
    event_rows = np.fromiter((row for row, _ in events), dtype=np.int64,
                             count=len(events))
    event_columns = np.fromiter((column for column, _ in events.values()),
                                dtype=np.int64, count=len(events))
    order = np.lexsort((event_columns, event_rows))

    event_items = list(events.items())
    starts = annotation["Starttime"].to_numpy(dtype=float, copy=False)
    ends = annotation["Endtime"].to_numpy(dtype=float, copy=False)
    return [(class_name, float(starts[row]), float(ends[row]), value)
            for (row, class_name), (_, value)
            in (event_items[index] for index in order)]


def _annotation_csvs(dataset_dir: str, partition: str, subset: str):
    """Iterate a sub-dataset's adjacent annotation CSV paths in filename order.

    Args:
        dataset_dir: Dataset root directory.
        partition: ``Training_Set`` or ``Validation_Set``.
        subset: Sub-dataset short name, e.g. ``BV`` or ``PB24``.

    Yields:
        The absolute path of the annotation CSV alongside the audio.
    """
    annotation_dir = os.path.join(dataset_dir, "Development_Set", partition,
                                  subset)
    with os.scandir(annotation_dir) as entries:
        csv_names = sorted(entry.name for entry in entries
                           if entry.is_file() and entry.name.endswith(".csv"))
    for csv_name in csv_names:
        yield os.path.join(annotation_dir, csv_name)


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load DCASE-2024-Task-5 event-level annotations.

    ``audio_path`` looks like
    ``Development_Set/{Training_Set|Validation_Set}/{subset}/{stem}.wav``.
    The path is generated from the annotation CSV's location and file stem,
    rather than the ``Audiofilename`` field, which is erroneous in some
    sub-datasets.

    Args:
        dataset_dir: DCASE-2024-Task-5 dataset root directory.

    Returns:
        A unified annotation object containing the full official class set and
        per-audio events.
    """
    training_classes = pd.read_csv(os.path.join(dataset_dir, _TRAIN_CLASS_CSV))
    validation_classes = pd.read_csv(
        os.path.join(dataset_dir, _VALIDATION_CLASS_CSV))
    training_class_names = _build_training_class_names(training_classes)
    validation_table = validation_classes[
        ["dataset", "recording", "class_name"]]
    validation_rows = validation_table.itertuples(index=False, name=None)
    validation_class_names = {
        (dataset, recording): class_name
        for dataset, recording, class_name in validation_rows
    }

    classes = sorted(set(training_class_names.values())
                     | set(validation_class_names.values()))
    strong_events = {}
    for partition, subsets in PARTITION_SUBSETS:
        for subset in subsets:
            for csv_path in _annotation_csvs(dataset_dir, partition, subset):
                annotation = pd.read_csv(csv_path)
                stem = os.path.splitext(os.path.basename(csv_path))[0]
                if partition == "Training_Set":
                    column_to_class = {
                        code: training_class_names[(subset, code)]
                        for code in annotation.columns[3:]
                    }
                else:
                    column_to_class = {
                        code: validation_class_names[(subset, stem)]
                        for code in annotation.columns[3:]
                    }
                audio_path = "/".join(
                    ("Development_Set", partition, subset, f"{stem}.wav"))
                strong_events[audio_path] = _extract_events(
                    annotation, column_to_class)

    return DatasetAnnotation(label_kind="multilabel", annotation_kind="strong",
                             classes=classes, strong_events=strong_events)
