"""Unified structure definitions for dataset annotation adapters."""

import dataclasses
from typing import Protocol


class LabelMapping(Protocol):
    """The minimal query protocol for ``weak_labels``/``strong_events`` (a dict naturally satisfies it).

    The builder's weak path uses ``in`` and ``[]``, while the strong path uses
    ``get``; the protocol uniformly requires all three so that the two kinds of
    mapping are interchangeable, and so that custom lazy mappings (e.g. resolving
    by YTID or lazily reading per file) don't fall back to implicit behavior such
    as the legacy iteration protocol due to missing methods.
    """

    def __contains__(self, audio_path: str) -> bool: ...

    def __getitem__(self, audio_path: str): ...

    def get(self, audio_path: str, default=None): ...


@dataclasses.dataclass(frozen=True)
class DatasetAnnotation:
    """The full set of annotations for one dataset (independent of split).

    Attributes:
        label_kind: The weak aggregation form, 'multiclass' or 'multilabel'.
        annotation_kind: The raw annotation granularity, 'weak' or 'strong'.
        classes: The full set of class names (the index is generated in
            lexicographic order by labels.build_label_index).
        weak_labels: Non-None when annotation_kind='weak';
            {audio_path: class_name} for multiclass,
            {audio_path: [class_name, ...]} for multilabel.
        strong_events: Non-None when annotation_kind='strong';
            {audio_path: [(class_name, start_sec, end_sec, value), ...]},
            where class_name is always a valid class name; value is one of three
            annotation states: 1.0 = confirmed present (POS), float("nan") =
            uncertain (UNK); NEG and unannotated cases never produce an event.
    """

    label_kind: str
    annotation_kind: str
    classes: list
    weak_labels: LabelMapping | None = None
    strong_events: LabelMapping | None = None
