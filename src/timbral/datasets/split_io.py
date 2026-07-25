"""Reading and structural validation of split.json."""

import dataclasses
import json

SPLIT_NAMES = ("train", "validation", "test")


@dataclasses.dataclass(frozen=True)
class SplitEntry:
    """A single split record: the audio's relative path and the time region it falls into within this split (0/inf for the whole recording)."""

    audio_path: str
    start: float
    end: float


def load_split_json(path: str) -> dict:
    """Read and validate split.json, returning {split_name: [SplitEntry, ...]}.

    Time fields accept either the string "inf" or a numeric value
    (float("inf") can parse the string directly).

    Validation rules:
      - All three keys train/validation/test must be present;
      - start < end;
      - The same audio_path may appear at most once within a given split
        (guaranteeing segment_id can be uniquely numbered by its index
        within the entry);
      - For the same audio shared across splits, the regions must either
        be exactly equal (a copy case, valid) or mutually disjoint
        (left-closed, right-open, consistent with the segmentation
        implementation); partial overlap raises an error.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    missing = [s for s in SPLIT_NAMES if s not in raw]
    if missing:
        raise ValueError(f"split.json is missing split(s): {missing}")
    splits = {}
    for split in SPLIT_NAMES:
        entries = [SplitEntry(e["audio_path"], float(e["start"]), float(e["end"]))
                   for e in raw[split]]
        seen = set()
        for e in entries:
            if e.audio_path in seen:
                raise ValueError(f"Duplicate audio_path in {split}: {e.audio_path}")
            seen.add(e.audio_path)
            if not e.start < e.end:
                raise ValueError(f"Invalid interval for {split}/{e.audio_path}: "
                                 f"start={e.start} end={e.end}")
        splits[split] = entries
    _check_cross_split_overlap(splits)
    return splits


def _check_cross_split_overlap(splits: dict) -> None:
    """For the same audio shared across splits, the regions are valid only if exactly equal (copy) or mutually disjoint (left-closed, right-open)."""
    intervals = {}
    for split, entries in splits.items():
        for e in entries:
            intervals.setdefault(e.audio_path, []).append((e.start, e.end, split))
    for audio_path, ivs in intervals.items():
        for i in range(len(ivs)):
            for j in range(i + 1, len(ivs)):
                (s1, e1, sp1), (s2, e2, sp2) = ivs[i], ivs[j]
                if (s1, e1) == (s2, e2):
                    continue
                if s1 < e2 and s2 < e1:
                    raise ValueError(f"{audio_path} has partially overlapping "
                                     f"intervals between {sp1}/{sp2}: "
                                     f"[{s1}, {e1}) and [{s2}, {e2})")
