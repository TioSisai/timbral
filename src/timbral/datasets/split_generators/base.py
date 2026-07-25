"""Shared utility library for generating audio classification dataset split JSON files.

All dataset default split generator modules must import this module to ensure:
  1. The format/fields/inf encoding of the output JSON are fully consistent;
  2. The stratified (weak single-label) and greedy duration (strong multi-label) split
     algorithms are fully consistent;
  3. The copy rule for missing validation/test is consistent.

Conventions:
  - split.json structure: {"train": [entry, ...], "validation": [...], "test": [...]}
  - entry structure: {"audio_path": <path relative to dataset root, posix style>,
    "start": 0.0, "end": "inf"}, where the end of a full-length segment is written as
    the standard JSON string "inf" (rather than the non-standard Infinity); the
    subsequent raw_prep step uniformly restores it to a numeric value via
    float(e["end"]) (float("inf")==inf, finite values pass through as-is).
  - split.json contains no label; the label is obtained during raw_prep by joining
    audio_path back to the original annotations.
  - When validation or test is missing, the copy rule fills it in from an existing
    split (see fill_missing_splits).
"""

import json
import math
import os
import random
from collections import defaultdict

INF = float("inf")
SPLIT_NAMES = ("train", "validation", "test")


def make_entry(audio_path, start=0.0, end=INF):
    """Construct a single split entry. audio_path is normalized to a posix relative
    path (no leading slash)."""
    ap = audio_path.replace(os.sep, "/").lstrip("/")
    return {"audio_path": ap, "start": float(start), "end": float(end)}


def stratified_split_single_label(item_ids, labels, ratios=(0.8, 0.1, 0.1), seed=0):
    """Stratified random split for weak single-label datasets (pure random, no
    co-source grouping).

    Each class is split independently according to ratios (each class has its own
    independent rng, so classes don't affect each other); within each group, items
    are shuffled with a fixed seed and then sliced by ratio, which precisely
    preserves the train/val/test proportions for every class. Pure Python,
    deterministic.

    Limitation: at rounding boundaries, extremely small classes have no guaranteed
    val/test coverage — classes with n<=2 go entirely to train, and classes with
    n<=5 may end up with an empty validation set (banker's rounding); if a long-tail
    dataset needs to be onboarded, add a coverage fallback similar to
    greedy_multilabel_duration_split.

    Args:
        item_ids: Iterable of unique ids for each sample (used to place it in a
            split, typically audio_path or a key).
        labels:   Same length as item_ids; the single class label of each sample.
        ratios:   (train, val, test) ratios, should sum to 1.0.
        seed:     Random seed.

    Returns:
        dict: {"train": [id, ...], "validation": [...], "test": [...]}
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1"
    groups = defaultdict(list)
    for iid, lab in zip(item_ids, labels):
        groups[lab].append(iid)
    result = {s: [] for s in SPLIT_NAMES}
    # The processing order of classes only determines the ordering within result
    # (deterministic); it does not affect any sample's assignment
    for lab in sorted(groups, key=lambda l: (len(groups[l]), str(l))):
        ids = sorted(groups[lab], key=str)  # deterministic
        rng = random.Random("{}:{}".format(seed, lab))
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        # Fix boundaries: ensure non-negative and sum to n
        n_train = min(max(n_train, 0), n)
        n_val = min(max(n_val, 0), n - n_train)
        n_test = n - n_train - n_val
        result["train"].extend(ids[:n_train])
        result["validation"].extend(ids[n_train:n_train + n_val])
        result["test"].extend(ids[n_train + n_val:])
    return result


def greedy_multilabel_duration_split(file_ids, file_labelsets, file_weights,
                                     ratios=(0.8, 0.1, 0.1)):
    """Greedy duration split for strong multi-label datasets (used for DataSED),
    with rare-class coverage guarantees.

    An improved iterative stratified greedy algorithm (in the spirit of Sechidis),
    weighted by duration:
      - Processing order: files "containing the rarest class" are processed first,
        so rare classes get placed preferentially;
      - When placing each file, the primary key ranks splits by the duration
        deficit of the file's [rarest class] (rather than the sum of all class
        deficits, which would let rare-class files get swept into train by
        large-class deficits); the secondary key is the sum of deficits of the
        remaining classes, and the tertiary key is the overall duration deficit;
      - Coverage fallback: for classes with >=3 files, guarantee at least 1 file
        in each of validation and test (if not met, migrate a file containing
        that class from train; only migrate out of train so as not to break
        coverage already established in other splits).
    Each file goes entirely into a single split (files are not segmented). Pure
    Python, deterministic.

    Args:
        file_ids:       Iterable of unique ids for each file.
        file_labelsets: dict file_id -> set of classes appearing in that file.
        file_weights:   dict file_id -> duration of that file (seconds, float).
        ratios:         (train, val, test) duration ratios.

    Returns:
        dict: {"train": [file_id, ...], "validation": [...], "test": [...]}
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1"
    file_ids = list(file_ids)
    total_w = sum(file_weights[f] for f in file_ids)
    desired = {s: ratios[i] * total_w for i, s in enumerate(SPLIT_NAMES)}
    class_w = defaultdict(float)
    class_files = defaultdict(list)
    for f in file_ids:
        for c in file_labelsets.get(f, ()):
            class_w[c] += file_weights[f]
            class_files[c].append(f)
    class_desired = {c: {s: ratios[i] * class_w[c] for i, s in enumerate(SPLIT_NAMES)}
                     for c in class_w}
    cur = {s: 0.0 for s in SPLIT_NAMES}
    class_cur = {c: {s: 0.0 for s in SPLIT_NAMES} for c in class_w}

    def rarity(f):
        labs = file_labelsets.get(f, ())
        return min((class_w[c] for c in labs), default=math.inf)

    order = sorted(file_ids, key=lambda f: (rarity(f), -file_weights[f], str(f)))
    result = {s: [] for s in SPLIT_NAMES}
    where = {}
    for f in order:
        labs = file_labelsets.get(f, ())
        w = file_weights[f]
        # The rarest class in this file (smallest total class duration; None if unlabeled)
        rare_c = min(labs, key=lambda c: (class_w[c], str(c))) if labs else None
        best, best_score = None, None
        for s in SPLIT_NAMES:
            rare_deficit = (class_desired[rare_c][s] - class_cur[rare_c][s]) if rare_c is not None else 0.0
            other_deficit = sum((class_desired[c][s] - class_cur[c][s]) for c in labs if c != rare_c)
            overall_deficit = desired[s] - cur[s]
            score = (rare_deficit, other_deficit, overall_deficit)
            if best_score is None or score > best_score:
                best_score, best = score, s
        result[best].append(f)
        where[f] = best
        cur[best] += w
        for c in labs:
            class_cur[c][best] += w

    # Coverage fallback: for classes with >=3 files, guarantee >=1 file each in
    # validation and test (only migrate out of train)
    for c in sorted(class_files, key=lambda c: (len(class_files[c]), str(c))):
        files_c = class_files[c]
        if len(files_c) < 3:
            continue
        for tgt in ("validation", "test"):
            if any(where[f] == tgt for f in files_c):
                continue
            cands = [f for f in files_c if where[f] == "train"]
            if not cands:
                continue
            # Migrate out the one causing the least disruption: fewest other
            # classes, shortest duration, deterministic
            f = min(cands, key=lambda f: (len(file_labelsets.get(f, ())), file_weights[f], str(f)))
            result["train"].remove(f)
            result[tgt].append(f)
            where[f] = tgt
    return result


def fill_missing_splits(splits):
    """Copy rule: when validation or test is missing, fill it in by copying the
    other existing split.

    Rule (as specified by the user): for any split missing validation or test,
    copy the existing one into the missing one. This only handles the case where
    exactly one of validation/test is missing:
      - validation present, test absent  -> test = deep copy of validation;
      - test present, validation absent  -> validation = deep copy of test.
    train must be present. Returns whether a copy occurred and in which direction
    (for logging).

    Args:
        splits: dict, containing at least "train"; validation/test may be missing
            or an empty list.

    Returns:
        (splits, note): note is a str describing the copy situation (an empty
        string if no copy occurred).
    """
    assert splits.get("train"), "train split must not be empty"
    has_val = bool(splits.get("validation"))
    has_test = bool(splits.get("test"))
    note = ""
    if has_val and not has_test:
        splits["test"] = [dict(e) for e in splits["validation"]]
        note = "test is a copy of validation (no official test set)"
    elif has_test and not has_val:
        splits["validation"] = [dict(e) for e in splits["test"]]
        note = "validation is a copy of test (no official validation set)"
    splits.setdefault("validation", [])
    splits.setdefault("test", [])
    return splits, note


def write_split_json(out_path, splits):
    """Write the splits out as default.json (indent=2 for human readability).

    inf/-inf are written as the standard JSON strings "inf"/"-inf" (readable by
    any parser, unlike the non-standard Infinity), while finite values remain
    numeric; when raw_prep reads them back it can uniformly restore them via
    float(e["end"]) (float("inf")==inf). Writes to a process-exclusive temporary
    file first, then atomically replaces the target, so no half-written file ever
    appears at the target path if generation is interrupted or run concurrently.
    """
    out_path = os.fspath(out_path)
    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def _enc(v):
        v = float(v)
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v

    # Normalize field order and types
    norm = {}
    for s in SPLIT_NAMES:
        norm[s] = [
            {"audio_path": e["audio_path"], "start": _enc(e["start"]), "end": _enc(e["end"])}
            for e in splits.get(s, [])
        ]
    tmp_path = "{}.tmp.{}".format(out_path, os.getpid())
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(norm, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)
    return {s: len(norm[s]) for s in SPLIT_NAMES}


def verify_split_json(out_path, require_disjoint_paths=True, allow_val_test_copy=True):
    """Read-back verification: checks structure/fields/types are correct;
    optionally checks whether audio_path is mutually exclusive across splits.

    When test is a copy of validation (copy rule), validation and test are
    allowed to be fully identical, so require_disjoint_paths only verifies that
    train is disjoint from (val∪test), and that val/test are also disjoint from
    each other if they are not equal. Returns a verification summary dict.
    """
    with open(out_path, encoding="utf-8") as f:
        obj = json.load(f)
    assert set(obj.keys()) >= {"train", "validation", "test"}, "missing split key(s)"
    counts = {}
    paths = {}
    for s in SPLIT_NAMES:
        assert isinstance(obj[s], list), "{} is not a list".format(s)
        ps = []
        for e in obj[s]:
            assert set(e.keys()) == {"audio_path", "start", "end"}, "field mismatch: {}".format(e)
            assert isinstance(e["audio_path"], str) and e["audio_path"]
            assert not e["audio_path"].startswith("/")
            # start/end are numeric, or the string encoding of inf/-inf ("inf"/"-inf")
            assert isinstance(e["start"], (int, float)) or e["start"] in ("inf", "-inf")
            assert isinstance(e["end"], (int, float)) or e["end"] in ("inf", "-inf")
            ps.append(e["audio_path"])
        counts[s] = len(ps)
        paths[s] = ps
    info = {"counts": counts}
    if require_disjoint_paths:
        tr, va, te = set(paths["train"]), set(paths["validation"]), set(paths["test"])
        info["train_val_overlap"] = len(tr & va)
        info["train_test_overlap"] = len(tr & te)
        val_test_equal = paths["validation"] == paths["test"]
        info["val_test_equal(copy)"] = val_test_equal
        info["train_val_disjoint_ok"] = (len(tr & va) == 0)
        info["train_test_disjoint_ok"] = (len(tr & te) == 0)
        if not val_test_equal:
            info["val_test_overlap"] = len(va & te)
    return info
