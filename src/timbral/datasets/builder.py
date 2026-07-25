"""Main data preparation orchestration: metadata construction -> HF map segmentation -> cache persistence."""

import dataclasses
import json
import os
import posixpath

import pandas as pd
from datasets import (ClassLabel, Dataset, DatasetDict, Features, Sequence,
                      Value)
from datasets.fingerprint import Hasher

from timbral.storage import map_tmp_context, resolve_cache_target, write_json

from . import adapters, audio_io, labels, segmentation, split_io
from .config import PrepConfig


def prepare_dataset(config: PrepConfig) -> str:
    """Run one data preparation pass and return cache_dir (skipped directly if it already exists and overwrite is not set).

    Pipeline: annotation loading -> metadata construction (split parsing +
    annotation join) -> per-split map segmentation -> save_to_disk directly
    to cache_dir. For S3 output, the map's temporary Arrow files are placed
    in ``$TMPDIR``, while the final artifact is written directly to the S3
    target path.
    """
    # Local and S3 are both handled uniformly through the fsspec
    # filesystem, eliminating dual branches; only the map temp directory
    # and save_to_disk's storage_options retain the S3-specific difference.
    cache_fs, cache_path, storage_options = resolve_cache_target(
        config.cache_dir)
    prep_config_path = posixpath.join(cache_path, "prep_config.json")
    if cache_fs.exists(cache_path):
        if cache_fs.isfile(prep_config_path):
            if not config.overwrite:
                # Prevents silently reusing a stale artifact when cache_dir
                # is explicitly given and parameters have changed: reuse is
                # only allowed if the hash matches.
                with cache_fs.open(prep_config_path, "r",
                                   encoding="utf-8") as f:
                    cached_hash = json.load(f)["config_hash"]
                if cached_hash != config.config_hash:
                    raise ValueError(
                        f"cache_dir already exists but config_hash does not "
                        f"match (old {cached_hash} / new "
                        f"{config.config_hash}); parameters or the split "
                        f"have changed. Pass --overwrite to rebuild: "
                        f"{config.cache_dir}")
                print(f"Cache already exists, skipping: {config.cache_dir}")
                return config.cache_dir
        elif not config.overwrite:
            raise ValueError(
                "cache_dir already exists but is missing "
                "prep_config.json, which may indicate an incompletely "
                f"written cache; pass --overwrite to rebuild: "
                f"{config.cache_dir}")
        cache_fs.rm(cache_path, recursive=True)

    annotation = adapters.load_annotation(config.dataset_name, config.dataset_dir)
    if config.label_type == "strong" and annotation.annotation_kind == "weak":
        raise ValueError(
            f"{config.dataset_name} only has weak annotations; cannot "
            f"produce label_type=strong")
    if annotation.annotation_kind == "strong":
        out_mode = "strong" if config.label_type == "strong" else "strong_to_weak"
    else:
        out_mode = annotation.label_kind
    label_index = labels.build_label_index(annotation.classes)

    splits_meta = _build_metadata(config, annotation, label_index)
    features = _build_features(config, label_index, out_mode)

    map_fn = _make_map_fn(config, out_mode, num_classes=len(label_index))
    with map_tmp_context(config.cache_dir, config.config_hash) as map_tmp_dir:
        processed = {}
        for split, df in splits_meta.items():
            ds = Dataset.from_pandas(df, preserve_index=False)
            # Cap the process count by the split size; in datasets 5.0.0,
            # num_proc=0 is equivalent to None (main process), while
            # num_proc=1 spawns one worker process. Explicitly setting
            # new_fingerprint makes the fingerprint vary only with
            # config_hash, decoupling it from the closure code, so that
            # state.json stays byte-stable across cache rebuilds; the split
            # name is hashed to a fixed length so the fingerprint length is
            # independent of the split name (when num_proc >= 1, map also
            # appends a shard suffix, and the total must stay within
            # datasets' 64-character limit).
            num_proc = min(config.num_proc, len(ds))
            processed[split] = ds.map(
                map_fn,
                batched=True,
                batch_size=config.batch_size,
                writer_batch_size=config.batch_size,
                num_proc=num_proc,
                remove_columns=ds.column_names,
                features=features,
                cache_file_name=os.path.join(map_tmp_dir, f"map-{split}.arrow"),
                new_fingerprint=(
                    f"raw-{config.config_hash}-{Hasher.hash(split)}"),
                desc=f"Segmenting {split}",
            )
        counts = {split: len(ds) for split, ds in processed.items()}
        DatasetDict(processed).save_to_disk(
            config.cache_dir, storage_options=storage_options)
        del processed

    write_json(cache_fs, posixpath.join(cache_path, "label_index.json"),
               label_index)
    # Write the completion marker last, to avoid a subsequent run
    # mistakenly hitting a cache that was not fully written.
    write_json(cache_fs, prep_config_path, dataclasses.asdict(config))

    print(f"Done: {config.cache_dir}")
    for split, n in counts.items():
        print(f"  {split}: {n} segments")
    return config.cache_dir


def _build_metadata(config: PrepConfig, annotation, label_index: dict) -> dict:
    """Build a metadata DataFrame per split (one row per split entry, annotations already joined).

    Columns: audio_abspath (absolute path, for loading) / audio_path /
    audio_id / entry_start / entry_end, plus, depending on the annotation
    form: label_idx (multiclass) | label_idxs (multilabel) |
    ev_target/ev_start/ev_end/ev_value (strong; target is always a valid
    class index, value is the tri-state annotation value 1.0/NaN).
    """
    splits = split_io.load_split_json(config.split_json)
    all_paths = sorted({e.audio_path for entries in splits.values()
                        for e in entries})
    audio_ids = {p: i for i, p in enumerate(all_paths)}

    result = {}
    for split, entries in splits.items():
        if not entries:
            # In datasets 5.0.0, map on an empty dataset returns early
            # without applying features=; an empty split would produce a
            # zero-column split, breaking schema consistency, so we
            # intercept it here.
            raise ValueError(f"Split {split} is empty; cannot build a "
                             f"schema-consistent cache. Please check "
                             f"split_json: {config.split_json}")
        rows = {
            "audio_abspath": [os.path.join(config.dataset_dir, e.audio_path)
                              for e in entries],
            "audio_path": [e.audio_path for e in entries],
            "audio_id": [audio_ids[e.audio_path] for e in entries],
            "entry_start": [e.start for e in entries],
            "entry_end": [e.end for e in entries],
        }
        if annotation.annotation_kind == "weak":
            missing = [e.audio_path for e in entries
                       if e.audio_path not in annotation.weak_labels]
            if missing:
                raise ValueError(f"{len(missing)} audio file(s) in {split} "
                                 f"are missing annotations, e.g.: "
                                 f"{missing[:5]}")
            if annotation.label_kind == "multiclass":
                rows["label_idx"] = [
                    label_index[annotation.weak_labels[e.audio_path]]
                    for e in entries]
            else:
                rows["label_idxs"] = [
                    sorted(label_index[c]
                           for c in annotation.weak_labels[e.audio_path])
                    for e in entries]
        else:
            ev_t, ev_s, ev_e, ev_v = [], [], [], []
            for e in entries:
                events = annotation.strong_events.get(e.audio_path, [])
                ev_t.append([label_index[c] for c, _, _, _ in events])
                ev_s.append([s for _, s, _, _ in events])
                ev_e.append([t for _, _, t, _ in events])
                ev_v.append([v for _, _, _, v in events])
            rows["ev_target"], rows["ev_start"] = ev_t, ev_s
            rows["ev_end"], rows["ev_value"] = ev_e, ev_v
        result[split] = pd.DataFrame(rows)
    return result


def _build_features(config: PrepConfig, label_index: dict,
                    out_mode: str) -> Features:
    """Build an explicit features schema according to the output mode."""
    seg_len = round(config.seg_sec * config.sr)
    raw_feature = (Sequence(Value("float32"), length=seg_len) if config.mono
                   else Sequence(Sequence(Value("float32"))))
    if out_mode == "multiclass":
        label_feature = ClassLabel(names=list(label_index))
    elif out_mode in ("multilabel", "strong_to_weak"):
        label_feature = Sequence(Value("float32"), length=len(label_index))
    else:
        # strong: target is always a valid ClassLabel, value is the
        # tri-state annotation value (1.0/NaN)
        label_feature = [{"target": ClassLabel(names=list(label_index)),
                          "start": Value("float64"), "end": Value("float64"),
                          "value": Value("float32")}]
    return Features({
        "audio_path": Value("string"),
        "audio_id": Value("int64"),
        "segment_id": Value("int64"),
        "raw": raw_feature,
        "sr": Value("int64"),
        "start": Value("float64"),
        "end": Value("float64"),
        "valid_sec": Value("float64"),
        "label": label_feature,
    })


def _make_map_fn(config: PrepConfig, out_mode: str, num_classes: int):
    """Build a batched map function: per recording, probe duration -> segment -> load -> generate segment-level labels.

    Supports ragged batches (output row count is independent of input row
    count); segment_id is a 0-based index within the entry.
    """
    seg_len = round(config.seg_sec * config.sr)

    def process_batch(batch):
        out = {key: [] for key in ("audio_path", "audio_id", "segment_id", "raw",
                                   "sr", "start", "end", "valid_sec", "label")}
        for i in range(len(batch["audio_path"])):
            path = batch["audio_abspath"][i]
            duration = audio_io.probe_duration(path)
            plan = segmentation.plan_segments(
                batch["entry_start"][i], batch["entry_end"][i], duration,
                config.seg_sec, config.hop_sec, config.tol_sec)
            if len(plan) == 0:
                # An empty plan occurs only when the measured audio
                # duration does not exceed the entry start (a
                # corrupted/truncated file, or a mismatch between the
                # split and the audio); silently dropping it would violate
                # the invariant that "every entry produces at least one
                # segment", so this must fail explicitly.
                raise ValueError(
                    f"Entry produced no segments: {path} (entry_start="
                    f"{batch['entry_start'][i]}, measured duration="
                    f"{duration})")
            for segment_id, (seg_start, valid_sec) in enumerate(plan.tolist()):
                raw = audio_io.load_segment(path, seg_start, valid_sec,
                                            config.sr, config.mono, seg_len)
                if out_mode == "multiclass":
                    label = batch["label_idx"][i]
                elif out_mode == "multilabel":
                    label = labels.encode_multihot(batch["label_idxs"][i],
                                                   num_classes)
                elif out_mode == "strong_to_weak":
                    label = labels.clipped_events_multihot(
                        batch["ev_target"][i], batch["ev_start"][i],
                        batch["ev_end"][i], batch["ev_value"][i], seg_start,
                        seg_start + valid_sec, num_classes)
                else:
                    label = labels.clip_events(
                        batch["ev_target"][i], batch["ev_start"][i],
                        batch["ev_end"][i], batch["ev_value"][i], seg_start,
                        seg_start + valid_sec)
                out["audio_path"].append(batch["audio_path"][i])
                out["audio_id"].append(batch["audio_id"][i])
                out["segment_id"].append(segment_id)
                out["raw"].append(raw)
                out["sr"].append(config.sr)
                out["start"].append(seg_start)
                out["end"].append(seg_start + valid_sec)
                out["valid_sec"].append(valid_sec)
                out["label"].append(label)
        return out

    return process_batch
