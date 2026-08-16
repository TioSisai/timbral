"""Main embedding extraction orchestration: raw cache loading ->
Transform/Encoder batched forward pass -> embedding cache written to disk.
"""

import dataclasses
import os
import posixpath

import numpy as np
import torch
from datasets import (Array2D, ClassLabel, DatasetDict, Features, Sequence,
                      Value, load_from_disk)
from datasets.fingerprint import Hasher
from torch.nn import functional as F

from timbral.models import create_model
from timbral.storage import map_tmp_context, resolve_cache_target, write_json

from .config import EmbPrepConfig
from .labels import clip_multihot, frame_multihot

# Provenance fields passed straight through unchanged at the Arrow layer (not
# part of the map return value; kept by simply not being removed, so float64
# fields incur zero precision loss; the raw waveform and sample rate sr are
# excluded from the output)
_PASSTHROUGH_KEYS = ("audio_path", "audio_id", "segment_id", "start", "end",
                     "valid_sec")


def prepare_embeddings(config: EmbPrepConfig) -> str:
    """Run one embedding extraction pass and return the output directory
    (skipped if the completion marker already exists and overwrite is not
    set).

    Flow: raw cache loading -> label mode inference -> create_model builds
    the Transform and Encoder and moves them to the target device -> (for
    frame granularity) a full-window probe of the constant frame count ->
    per-split batched map forward pass -> save_to_disk -> write
    label_index.json -> finally write emb_config.json (which doubles as the
    completion marker). For local output, the temporary Arrow files produced
    by map are placed in a temporary directory next to the output directory;
    for S3 output they are placed under ``$TMPDIR`` and the final artifacts
    are written directly to the S3 target path; temporary directories are
    cleaned up when done in either case.

    Args:
        config: The full configuration produced by resolve_config.

    Returns:
        The final three-level output directory (same as config.output_dir).

    Raises:
        ValueError: The output directory already exists but is missing the
            completion marker and overwrite was not specified, or the S3
            output root is a bucket-root path.
    """
    out_dir = config.output_dir
    # Local and S3 targets are both handled through the fsspec filesystem,
    # which eliminates a dual code path; only the map temporary directory and
    # the storage_options passed to save_to_disk retain S3-specific
    # differences.
    out_fs, out_path, storage_options = resolve_cache_target(out_dir)
    emb_config_path = posixpath.join(out_path, "emb_config.json")
    if out_fs.exists(out_path):
        if out_fs.isfile(emb_config_path):
            if not config.overwrite:
                print(f"Embedding cache already exists, skipping: {out_dir}")
                return out_dir
        elif not config.overwrite:
            raise ValueError(
                "Output directory already exists but is missing "
                "emb_config.json, which may indicate an incompletely "
                f"written cache; pass --overwrite to rebuild it: {out_dir}")
        out_fs.rm(out_path, recursive=True)

    dataset_dict = load_from_disk(config.cache_dir)
    first_split = next(iter(dataset_dict.values()))
    label_mode = _infer_label_mode(first_split.features)

    device = _resolve_device(config.device)
    transform, encoder = create_model(
        config.model_name, granularity=config.granularity,
        pretrained_dir=config.pretrained_dir, **config.model_kwargs)
    transform = transform.to(device).eval()
    encoder = encoder.to(device).eval()
    if config.sr < transform.target_sample_rate:
        print(f"Note: raw cache sample rate {config.sr} Hz is below the "
              f"model's native sample rate {transform.target_sample_rate} "
              "Hz; silent upsampling cannot recover already-lost high "
              "frequencies")
    num_frames = (_probe_frame_count(transform, encoder, config.sr,
                                     config.seg_sec)
                  if config.granularity == "frame" else None)

    map_fn = _make_map_fn(config, transform, encoder, label_mode, num_frames)
    # Only drop columns that are excluded from the output or replaced/rebuilt:
    # raw/sr are always dropped; label is dropped only when it gets
    # transformed (clip aggregation or frame labels for a strong cache); for
    # a weak cache, label passes through unchanged at both granularities.
    removed_columns = ["raw", "sr"]
    if label_mode == "strong":
        removed_columns.append("label")
    with map_tmp_context(out_dir, config.emb_hash) as map_tmp_dir:
        processed = {}
        for split, ds in dataset_dict.items():
            # num_proc is intentionally omitted (in datasets 5.0.0, 1 spawns
            # a real subprocess, while only None runs in the main process);
            # new_fingerprint is set explicitly to avoid a dill hash of the
            # model weights captured in the closure; the split name is
            # reduced to a fixed-length hash so the fingerprint length is
            # independent of the split name (it must stay within datasets'
            # 64-character limit).
            # Batches are taken in numpy format to avoid per-element boxing
            # (map formats all columns; the columns= restriction does not
            # take effect inside map); the format is reset before writing to
            # disk so the numpy format does not get written into state.json
            # by save_to_disk and change cache-loading behavior.
            processed[split] = ds.with_format("numpy").map(
                map_fn,
                batched=True,
                batch_size=config.batch_size,
                # At frame granularity each row is on the order of MB, so the
                # writer batch follows the forward-pass batch to bound memory
                # usage; at clip granularity each row is only ~D×4B, so the
                # writer batch is decoupled and made larger to avoid a large
                # number of small record batches slowing down writes and
                # downstream loading.
                writer_batch_size=(config.batch_size
                                   if config.granularity == "frame" else 1000),
                remove_columns=removed_columns,
                features=_build_features(ds.features, encoder.embedding_dim,
                                         label_mode, len(config.label_index),
                                         config.granularity, num_frames),
                cache_file_name=os.path.join(map_tmp_dir, f"map-{split}.arrow"),
                new_fingerprint=(
                    f"emb-{config.emb_hash}-{Hasher.hash(split)}"),
                desc=f"Embedding extraction {split}",
            ).with_format(None)
        counts = {split: len(ds) for split, ds in processed.items()}
        DatasetDict(processed).save_to_disk(out_dir,
                                            storage_options=storage_options)
        del processed

    with open(os.path.join(config.cache_dir, "label_index.json"),
              encoding="utf-8") as f:
        label_index_text = f.read()
    with out_fs.open(posixpath.join(out_path, "label_index.json"),
                     "w", encoding="utf-8") as f:
        f.write(label_index_text)
    # label_index has already been written to disk as a separate sidecar
    # file, so it is not duplicated in the config snapshot.
    snapshot = dataclasses.asdict(config)
    del snapshot["label_index"]
    # Write the completion marker last, so a subsequent run does not
    # mistakenly match an incompletely written cache.
    write_json(out_fs, emb_config_path, snapshot)

    print(f"Done: {out_dir}")
    for split, n in counts.items():
        print(f"  {split}: {n} segments")
    return out_dir


def _resolve_device(device: str) -> torch.device:
    """Resolve the target device string: "auto" selects cuda > mps > cpu
    automatically.
    """
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _probe_frame_count(transform, encoder, sample_rate: int,
                       seg_sec: float) -> int:
    """Probe the constant frame count T_full for frame granularity across
    the whole dataset via a full-window dummy forward pass.

    The physical slice window in the raw cache is always seg_sec, each row's
    valid_sec is <= seg_sec, and the Encoder's output frame count is
    non-decreasing in the valid duration; hence the frame count from a
    full-window forward pass is the maximum frame count over the whole
    dataset. Every batch is padded uniformly to this value, so the output is
    independent of batch_size.
    """
    waveform = torch.zeros((1, round(seg_sec * sample_rate)),
                           dtype=torch.float32)
    valid_seconds = torch.tensor([seg_sec], dtype=torch.float32)
    with torch.inference_mode():
        features = transform(waveform, sample_rate=sample_rate,
                             valid_seconds=valid_seconds)
        out = encoder(features.pop("input_features"), **features)
    return out["embedding"].shape[1]


def _infer_label_mode(features: Features) -> str:
    """Infer the label mode from the cache's features: multiclass /
    multilabel / strong.

    ClassLabel -> multiclass; a fixed-length float32 sequence -> multilabel
    (including the tri-state multi-hot produced by strong-to-weak downgrade
    aggregation); anything else (a nested event list) -> strong.
    """
    label_feature = features["label"]
    if isinstance(label_feature, ClassLabel):
        return "multiclass"
    if (isinstance(label_feature, Sequence)
            and isinstance(label_feature.feature, Value)
            and label_feature.feature.dtype == "float32"
            and label_feature.length != -1):
        return "multilabel"
    return "strong"


def _build_features(input_features: Features, embedding_dim: int,
                    label_mode: str, num_classes: int, granularity: str,
                    num_frames: int | None) -> Features:
    """Build the explicit output features: provenance fields follow the
    input, embedding is a fixed-length float32.

    At frame granularity, embedding / geometry / valid_mask are shaped
    according to the probed frame count num_frames; label is a
    [num_frames, num_classes] frame label in strong mode, and passes through
    the input unchanged in weak mode. At clip granularity, label is a
    [num_classes] float32 tri-state multi-hot in the strong aggregated mode,
    and passes through the input unchanged in other modes.
    """
    passthrough = {key: input_features[key] for key in _PASSTHROUGH_KEYS}
    if granularity == "frame":
        if label_mode == "strong":
            label_feature = Array2D(shape=(num_frames, num_classes),
                                    dtype="float32")
        else:
            label_feature = input_features["label"]
        return Features({
            **passthrough,
            "embedding": Array2D(shape=(num_frames, embedding_dim),
                                 dtype="float32"),
            "label": label_feature,
            "geometry": Array2D(shape=(num_frames, 2), dtype="float32"),
            "valid_mask": Sequence(Value("bool"), length=num_frames),
        })
    if label_mode == "strong":
        label_feature = Sequence(Value("float32"), length=num_classes)
    else:
        label_feature = input_features["label"]
    return Features({
        **passthrough,
        "embedding": Sequence(Value("float32"), length=embedding_dim),
        "label": label_feature,
    })


def _make_map_fn(config: EmbPrepConfig, transform, encoder, label_mode: str,
                 num_frames: int | None):
    """Build the batched map function: waveform batch forward pass ->
    embedding (+ label + geometry + mask).

    Upstream batches are taken in numpy format, so batch["raw"] is already a
    [B, L] float32 ndarray and converts to torch with zero copy; valid_sec is
    passed straight through in seconds as the float32 valid_seconds argument
    to Transform, and Transform's remaining output keys are exactly the
    kwargs for the Encoder's forward pass. Only the replaced columns are
    returned; provenance fields and an untransformed label pass through
    unchanged at the Arrow layer. At frame granularity, embedding / geometry
    / valid_mask are all padded to the probed frame count num_frames (zero
    padding + mask False); for a strong cache, each row's event list and that
    row's geometry produce a [num_frames, C] tri-state frame label (the
    padding frame slot [0, 0] is naturally all zero); at clip granularity,
    for a strong cache each row's event list is aggregated into a [C]
    tri-state multi-hot.
    """
    num_classes = len(config.label_index)
    sample_rate = config.sr

    def embed_batch(batch):
        waveform = torch.from_numpy(np.asarray(batch["raw"], dtype=np.float32))
        valid_seconds = torch.from_numpy(
            np.asarray(batch["valid_sec"], dtype=np.float32))
        with torch.inference_mode():
            features = transform(waveform, sample_rate=sample_rate,
                                 valid_seconds=valid_seconds)
            out = encoder(features.pop("input_features"), **features)
            if config.granularity == "frame":
                pad = num_frames - out["embedding"].shape[1]
                embedding = (F.pad(out["embedding"], (0, 0, 0, pad))
                             .cpu().numpy().astype(np.float32, copy=False))
                geometry = (F.pad(out["geometry"], (0, 0, 0, pad))
                            .cpu().numpy().astype(np.float32, copy=False))
                valid_mask = F.pad(out["valid_mask"], (0, pad)).cpu().numpy()
            else:
                embedding = (out["embedding"].cpu().numpy()
                             .astype(np.float32, copy=False))
        # In datasets 5.0.0, a List(struct) event field comes out row-wise as
        # a list[dict] (verified empirically, not a dict-of-lists); a slice
        # with no events is [].
        if config.granularity == "frame":
            result = {
                "embedding": list(embedding),
                "geometry": list(geometry),
                "valid_mask": list(valid_mask),
            }
            if label_mode == "strong":
                result["label"] = [
                    frame_multihot([ev["target"] for ev in events],
                                   [ev["start"] for ev in events],
                                   [ev["end"] for ev in events],
                                   [ev["value"] for ev in events],
                                   geometry[row], num_classes)
                    for row, events in enumerate(batch["label"])]
            return result
        result = {"embedding": embedding}
        if label_mode == "strong":
            result["label"] = [
                clip_multihot([ev["target"] for ev in events],
                              [ev["value"] for ev in events], num_classes)
                for events in batch["label"]]
        return result

    return embed_batch
