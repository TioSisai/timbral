"""Embedding extraction config resolution: raw cache metadata reading,
three-level output path derivation, and emb_hash computation.
"""

import dataclasses
import json
import os
import posixpath
from pathlib import Path

from datasets.fingerprint import Hasher

from timbral.models import PUBLIC_PARAMETER_NAMES
from timbral.storage import is_s3_path


@dataclasses.dataclass(frozen=True)
class EmbPrepConfig:
    """Full configuration for one embedding extraction run (see
    scripts/emb_prep.py for the semantics of the input parameters).

    Derived fields beyond the command-line input parameters:

    Attributes:
        output_dir: The final three-level output directory, always
            ``{output root}/{dataset_name}/{model_name with "/" replaced by
            "--"}/{emb_hash}``.
        dataset_name: Taken from the raw cache's prep_config.json.
        sr: The raw cache's sample rate, passed to Transform as sample_rate
            during the forward pass.
        seg_sec: The raw cache's slice window length in seconds, used at
            frame granularity to probe the constant frame count.
        label_type: The raw cache's label type (weak/strong).
        raw_config_hash: The raw cache's config_hash.
        model_kwargs: Model-specific constructor parameters, forwarded
            to create_model; an empty mapping means none were given.
        emb_hash: Computed from {raw_config_hash, model_name,
            granularity}, plus model_kwargs when it is non-empty.
            Execution parameters such as device/batch size are not
            included, and an empty model_kwargs is left out entirely so
            that artifacts built before this parameter existed keep
            hashing to the same value.
        label_index: The contents of the raw cache's label_index.json
            (class_name -> index).
        raw_prep_config: The full parameter snapshot from the raw cache's
            prep_config.json.
    """

    cache_dir: str
    model_name: str
    granularity: str
    output_dir: str
    device: str
    batch_size: int
    pretrained_dir: str | None
    model_kwargs: dict
    overwrite: bool
    dataset_name: str
    sr: int
    seg_sec: float
    label_type: str
    raw_config_hash: str
    emb_hash: str
    label_index: dict
    raw_prep_config: dict


def resolve_config(cache_dir, model_name, granularity, output_dir=None,
                   device="auto", batch_size=32, pretrained_dir=None,
                   model_kwargs=None, overwrite=False) -> EmbPrepConfig:
    """Read the raw cache metadata, compute emb_hash, and derive the
    three-level output path.

    Args:
        cache_dir: The DatasetDict cache directory produced by raw_prep
            (a local path); must contain prep_config.json and
            label_index.json.
        model_name: A model name registered in timbral.models.
        granularity: Output granularity, ``clip`` or ``frame``; a weak cache
            combined with frame outputs frame embeddings with the clip label
            passed through unchanged (the weakly labeled SED scenario).
        output_dir: The output root directory prefix; supports both local
            paths and ``s3://`` paths (the trailing slash is normalized
            before joining as a posix path). When ``None``, the current
            working directory is used; the final output directory is always
            the three-level structure ``{output_dir}/{dataset_name}/
            {model_name with "/" replaced by "--"}/{emb_hash}``.
        device: Target device string, resolved by builder to move the model
            components.
        batch_size: The batch_size and writer_batch_size used by map.
        pretrained_dir: A custom weights directory, passed through to
            timbral.models.create_model.
        model_kwargs: Model-specific constructor parameters, passed
            through to timbral.models.create_model; ``None`` and an
            empty mapping are equivalent and stay out of emb_hash. The
            public parameters of create_model are rejected here: they
            are either fixed by this pipeline or carried by their own
            parameter.
        overwrite: Whether to force a rebuild when the output already
            exists.

    Returns:
        An immutable config object with all fields populated.

    Raises:
        FileNotFoundError: cache_dir is missing prep_config.json or
            label_index.json.
        ValueError: model_kwargs carries one of create_model's public
            parameters.
    """
    cache_dir = os.fspath(cache_dir)
    with open(os.path.join(cache_dir, "prep_config.json"),
              encoding="utf-8") as f:
        raw_prep_config = json.load(f)
    with open(os.path.join(cache_dir, "label_index.json"),
              encoding="utf-8") as f:
        label_index = json.load(f)

    raw_config_hash = raw_prep_config["config_hash"]
    model_kwargs = dict(model_kwargs) if model_kwargs else {}
    # create_model's public parameters would be accepted silently by its
    # **kwargs, letting model_kwargs switch off pretrained weights and
    # cache randomly initialized features under a directory that gives no
    # hint of it. granularity and pretrained_dir have their own
    # parameters, and pretrained is fixed by this pipeline.
    public_conflicts = sorted(model_kwargs.keys() & PUBLIC_PARAMETER_NAMES)
    if public_conflicts:
        raise ValueError(
            f"model_kwargs must not contain the public parameters "
            f"{public_conflicts}; granularity and pretrained_dir have "
            "their own parameters, and pretrained is fixed."
        )
    hash_fields = {
        "raw_config_hash": raw_config_hash,
        "model_name": model_name,
        "granularity": granularity,
    }
    # Adding the key at all changes the digest, so an empty mapping is
    # omitted rather than hashed as {}: runs that predate this
    # parameter keep resolving to their existing output directory.
    if model_kwargs:
        hash_fields["model_kwargs"] = model_kwargs
    emb_hash = Hasher.hash(hash_fields)
    if is_s3_path(output_dir):
        final_output_dir = posixpath.join(
            output_dir.rstrip("/"), raw_prep_config["dataset_name"],
            model_name.replace("/", "--"), emb_hash)
    else:
        output_root = Path(output_dir) if output_dir is not None else Path.cwd()
        final_output_dir = str(output_root / raw_prep_config["dataset_name"]
                               / model_name.replace("/", "--") / emb_hash)

    return EmbPrepConfig(
        cache_dir=cache_dir,
        model_name=model_name,
        granularity=granularity,
        output_dir=final_output_dir,
        device=device,
        batch_size=batch_size,
        pretrained_dir=(None if pretrained_dir is None
                        else os.fspath(pretrained_dir)),
        model_kwargs=model_kwargs,
        overwrite=overwrite,
        dataset_name=raw_prep_config["dataset_name"],
        sr=raw_prep_config["sr"],
        seg_sec=raw_prep_config["seg_sec"],
        label_type=raw_prep_config["label_type"],
        raw_config_hash=raw_config_hash,
        emb_hash=emb_hash,
        label_index=label_index,
        raw_prep_config=raw_prep_config,
    )
