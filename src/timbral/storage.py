"""Shared storage utilities: S3 storage options construction, cache
target resolution, and map temporary directory management.

A top-level shared module at the same level as ``timbral.paths``, used by
all components and the orchestration layer so that the same storage logic
never exists as multiple private copies.
"""

import contextlib
import json
import os
import posixpath
import shutil
import tempfile

import fsspec

_S3_PREFIX = "s3://"


def is_s3_path(path) -> bool:
    """Determine whether a path is an s3:// object storage path.

    Args:
        path: The path to check (any type; non-strings are always treated
            as local paths).

    Returns:
        Whether the path is an s3:// path.
    """
    return isinstance(path, str) and path.startswith(_S3_PREFIX)


def s3_storage_options() -> dict:
    """Construct s3fs parameters compatible with Allas.

    Returns:
        The storage parameters passed to fsspec and Hugging Face datasets.
    """
    options = {
        "config_kwargs": {
            "request_checksum_calculation": "when_required",
            "response_checksum_validation": "when_required",
        }
    }
    endpoint = os.environ.get("AWS_S3_ENDPOINT")
    if endpoint:
        options["endpoint_url"] = (endpoint if "://" in endpoint
                                   else f"https://{endpoint}")
    return options


def resolve_cache_target(cache_dir: str) -> tuple:
    """Resolve local and S3 cache targets uniformly, eliminating a dual
    code path for callers.

    Args:
        cache_dir: A local path or an s3:// path.

    Returns:
        A ``(fs, path, storage_options)`` tuple: the fsspec filesystem, the
        path within the protocol, and the storage parameters passed to
        ``save_to_disk`` (``None`` for local paths).

    Raises:
        ValueError: The S3 path is a bucket root (it must include a
            directory within the bucket).
    """
    is_s3 = is_s3_path(cache_dir)
    storage_options = s3_storage_options() if is_s3 else None
    fs, path = fsspec.core.url_to_fs(cache_dir, **(storage_options or {}))
    if is_s3 and not posixpath.dirname(path.rstrip("/")):
        raise ValueError(
            "S3 output directory must include a directory within the "
            f"bucket; the bucket-root path cannot be used: {cache_dir}")
    return fs, path, storage_options


def map_tmp_context(cache_dir: str, tag: str):
    """Choose the temporary Arrow directory for map: ``$TMPDIR`` for S3
    output, a directory next to the target for local output.

    Placing the local temporary directory next to the target rather than
    under ``$TMPDIR`` is deliberate: map's temporary Arrow files are on the
    same order of size as the final artifacts, and keeping them on the same
    filesystem avoids a cross-disk double write and possible insufficient
    capacity on the node's local disk.

    Args:
        cache_dir: The final output directory (a local path or an s3://
            path).
        tag: An identifier embedded in the temporary directory name to
            distinguish different runs (usually a config hash).

    Returns:
        A context manager that yields the temporary directory path and
        cleans it up on exit.
    """
    if is_s3_path(cache_dir):
        return tempfile.TemporaryDirectory(prefix=f"timbral-map-{tag}-",
                                           dir=os.environ.get("TMPDIR"))
    parent = os.path.dirname(cache_dir.rstrip("/"))
    return local_map_tmp(os.path.join(parent, f".maptmp-{tag}"))


def write_json(fs, path: str, obj) -> None:
    """Write JSON through an fsspec filesystem in a consistent format
    (indent=4, non-ASCII preserved).

    Args:
        fs: The fsspec filesystem.
        path: The target path within the protocol.
        obj: A JSON-serializable object.
    """
    with fs.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


@contextlib.contextmanager
def local_map_tmp(path: str):
    """Create a local temporary directory for map, removing it on exit
    (including on an exception); cleans up any leftover directory with
    the same name first.

    Args:
        path: The target path for the temporary directory.

    Yields:
        The created temporary directory path (same as the input).
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    try:
        yield path
    finally:
        shutil.rmtree(path)
