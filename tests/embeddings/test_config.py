"""Config-resolution tests for timbral.embeddings.config: manually constructs minimal cache metadata."""

import json
from pathlib import Path

import pytest
from datasets.fingerprint import Hasher

from timbral.embeddings.config import resolve_config

_PREP_CONFIG = {
    "dataset_name": "FakeSet",
    "sr": 8000,
    "seg_sec": 1.0,
    "label_type": "weak",
    "config_hash": "abc123",
}
_LABEL_INDEX = {"cat": 0, "dog": 1}


@pytest.fixture()
def cache_dir(tmp_path):
    """Write out a minimal raw cache directory containing only metadata files."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "prep_config.json").write_text(
        json.dumps(_PREP_CONFIG), encoding="utf-8")
    (cache / "label_index.json").write_text(
        json.dumps(_LABEL_INDEX), encoding="utf-8")
    return str(cache)


def test_emb_hash_and_local_output_path(cache_dir, tmp_path):
    cfg = resolve_config(cache_dir, "fake/enc", "clip",
                         output_dir=str(tmp_path / "out"))
    expected_hash = Hasher.hash({
        "raw_config_hash": "abc123",
        "model_name": "fake/enc",
        "granularity": "clip",
    })
    assert cfg.emb_hash == expected_hash
    assert cfg.output_dir == str(
        tmp_path / "out" / "FakeSet" / "fake--enc" / expected_hash)
    # granularity participates in the hash: frame yields a different output
    # directory
    frame_cfg = resolve_config(cache_dir, "fake/enc", "frame",
                               output_dir=str(tmp_path / "out"))
    assert frame_cfg.emb_hash != cfg.emb_hash


def test_default_output_root_is_cwd(cache_dir, tmp_path, monkeypatch):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    cfg = resolve_config(cache_dir, "fake/enc", "clip")
    assert cfg.output_dir == str(
        work_dir / "FakeSet" / "fake--enc" / cfg.emb_hash)


def test_s3_output_root_posix_join(cache_dir):
    cfg = resolve_config(cache_dir, "fake/enc", "clip",
                         output_dir="s3://bucket/emb/")
    assert cfg.output_dir == (
        f"s3://bucket/emb/FakeSet/fake--enc/{cfg.emb_hash}")


def test_weak_cache_allows_frame_granularity(cache_dir, tmp_path):
    # weak cache + frame: frame embeddings with passthrough clip labels
    # (weakly-labeled SED scenario)
    cfg = resolve_config(cache_dir, "fake/enc", "frame",
                         output_dir=str(tmp_path / "out"))
    assert cfg.label_type == "weak"
    assert cfg.granularity == "frame"


def test_metadata_fields_populated(cache_dir, tmp_path):
    cfg = resolve_config(cache_dir, "fake/enc", "clip",
                         output_dir=str(tmp_path / "out"),
                         pretrained_dir=Path("/weights"))
    assert cfg.dataset_name == "FakeSet"
    assert cfg.sr == 8000
    assert cfg.seg_sec == 1.0
    assert cfg.raw_config_hash == "abc123"
    assert cfg.label_index == _LABEL_INDEX
    assert cfg.raw_prep_config == _PREP_CONFIG
    assert cfg.pretrained_dir == "/weights"


def test_missing_metadata_file_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_config(str(empty), "fake/enc", "clip")
