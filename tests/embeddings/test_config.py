"""Config-resolution tests for timbral.embeddings.config: manually constructs minimal cache metadata."""

import dataclasses
import json
from pathlib import Path

import pytest
from datasets.fingerprint import Hasher

from timbral.embeddings.config import resolve_config
from timbral.models import PUBLIC_PARAMETER_NAMES

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


def test_empty_model_kwargs_preserves_historical_hash(cache_dir, tmp_path):
    # Regression guard: model_kwargs was added after artifacts had already
    # been built, so a run that does not use it must keep resolving to the
    # exact hash produced by the original three-field digest.
    historical_hash = Hasher.hash({
        "raw_config_hash": "abc123",
        "model_name": "fake/enc",
        "granularity": "clip",
    })
    # Recomputing the digest here only proves the hashed field set is
    # unchanged, so the literal value existing artifacts were addressed
    # with is pinned too: a Hasher change under a datasets upgrade would
    # silently orphan every directory built so far.
    assert historical_hash == "09c433b6219f8d8a"
    omitted = resolve_config(cache_dir, "fake/enc", "clip",
                             output_dir=str(tmp_path / "out"))
    assert omitted.emb_hash == historical_hash
    # None and {} are equivalent to omitting the parameter: neither adds a
    # "model_kwargs" key to the hashed mapping
    for model_kwargs in (None, {}):
        cfg = resolve_config(cache_dir, "fake/enc", "clip",
                             output_dir=str(tmp_path / "out"),
                             model_kwargs=model_kwargs)
        assert cfg.emb_hash == historical_hash
        assert cfg.model_kwargs == {}
        assert cfg.output_dir == omitted.output_dir


def test_non_empty_model_kwargs_changes_hash(cache_dir, tmp_path):
    base = resolve_config(cache_dir, "fake/enc", "clip",
                          output_dir=str(tmp_path / "out"))
    twelve = resolve_config(cache_dir, "fake/enc", "clip",
                            output_dir=str(tmp_path / "out"),
                            model_kwargs={"n_blocks": 12})
    six = resolve_config(cache_dir, "fake/enc", "clip",
                         output_dir=str(tmp_path / "out"),
                         model_kwargs={"n_blocks": 6})
    assert twelve.emb_hash != base.emb_hash
    assert six.emb_hash != base.emb_hash
    # different mappings never collide, and the hash reaches the third
    # level of the output path
    assert twelve.emb_hash != six.emb_hash
    assert twelve.output_dir.endswith(twelve.emb_hash)
    assert twelve.output_dir != six.output_dir


def test_model_kwargs_key_order_does_not_change_hash(cache_dir, tmp_path):
    first = resolve_config(cache_dir, "fake/enc", "clip",
                           output_dir=str(tmp_path / "out"),
                           model_kwargs={"arch": "base", "n_blocks": 12})
    second = resolve_config(cache_dir, "fake/enc", "clip",
                            output_dir=str(tmp_path / "out"),
                            model_kwargs={"n_blocks": 12, "arch": "base"})
    assert first.emb_hash == second.emb_hash


def test_model_kwargs_is_a_detached_plain_dict(cache_dir, tmp_path):
    caller_kwargs = {"n_blocks": 12}
    cfg = resolve_config(cache_dir, "fake/enc", "clip",
                         output_dir=str(tmp_path / "out"),
                         model_kwargs=caller_kwargs)
    assert type(cfg.model_kwargs) is dict
    assert cfg.model_kwargs == {"n_blocks": 12}
    # the config keeps a copy: later edits by the caller cannot desync the
    # stored parameters from the emb_hash they were computed with
    caller_kwargs["n_blocks"] = 1
    caller_kwargs["arch"] = "base"
    assert cfg.model_kwargs == {"n_blocks": 12}


@pytest.mark.parametrize("public_name, value",
                         [("granularity", "clip"),
                          ("pretrained", False),
                          ("pretrained_dir", "/weights")])
def test_model_kwargs_rejects_create_model_public_parameters(
        cache_dir, tmp_path, public_name, value):
    # create_model declares these three explicitly, so **model_kwargs would
    # bind to them instead of being rejected as unknown. The damaging case
    # is pretrained=False: it would silently cache randomly initialized
    # features under a directory whose name gives no hint of it.
    with pytest.raises(ValueError, match=public_name):
        resolve_config(cache_dir, "fake/enc", "clip",
                       output_dir=str(tmp_path / "out"),
                       model_kwargs={public_name: value})


def test_model_kwargs_rejection_lists_every_conflicting_name(
        cache_dir, tmp_path):
    with pytest.raises(ValueError) as error:
        resolve_config(cache_dir, "fake/enc", "clip",
                       output_dir=str(tmp_path / "out"),
                       model_kwargs={"pretrained": False, "n_blocks": 12,
                                     "granularity": "frame"})
    message = str(error.value)
    assert "'granularity'" in message and "'pretrained'" in message
    # a genuine model-specific parameter is not implicated
    assert "n_blocks" not in message


def test_model_kwargs_rejection_covers_the_whole_public_set(
        cache_dir, tmp_path):
    # The guard reads the registry's single source of truth, so a public
    # parameter added to create_model later is rejected without touching
    # this component.
    for public_name in PUBLIC_PARAMETER_NAMES:
        with pytest.raises(ValueError):
            resolve_config(cache_dir, "fake/enc", "clip",
                           output_dir=str(tmp_path / "out"),
                           model_kwargs={public_name: "x"})


def test_model_kwargs_appears_in_config_snapshot(cache_dir, tmp_path):
    cfg = resolve_config(cache_dir, "fake/enc", "clip",
                         output_dir=str(tmp_path / "out"),
                         model_kwargs={"n_blocks": 12})
    # emb_config.json is written from this snapshot, so the parameters must
    # survive both asdict and a JSON round trip
    snapshot = dataclasses.asdict(cfg)
    assert snapshot["model_kwargs"] == {"n_blocks": 12}
    reloaded = json.loads(json.dumps(snapshot))
    assert reloaded["model_kwargs"] == {"n_blocks": 12}
