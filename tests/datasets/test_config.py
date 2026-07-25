"""Unit tests for timbral.datasets.config: path resolution, default split generation, and config hashing."""

import json
from pathlib import Path

from datasets.fingerprint import Hasher
import pytest

from timbral.datasets import config


def _write_split(path: Path, audio_path: str = "audio.wav") -> Path:
    """Write a minimal valid split file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "train": [{"audio_path": audio_path, "start": 0.0, "end": "inf"}],
        "validation": [],
        "test": [],
    }), encoding="utf-8")
    return path


def test_project_root_points_to_repo_root():
    # Derive the repo root independently from the test file's own location,
    # to avoid tautologically reusing the implementation's expression.
    assert config.PROJECT_ROOT == Path(__file__).resolve().parents[2]


def test_dataset_dir_is_required(tmp_path):
    split_json = _write_split(tmp_path / "split.json")
    with pytest.raises(TypeError):
        config.resolve_config("FakeSet", split_json=str(split_json))
    with pytest.raises(ValueError, match="dataset_dir must be passed explicitly"):
        config.resolve_config("FakeSet", None, split_json=str(split_json))


def test_existing_default_split_uses_project_root_and_cwd(
        monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    split_json = _write_split(
        project_root / "assets" / "datasets" / "splits" / "FakeSet"
        / "default.json")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.setattr(config, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(work_dir)

    def fail_if_called(*args, **kwargs):
        pytest.fail("The split generator should not be called when the default split already exists")

    monkeypatch.setattr(config.split_generators, "generate", fail_if_called)
    cfg = config.resolve_config("FakeSet", "/datasets/FakeSet")

    expected_split_hash = Hasher.hash(split_json.read_bytes())
    expected_config_hash = Hasher.hash({
        "dataset_name": "FakeSet",
        "split_json_hash": expected_split_hash,
        "sr": 16000,
        "mono": True,
        "seg_sec": 10.0,
        "hop_sec": 10.0,
        "tol_sec": 0.0,
        "label_type": "weak",
    })
    assert cfg.dataset_dir == "/datasets/FakeSet"
    assert cfg.split_json == str(split_json)
    assert cfg.split_json_hash == expected_split_hash
    assert cfg.config_hash == expected_config_hash
    assert cfg.cache_dir == str(work_dir / "FakeSet" / expected_config_hash)


def test_missing_default_split_runs_generator_with_dataset_dir(
        monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    received = []

    def fake_generate(dataset_dir):
        received.append(dataset_dir)
        return {"train": [{"audio_path": "a.wav", "start": 0.0, "end": "inf"}],
                "validation": [], "test": []}

    monkeypatch.setitem(
        config.split_generators.GENERATORS, "FakeSet", fake_generate)
    monkeypatch.setattr(config, "PROJECT_ROOT", project_root)

    cfg = config.resolve_config("FakeSet", "/datasets/custom-root")

    default_path = (project_root / "assets" / "datasets" / "splits"
                    / "FakeSet" / "default.json")
    assert cfg.split_json == str(default_path)
    assert default_path.is_file()
    assert received == ["/datasets/custom-root"]


def test_missing_default_generator_raises(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    monkeypatch.setattr(config, "PROJECT_ROOT", project_root)

    with pytest.raises(FileNotFoundError, match="no corresponding split generator was found"):
        config.resolve_config("FakeSet", "/datasets/FakeSet")


def test_generator_failure_propagates_naturally(monkeypatch, tmp_path):
    project_root = tmp_path / "project"

    def broken_generate(dataset_dir):
        raise RuntimeError("generator internal error")

    monkeypatch.setitem(
        config.split_generators.GENERATORS, "FakeSet", broken_generate)
    monkeypatch.setattr(config, "PROJECT_ROOT", project_root)

    with pytest.raises(RuntimeError, match="generator internal error"):
        config.resolve_config("FakeSet", "/datasets/FakeSet")


def test_explicit_missing_split_never_runs_generator(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    called = []

    def fake_generate(dataset_dir):
        called.append(dataset_dir)
        return {"train": [], "validation": [], "test": []}

    monkeypatch.setitem(
        config.split_generators.GENERATORS, "FakeSet", fake_generate)
    monkeypatch.setattr(config, "PROJECT_ROOT", project_root)

    explicit_default_path = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError, match="Explicitly specified"):
        config.resolve_config(
            "FakeSet", "/datasets/FakeSet",
            split_json=str(explicit_default_path))
    assert not called


def test_explicit_paths_are_preserved(monkeypatch, tmp_path):
    split_json = _write_split(tmp_path / "custom.json")

    def fail_if_called(*args, **kwargs):
        pytest.fail("The split generator should not be called when an explicit split exists")

    monkeypatch.setattr(config.split_generators, "generate", fail_if_called)
    cfg = config.resolve_config(
        "FakeSet", "/datasets/FakeSet",
        split_json=str(split_json), cache_dir="s3://bucket/cache")
    assert cfg.split_json == str(split_json)
    assert cfg.cache_dir == "s3://bucket/cache"


def test_config_hash_uses_only_declared_fields(tmp_path):
    split_json = _write_split(tmp_path / "split.json")
    common = {"split_json": str(split_json), "sr": 8000, "seg_sec": 2.0}
    cfg_a = config.resolve_config(
        "FakeSet", "/dataset/a", cache_dir=str(tmp_path / "cache-a"), **common)
    cfg_b = config.resolve_config(
        "FakeSet", "/dataset/b", cache_dir=str(tmp_path / "cache-b"), **common)
    cfg_strong = config.resolve_config(
        "FakeSet", "/dataset/a", label_type="strong", **common)

    assert cfg_a.config_hash == cfg_b.config_hash
    assert cfg_a.config_hash != cfg_strong.config_hash


def test_config_hash_is_sensitive_to_split_content(tmp_path):
    split_a = _write_split(tmp_path / "a.json", "a.wav")
    split_b = _write_split(tmp_path / "b.json", "b.wav")
    cfg_a = config.resolve_config(
        "FakeSet", "/datasets/FakeSet", split_json=str(split_a))
    cfg_b = config.resolve_config(
        "FakeSet", "/datasets/FakeSet", split_json=str(split_b))
    assert cfg_a.config_hash != cfg_b.config_hash


def test_invalid_label_type_raises(tmp_path):
    split_json = _write_split(tmp_path / "split.json")
    with pytest.raises(ValueError, match="label_type only supports"):
        config.resolve_config(
            "FakeSet", "/datasets/FakeSet",
            split_json=str(split_json), label_type="frame")
