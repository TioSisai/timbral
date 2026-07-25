"""Tests for timbral.datasets.split_generators: registry and dispatch contract."""

import inspect
import json
from pathlib import Path

import pytest

from timbral.datasets import split_generators
from timbral.datasets.adapters import ADAPTERS


def test_generators_cover_exactly_all_adapted_datasets():
    # Structural invariant: every supported dataset must have both an annotation adapter and a split generator
    assert split_generators.GENERATORS.keys() == ADAPTERS.keys()


@pytest.mark.parametrize("dataset_name",
                         sorted(split_generators.GENERATORS),
                         ids=str)
def test_generator_signature_is_uniform(dataset_name):
    generate_fn = split_generators.GENERATORS[dataset_name]
    parameters = inspect.signature(generate_fn).parameters
    assert list(parameters) == ["dataset_dir"]


@pytest.mark.parametrize("dataset_name",
                         sorted(split_generators.GENERATORS),
                         ids=str)
def test_generator_has_no_machine_specific_paths(dataset_name):
    generate_fn = split_generators.GENERATORS[dataset_name]
    source = Path(inspect.getfile(generate_fn)).read_text(encoding="utf-8")
    assert "/data/" not in source
    assert "/projappl/" not in source


def test_unregistered_dataset_raises_keyerror_with_listing():
    with pytest.raises(KeyError, match="No split generator registered"):
        split_generators.generate("NoSuchSet", "/datasets/NoSuchSet")


def test_generate_writes_and_verifies_output(monkeypatch, tmp_path):
    def fake_generate(dataset_dir):
        return {"train": [{"audio_path": "a.wav", "start": 0.0, "end": "inf"}],
                "validation": [], "test": []}

    monkeypatch.setitem(split_generators.GENERATORS, "FakeSet", fake_generate)
    output_path = tmp_path / "FakeSet" / "default.json"

    result = split_generators.generate(
        "FakeSet", "/datasets/FakeSet", output_path=output_path)

    assert result["output_path"] == str(output_path)
    assert result["counts"] == {"train": 1, "validation": 0, "test": 0}
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["train"] == [
        {"audio_path": "a.wav", "start": 0.0, "end": "inf"}]


def test_generate_accepts_output_filename_without_parent(monkeypatch, tmp_path):
    def fake_generate(dataset_dir):
        return {"train": [{"audio_path": "a.wav", "start": 0.0, "end": "inf"}],
                "validation": [], "test": []}

    monkeypatch.setitem(split_generators.GENERATORS, "FakeSet", fake_generate)
    monkeypatch.chdir(tmp_path)

    result = split_generators.generate(
        "FakeSet", "/datasets/FakeSet", output_path="default.json")

    assert result["output_path"] == "default.json"
    assert (tmp_path / "default.json").is_file()


def test_default_split_path_points_into_assets():
    path = split_generators.default_split_path("ESC-50")
    repo_root = Path(__file__).resolve().parents[2]
    assert path == (repo_root / "assets" / "datasets" / "splits"
                    / "ESC-50" / "default.json")
