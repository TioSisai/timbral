"""Command-line argument tests for scripts.raw_prep."""

import pytest

from scripts import raw_prep


def test_dataset_dir_is_required():
    with pytest.raises(SystemExit) as exc_info:
        raw_prep.parse_args(["--dataset_name", "FakeSet"])
    assert exc_info.value.code == 2


def test_explicit_dataset_dir_is_parsed():
    args = raw_prep.parse_args([
        "--dataset_name", "FakeSet",
        "--dataset_dir", "/datasets/FakeSet",
    ])
    assert args.dataset_name == "FakeSet"
    assert args.dataset_dir == "/datasets/FakeSet"
