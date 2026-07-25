"""Command-line argument tests for scripts.emb_prep."""

import pytest

from scripts import emb_prep


def test_required_args_are_enforced():
    with pytest.raises(SystemExit) as exc_info:
        emb_prep.parse_args(["--cache_dir", "/cache/EmbSet/abc"])
    assert exc_info.value.code == 2


def test_invalid_granularity_choice_exits():
    with pytest.raises(SystemExit) as exc_info:
        emb_prep.parse_args([
            "--cache_dir", "/cache/EmbSet/abc",
            "--model_name", "fake/enc",
            "--granularity", "token",
        ])
    assert exc_info.value.code == 2


def test_defaults_are_applied():
    args = emb_prep.parse_args([
        "--cache_dir", "/cache/EmbSet/abc",
        "--model_name", "fake/enc",
        "--granularity", "clip",
    ])
    assert args.cache_dir == "/cache/EmbSet/abc"
    assert args.model_name == "fake/enc"
    assert args.granularity == "clip"
    assert args.output_dir is None
    assert args.device == "auto"
    assert args.batch_size == 32
    assert args.pretrained_dir is None
    assert args.overwrite is False


def test_full_args_are_parsed():
    args = emb_prep.parse_args([
        "--cache_dir", "/cache/EmbSet/abc",
        "--model_name", "fake/enc",
        "--granularity", "frame",
        "--output_dir", "/out",
        "--device", "cuda:1",
        "--batch_size", "8",
        "--pretrained_dir", "/weights",
        "--overwrite",
    ])
    assert args.granularity == "frame"
    assert args.output_dir == "/out"
    assert args.device == "cuda:1"
    assert args.batch_size == 8
    assert args.pretrained_dir == "/weights"
    assert args.overwrite is True
