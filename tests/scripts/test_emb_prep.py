"""Command-line argument tests for scripts.emb_prep."""

import inspect
import sys

import pytest

from scripts import emb_prep
from timbral.embeddings.config import resolve_config


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


def test_model_kwargs_defaults_to_none():
    args = emb_prep.parse_args([
        "--cache_dir", "/cache/EmbSet/abc",
        "--model_name", "fake/enc",
        "--granularity", "clip",
    ])
    assert args.model_kwargs is None


def test_model_kwargs_json_object_is_parsed():
    args = emb_prep.parse_args([
        "--cache_dir", "/cache/EmbSet/abc",
        "--model_name", "atst-frame-base",
        "--granularity", "frame",
        "--model_kwargs", '{"n_blocks": 12}',
    ])
    assert args.model_kwargs == {"n_blocks": 12}
    # JSON typing is preserved: the encoder receives an int, not "12"
    assert isinstance(args.model_kwargs["n_blocks"], int)


def test_invalid_model_kwargs_json_exits():
    with pytest.raises(SystemExit) as exc_info:
        emb_prep.parse_args([
            "--cache_dir", "/cache/EmbSet/abc",
            "--model_name", "atst-frame-base",
            "--granularity", "frame",
            "--model_kwargs", "{n_blocks: 12}",
        ])
    assert exc_info.value.code == 2


def test_model_kwargs_json_array_exits():
    # valid JSON, but not a mapping that can be splatted into create_model
    with pytest.raises(SystemExit) as exc_info:
        emb_prep.parse_args([
            "--cache_dir", "/cache/EmbSet/abc",
            "--model_name", "atst-frame-base",
            "--granularity", "frame",
            "--model_kwargs", '[{"n_blocks": 12}]',
        ])
    assert exc_info.value.code == 2


def test_model_kwargs_reaches_resolve_config(monkeypatch):
    # main dispatches with **vars(args), so every parsed name must be a
    # resolve_config parameter and the parsed mapping must arrive intact
    calls = {}

    def fake_resolve_config(**kwargs):
        calls["resolve"] = kwargs
        return "resolved-config"

    def fake_prepare_embeddings(config):
        calls["prepare"] = config

    monkeypatch.setattr(emb_prep, "resolve_config", fake_resolve_config)
    monkeypatch.setattr(emb_prep, "prepare_embeddings",
                        fake_prepare_embeddings)
    monkeypatch.setattr(sys, "argv", [
        "emb_prep.py",
        "--cache_dir", "/cache/EmbSet/abc",
        "--model_name", "atst-frame-base",
        "--granularity", "frame",
        "--model_kwargs", '{"n_blocks": 12}',
    ])

    emb_prep.main()

    assert calls["resolve"]["model_kwargs"] == {"n_blocks": 12}
    assert calls["resolve"]["model_name"] == "atst-frame-base"
    assert calls["prepare"] == "resolved-config"
    # the real signature accepts exactly what the parser produced
    inspect.signature(resolve_config).bind(**calls["resolve"])
