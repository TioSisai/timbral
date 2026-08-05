"""Weight-free unit tests for the model registry."""

from __future__ import annotations

import pytest

import timbral.models
from timbral.models import (
    ModelPair,
    ModelSpec,
    create_model,
    list_models,
    register_model,
)
from timbral.models import registry as registry_module
from timbral.models.encoders import (
    AstEncoder,
    BeatsEncoder,
    ClapHtsatEncoder,
    PannsCnn14Encoder,
    Wav2Vec2Encoder,
)
from timbral.models.transforms import (
    AstKaldiFbankTransform,
    BeatsKaldiFbankTransform,
    ClapLogmelTransform,
    PannsLogmelTransform,
    Wav2Vec2WaveformTransform,
)

_BEATS_MODELS = (
    "beats_iter1",
    "beats_iter2",
    "beats_iter3",
    "beats_iter3_plus_as20k",
    "beats_iter3_plus_as2m",
    "fine_tuned_beats_iter1_cpt1",
    "fine_tuned_beats_iter1_cpt2",
    "fine_tuned_beats_iter2_cpt1",
    "fine_tuned_beats_iter2_cpt2",
    "fine_tuned_beats_iter3_cpt1",
    "fine_tuned_beats_iter3_cpt2",
    "fine_tuned_beats_iter3_plus_as20k_cpt1",
    "fine_tuned_beats_iter3_plus_as20k_cpt2",
    "fine_tuned_beats_iter3_plus_as2m_cpt1",
    "fine_tuned_beats_iter3_plus_as2m_cpt2",
)

_BUILTIN_MODELS = (
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    *_BEATS_MODELS[:5],
    "facebook/wav2vec2-base",
    *_BEATS_MODELS[5:],
    "laion/clap-htsat-fused",
    "panns-16k-cnn14-max_mean",
    "panns-32k-cnn14-decision_level_max",
    "panns-32k-cnn14-max_mean",
)

_EXPECTED_COMPONENT_TYPES = {
    "MIT/ast-finetuned-audioset-10-10-0.4593": (
        AstKaldiFbankTransform,
        AstEncoder,
    ),
    "facebook/wav2vec2-base": (
        Wav2Vec2WaveformTransform,
        Wav2Vec2Encoder,
    ),
    "laion/clap-htsat-fused": (ClapLogmelTransform, ClapHtsatEncoder),
    "panns-16k-cnn14-max_mean": (PannsLogmelTransform, PannsCnn14Encoder),
    "panns-32k-cnn14-decision_level_max": (
        PannsLogmelTransform,
        PannsCnn14Encoder,
    ),
    "panns-32k-cnn14-max_mean": (PannsLogmelTransform, PannsCnn14Encoder),
    **{
        name: (BeatsKaldiFbankTransform, BeatsEncoder)
        for name in _BEATS_MODELS
    },
}

_PANNS_EXPECTED_CONFIGS = {
    "panns-16k-cnn14-max_mean": {
        "target_sample_rate": 16000,
        "variant": "max_mean",
        "n_fft": 512,
        "win_length": 512,
        "hop_length": 160,
        "n_mels": 64,
        "f_min": 50.0,
        "f_max": 8000.0,
    },
    "panns-32k-cnn14-max_mean": {
        "target_sample_rate": 32000,
        "variant": "max_mean",
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 320,
        "n_mels": 64,
        "f_min": 50.0,
        "f_max": 14000.0,
    },
    "panns-32k-cnn14-decision_level_max": {
        "target_sample_rate": 32000,
        "variant": "decision_level_max",
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 320,
        "n_mels": 64,
        "f_min": 50.0,
        "f_max": 14000.0,
    },
}


def test_list_models_returns_sorted_builtin_names():
    assert list_models() == list(_BUILTIN_MODELS)
    assert list_models() == sorted(_BUILTIN_MODELS)


def test_top_level_reexports_registry_symbols():
    assert timbral.models.ModelPair is ModelPair
    assert timbral.models.ModelSpec is ModelSpec
    assert timbral.models.create_model is create_model
    assert timbral.models.list_models is list_models
    assert timbral.models.register_model is register_model


@pytest.mark.parametrize("name", _BUILTIN_MODELS)
def test_create_model_returns_expected_pair(name):
    pair = create_model(name, granularity="clip", pretrained=False)

    expected_transform_type, expected_encoder_type = (
        _EXPECTED_COMPONENT_TYPES[name]
    )
    assert isinstance(pair, ModelPair)
    assert type(pair.transform) is expected_transform_type
    assert type(pair.encoder) is expected_encoder_type
    assert pair.encoder.granularity == "clip"
    if name in _BEATS_MODELS:
        assert pair.encoder.checkpoint == name

    transform, encoder = pair
    assert transform is pair.transform
    assert encoder is pair.encoder


@pytest.mark.parametrize("name", sorted(_PANNS_EXPECTED_CONFIGS))
def test_panns_components_hold_frozen_configuration(name):
    expected = _PANNS_EXPECTED_CONFIGS[name]
    transform, encoder = create_model(
        name, granularity="frame", pretrained=False
    )

    for attribute, value in expected.items():
        assert getattr(transform, attribute) == value
    assert encoder.target_sample_rate == expected["target_sample_rate"]
    assert encoder.variant == expected["variant"]
    assert encoder.granularity == "frame"


_EXPECTED_METADATA = {
    "MIT/ast-finetuned-audioset-10-10-0.4593": (16000, 768),
    "facebook/wav2vec2-base": (16000, 768),
    "laion/clap-htsat-fused": (48000, 512),
    "panns-16k-cnn14-max_mean": (16000, 2048),
    "panns-32k-cnn14-decision_level_max": (32000, 2048),
    "panns-32k-cnn14-max_mean": (32000, 2048),
    "beats_iter1": (16000, 768),
    "fine_tuned_beats_iter3_plus_as2m_cpt2": (16000, 768),
}


@pytest.mark.parametrize("name", sorted(_EXPECTED_METADATA))
def test_pair_exposes_metadata_attributes(name):
    expected_sample_rate, expected_dim = _EXPECTED_METADATA[name]
    transform, encoder = create_model(
        name, granularity="clip", pretrained=False
    )
    assert transform.target_sample_rate == expected_sample_rate
    assert encoder.embedding_dim == expected_dim
    assert type(encoder).embedding_dim == expected_dim


def test_frame_granularity_rejected_for_clip_only_model():
    with pytest.raises(ValueError, match="does not support 'frame'"):
        create_model(
            "laion/clap-htsat-fused", granularity="frame", pretrained=False
        )


def test_unknown_name_raises_key_error_listing_registered():
    with pytest.raises(KeyError) as exc_info:
        create_model("unknown-model", granularity="clip", pretrained=False)

    message = str(exc_info.value)
    for name in _BUILTIN_MODELS:
        assert name in message


def test_unrouted_keyword_argument_raises_type_error():
    with pytest.raises(TypeError, match="are not declared by"):
        create_model(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            granularity="clip",
            pretrained=False,
            unknown_argument=1,
        )


def test_fixed_kwargs_cannot_be_overridden():
    with pytest.raises(TypeError, match="are fixed by registered name"):
        create_model(
            "panns-16k-cnn14-max_mean",
            granularity="clip",
            pretrained=False,
            n_fft=256,
        )


def test_wav2vec2_pins_official_normalization():
    transform, _ = create_model(
        "facebook/wav2vec2-base", granularity="clip", pretrained=False
    )
    assert transform.do_normalize is True

    with pytest.raises(TypeError, match="are fixed by registered name"):
        create_model(
            "facebook/wav2vec2-base",
            granularity="clip",
            pretrained=False,
            do_normalize=False,
        )


@pytest.mark.parametrize(
    "parameter_name",
    ("granularity", "pretrained", "pretrained_dir"),
)
def test_fixed_kwargs_reject_public_parameters(parameter_name):
    with pytest.raises(ValueError, match="public parameters"):
        ModelSpec(
            transform_cls=AstKaldiFbankTransform,
            encoder_cls=AstEncoder,
            fixed_kwargs={parameter_name: object()},
        )


def test_register_model_routes_by_signature_and_overwrites(monkeypatch):
    monkeypatch.setattr(
        registry_module, "MODELS", dict(registry_module.MODELS)
    )

    class FakeTransform:
        def __init__(self, *, pretrained, scale=1):
            self.pretrained = pretrained
            self.scale = scale

    class FakeEncoder:
        def __init__(self, *, granularity, scale=1):
            self.granularity = granularity
            self.scale = scale

    register_model(
        "fake-model",
        ModelSpec(transform_cls=FakeTransform, encoder_cls=FakeEncoder),
    )
    assert "fake-model" in list_models()

    transform, encoder = create_model(
        "fake-model", granularity="clip", pretrained=False, scale=3
    )
    assert transform.pretrained is False
    assert transform.scale == 3
    assert encoder.granularity == "clip"
    assert encoder.scale == 3

    class OtherEncoder:
        def __init__(self, *, granularity):
            self.granularity = granularity

    register_model(
        "fake-model",
        ModelSpec(transform_cls=FakeTransform, encoder_cls=OtherEncoder),
    )
    pair = create_model("fake-model", granularity="frame")
    assert type(pair.encoder) is OtherEncoder
    assert pair.transform.pretrained is True
