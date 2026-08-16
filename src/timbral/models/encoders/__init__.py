"""Audio encoder interfaces."""

from .ast_encoder import AstEncoder
from .atst import AtstClipEncoder, AtstFrameEncoder
from .base import BaseEncoder, Granularity
from .beats import BeatsEncoder
from .clap import ClapHtsatEncoder
from .panns import PannsCnn14Encoder, PannsVariant
from .wav2vec2 import Wav2Vec2Encoder

__all__ = (
    "AstEncoder",
    "AtstClipEncoder",
    "AtstFrameEncoder",
    "BaseEncoder",
    "BeatsEncoder",
    "ClapHtsatEncoder",
    "Granularity",
    "PannsCnn14Encoder",
    "PannsVariant",
    "Wav2Vec2Encoder",
)
