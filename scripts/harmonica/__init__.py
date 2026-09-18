"""Reusable audio-to-Delta-Force harmonica transcription package."""

import sys

_package = sys.modules[__name__]
if __name__ == "scripts.harmonica":
    sys.modules.setdefault("harmonica", _package)
elif __name__ == "harmonica":
    sys.modules.setdefault("scripts.harmonica", _package)

from .exceptions import (
    HarmonicaError,
    InputValidationError,
    InstrumentRangeError,
    RhythmError,
    TranscriptionError,
)
from .models import Fingering, NoteEvent, QuantizedNote, TempoGrid

__all__ = [
    "Fingering",
    "HarmonicaError",
    "InputValidationError",
    "InstrumentRangeError",
    "NoteEvent",
    "QuantizedNote",
    "RhythmError",
    "TempoGrid",
    "TranscriptionError",
]
