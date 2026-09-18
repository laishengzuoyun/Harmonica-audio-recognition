"""Exception types shared by the harmonica transcription modules."""

import sys


class HarmonicaError(RuntimeError):
    """Base exception for harmonica transcription failures."""


class InputValidationError(HarmonicaError):
    """Raised when an input cannot be validated."""


class TranscriptionError(HarmonicaError):
    """Raised when audio transcription fails."""


class RhythmError(HarmonicaError):
    """Raised when rhythm analysis fails."""


class InstrumentRangeError(HarmonicaError):
    """Raised when a note is outside the instrument range."""


__all__ = [
    "HarmonicaError",
    "InputValidationError",
    "TranscriptionError",
    "RhythmError",
    "InstrumentRangeError",
]

_module = sys.modules[__name__]
if __name__ == "scripts.harmonica.exceptions":
    sys.modules.setdefault("harmonica.exceptions", _module)
elif __name__ == "harmonica.exceptions":
    sys.modules.setdefault("scripts.harmonica.exceptions", _module)
