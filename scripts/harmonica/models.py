"""Core immutable data types for harmonica transcription."""

from dataclasses import dataclass

from .exceptions import (
    HarmonicaError,
    InputValidationError,
    InstrumentRangeError,
    RhythmError,
    TranscriptionError,
)

__all__ = [
    "HarmonicaError",
    "InputValidationError",
    "TranscriptionError",
    "RhythmError",
    "InstrumentRangeError",
    "NoteEvent",
    "TempoGrid",
    "QuantizedNote",
    "Fingering",
]


@dataclass(frozen=True)
class NoteEvent:
    start: float
    end: float
    midi: int
    confidence: float
    onset_strength: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class TempoGrid:
    bpm: float
    anchor: float
    source: str
    consistency: float


@dataclass(frozen=True)
class QuantizedNote:
    start: float
    end: float
    source_midi: int
    play_midi: int
    confidence: float
    bar: int
    slot: int
    duration_slots: int

    def __post_init__(self) -> None:
        if self.bar < 1:
            raise ValueError("bar must be >= 1")
        if not 1 <= self.slot <= 16:
            raise ValueError("slot must be between 1 and 16")
        if self.duration_slots < 1:
            raise ValueError("duration_slots must be >= 1")


@dataclass(frozen=True)
class Fingering:
    shift: int
    sharp: bool
    key_index: int
