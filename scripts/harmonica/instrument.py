"""Game harmonica fingering and low-cost melody transposition."""

from __future__ import annotations

from typing import Iterable

from .models import Fingering, InstrumentRangeError, NoteEvent

KEYS = ("Z", "X", "C", "V", "B", "N", "M", ",")
DEGREES = (1, 2, 3, 4, 5, 6, 7, 1)
OFFSETS = (0, 2, 4, 5, 7, 9, 11, 12)


def fingering_candidates(midi: int) -> list[Fingering]:
    """Return every control spelling of a MIDI pitch."""
    delta = int(midi) - 60
    result: list[Fingering] = []
    for shift in (-1, 0, 1):
        for sharp in (False, True):
            for key_index, offset in enumerate(OFFSETS):
                if 12 * shift + offset + int(sharp) == delta:
                    result.append(Fingering(shift, sharp, key_index))
    return result


def playable_pitches() -> set[int]:
    return {midi for midi in range(128) if fingering_candidates(midi)}


def _modifier_count(fingering: Fingering) -> int:
    return int(fingering.shift != 0) + int(fingering.sharp) + int(fingering.key_index == 7)


def canonical_fingering(midi: int) -> Fingering:
    candidates = fingering_candidates(midi)
    if not candidates:
        raise InstrumentRangeError("音符超出口琴音域（C3–C#6）")
    return min(
        candidates,
        key=lambda f: (_modifier_count(f), abs(f.shift), int(f.sharp), f.key_index),
    )


def modifier_state(fingering: Fingering) -> tuple[int, bool]:
    return fingering.shift, fingering.sharp


def continuous_token(midi: int) -> str:
    fingering = canonical_fingering(midi)
    labels: list[str] = []
    if fingering.shift < 0:
        labels.append("左")
    elif fingering.shift > 0:
        labels.append("右")
    if fingering.sharp:
        labels.append("中")
    if fingering.key_index == 7:
        labels.append(",")
    suffix = f"({'+'.join(labels)})" if labels else ""
    return f"{DEGREES[fingering.key_index]}{suffix}"


def _score_transpose(pitches: Iterable[int], transpose: int) -> tuple[int, int, int, int, int, int]:
    shifted = [pitch + transpose for pitch in pitches]
    out_of_range = sum(not fingering_candidates(pitch) for pitch in shifted)
    states = [modifier_state(canonical_fingering(pitch)) for pitch in shifted if fingering_candidates(pitch)]
    state_changes = sum(a != b for a, b in zip(states, states[1:]))
    fingerings = [canonical_fingering(pitch) for pitch in shifted if fingering_candidates(pitch)]
    modifiers = sum(_modifier_count(fingering) > 0 for fingering in fingerings)
    sharps = sum(fingering.sharp for fingering in fingerings)
    return (out_of_range, state_changes, modifiers, sharps, abs(transpose), transpose)


def choose_transpose(notes: Iterable[NoteEvent]) -> int:
    pitches = [note.midi for note in notes]
    if not pitches:
        raise InstrumentRangeError("旋律为空，无法移调")
    best = min(range(-24, 25), key=lambda shift: _score_transpose(pitches, shift))
    if _score_transpose(pitches, best)[0]:
        raise InstrumentRangeError("旋律跨度过宽，无法移入口琴音域（C3–C#6）")
    return best

