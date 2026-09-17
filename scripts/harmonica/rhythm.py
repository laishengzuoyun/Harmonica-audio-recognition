"""Tempo estimation and sixteenth-note quantization for melody events."""

from __future__ import annotations

import librosa
import numpy as np

from .models import NoteEvent, QuantizedNote, RhythmError, TempoGrid


def _interval_grid(notes: list[NoteEvent]) -> TempoGrid:
    """Estimate a grid from repeated note-onset intervals."""
    starts = np.asarray([note.start for note in notes], dtype=float)
    intervals = np.diff(starts)
    intervals = intervals[(intervals >= 0.06) & (intervals <= 2.0)]
    candidates: list[tuple[float, float, int, float]] = []

    for interval in intervals:
        for multiple in range(1, 9):
            slot_seconds = float(interval) / multiple
            bpm = 60.0 / (slot_seconds * 4.0)
            if not 45.0 <= bpm <= 210.0:
                continue
            residuals = np.abs(intervals / slot_seconds - np.rint(intervals / slot_seconds))
            consistency = float(np.mean(residuals <= 0.18))
            candidates.append(
                (consistency, -float(np.median(residuals)), -multiple, bpm)
            )

    if not candidates:
        raise RhythmError("Unable to infer a usable rhythm grid from note intervals.")

    consistency, _, _, bpm = max(candidates)
    if consistency < 0.55:
        raise RhythmError("Note intervals are too inconsistent to infer a rhythm grid.")

    return TempoGrid(float(bpm), float(notes[0].start), "intervals", consistency)


def detect_tempo_grid(
    accompaniment: np.ndarray, sample_rate: int, notes: list[NoteEvent]
) -> TempoGrid:
    """Prefer a consistent accompaniment beat track, then use melody intervals."""
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=accompaniment, sr=sample_rate)
        beat_frames = np.asarray(beat_frames, dtype=int)
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
        tempo_values = np.asarray(tempo, dtype=float).reshape(-1)
        bpm = float(tempo_values[0]) if tempo_values.size else float("nan")
    except Exception:
        beat_times = np.asarray([], dtype=float)
        bpm = float("nan")

    if 45.0 <= bpm <= 210.0 and len(beat_times) >= 4:
        gaps = np.diff(beat_times)
        median_gap = float(np.median(gaps)) if len(gaps) else 0.0
        if median_gap > 0.0:
            consistency = float(np.mean(np.abs(gaps - median_gap) / median_gap <= 0.08))
            if consistency >= 0.60:
                return TempoGrid(bpm, float(beat_times[0]), "beats", consistency)

    return _interval_grid(notes)


def quantize_notes(
    notes: list[NoteEvent], grid: TempoGrid, transpose: int
) -> list[QuantizedNote]:
    """Place note events on a four-four sixteenth-note grid."""
    if not notes:
        raise RhythmError("Cannot quantize an empty melody.")
    if not np.isfinite(grid.bpm) or grid.bpm <= 0.0:
        raise RhythmError("Tempo must be a positive finite BPM value.")

    slot_seconds = 60.0 / grid.bpm / 4.0
    starts = [int(np.rint((note.start - grid.anchor) / slot_seconds)) for note in notes]
    base_slot = (starts[0] // 16) * 16
    quantized: list[QuantizedNote] = []

    for note, start_slot in zip(notes, starts):
        end_slot = int(np.rint((note.end - grid.anchor) / slot_seconds))
        end_slot = max(end_slot, start_slot + 1)
        relative_slot = start_slot - base_slot
        quantized.append(
            QuantizedNote(
                start=note.start,
                end=note.end,
                source_midi=note.midi,
                play_midi=note.midi + transpose,
                confidence=note.confidence,
                bar=relative_slot // 16 + 1,
                slot=relative_slot % 16 + 1,
                duration_slots=end_slot - start_slot,
            )
        )

    return quantized


def section_starts(
    notes: list[NoteEvent] | list[QuantizedNote],
    gap_seconds: float = 1.2,
    max_bars: int = 8,
) -> list[int]:
    """Return note indexes where readable score sections should begin."""
    if not notes:
        return []

    starts = [0]
    last_section_bar = getattr(notes[0], "bar", None)
    previous = notes[0]
    for index, note in enumerate(notes[1:], start=1):
        section_bar = getattr(note, "bar", None)
        has_time_gap = note.start - previous.end >= gap_seconds
        has_bar_span = (
            section_bar is not None
            and last_section_bar is not None
            and section_bar - last_section_bar >= max_bars
        )
        if has_time_gap or has_bar_span:
            starts.append(index)
            last_section_bar = section_bar
        previous = note

    return starts
