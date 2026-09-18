"""Tempo estimation and sixteenth-note quantization for melody events."""

from __future__ import annotations

import librosa
import numpy as np

from .exceptions import RhythmError
from .models import NoteEvent, QuantizedNote, TempoGrid


def _interval_grid(notes: list[NoteEvent]) -> TempoGrid:
    """Estimate a grid from repeated note-onset intervals.

    Half/double-time aliases can align equally well; lower subdivision multiples
    win those ties so a uniform 0.15-second onset pattern remains 100 BPM.
    """
    starts = np.asarray([note.start for note in notes], dtype=float)
    intervals = np.diff(starts)
    intervals = intervals[(intervals >= 0.06) & (intervals <= 2.0)]
    candidates: list[tuple[float, float, float, float, int, float]] = []

    for interval in intervals:
        for multiple in range(1, 9):
            slot_seconds = float(interval) / multiple
            bpm = 60.0 / (slot_seconds * 4.0)
            if not 45.0 <= bpm <= 210.0:
                continue
            elapsed = starts - starts[0]
            for _ in range(3):
                slot_indices = np.rint(elapsed / slot_seconds)
                denominator = float(np.dot(slot_indices, slot_indices))
                if denominator == 0.0:
                    break
                slot_seconds = float(np.dot(slot_indices, elapsed) / denominator)
            if not np.isfinite(slot_seconds) or slot_seconds <= 0.0:
                continue
            bpm = 60.0 / (slot_seconds * 4.0)
            if not 45.0 <= bpm <= 210.0:
                continue

            phase_residuals = np.abs(
                elapsed / slot_seconds - np.rint(elapsed / slot_seconds)
            )
            interval_residuals = np.abs(
                intervals / slot_seconds - np.rint(intervals / slot_seconds)
            )
            phase_consistency = float(np.mean(phase_residuals <= 0.18))
            interval_consistency = float(np.mean(interval_residuals <= 0.18))
            candidates.append(
                (
                    phase_consistency,
                    interval_consistency,
                    -float(np.median(phase_residuals)),
                    -float(np.median(interval_residuals)),
                    -multiple,
                    bpm,
                )
            )

    if not candidates:
        raise RhythmError("Unable to infer a usable rhythm grid from note intervals.")

    consistency, _, _, _, _, bpm = max(candidates)
    if consistency < 0.55:
        raise RhythmError("Note onsets are too inconsistent to infer a rhythm grid.")

    return TempoGrid(float(bpm), float(notes[0].start), "intervals", consistency)


def detect_tempo_grid(
    accompaniment: np.ndarray, sample_rate: int, notes: list[NoteEvent]
) -> TempoGrid:
    """Prefer a consistent accompaniment beat track, then use melody intervals."""
    try:
        _, beat_frames = librosa.beat.beat_track(y=accompaniment, sr=sample_rate)
        beat_frames = np.asarray(beat_frames, dtype=int)
        beat_times = np.asarray(
            librosa.frames_to_time(beat_frames, sr=sample_rate), dtype=float
        )
    except Exception:
        beat_times = np.asarray([], dtype=float)

    if len(beat_times) >= 4 and np.all(np.isfinite(beat_times)):
        gaps = np.diff(beat_times)
        median_gap = float(np.median(gaps)) if len(gaps) else 0.0
        if median_gap > 0.0:
            beat_indices = np.arange(len(beat_times), dtype=float)
            period, anchor = np.polyfit(beat_indices, beat_times, 1)
            bpm = 60.0 / period if period > 0.0 else float("nan")
            gap_consistency = float(
                np.mean(np.abs(gaps - median_gap) / median_gap <= 0.08)
            )
            fit_residuals = np.abs(beat_times - (anchor + period * beat_indices))
            fit_consistency = float(np.mean(fit_residuals / period <= 0.08))
            if (
                np.isfinite(period)
                and period > 0.0
                and 45.0 <= bpm <= 210.0
                and gap_consistency >= 0.60
                and fit_consistency >= 0.60
            ):
                return TempoGrid(
                    bpm,
                    float(anchor),
                    "beats",
                    min(gap_consistency, fit_consistency),
                )
            if (
                np.all(gaps > 0.0)
                and np.isfinite(bpm)
                and 45.0 <= bpm <= 210.0
                and gap_consistency >= 0.60
            ):
                return TempoGrid(
                    float(bpm),
                    float(beat_times[0]),
                    "beats-adaptive",
                    gap_consistency,
                    tuple(float(value) for value in beat_times),
                )

    return _interval_grid(notes)


def _grid_slots(times: list[float], grid: TempoGrid) -> list[int]:
    """Map seconds to sixteenth-note slots on a fixed or observed beat grid."""
    if not grid.beat_times:
        slot_seconds = 60.0 / grid.bpm / 4.0
        return [int(np.rint((value - grid.anchor) / slot_seconds)) for value in times]

    beats = np.asarray(grid.beat_times, dtype=float)
    if (
        len(beats) < 2
        or not np.all(np.isfinite(beats))
        or not np.all(np.diff(beats) > 0)
    ):
        raise RhythmError("Adaptive beat times must be finite and strictly increasing.")

    values = np.asarray(times, dtype=float)
    beat_positions = np.interp(values, beats, np.arange(len(beats), dtype=float))
    before = values < beats[0]
    after = values > beats[-1]
    beat_positions[before] = (values[before] - beats[0]) / (beats[1] - beats[0])
    beat_positions[after] = (len(beats) - 1) + (
        (values[after] - beats[-1]) / (beats[-1] - beats[-2])
    )
    return np.rint(beat_positions * 4.0).astype(int).tolist()


def quantize_notes(
    notes: list[NoteEvent], grid: TempoGrid, transpose: int
) -> list[QuantizedNote]:
    """Place note events on a four-four sixteenth-note grid."""
    if not notes:
        raise RhythmError("Cannot quantize an empty melody.")
    if not np.isfinite(grid.bpm) or grid.bpm <= 0.0:
        raise RhythmError("Tempo must be a positive finite BPM value.")
    if not np.isfinite(grid.anchor):
        raise RhythmError("Tempo grid anchor must be finite.")

    previous_start = float("-inf")
    for note in notes:
        if not np.isfinite(note.start) or not np.isfinite(note.end):
            raise RhythmError("Note times must be finite.")
        if note.end < note.start:
            raise RhythmError("Note end must not precede its start.")
        if note.start < previous_start:
            raise RhythmError("Notes must be in chronological order.")
        previous_start = note.start

    desired_starts = _grid_slots([note.start for note in notes], grid)
    starts = [desired_starts[0]]
    for desired_start in desired_starts[1:]:
        starts.append(max(desired_start, starts[-1] + 1))
    desired_ends = _grid_slots([note.end for note in notes], grid)
    base_slot = (starts[0] // 16) * 16
    quantized: list[QuantizedNote] = []

    for index, (note, start_slot, desired_end) in enumerate(
        zip(notes, starts, desired_ends)
    ):
        end_slot = max(desired_end, start_slot + 1)
        if index + 1 < len(starts):
            end_slot = min(end_slot, starts[index + 1])
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
