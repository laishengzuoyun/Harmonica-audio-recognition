"""Pitch-frame extraction and cleanup helpers for vocal transcription."""

from collections import Counter
from dataclasses import dataclass

import librosa
import numpy as np

from .models import NoteEvent, TranscriptionError


@dataclass(frozen=True)
class PitchFrames:
    """Frame-level pitch analysis produced from a monophonic audio signal."""

    times: np.ndarray
    midi: np.ndarray
    voiced_prob: np.ndarray
    rms_db: np.ndarray
    pitch: np.ndarray
    hop_seconds: float


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open index spans of the contiguous true regions in *mask*."""
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def smooth_quantized_pitch(
    midi: np.ndarray, valid: np.ndarray, radius: int = 4
) -> np.ndarray:
    """Quantize valid pitch frames and smooth each run with its local mode."""
    rounded = np.rint(midi).astype(float)
    rounded[~valid] = np.nan
    result = rounded.copy()

    for start, end in contiguous_runs(valid):
        for index in range(start, end):
            left = max(start, index - radius)
            right = min(end, index + radius + 1)
            values = rounded[left:right]
            values = values[np.isfinite(values)].astype(int)
            if len(values):
                result[index] = Counter(values.tolist()).most_common(1)[0][0]
    return result


def bridge_tiny_gaps(pitch: np.ndarray, max_frames: int) -> np.ndarray:
    """Fill short unvoiced gaps only when the surrounding pitches agree."""
    result = np.asarray(pitch, dtype=float).copy()
    for start, end in contiguous_runs(~np.isfinite(result)):
        if end - start > max_frames or start == 0 or end == len(result):
            continue
        if result[start - 1] == result[end]:
            result[start:end] = result[start - 1]
    return result


def extract_pitch_frames(
    audio: np.ndarray, sample_rate: int, hop_length: int = 256
) -> PitchFrames:
    """Extract cleaned, integer MIDI pitch frames from a vocal audio signal."""
    frame_length = 2048
    signal = np.asarray(audio, dtype=float).reshape(-1)
    hop_seconds = hop_length / sample_rate
    if len(signal) == 0:
        empty = np.array([], dtype=float)
        return PitchFrames(empty, empty, empty, empty, empty, hop_seconds)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        signal,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        fill_na=np.nan,
    )
    midi = librosa.hz_to_midi(f0)
    rms = librosa.feature.rms(
        y=signal, frame_length=frame_length, hop_length=hop_length
    )[0][: len(f0)]
    reference = max(float(np.max(rms, initial=0.0)), 1e-9)
    rms_db = librosa.amplitude_to_db(rms, ref=reference)
    finite_energy = rms_db[np.isfinite(rms_db)]
    energy_floor = (
        max(-48.0, float(np.quantile(finite_energy, 0.2)))
        if len(finite_energy)
        else -48.0
    )

    valid = (
        np.asarray(voiced_flag, dtype=bool)
        & np.isfinite(midi)
        & (voiced_prob >= 0.55)
        & (rms_db >= energy_floor)
        & (midi >= 36)
        & (midi <= 84)
    )
    pitch = smooth_quantized_pitch(midi, valid)
    pitch = bridge_tiny_gaps(pitch, max_frames=round(0.06 * sample_rate / hop_length))
    times = librosa.frames_to_time(
        np.arange(len(f0)), sr=sample_rate, hop_length=hop_length
    )

    return PitchFrames(
        times=times,
        midi=midi,
        voiced_prob=voiced_prob,
        rms_db=rms_db,
        pitch=pitch,
        hop_seconds=hop_seconds,
    )


def detect_onsets(
    audio: np.ndarray, sr: int, hop: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Detect vocal attacks using both spectral and energy changes."""
    signal = np.asarray(audio, dtype=float).reshape(-1)
    if len(signal) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    spectral = librosa.onset.onset_strength(
        y=signal, sr=sr, hop_length=hop
    )
    rms = librosa.feature.rms(y=signal, frame_length=2048, hop_length=hop)[0]
    frame_count = min(len(spectral), len(rms))
    spectral = spectral[:frame_count]
    rms = rms[:frame_count]
    rms_change = np.maximum(np.diff(rms, prepend=rms[0]), 0.0)

    def normalized(values: np.ndarray) -> np.ndarray:
        maximum = float(np.max(values, initial=0.0))
        if maximum <= 0.0 or not np.isfinite(maximum):
            return np.zeros_like(values, dtype=float)
        return np.nan_to_num(values / maximum, nan=0.0, posinf=0.0, neginf=0.0)

    envelope = np.maximum(normalized(spectral), normalized(rms_change))
    wait_frames = max(1, int(np.ceil(0.06 * sr / hop)))
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sr,
        hop_length=hop,
        backtrack=False,
        units="frames",
        delta=0.07,
        wait=wait_frames,
    )
    return np.asarray(onset_frames, dtype=int), envelope


def _raw_pitch_segments(pitch: np.ndarray) -> list[tuple[int, int, int]]:
    """Return half-open runs of the same finite, integer MIDI pitch."""
    values = np.asarray(pitch, dtype=float).reshape(-1)
    segments: list[tuple[int, int, int]] = []
    start: int | None = None
    current_midi: int | None = None

    for index, value in enumerate(values):
        midi = int(np.rint(value)) if np.isfinite(value) else None
        if midi == current_midi and midi is not None:
            continue
        if start is not None and current_midi is not None:
            segments.append((start, index, current_midi))
        start = index if midi is not None else None
        current_midi = midi

    if start is not None and current_midi is not None:
        segments.append((start, len(values), current_midi))
    return segments


def frames_to_notes(
    times: np.ndarray,
    pitch: np.ndarray,
    voiced_prob: np.ndarray,
    onset_frames: np.ndarray,
    onset_envelope: np.ndarray,
    hop_seconds: float,
) -> list[NoteEvent]:
    """Turn cleaned pitch frames into onset-aware note events."""
    frame_times = np.asarray(times, dtype=float).reshape(-1)
    pitches = np.asarray(pitch, dtype=float).reshape(-1)
    probabilities = np.asarray(voiced_prob, dtype=float).reshape(-1)
    envelope = np.asarray(onset_envelope, dtype=float).reshape(-1)
    frame_count = min(len(frame_times), len(pitches), len(probabilities))
    if frame_count == 0:
        return []

    onset_max = float(np.max(envelope, initial=0.0))
    normalized_onsets = (
        np.nan_to_num(envelope / onset_max, nan=0.0, posinf=0.0, neginf=0.0)
        if onset_max > 0.0 and np.isfinite(onset_max)
        else np.zeros_like(envelope)
    )
    minimum_onset_gap = max(1, int(np.ceil(0.06 / hop_seconds)))
    spaced_onsets: list[int] = []
    for onset in sorted(set(np.asarray(onset_frames, dtype=int).tolist())):
        if onset < 0 or onset >= frame_count:
            continue
        if not spaced_onsets or onset - spaced_onsets[-1] >= minimum_onset_gap:
            spaced_onsets.append(onset)

    minimum_segment_frames = max(1, int(np.ceil(0.07 / hop_seconds)))
    notes: list[NoteEvent] = []
    for start, end, midi in _raw_pitch_segments(pitches[:frame_count]):
        cuts = [start]
        for onset in spaced_onsets:
            if (
                start < onset < end
                and onset - cuts[-1] >= minimum_segment_frames
                and end - onset >= minimum_segment_frames
            ):
                cuts.append(onset)
        cuts.append(end)

        for piece_start, piece_end in zip(cuts, cuts[1:]):
            piece_pitch = pitches[piece_start:piece_end]
            piece_probability = probabilities[piece_start:piece_end]
            probability = float(np.nanmean(piece_probability))
            if not np.isfinite(probability):
                probability = 0.0
            pitch_std = float(np.nanstd(piece_pitch))
            stability = max(0.0, 1.0 - pitch_std / 0.7)
            onset_strength = (
                float(normalized_onsets[piece_start])
                if piece_start < len(normalized_onsets)
                else 0.0
            )
            confidence = float(
                np.clip(
                    0.65 * probability + 0.25 * stability + 0.10 * onset_strength,
                    0.0,
                    1.0,
                )
            )
            notes.append(
                NoteEvent(
                    start=float(frame_times[piece_start]),
                    end=float(frame_times[piece_end - 1] + hop_seconds),
                    midi=midi,
                    confidence=confidence,
                    onset_strength=onset_strength,
                )
            )

    return clean_notes(notes)


def clean_notes(
    notes: list[NoteEvent], min_duration: float = 0.075
) -> list[NoteEvent]:
    """Remove short artifacts, folding them into matching neighbours."""
    cleaned = list(notes)
    while True:
        short_index = next(
            (index for index, note in enumerate(cleaned) if note.duration < min_duration),
            None,
        )
        if short_index is None:
            return cleaned

        index = short_index
        if (
            0 < index < len(cleaned) - 1
            and cleaned[index - 1].midi == cleaned[index + 1].midi
        ):
            previous, artifact, following = cleaned[index - 1 : index + 2]
            cleaned[index - 1 : index + 2] = [
                NoteEvent(
                    start=previous.start,
                    end=following.end,
                    midi=previous.midi,
                    confidence=max(
                        previous.confidence, artifact.confidence, following.confidence
                    ),
                    onset_strength=max(
                        previous.onset_strength,
                        artifact.onset_strength,
                        following.onset_strength,
                    ),
                )
            ]
        else:
            del cleaned[index]


def validate_melody(notes: list[NoteEvent]) -> None:
    """Reject transcriptions too weak or short to form a usable melody."""
    if len(notes) < 12:
        raise TranscriptionError("旋律音符不足：至少需要 12 个音符。")
    if sum(note.duration for note in notes) < 8.0:
        raise TranscriptionError("旋律有效时长不足：音符总时长至少需要 8 秒。")
    if float(np.median([note.confidence for note in notes])) < 0.55:
        raise TranscriptionError("旋律置信度过低：中位置信度需达到 0.55。")


def transcribe_vocals(
    audio: np.ndarray, sr: int
) -> tuple[list[NoteEvent], PitchFrames]:
    """Transcribe monophonic vocals into validated note events."""
    frames = extract_pitch_frames(audio, sr)
    onset_frames, onset_envelope = detect_onsets(audio, sr)
    notes = frames_to_notes(
        frames.times,
        frames.pitch,
        frames.voiced_prob,
        onset_frames,
        onset_envelope,
        frames.hop_seconds,
    )
    validate_melody(notes)
    return notes, frames
