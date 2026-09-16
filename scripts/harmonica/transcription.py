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
