from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass
class Event:
    start: float
    end: float
    midi: int
    confidence: float

    @property
    def duration(self) -> float:
        return self.end - self.start


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
DEGREES = {0: "1", 2: "2", 4: "3", 5: "4", 7: "5", 9: "6", 11: "7"}


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def game_notation(midi: int, base_c: int = 60) -> str:
    """Map a MIDI pitch onto the game's C-major row plus octave/sharp modifiers."""
    rel = midi - base_c
    octave, pc = divmod(rel, 12)
    if pc in DEGREES:
        degree = DEGREES[pc]
        sharp = False
    else:
        # The game exposes sharps. Spell black keys as the sharp of the lower degree.
        lower_pc = (pc - 1) % 12
        degree = DEGREES[lower_pc]
        sharp = True
    if octave <= -1:
        band = "L"
    elif octave == 0:
        band = "M"
    else:
        band = "H"
    return f"{band}{'#' if sharp else ''}{degree}"


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def smooth_quantized_pitch(midi: np.ndarray, valid: np.ndarray, radius: int = 4) -> np.ndarray:
    rounded = np.rint(midi).astype(float)
    rounded[~valid] = np.nan
    result = rounded.copy()
    for start, end in contiguous_runs(valid):
        for i in range(start, end):
            left = max(start, i - radius)
            right = min(end, i + radius + 1)
            values = rounded[left:right]
            values = values[np.isfinite(values)].astype(int)
            if len(values):
                result[i] = Counter(values.tolist()).most_common(1)[0][0]
    return result


def bridge_tiny_gaps(pitch: np.ndarray, max_frames: int) -> np.ndarray:
    out = pitch.copy()
    nan_runs = contiguous_runs(~np.isfinite(out))
    for start, end in nan_runs:
        if end - start > max_frames or start == 0 or end == len(out):
            continue
        if out[start - 1] == out[end]:
            out[start:end] = out[start - 1]
    return out


def pitch_runs(pitch: np.ndarray, times: np.ndarray, probs: np.ndarray, hop_seconds: float) -> list[Event]:
    events: list[Event] = []
    i = 0
    while i < len(pitch):
        if not np.isfinite(pitch[i]):
            i += 1
            continue
        value = int(pitch[i])
        j = i + 1
        while j < len(pitch) and np.isfinite(pitch[j]) and int(pitch[j]) == value:
            j += 1
        events.append(
            Event(
                start=float(times[i]),
                end=float(times[j - 1] + hop_seconds),
                midi=value,
                confidence=float(np.nanmean(probs[i:j])),
            )
        )
        i = j
    return events


def clean_events(events: list[Event], min_duration: float = 0.075) -> list[Event]:
    # Fold very short pitch flips into a compatible neighbour. These are usually
    # vibrato boundaries or consonant artefacts rather than intentional notes.
    events = [Event(e.start, e.end, e.midi, e.confidence) for e in events]
    changed = True
    while changed:
        changed = False
        for i, event in enumerate(events):
            if event.duration >= min_duration:
                continue
            prev = events[i - 1] if i else None
            nxt = events[i + 1] if i + 1 < len(events) else None
            replacement: Event | None = None
            if prev and nxt and prev.midi == nxt.midi and nxt.start - prev.end < 0.16:
                prev.end = nxt.end
                prev.confidence = max(prev.confidence, nxt.confidence)
                del events[i : i + 2]
                changed = True
                break
            if prev and abs(prev.midi - event.midi) <= 1 and event.start - prev.end < 0.05:
                prev.end = event.end
                replacement = prev
            elif nxt and abs(nxt.midi - event.midi) <= 1 and nxt.start - event.end < 0.05:
                nxt.start = event.start
                replacement = nxt
            if replacement is not None:
                del events[i]
                changed = True
                break
    return [e for e in events if e.duration >= min_duration]


def phrase_groups(events: list[Event], gap: float = 0.42) -> list[list[Event]]:
    phrases: list[list[Event]] = []
    current: list[Event] = []
    for event in events:
        if current and event.start - current[-1].end >= gap:
            phrases.append(current)
            current = []
        current.append(event)
    if current:
        phrases.append(current)
    return phrases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vocals", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--transpose", type=int, default=5)
    parser.add_argument("--sr", type=int, default=22050)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(args.vocals, sr=args.sr, mono=True)
    hop = 256
    frame_length = 2048
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop,
        fill_na=np.nan,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0][: len(f0)]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    midi = librosa.hz_to_midi(f0)

    # The source vocal is comfortably inside this range. The range limit removes
    # residual bass/guitar leakage left by source separation.
    valid = (
        voiced_flag
        & np.isfinite(midi)
        & (voiced_prob >= 0.72)
        & (rms_db >= -42.0)
        & (midi >= 48)
        & (midi <= 74)
    )
    pitch = smooth_quantized_pitch(midi, valid)
    pitch = bridge_tiny_gaps(pitch, max_frames=5)
    events = clean_events(pitch_runs(pitch, times, voiced_prob, hop / sr))

    with (args.output / "pitch_frames.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            times=times,
            f0=f0,
            midi=midi,
            voiced_prob=voiced_prob,
            rms_db=rms_db,
            valid=valid,
            pitch=pitch,
        )

    fields = [
        "start",
        "end",
        "duration",
        "source_midi",
        "source_note",
        "play_midi",
        "play_note",
        "game",
        "confidence",
    ]
    with (args.output / "note_events.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            play_midi = event.midi + args.transpose
            writer.writerow(
                {
                    "start": f"{event.start:.3f}",
                    "end": f"{event.end:.3f}",
                    "duration": f"{event.duration:.3f}",
                    "source_midi": event.midi,
                    "source_note": note_name(event.midi),
                    "play_midi": play_midi,
                    "play_note": note_name(play_midi),
                    "game": game_notation(play_midi),
                    "confidence": f"{event.confidence:.3f}",
                }
            )

    phrases = phrase_groups(events)
    payload = []
    for phrase in phrases:
        payload.append(
            {
                "start": round(phrase[0].start, 3),
                "end": round(phrase[-1].end, 3),
                "notes": [game_notation(e.midi + args.transpose) for e in phrase],
                "source_notes": [note_name(e.midi) for e in phrase],
                "durations": [round(e.duration, 3) for e in phrase],
                "confidence": round(float(np.mean([e.confidence for e in phrase])), 3),
            }
        )
    (args.output / "phrases.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"duration_seconds={len(y) / sr:.3f}")
    print(f"voiced_frames={int(valid.sum())}/{len(valid)}")
    print(f"events={len(events)} phrases={len(phrases)}")
    print(f"source_range={note_name(min(e.midi for e in events))}..{note_name(max(e.midi for e in events))}")
    print(
        f"play_range={note_name(min(e.midi for e in events) + args.transpose)}.."
        f"{note_name(max(e.midi for e in events) + args.transpose)}"
    )


if __name__ == "__main__":
    main()
