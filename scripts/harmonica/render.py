"""Render quantized harmonica notes to human- and machine-readable files."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from .instrument import KEYS, canonical_fingering, continuous_token
from .models import HarmonicaError, QuantizedNote, TempoGrid

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(midi: int) -> str:
    """Return a MIDI note name using scientific pitch notation."""
    value = int(midi)
    return f"{NOTE_NAMES[value % 12]}{value // 12 - 1}"


def _format_time(seconds: float) -> str:
    minutes, remainder = divmod(max(0.0, float(seconds)), 60.0)
    return f"{int(minutes):02d}:{remainder:04.1f}"


def _validate_sections(notes: Sequence[QuantizedNote], sections: Sequence[int]) -> None:
    if not sections or sections[0] != 0:
        raise HarmonicaError("分段 section 必须从第 0 个音符开始")
    if any(not isinstance(index, int) for index in sections):
        raise HarmonicaError("分段 section 索引必须是整数")
    if list(sections) != sorted(set(sections)):
        raise HarmonicaError("分段 section 索引必须严格递增")
    if any(index < 0 or index >= len(notes) for index in sections):
        raise HarmonicaError("分段 section 索引超出音符范围")


def continuous_markdown(
    title: str, notes: Sequence[QuantizedNote], sections: Sequence[int]
) -> str:
    """Create a compact, continuous key guide split into musical sections."""
    _validate_sections(notes, sections)
    lines = [
        f"# {title}—连续按键谱",
        "",
        "> 括号表示需要同时按住的鼠标键；全角竖线表示小节线。",
        "",
    ]
    boundaries = list(sections) + [len(notes)]
    for section_number, (start, stop) in enumerate(
        zip(boundaries, boundaries[1:]), start=1
    ):
        section_notes = notes[start:stop]
        lines.append(
            f"## 第 {section_number} 段（{_format_time(section_notes[0].start)}）"
        )
        lines.append("")
        tokens: list[str] = []
        previous_bar: int | None = None
        for note in section_notes:
            if previous_bar is not None and note.bar != previous_bar:
                tokens.append("｜")
            tokens.append(continuous_token(note.play_midi))
            previous_bar = note.bar
        lines.append(" ".join(tokens))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def detailed_markdown(
    title: str, notes: Sequence[QuantizedNote], sections: Sequence[int]
) -> str:
    """Create a sixteenth-note grid, including rests and held-note cells."""
    _validate_sections(notes, sections)
    cells: dict[int, str] = {}
    for note in notes:
        onset = (note.bar - 1) * 16 + note.slot - 1
        cells[onset] = continuous_token(note.play_midi)
        for position in range(onset + 1, onset + note.duration_slots):
            cells.setdefault(position, "—")

    sections_at_bar: dict[int, list[tuple[int, float]]] = {}
    for number, index in enumerate(sections, 1):
        sections_at_bar.setdefault(notes[index].bar, []).append(
            (number, notes[index].start)
        )
    final_position = max(cells)
    final_bar = final_position // 16 + 1
    lines = [
        f"# {title}—详细节奏谱",
        "",
        "> 每小节 16 格，每 4 格为一拍；“·”是休止，“—”是延音。",
        "",
    ]
    for bar in range(1, final_bar + 1):
        for section_number, timestamp in sections_at_bar.get(bar, []):
            lines.append(
                f"## 第 {section_number} 段（{_format_time(timestamp)}）"
            )
            lines.append("")
        bar_start = (bar - 1) * 16
        values = [cells.get(bar_start + offset, "·") for offset in range(16)]
        beats = [" ".join(values[start : start + 4]) for start in range(0, 16, 4)]
        lines.append(f"**第 {bar} 小节**  | " + " | ".join(beats) + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _game_input(midi: int) -> str:
    fingering = canonical_fingering(midi)
    controls: list[str] = []
    if fingering.shift < 0:
        controls.append("左")
    elif fingering.shift > 0:
        controls.append("右")
    if fingering.sharp:
        controls.append("中")
    controls.append(KEYS[fingering.key_index])
    return "+".join(controls)


def write_csv(notes: Sequence[QuantizedNote], path: Path) -> None:
    """Write note details as an Excel-friendly UTF-8 CSV."""
    fields = (
        "start",
        "end",
        "source_note",
        "play_note",
        "confidence",
        "bar",
        "slot",
        "duration_slots",
        "game_input",
    )
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for note in notes:
            writer.writerow(
                {
                    "start": note.start,
                    "end": note.end,
                    "source_note": note_name(note.source_midi),
                    "play_note": note_name(note.play_midi),
                    "confidence": note.confidence,
                    "bar": note.bar,
                    "slot": note.slot,
                    "duration_slots": note.duration_slots,
                    "game_input": _game_input(note.play_midi),
                }
            )


def _score_seconds(notes: Sequence[QuantizedNote], bpm: float) -> float:
    slot_seconds = 60.0 / float(bpm) / 4.0
    final_slot = max(
        (note.bar - 1) * 16 + note.slot - 1 + note.duration_slots for note in notes
    )
    return final_slot * slot_seconds


def write_midi(
    notes: Sequence[QuantizedNote], grid: TempoGrid, path: Path
) -> float:
    """Write a tempo-aware harmonica MIDI and return its decoded duration."""
    ticks_per_beat = 480
    ticks_per_slot = ticks_per_beat // 4
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(grid.bpm), time=0)
    )
    track.append(mido.Message("program_change", program=22, time=0))

    events: list[tuple[int, int, int, mido.Message]] = []
    for order, note in enumerate(notes):
        start_slot = (note.bar - 1) * 16 + note.slot - 1
        start_tick = start_slot * ticks_per_slot
        end_tick = (start_slot + note.duration_slots) * ticks_per_slot
        events.append(
            (start_tick, 1, order, mido.Message("note_on", note=note.play_midi, velocity=88))
        )
        events.append(
            (end_tick, 0, order, mido.Message("note_off", note=note.play_midi, velocity=0))
        )

    previous_tick = 0
    for tick, _priority, _order, message in sorted(events, key=lambda event: event[:3]):
        message.time = tick - previous_tick
        track.append(message)
        previous_tick = tick
    midi.save(Path(path))
    return float(mido.MidiFile(Path(path)).length)


def write_preview(
    notes: Sequence[QuantizedNote],
    grid: TempoGrid,
    path: Path,
    sample_rate: int = 22_050,
) -> float:
    """Synthesize a deterministic, lightweight harmonica-like PCM preview."""
    total_seconds = _score_seconds(notes, grid.bpm)
    frame_count = max(1, round(total_seconds * sample_rate))
    waveform = np.zeros(frame_count, dtype=np.float64)
    slot_seconds = 60.0 / float(grid.bpm) / 4.0
    random = np.random.default_rng(20_260_916)

    for note in notes:
        start_slot = (note.bar - 1) * 16 + note.slot - 1
        start_frame = round(start_slot * slot_seconds * sample_rate)
        end_frame = min(
            frame_count,
            round((start_slot + note.duration_slots) * slot_seconds * sample_rate),
        )
        length = max(0, end_frame - start_frame)
        if length == 0:
            continue
        time = np.arange(length, dtype=np.float64) / sample_rate
        frequency = 440.0 * 2.0 ** ((note.play_midi - 69) / 12.0)
        tone = (
            np.sin(2.0 * np.pi * frequency * time)
            + 0.35 * np.sin(2.0 * np.pi * frequency * 2.0 * time)
            + 0.18 * np.sin(2.0 * np.pi * frequency * 3.0 * time)
            + 0.008 * random.standard_normal(length)
        )
        envelope = np.ones(length, dtype=np.float64)
        attack = min(length, max(1, round(0.020 * sample_rate)))
        release = min(length, max(1, round(0.040 * sample_rate)))
        envelope[:attack] *= np.linspace(0.0, 1.0, attack)
        envelope[-release:] *= np.linspace(1.0, 0.0, release)
        waveform[start_frame:end_frame] += tone * envelope

    peak = float(np.max(np.abs(waveform)))
    if peak > 0.0:
        waveform *= 0.92 / peak
    sf.write(Path(path), waveform, sample_rate, subtype="PCM_16")
    info = sf.info(Path(path))
    return float(info.frames / info.samplerate)


def render_all(
    title: str,
    notes: Sequence[QuantizedNote],
    grid: TempoGrid,
    transpose: int,
    output: Path,
    sections: Sequence[int],
    analysis: Mapping[str, object] | None = None,
) -> list[Path]:
    """Render markdown, CSV, MIDI, WAV, and JSON outputs for one melody."""
    notes = list(notes)
    if not notes:
        raise HarmonicaError("无法渲染空 empty 旋律")
    if not math.isfinite(grid.bpm) or grid.bpm <= 0:
        raise HarmonicaError("BPM tempo 必须是正的有限数")
    _validate_sections(notes, sections)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        output / f"{title}-连续按键谱.md",
        output / f"{title}-详细节奏谱.md",
        output / f"{title}-音符明细.csv",
        output / f"{title}-主旋律.mid",
        output / f"{title}-口琴试听.wav",
        output / f"{title}-分析报告.json",
    ]
    paths[0].write_text(continuous_markdown(title, notes, sections), encoding="utf-8")
    paths[1].write_text(detailed_markdown(title, notes, sections), encoding="utf-8")
    write_csv(notes, paths[2])
    midi_seconds = write_midi(notes, grid, paths[3])
    wav_seconds = write_preview(notes, grid, paths[4])

    source_pitches = [note.source_midi for note in notes]
    play_pitches = [note.play_midi for note in notes]
    report: dict[str, object] = {
        "bpm": grid.bpm,
        "grid_source": grid.source,
        "grid_consistency": grid.consistency,
        "transpose": transpose,
        "note_count": len(notes),
        "source_range": [
            note_name(min(source_pitches)),
            note_name(max(source_pitches)),
        ],
        "play_range": [note_name(min(play_pitches)), note_name(max(play_pitches))],
        "source_midi_range": [min(source_pitches), max(source_pitches)],
        "play_midi_range": [min(play_pitches), max(play_pitches)],
        "median_confidence": float(np.median([note.confidence for note in notes])),
        "midi_seconds": midi_seconds,
        "wav_seconds": wav_seconds,
        "filtered_frame_count": 0,
        "filtered_note_count": 0,
        "warnings": [],
    }
    if analysis is not None:
        report.update(analysis)
    paths[5].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return paths
