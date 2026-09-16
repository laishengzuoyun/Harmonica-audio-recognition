from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np
import soundfile as sf


TPB_REFERENCE = 960
REFERENCE_BPM = 138
SLOW_BPM = 68.5
SLOT_TICKS = 480  # One sixteenth note when the song is counted at 69 BPM.
BAR_TICKS = SLOT_TICKS * 16
SOURCE_OCTAVE_CORRECTION = -12
TRANSPOSE = 5  # Original G major -> game-friendly C major.
AUDIO_TIME_SCALE = 1.007
AUDIO_TIME_OFFSET = 0.75

KEYS = ["Z", "X", "C", "V", "B", "N", "M", ","]
DEGREES = [1, 2, 3, 4, 5, 6, 7, 1]
OFFSETS = [0, 2, 4, 5, 7, 9, 11, 12]
PC_TO_DEGREE = {
    0: (1, False),
    1: (1, True),
    2: (2, False),
    3: (2, True),
    4: (3, False),
    5: (4, False),
    6: (4, True),
    7: (5, False),
    8: (5, True),
    9: (6, False),
    10: (6, True),
    11: (7, False),
}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass(frozen=True)
class Note:
    start_tick: int
    end_tick: int
    reference_midi: int
    velocity: int

    @property
    def source_midi(self) -> int:
        return self.reference_midi + SOURCE_OCTAVE_CORRECTION

    @property
    def play_midi(self) -> int:
        return self.source_midi + TRANSPOSE


@dataclass(frozen=True)
class Fingering:
    shift: int
    sharp: bool
    key_index: int

    @property
    def key(self) -> str:
        return KEYS[self.key_index]

    @property
    def combo(self) -> str:
        parts: list[str] = []
        if self.shift < 0:
            parts.append("左键")
        elif self.shift > 0:
            parts.append("右键")
        if self.sharp:
            parts.append("中键")
        parts.append(self.key)
        return "+".join(parts)


SECTIONS = {
    9: ("主歌 A", "00:29"),
    13: ("唱名过门", "00:43"),
    15: ("主歌 B", "00:50"),
    17: ("主歌 C（密集短音）", "00:57"),
    21: ("过渡", "01:10"),
    25: ("副歌 1", "01:25"),
    41: ("间奏", "02:21"),
    45: ("主歌重复", "02:35"),
    49: ("过渡 2", "02:48"),
    53: ("副歌 2", "03:03"),
    69: ("尾句／进入快速念白", "03:56"),
}


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def numbered_symbol(midi: int) -> str:
    relative = midi - 60
    octave, pc = divmod(relative, 12)
    degree, sharp = PC_TO_DEGREE[pc]
    if octave < 0:
        octave_mark = "↓" * abs(octave)
    elif octave > 0:
        octave_mark = "↑" * octave
    else:
        octave_mark = ""
    return f"{octave_mark}{'♯' if sharp else ''}{degree}"


def continuous_token(midi: int) -> str:
    relative = midi - 60
    octave, pc = divmod(relative, 12)
    degree, sharp = PC_TO_DEGREE[pc]

    if octave == 1 and degree == 1:
        return f"1({'中+' if sharp else ''},)"

    modifiers: list[str] = []
    if octave < 0:
        modifiers.extend(["左"] * abs(octave))
    elif octave > 0:
        modifiers.extend(["右"] * octave)
    if sharp:
        modifiers.append("中")
    return f"{degree}({'+'.join(modifiers)})" if modifiers else str(degree)


def fingerings(midi: int) -> list[Fingering]:
    relative = midi - 60
    result: list[Fingering] = []
    for shift in (-1, 0, 1):
        for sharp in (False, True):
            for key_index, offset in enumerate(OFFSETS):
                if 12 * shift + offset + int(sharp) == relative:
                    result.append(Fingering(shift, sharp, key_index))
    return result


def canonical_fingering(midi: int) -> Fingering:
    options = fingerings(midi)
    if not options:
        raise ValueError(f"Pitch {note_name(midi)} is outside the game instrument range")
    return min(
        options,
        key=lambda f: (
            int(f.shift != 0) + int(f.sharp),
            int(f.shift != 0),
            int(f.sharp),
            abs(f.shift),
            f.key_index,
        ),
    )


def extract_track_notes(midi_path: Path, track_name: str) -> list[Note]:
    midi = mido.MidiFile(midi_path)
    if midi.ticks_per_beat != TPB_REFERENCE:
        raise ValueError(f"Unexpected MIDI tick division: {midi.ticks_per_beat}")
    track = next((t for t in midi.tracks if t.name.lower() == track_name.lower()), None)
    if track is None:
        raise ValueError(f"Track {track_name!r} not found")

    tick = 0
    active: dict[tuple[int, int], list[tuple[int, int]]] = {}
    notes: list[Note] = []
    for message in track:
        tick += message.time
        if message.type == "note_on" and message.velocity > 0:
            active.setdefault((message.channel, message.note), []).append((tick, message.velocity))
        elif message.type in {"note_off", "note_on"} and (
            message.type == "note_off" or message.velocity == 0
        ):
            key = (message.channel, message.note)
            if key not in active or not active[key]:
                continue
            start, velocity = active[key].pop(0)
            notes.append(Note(start, tick, message.note, velocity))
    return sorted(notes, key=lambda n: (n.start_tick, n.end_tick, n.reference_midi))


def reference_seconds(tick: int) -> float:
    return tick / TPB_REFERENCE * 60.0 / REFERENCE_BPM


def audio_seconds(tick: int) -> float:
    return AUDIO_TIME_SCALE * reference_seconds(tick) + AUDIO_TIME_OFFSET


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - 60 * minutes
    return f"{minutes:02d}:{remainder:04.1f}"


def build_grid(notes: list[Note]) -> tuple[int, int, dict[int, list[str]]]:
    first_bar = min(n.start_tick for n in notes) // BAR_TICKS + 1
    last_bar = math.ceil(max(n.end_tick for n in notes) / BAR_TICKS)
    grid = {bar: ["·"] * 16 for bar in range(first_bar, last_bar + 1)}
    for note in notes:
        start_slot = round(note.start_tick / SLOT_TICKS)
        end_slot = round(note.end_tick / SLOT_TICKS)
        if end_slot <= start_slot:
            end_slot = start_slot + 1
        for absolute_slot in range(start_slot, end_slot):
            bar = absolute_slot // 16 + 1
            slot = absolute_slot % 16
            grid[bar][slot] = numbered_symbol(note.play_midi) if absolute_slot == start_slot else "—"
    return first_bar, last_bar, grid


def markdown_score(notes: list[Note]) -> str:
    first_bar, last_bar, grid = build_grid(notes)
    lines = [
        "# 《晴天》——《三角洲行动》口琴谱",
        "",
        "版本：周杰伦原唱录音的单音主旋律；4/4 拍，原速约 **68.5 BPM**，练习时可先降到 55–60 BPM。",
        "",
        "为减少鼠标修饰键，谱面已将原曲 G 调整体上移 5 个半音到 C 调。音域为 G3–D5，在游戏口琴的完整音域内。",
        "",
        "## 记号",
        "",
        "- `↓5`：低音 5，按住鼠标左键后按 `B`。",
        "- `1`–`7`：中音，对应 `Z X C V B N M`。",
        "- `↑1`：高音 1，直接按逗号键 `,`；`↑2`及以上用鼠标右键加对应数字键。",
        "- `♯`：同时按住鼠标中键。",
        "- `—`：延长前一音；`·`：休止。",
        "- 每行一小节，每个 `|` 内是一拍，每拍四格（十六分音符网格）。",
        "",
        "## 完整谱",
        "",
    ]
    melody_bar = 0
    for absolute_bar in range(first_bar, last_bar + 1):
        if absolute_bar in SECTIONS:
            title, timestamp = SECTIONS[absolute_bar]
            lines.extend([f"### {title}（原音约 {timestamp}）", ""])
        melody_bar += 1
        cells = grid[absolute_bar]
        groups = [" ".join(cells[i : i + 4]) for i in range(0, 16, 4)]
        lines.append(f"{melody_bar:02d}  | " + " | ".join(groups) + " |")
        if absolute_bar in {12, 14, 16, 20, 24, 40, 44, 48, 52, 68, 69}:
            lines.append("")
    lines.extend(
        [
            "",
            "## 纯音符速查",
            "",
            "下面删掉了延音和休止，适合先记指法；`/` 表示换小节。",
            "",
        ]
    )
    current_section: list[str] = []
    current_title = ""
    for absolute_bar in range(first_bar, last_bar + 1):
        if absolute_bar in SECTIONS:
            if current_section:
                lines.extend([f"**{current_title}**  ", " / ".join(current_section), ""])
            current_title = SECTIONS[absolute_bar][0]
            current_section = []
        current_section.append(" ".join(token for token in grid[absolute_bar] if token not in {"·", "—"}) or "·")
    if current_section:
        lines.extend([f"**{current_title}**  ", " / ".join(current_section), ""])
    lines.extend(
        [
            "",
            "## 范围与取舍",
            "",
            "谱面覆盖从约 00:29 开始的主旋律，到 04:02 左右进入快速念白。后面的念白没有稳定单音音高，因此不强行塞进口琴谱。",
            "",
        ]
    )
    return "\n".join(lines)


def continuous_markdown(notes: list[Note]) -> str:
    first_bar, last_bar, _ = build_grid(notes)
    notes_by_bar: dict[int, list[Note]] = {
        bar: [] for bar in range(first_bar, last_bar + 1)
    }
    for note in notes:
        bar = note.start_tick // BAR_TICKS + 1
        notes_by_bar[bar].append(note)

    lines = [
        "# 《晴天》——连续按键版",
        "",
        "按数字从左到右连续演奏，`｜` 代表换小节。",
        "本页只显示按键顺序；时值、延音和休止请查看《晴天-三角洲口琴谱》详细节奏谱。",
        "",
        "- `5(左)`：鼠标左键＋数字 5",
        "- `4(中)`：鼠标中键＋数字 4",
        "- `2(右)`：鼠标右键＋数字 2",
        "- `1(,)`：直接按逗号键",
        "",
    ]

    current_bars: list[str] = []
    for bar in range(first_bar, last_bar + 1):
        if bar in SECTIONS:
            if current_bars:
                lines.extend(["｜".join(current_bars), ""])
            title, timestamp = SECTIONS[bar]
            lines.extend([f"## {title}（{timestamp}）", ""])
            current_bars = []
        tokens = [continuous_token(note.play_midi) for note in notes_by_bar[bar]]
        if tokens:
            current_bars.append(" ".join(tokens))
    if current_bars:
        lines.extend(["｜".join(current_bars), ""])
    return "\n".join(lines)


def write_csv(notes: list[Note], output: Path) -> None:
    fields = [
        "audio_start",
        "audio_end",
        "bar",
        "slot",
        "duration_slots",
        "source_note",
        "play_note",
        "numbered",
        "game_input",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        first_bar = min(n.start_tick for n in notes) // BAR_TICKS + 1
        for note in notes:
            absolute_slot = round(note.start_tick / SLOT_TICKS)
            absolute_bar = absolute_slot // 16 + 1
            writer.writerow(
                {
                    "audio_start": format_time(audio_seconds(note.start_tick)),
                    "audio_end": format_time(audio_seconds(note.end_tick)),
                    "bar": absolute_bar - first_bar + 1,
                    "slot": absolute_slot % 16 + 1,
                    "duration_slots": max(1, round((note.end_tick - note.start_tick) / SLOT_TICKS)),
                    "source_note": note_name(note.source_midi),
                    "play_note": note_name(note.play_midi),
                    "numbered": numbered_symbol(note.play_midi),
                    "game_input": canonical_fingering(note.play_midi).combo,
                }
            )


def write_midi(notes: list[Note], output: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    meta = mido.MidiTrack()
    midi.tracks.append(meta)
    meta.append(mido.MetaMessage("track_name", name="Sunny Day - Delta Force Harmonica", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(SLOW_BPM), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    melody = mido.MidiTrack()
    midi.tracks.append(melody)
    melody.append(mido.MetaMessage("track_name", name="Harmonica melody", time=0))
    melody.append(mido.Message("program_change", program=22, channel=0, time=0))
    first_tick = (min(n.start_tick for n in notes) // BAR_TICKS) * BAR_TICKS
    events: list[tuple[int, int, mido.Message]] = []
    for note in notes:
        start = (note.start_tick - first_tick) // 4
        end = (note.end_tick - first_tick) // 4
        events.append((start, 1, mido.Message("note_on", note=note.play_midi, velocity=88, channel=0)))
        events.append((end, 0, mido.Message("note_off", note=note.play_midi, velocity=0, channel=0)))
    events.sort(key=lambda item: (item[0], item[1]))
    previous = 0
    for tick, _, message in events:
        message.time = tick - previous
        melody.append(message)
        previous = tick
    melody.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(output)


def write_preview(notes: list[Note], output: Path, sample_rate: int = 22050) -> None:
    first_time = audio_seconds(min(n.start_tick for n in notes))
    last_time = audio_seconds(max(n.end_tick for n in notes))
    lead_in = 0.5
    length = int((last_time - first_time + 1.2) * sample_rate)
    audio = np.zeros(length, dtype=np.float64)
    rng = np.random.default_rng(20260916)
    for note in notes:
        start = lead_in + audio_seconds(note.start_tick) - first_time
        end = lead_in + audio_seconds(note.end_tick) - first_time
        start_sample = max(0, int(start * sample_rate))
        end_sample = min(length, int(end * sample_rate))
        if end_sample <= start_sample:
            continue
        t = np.arange(end_sample - start_sample, dtype=np.float64) / sample_rate
        frequency = 440.0 * 2 ** ((note.play_midi - 69) / 12)
        vibrato = 1.0 + 0.0025 * np.sin(2 * np.pi * 5.3 * t)
        phase = 2 * np.pi * frequency * np.cumsum(vibrato) / sample_rate
        wave = (
            np.sin(phase)
            + 0.42 * np.sin(2 * phase + 0.08)
            + 0.22 * np.sin(3 * phase + 0.17)
            + 0.11 * np.sin(4 * phase + 0.31)
            + 0.06 * np.sin(5 * phase + 0.47)
        )
        attack = min(len(t), int(0.025 * sample_rate))
        release = min(len(t), int(0.045 * sample_rate))
        envelope = np.ones_like(t)
        if attack:
            envelope[:attack] = np.linspace(0.0, 1.0, attack)
        if release:
            envelope[-release:] *= np.linspace(1.0, 0.0, release)
        breath = rng.standard_normal(len(t)) * 0.012
        audio[start_sample:end_sample] += 0.18 * envelope * (wave + breath)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio *= 0.92 / peak
    sf.write(output, audio.astype(np.float32), sample_rate, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("midi", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    notes = extract_track_notes(args.midi, "Lead")
    if not all(fingerings(note.play_midi) for note in notes):
        raise ValueError("Transposed score exceeds the game instrument range")

    score_path = args.output / "晴天-三角洲口琴谱.md"
    score_path.write_text(markdown_score(notes), encoding="utf-8")
    continuous_path = args.output / "晴天-连续按键版.md"
    continuous_path.write_text(continuous_markdown(notes), encoding="utf-8")
    write_csv(notes, args.output / "晴天-音符明细.csv")
    write_midi(notes, args.output / "晴天-口琴主旋律.mid")
    write_preview(notes, args.output / "晴天-口琴主旋律试听.wav")

    print(f"notes={len(notes)}")
    print(f"score={score_path}")
    print(f"continuous_score={continuous_path}")
    print(f"source_range={note_name(min(n.source_midi for n in notes))}..{note_name(max(n.source_midi for n in notes))}")
    print(f"play_range={note_name(min(n.play_midi for n in notes))}..{note_name(max(n.play_midi for n in notes))}")
    print(f"audio_span={format_time(audio_seconds(min(n.start_tick for n in notes)))}..{format_time(audio_seconds(max(n.end_tick for n in notes)))}")


if __name__ == "__main__":
    main()
