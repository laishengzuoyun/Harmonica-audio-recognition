# 音频自动提取《三角洲行动》口琴谱 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 Windows 本地一键工具，把 MP3/WAV/FLAC 中的单音人声主旋律自动转换为《三角洲行动》口琴连续按键谱、节奏谱、CSV、MIDI、WAV 试听和分析报告。

**Architecture:** 主程序只负责流程编排和事务式输出；音频分离、旋律转录、节拍量化、游戏指法和文件渲染分别放在独立模块。转录采用 Demucs 人声分离、pYIN 音高跟踪和发音起点检测的混合方法；现有《晴天》脚本保留为对照基线，新流程不依赖在线 MIDI。

**Tech Stack:** Python 3.12、Demucs 4.1、PyTorch、librosa、NumPy、SciPy、SoundFile、Mido、Windows 批处理、`unittest`

---

## 文件结构

- Create: `requirements-audio.txt` — 锁定直接音频依赖版本。
- Create: `提取口琴谱.bat` — 拖放入口、环境初始化和中文提示。
- Create: `scripts/extract_harmonica_score.py` — 命令行参数、阶段进度、事务式输出和失败保存。
- Create: `scripts/harmonica/__init__.py` — 包入口。
- Create: `scripts/harmonica/models.py` — 跨模块数据类型与领域异常。
- Create: `scripts/harmonica/audio.py` — 输入验证、磁盘检查、Demucs 和波形读取。
- Create: `scripts/harmonica/transcription.py` — pYIN、能量门限、音高平滑、起点拆分和置信度。
- Create: `scripts/harmonica/rhythm.py` — 拍点检测、间隔回退、十六分网格和段落边界。
- Create: `scripts/harmonica/instrument.py` — 游戏音域、规范指法、连续记号和自动移调。
- Create: `scripts/harmonica/render.py` — Markdown、CSV、MIDI、WAV 和 JSON。
- Create: `tests/test_models.py`
- Create: `tests/test_instrument.py`
- Create: `tests/test_transcription.py`
- Create: `tests/test_rhythm.py`
- Create: `tests/test_audio.py`
- Create: `tests/test_render.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_synthetic_pipeline.py`
- Create: `tests/test_sunny_regression.py`
- Modify: `tests/test_continuous_score.py` — 缺少本地参考 MIDI 时跳过旧回归测试。

## 统一接口

所有模块使用 `scripts/harmonica/models.py` 中的数据结构，避免相互猜测字段名：

```python
@dataclass(frozen=True)
class NoteEvent:
    start: float
    end: float
    midi: int
    confidence: float
    onset_strength: float = 0.0

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

@dataclass(frozen=True)
class Fingering:
    shift: int
    sharp: bool
    key_index: int
```

---

### Task 1: 建立包骨架、锁定依赖和公共数据类型

**Files:**
- Create: `requirements-audio.txt`
- Create: `scripts/harmonica/__init__.py`
- Create: `scripts/harmonica/models.py`
- Create: `tests/test_models.py`
- Modify: `tests/test_continuous_score.py:31-34`

- [ ] **Step 1: 写公共模型的失败测试**

```python
# tests/test_models.py
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from harmonica.models import NoteEvent, QuantizedNote, TempoGrid


class ModelTests(unittest.TestCase):
    def test_note_duration(self) -> None:
        self.assertAlmostEqual(NoteEvent(1.0, 1.25, 60, 0.8).duration, 0.25)

    def test_quantized_note_rejects_non_positive_duration(self) -> None:
        with self.assertRaisesRegex(ValueError, "duration_slots"):
            QuantizedNote(0.0, 0.1, 60, 60, 0.8, 1, 1, 0)

    def test_grid_source_is_explicit(self) -> None:
        grid = TempoGrid(120.0, 0.5, "beats", 0.9)
        self.assertEqual(grid.source, "beats")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_models -v`

Expected: FAIL，包含 `ModuleNotFoundError: No module named 'harmonica'`。

- [ ] **Step 3: 添加模型、异常和固定依赖**

```python
# scripts/harmonica/models.py
from __future__ import annotations

from dataclasses import dataclass


class HarmonicaError(RuntimeError):
    """用户可理解、无需回溯栈即可报告的转谱错误。"""


class InputValidationError(HarmonicaError):
    pass


class TranscriptionError(HarmonicaError):
    pass


class RhythmError(HarmonicaError):
    pass


class InstrumentRangeError(HarmonicaError):
    pass


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
        if self.bar < 1 or not 1 <= self.slot <= 16:
            raise ValueError("bar and slot must use one-based score positions")
        if self.duration_slots < 1:
            raise ValueError("duration_slots must be positive")


@dataclass(frozen=True)
class Fingering:
    shift: int
    sharp: bool
    key_index: int
```

```python
# scripts/harmonica/__init__.py
"""Audio-to-Delta-Force-harmonica transcription package."""
```

```text
# requirements-audio.txt
demucs==4.1.0
librosa==1.0.0
mido==1.3.3
numpy==2.5.3
scipy==1.18.1
soundfile==0.14.0
torch==2.14.0
```

在 `tests/test_continuous_score.py` 的 `setUpClass` 开头加入：

```python
if not MIDI_PATH.exists():
    raise unittest.SkipTest("local Sunny Day reference MIDI is not available")
```

- [ ] **Step 4: 运行公共模型和旧测试**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_models tests.test_continuous_score -v`

Expected: 15 tests PASS；若参考 MIDI 不存在，则旧文档测试显示 SKIP 而不是 FAIL。

- [ ] **Step 5: 提交包骨架**

```powershell
git add requirements-audio.txt scripts/harmonica tests/test_models.py tests/test_continuous_score.py
git commit -m "build: add transcription package foundation"
```

---

### Task 2: 实现游戏口琴指法和低操作成本移调

**Files:**
- Create: `scripts/harmonica/instrument.py`
- Create: `tests/test_instrument.py`

- [ ] **Step 1: 写完整音域、重叠指法、记号和移调优先级测试**

```python
# tests/test_instrument.py
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from harmonica.instrument import (
    canonical_fingering,
    choose_transpose,
    continuous_token,
    playable_pitches,
)
from harmonica.models import InstrumentRangeError, NoteEvent


class InstrumentTests(unittest.TestCase):
    def test_playable_range_is_contiguous_c3_to_c_sharp_6(self) -> None:
        self.assertEqual(playable_pitches(), set(range(48, 86)))

    def test_middle_c_prefers_plain_z_over_left_comma(self) -> None:
        fingering = canonical_fingering(60)
        self.assertEqual((fingering.shift, fingering.sharp, fingering.key_index), (0, False, 0))

    def test_high_c_prefers_plain_comma(self) -> None:
        self.assertEqual(continuous_token(72), "1(,)")

    def test_low_and_sharp_tokens_put_mouse_beside_number(self) -> None:
        self.assertEqual(continuous_token(55), "5(左)")
        self.assertEqual(continuous_token(66), "4(中)")
        self.assertEqual(continuous_token(56), "5(左+中)")

    def test_out_of_range_pitch_is_rejected(self) -> None:
        with self.assertRaises(InstrumentRangeError):
            canonical_fingering(47)

    def test_transpose_prefers_fewer_mouse_notes_before_small_shift(self) -> None:
        notes = [NoteEvent(i * 0.2, i * 0.2 + 0.15, p, 0.9) for i, p in enumerate([55, 57, 59, 60])]
        result = choose_transpose(notes)
        self.assertEqual(result, 5)

    def test_transpose_rejects_melody_wider_than_instrument(self) -> None:
        notes = [NoteEvent(0, 1, 40, 0.9), NoteEvent(1, 2, 90, 0.9)]
        with self.assertRaises(InstrumentRangeError):
            choose_transpose(notes)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行并确认失败**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_instrument -v`

Expected: FAIL，包含 `No module named 'harmonica.instrument'`。

- [ ] **Step 3: 实现候选指法、规范选择、连续记号和移调评分**

```python
# scripts/harmonica/instrument.py
from __future__ import annotations

from collections.abc import Iterable, Sequence

from .models import Fingering, InstrumentRangeError, NoteEvent

KEYS = ("Z", "X", "C", "V", "B", "N", "M", ",")
DEGREES = (1, 2, 3, 4, 5, 6, 7, 1)
OFFSETS = (0, 2, 4, 5, 7, 9, 11, 12)


def fingering_candidates(midi: int) -> list[Fingering]:
    relative = midi - 60
    return [
        Fingering(shift, sharp, key_index)
        for shift in (-1, 0, 1)
        for sharp in (False, True)
        for key_index, offset in enumerate(OFFSETS)
        if 12 * shift + offset + int(sharp) == relative
    ]


def playable_pitches() -> set[int]:
    return {midi for midi in range(0, 128) if fingering_candidates(midi)}


def canonical_fingering(midi: int) -> Fingering:
    candidates = fingering_candidates(midi)
    if not candidates:
        raise InstrumentRangeError(f"音高 MIDI {midi} 超出游戏口琴音域 C3–C#6")
    return min(
        candidates,
        key=lambda item: (
            int(item.shift != 0) + int(item.sharp),
            int(item.shift != 0),
            int(item.sharp),
            item.key_index,
        ),
    )


def modifier_state(fingering: Fingering) -> tuple[int, bool]:
    return fingering.shift, fingering.sharp


def continuous_token(midi: int) -> str:
    fingering = canonical_fingering(midi)
    degree = DEGREES[fingering.key_index]
    parts: list[str] = []
    if fingering.shift < 0:
        parts.append("左")
    elif fingering.shift > 0:
        parts.append("右")
    if fingering.sharp:
        parts.append("中")
    if KEYS[fingering.key_index] == ",":
        parts.append(",")
    return f"{degree}({'+'.join(parts)})" if parts else str(degree)


def _score_transpose(pitches: Sequence[int], transpose: int) -> tuple[int, int, int, int, int, int]:
    fingerings: list[Fingering | None] = []
    for pitch in pitches:
        candidates = fingering_candidates(pitch + transpose)
        fingerings.append(canonical_fingering(pitch + transpose) if candidates else None)
    out_of_range = sum(item is None for item in fingerings)
    valid = [item for item in fingerings if item is not None]
    states = [modifier_state(item) for item in valid]
    changes = sum(left != right for left, right in zip(states, states[1:]))
    modifier_notes = sum(item.shift != 0 or item.sharp for item in valid)
    sharp_notes = sum(item.sharp for item in valid)
    return out_of_range, changes, modifier_notes, sharp_notes, abs(transpose), transpose


def choose_transpose(notes: Iterable[NoteEvent]) -> int:
    pitches = [note.midi for note in notes]
    if not pitches:
        raise InstrumentRangeError("没有可用于移调的音符")
    ranked = sorted((_score_transpose(pitches, shift), shift) for shift in range(-24, 25))
    best_score, best_shift = ranked[0]
    if best_score[0] != 0:
        raise InstrumentRangeError("旋律音域比游戏口琴音域更宽，无法完整移调")
    return best_shift
```

- [ ] **Step 4: 运行指法测试并修正预期分数**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_instrument -v`

Expected: 7 tests PASS。

- [ ] **Step 5: 提交游戏指法模块**

```powershell
git add scripts/harmonica/instrument.py tests/test_instrument.py
git commit -m "feat: map melody notes to game harmonica controls"
```

---

### Task 3: 实现 pYIN 音高帧、平滑和短间隙处理

**Files:**
- Create: `scripts/harmonica/transcription.py`
- Create: `tests/test_transcription.py`

- [ ] **Step 1: 写音高平滑和短间隙测试**

```python
# tests/test_transcription.py
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from harmonica.transcription import bridge_tiny_gaps, smooth_quantized_pitch


class PitchCleanupTests(unittest.TestCase):
    def test_mode_smoothing_removes_one_frame_vibrato_flip(self) -> None:
        midi = np.array([60.1, 60.2, 61.0, 60.0, 59.9])
        valid = np.ones(5, dtype=bool)
        np.testing.assert_array_equal(smooth_quantized_pitch(midi, valid, radius=2), [60, 60, 60, 60, 60])

    def test_bridge_only_fills_short_gap_between_same_pitch(self) -> None:
        pitch = np.array([60.0, np.nan, np.nan, 60.0, 62.0])
        filled = bridge_tiny_gaps(pitch, max_frames=2)
        np.testing.assert_array_equal(filled, [60, 60, 60, 60, 62])

    def test_bridge_does_not_join_different_pitches(self) -> None:
        pitch = np.array([60.0, np.nan, 62.0])
        self.assertTrue(np.isnan(bridge_tiny_gaps(pitch, 2)[1]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行并确认导入失败**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_transcription.PitchCleanupTests -v`

Expected: FAIL，包含 `No module named 'harmonica.transcription'`。

- [ ] **Step 3: 实现 PitchFrames、连续区间和平滑基础**

```python
# scripts/harmonica/transcription.py（第一部分）
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import librosa
import numpy as np

from .models import NoteEvent, TranscriptionError


@dataclass(frozen=True)
class PitchFrames:
    times: np.ndarray
    midi: np.ndarray
    voiced_prob: np.ndarray
    rms_db: np.ndarray
    pitch: np.ndarray
    hop_seconds: float


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def smooth_quantized_pitch(midi: np.ndarray, valid: np.ndarray, radius: int = 4) -> np.ndarray:
    rounded = np.rint(midi).astype(float)
    rounded[~valid] = np.nan
    result = rounded.copy()
    for start, end in contiguous_runs(valid):
        for index in range(start, end):
            values = rounded[max(start, index - radius):min(end, index + radius + 1)]
            integers = values[np.isfinite(values)].astype(int)
            if len(integers):
                result[index] = Counter(integers.tolist()).most_common(1)[0][0]
    return result


def bridge_tiny_gaps(pitch: np.ndarray, max_frames: int) -> np.ndarray:
    output = pitch.copy()
    for start, end in contiguous_runs(~np.isfinite(output)):
        if end - start <= max_frames and start > 0 and end < len(output) and output[start - 1] == output[end]:
            output[start:end] = output[start - 1]
    return output
```

- [ ] **Step 4: 运行清理测试**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_transcription.PitchCleanupTests -v`

Expected: 3 tests PASS。

- [ ] **Step 5: 写稳定正弦音的 pYIN 测试**

向同一测试文件加入：

```python
from harmonica.transcription import extract_pitch_frames


class PitchExtractionTests(unittest.TestCase):
    def test_a4_sine_is_detected_near_midi_69(self) -> None:
        sample_rate = 22050
        time = np.arange(sample_rate, dtype=float) / sample_rate
        audio = 0.3 * np.sin(2 * np.pi * 440.0 * time)
        frames = extract_pitch_frames(audio, sample_rate)
        detected = frames.pitch[np.isfinite(frames.pitch)]
        self.assertGreater(len(detected), 20)
        self.assertAlmostEqual(float(np.median(detected)), 69.0, delta=0.5)
```

- [ ] **Step 6: 实现自适应能量门限和 pYIN 帧提取**

```python
def extract_pitch_frames(audio: np.ndarray, sample_rate: int, hop_length: int = 256) -> PitchFrames:
    frame_length = 2048
    f0, voiced, probability = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        fill_na=np.nan,
    )
    midi = librosa.hz_to_midi(f0)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0][:len(f0)]
    rms_db = librosa.amplitude_to_db(rms, ref=max(float(np.max(rms)), 1e-9))
    finite_energy = rms_db[np.isfinite(rms_db)]
    energy_floor = max(-48.0, float(np.quantile(finite_energy, 0.20)))
    valid = (
        voiced
        & np.isfinite(midi)
        & (probability >= 0.55)
        & (rms_db >= energy_floor)
        & (midi >= 36)
        & (midi <= 84)
    )
    pitch = smooth_quantized_pitch(midi, valid)
    pitch = bridge_tiny_gaps(pitch, max_frames=max(1, round(0.06 * sample_rate / hop_length)))
    return PitchFrames(
        librosa.frames_to_time(np.arange(len(f0)), sr=sample_rate, hop_length=hop_length),
        midi,
        probability,
        rms_db,
        pitch,
        hop_length / sample_rate,
    )
```

- [ ] **Step 7: 运行转录基础测试并提交**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_transcription -v`

Expected: 4 tests PASS。

```powershell
git add scripts/harmonica/transcription.py tests/test_transcription.py
git commit -m "feat: extract and smooth vocal pitch frames"
```

---

### Task 4: 使用发音起点拆分重复音并执行质量门槛

**Files:**
- Modify: `scripts/harmonica/transcription.py`
- Modify: `tests/test_transcription.py`

- [ ] **Step 1: 写三个同音起点、伪短音合并和低质量失败测试**

```python
from harmonica.models import TranscriptionError
from harmonica.transcription import frames_to_notes, validate_melody


class NoteSegmentationTests(unittest.TestCase):
    def test_three_onsets_split_one_stable_pitch_into_three_notes(self) -> None:
        pitch = np.full(100, 60.0)
        times = np.arange(100) * 0.01
        notes = frames_to_notes(
            times=times,
            pitch=pitch,
            voiced_prob=np.full(100, 0.9),
            onset_frames=np.array([0, 33, 66]),
            onset_envelope=np.array([1.0 if i in {0, 33, 66} else 0.0 for i in range(100)]),
            hop_seconds=0.01,
        )
        self.assertEqual([note.midi for note in notes], [60, 60, 60])

    def test_seventy_millisecond_pitch_flip_is_folded_into_neighbour(self) -> None:
        pitch = np.array([60.0] * 20 + [61.0] * 5 + [60.0] * 20)
        times = np.arange(len(pitch)) * 0.01
        notes = frames_to_notes(times, pitch, np.full(len(pitch), 0.9), np.array([0]), np.ones(len(pitch)), 0.01)
        self.assertEqual([note.midi for note in notes], [60])

    def test_quality_gate_rejects_short_or_uncertain_melody(self) -> None:
        with self.assertRaises(TranscriptionError):
            validate_melody([NoteEvent(i * 0.1, i * 0.1 + 0.08, 60, 0.4) for i in range(11)])
```

- [ ] **Step 2: 运行新测试并确认函数尚未定义**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_transcription.NoteSegmentationTests -v`

Expected: FAIL，包含 `cannot import name 'frames_to_notes'`。

- [ ] **Step 3: 实现起点检测、片段拆分、短音清理和质量门槛**

在 `transcription.py` 加入以下公开函数：

```python
def detect_onsets(audio: np.ndarray, sample_rate: int, hop_length: int = 256) -> tuple[np.ndarray, np.ndarray]:
    spectral = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0][:len(spectral)]
    energy_change = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    spectral /= max(float(np.max(spectral)), 1e-9)
    energy_change /= max(float(np.max(energy_change)), 1e-9)
    envelope = np.maximum(spectral, energy_change)
    frames = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sample_rate,
        hop_length=hop_length,
        backtrack=False,
        units="frames",
        delta=0.07,
        wait=max(1, round(0.06 * sample_rate / hop_length)),
    )
    return np.asarray(frames, dtype=int), envelope


def _raw_pitch_segments(pitch: np.ndarray) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    index = 0
    while index < len(pitch):
        if not np.isfinite(pitch[index]):
            index += 1
            continue
        value = int(pitch[index])
        end = index + 1
        while end < len(pitch) and np.isfinite(pitch[end]) and int(pitch[end]) == value:
            end += 1
        segments.append((index, end, value))
        index = end
    return segments


def frames_to_notes(
    times: np.ndarray,
    pitch: np.ndarray,
    voiced_prob: np.ndarray,
    onset_frames: np.ndarray,
    onset_envelope: np.ndarray,
    hop_seconds: float,
) -> list[NoteEvent]:
    boundaries = set(int(frame) for frame in onset_frames)
    notes: list[NoteEvent] = []
    min_frames = max(1, round(0.07 / hop_seconds))
    for start, end, midi in _raw_pitch_segments(pitch):
        cuts = [start] + [frame for frame in sorted(boundaries) if start + min_frames <= frame <= end - min_frames] + [end]
        for left, right in zip(cuts, cuts[1:]):
            if right - left < min_frames:
                continue
            probability = float(np.nanmean(voiced_prob[left:right]))
            stability = max(0.0, 1.0 - float(np.nanstd(pitch[left:right])) / 0.7)
            onset = float(onset_envelope[left]) if left < len(onset_envelope) else 0.0
            onset_normalized = onset / max(float(np.max(onset_envelope)), 1e-9)
            confidence = 0.65 * probability + 0.25 * stability + 0.10 * onset_normalized
            notes.append(NoteEvent(float(times[left]), float(times[right - 1] + hop_seconds), midi, confidence, onset_normalized))
    return clean_notes(notes)


def clean_notes(notes: list[NoteEvent], min_duration: float = 0.075) -> list[NoteEvent]:
    cleaned = list(notes)
    changed = True
    while changed:
        changed = False
        for index, note in enumerate(cleaned):
            if note.duration >= min_duration:
                continue
            previous = cleaned[index - 1] if index else None
            following = cleaned[index + 1] if index + 1 < len(cleaned) else None
            if previous and following and previous.midi == following.midi:
                merged = NoteEvent(previous.start, following.end, previous.midi, max(previous.confidence, following.confidence))
                cleaned[index - 1:index + 2] = [merged]
                changed = True
                break
            del cleaned[index]
            changed = True
            break
    return cleaned


def validate_melody(notes: list[NoteEvent]) -> None:
    if len(notes) < 12:
        raise TranscriptionError(f"只检测到 {len(notes)} 个可靠音符，至少需要 12 个")
    duration = sum(note.duration for note in notes)
    if duration < 8.0:
        raise TranscriptionError(f"可靠旋律总长度只有 {duration:.1f} 秒，至少需要 8 秒")
    median_confidence = float(np.median([note.confidence for note in notes]))
    if median_confidence < 0.55:
        raise TranscriptionError(f"旋律置信度 {median_confidence:.2f} 低于 0.55")


def transcribe_vocals(audio: np.ndarray, sample_rate: int) -> tuple[list[NoteEvent], PitchFrames]:
    frames = extract_pitch_frames(audio, sample_rate)
    onset_frames, onset_envelope = detect_onsets(audio, sample_rate)
    notes = frames_to_notes(frames.times, frames.pitch, frames.voiced_prob, onset_frames, onset_envelope, frames.hop_seconds)
    validate_melody(notes)
    return notes, frames
```

- [ ] **Step 4: 运行全部转录测试**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_transcription -v`

Expected: 7 tests PASS。

- [ ] **Step 5: 提交重复音拆分**

```powershell
git add scripts/harmonica/transcription.py tests/test_transcription.py
git commit -m "feat: split repeated notes with vocal onsets"
```

---

### Task 5: 检测速度、回退网格并量化到十六分音符

**Files:**
- Create: `scripts/harmonica/rhythm.py`
- Create: `tests/test_rhythm.py`

- [ ] **Step 1: 写节拍检测、间隔回退、量化和段落测试**

```python
# tests/test_rhythm.py
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from harmonica.models import NoteEvent, RhythmError, TempoGrid
from harmonica.rhythm import detect_tempo_grid, quantize_notes, section_starts


class RhythmTests(unittest.TestCase):
    def test_click_track_detects_tempo_near_120(self) -> None:
        sr = 22050
        audio = librosa.clicks(times=np.arange(0.5, 8.0, 0.5), sr=sr, length=sr * 8)
        grid = detect_tempo_grid(audio, sr, [])
        self.assertAlmostEqual(grid.bpm, 120.0, delta=3.0)
        self.assertEqual(grid.source, "beats")

    def test_interval_fallback_recovers_100_bpm_grid(self) -> None:
        notes = [NoteEvent(i * 0.15, i * 0.15 + 0.1, 60, 0.9) for i in range(40)]
        grid = detect_tempo_grid(np.zeros(22050), 22050, notes)
        self.assertAlmostEqual(grid.bpm, 100.0, delta=2.0)
        self.assertEqual(grid.source, "intervals")

    def test_quantize_uses_one_based_bar_and_slot_and_minimum_length(self) -> None:
        notes = [NoteEvent(1.01, 1.02, 60, 0.9), NoteEvent(1.26, 1.48, 62, 0.8)]
        result = quantize_notes(notes, TempoGrid(120.0, 1.0, "beats", 0.9), transpose=0)
        self.assertEqual((result[0].bar, result[0].slot, result[0].duration_slots), (1, 1, 1))
        self.assertEqual((result[1].bar, result[1].slot), (1, 3))

    def test_large_silence_starts_new_section(self) -> None:
        notes = [
            NoteEvent(0.0, 0.2, 60, 0.9),
            NoteEvent(0.3, 0.5, 62, 0.9),
            NoteEvent(2.0, 2.2, 64, 0.9),
        ]
        self.assertEqual(section_starts(notes), [0, 2])


if __name__ == "__main__":
    unittest.main()
```

测试文件顶部还需加入 `import librosa`。

- [ ] **Step 2: 运行并确认模块不存在**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_rhythm -v`

Expected: FAIL，包含 `No module named 'harmonica.rhythm'`。

- [ ] **Step 3: 实现速度检测和间隔回退**

```python
# scripts/harmonica/rhythm.py
from __future__ import annotations

import librosa
import numpy as np

from .models import NoteEvent, QuantizedNote, RhythmError, TempoGrid


def _interval_grid(notes: list[NoteEvent]) -> TempoGrid:
    intervals = np.diff([note.start for note in notes])
    intervals = intervals[(intervals >= 0.06) & (intervals <= 2.0)]
    candidates: list[tuple[float, float, int, float]] = []
    for interval in intervals:
        for multiple in range(1, 9):
            slot = interval / multiple
            bpm = 60.0 / (slot * 4.0)
            if 45.0 <= bpm <= 210.0:
                residual = np.abs(intervals / slot - np.rint(intervals / slot))
                consistency = float(np.mean(residual <= 0.18))
                candidates.append((consistency, -float(np.median(residual)), -multiple, bpm))
    if not candidates:
        raise RhythmError("无法从音符间隔估计稳定节拍")
    consistency, _, _, bpm = max(candidates)
    if consistency < 0.55:
        raise RhythmError(f"音符间隔网格一致性只有 {consistency:.2f}")
    return TempoGrid(bpm, notes[0].start, "intervals", consistency)


def detect_tempo_grid(accompaniment: np.ndarray, sample_rate: int, notes: list[NoteEvent]) -> TempoGrid:
    tempo, beat_frames = librosa.beat.beat_track(y=accompaniment, sr=sample_rate)
    bpm = float(np.asarray(tempo).reshape(-1)[0]) if np.size(tempo) else 0.0
    beat_times = librosa.frames_to_time(np.asarray(beat_frames), sr=sample_rate)
    if 45.0 <= bpm <= 210.0 and len(beat_times) >= 4:
        gaps = np.diff(beat_times)
        consistency = float(np.mean(np.abs(gaps - np.median(gaps)) <= 0.08 * np.median(gaps)))
        if consistency >= 0.60:
            return TempoGrid(bpm, float(beat_times[0]), "beats", consistency)
    return _interval_grid(notes)
```

- [ ] **Step 4: 实现量化和段落边界**

```python
def quantize_notes(notes: list[NoteEvent], grid: TempoGrid, transpose: int) -> list[QuantizedNote]:
    slot_seconds = 60.0 / grid.bpm / 4.0
    raw_starts = [round((note.start - grid.anchor) / slot_seconds) for note in notes]
    base_slot = (min(raw_starts) // 16) * 16
    result: list[QuantizedNote] = []
    for note, raw_start in zip(notes, raw_starts):
        raw_end = max(raw_start + 1, round((note.end - grid.anchor) / slot_seconds))
        normalized = raw_start - base_slot
        result.append(
            QuantizedNote(
                note.start,
                note.end,
                note.midi,
                note.midi + transpose,
                note.confidence,
                normalized // 16 + 1,
                normalized % 16 + 1,
                raw_end - raw_start,
            )
        )
    return result


def section_starts(notes: list[NoteEvent | QuantizedNote], gap_seconds: float = 1.2, max_bars: int = 8) -> list[int]:
    if not notes:
        return []
    starts = [0]
    last_section_bar = getattr(notes[0], "bar", 1)
    for index in range(1, len(notes)):
        current_bar = getattr(notes[index], "bar", last_section_bar)
        if notes[index].start - notes[index - 1].end >= gap_seconds or current_bar - last_section_bar >= max_bars:
            starts.append(index)
            last_section_bar = current_bar
    return starts
```

- [ ] **Step 5: 运行节奏测试并提交**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_rhythm -v`

Expected: 4 tests PASS。

```powershell
git add scripts/harmonica/rhythm.py tests/test_rhythm.py
git commit -m "feat: quantize melody onto detected rhythm grid"
```

---

### Task 6: 验证输入并通过 Demucs 分离音轨

**Files:**
- Create: `scripts/harmonica/audio.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: 写扩展名、空间和 Demucs 命令测试**

```python
# tests/test_audio.py
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from harmonica.audio import separate_audio, validate_input
from harmonica.models import InputValidationError


class AudioTests(unittest.TestCase):
    def test_rejects_m4a_in_first_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "song.m4a"
            path.write_bytes(b"x")
            with self.assertRaisesRegex(InputValidationError, "MP3、WAV 或 FLAC"):
                validate_input(path, Path(directory))

    @mock.patch("harmonica.audio.shutil.disk_usage")
    def test_requires_three_gibibytes_free(self, disk_usage: mock.Mock) -> None:
        disk_usage.return_value = mock.Mock(free=2 * 1024**3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "song.wav"
            path.write_bytes(b"x")
            with self.assertRaisesRegex(InputValidationError, "3 GB"):
                validate_input(path, Path(directory))

    @mock.patch("harmonica.audio.subprocess.run")
    def test_demucs_is_invoked_without_shell_string(self, run: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "有 空格.mp3"
            source.write_bytes(b"x")
            stem_dir = root / "htdemucs" / source.stem
            stem_dir.mkdir(parents=True)
            (stem_dir / "vocals.wav").write_bytes(b"v")
            (stem_dir / "no_vocals.wav").write_bytes(b"n")
            run.return_value = subprocess.CompletedProcess([], 0)
            vocals, accompaniment = separate_audio(source, root)
            self.assertEqual(vocals.name, "vocals.wav")
            self.assertEqual(accompaniment.name, "no_vocals.wav")
            self.assertIn("--two-stems=vocals", run.call_args.args[0])
            self.assertFalse(run.call_args.kwargs.get("shell", False))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行并确认模块不存在**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_audio -v`

Expected: FAIL，包含 `No module named 'harmonica.audio'`。

- [ ] **Step 3: 实现输入检查、波形读取和分离封装**

```python
# scripts/harmonica/audio.py
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np

from .models import InputValidationError

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac"}
MIN_FREE_BYTES = 3 * 1024**3


def validate_input(source: Path, output_root: Path) -> None:
    if not source.is_file():
        raise InputValidationError(f"找不到输入文件：{source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise InputValidationError("第一版只支持 MP3、WAV 或 FLAC；请先转换其他格式")
    output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < MIN_FREE_BYTES:
        raise InputValidationError("输出磁盘可用空间不足 3 GB，未启动人声分离")


def separate_audio(source: Path, work_root: Path) -> tuple[Path, Path]:
    command = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        "htdemucs",
        "--two-stems=vocals",
        "-o",
        str(work_root),
        str(source),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise InputValidationError(f"人声分离失败，Demucs 返回代码 {error.returncode}") from error
    stem_dir = work_root / "htdemucs" / source.stem
    vocals = stem_dir / "vocals.wav"
    accompaniment = stem_dir / "no_vocals.wav"
    if not vocals.is_file() or not accompaniment.is_file():
        raise InputValidationError("人声分离结束，但没有找到 vocals.wav 和 no_vocals.wav")
    return vocals, accompaniment


def load_mono(path: Path, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
    audio, actual_rate = librosa.load(path, sr=sample_rate, mono=True)
    if not len(audio):
        raise InputValidationError(f"音频为空：{path}")
    return np.asarray(audio, dtype=np.float32), actual_rate
```

- [ ] **Step 4: 运行音频测试并提交**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_audio -v`

Expected: 3 tests PASS。

```powershell
git add scripts/harmonica/audio.py tests/test_audio.py
git commit -m "feat: validate and separate input audio"
```

---

### Task 7: 渲染连续谱、节奏谱和机器可读文件

**Files:**
- Create: `scripts/harmonica/render.py`
- Create: `tests/test_render.py`

- [ ] **Step 1: 写连续记号、十六分网格、CSV、MIDI/WAV 时长和报告测试**

```python
# tests/test_render.py
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import mido
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from harmonica.models import QuantizedNote, TempoGrid
from harmonica.render import render_all


class RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.notes = [
            QuantizedNote(0.0, 0.2, 55, 55, 0.9, 1, 1, 2),
            QuantizedNote(0.25, 0.5, 60, 60, 0.8, 1, 3, 2),
            QuantizedNote(2.0, 2.2, 72, 72, 0.85, 2, 1, 1),
        ]

    def test_render_all_writes_six_files_with_expected_notation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = render_all("测试歌", self.notes, TempoGrid(120, 0, "beats", 0.9), 0, Path(directory), [0])
            self.assertEqual(len(paths), 6)
            continuous = (Path(directory) / "测试歌-连续按键谱.md").read_text(encoding="utf-8")
            detailed = (Path(directory) / "测试歌-详细节奏谱.md").read_text(encoding="utf-8")
            self.assertIn("5(左)", continuous)
            self.assertIn("1(,)", continuous)
            self.assertIn("｜", continuous)
            self.assertIn("—", detailed)
            self.assertIn("·", detailed)

    def test_csv_and_json_are_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render_all(
                "测试歌", self.notes, TempoGrid(120, 0, "beats", 0.9), 0, root, [0, 2],
                analysis={"filtered_frame_count": 7, "filtered_note_count": 2, "warnings": ["节拍使用回退网格"]},
            )
            rows = list(csv.DictReader((root / "测试歌-音符明细.csv").open(encoding="utf-8-sig")))
            report = json.loads((root / "测试歌-分析报告.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 3)
            self.assertEqual(report["note_count"], 3)
            self.assertEqual(report["filtered_frame_count"], 7)
            self.assertEqual(report["filtered_note_count"], 2)
            self.assertEqual(report["warnings"], ["节拍使用回退网格"])

    def test_midi_and_wav_durations_match_within_point_one_second(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render_all("测试歌", self.notes, TempoGrid(120, 0, "beats", 0.9), 0, root, [0, 2])
            midi = mido.MidiFile(root / "测试歌-主旋律.mid")
            audio, sample_rate = sf.read(root / "测试歌-口琴试听.wav")
            self.assertLess(abs(midi.length - len(audio) / sample_rate), 0.1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行并确认模块不存在**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_render -v`

Expected: FAIL，包含 `No module named 'harmonica.render'`。

- [ ] **Step 3: 实现 Markdown 与 CSV 渲染**

`scripts/harmonica/render.py` 必须提供以下公开接口；Markdown 中的每个音符均调用 `continuous_token(note.play_midi)`，不得复制指法算法：

```python
from __future__ import annotations

import csv
import json
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from .instrument import KEYS, canonical_fingering, continuous_token
from .models import QuantizedNote, TempoGrid

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _format_time(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def continuous_markdown(title: str, notes: list[QuantizedNote], sections: list[int]) -> str:
    section_set = set(sections)
    lines = [f"# 《{title}》——连续按键谱", "", "`｜` 表示换小节，鼠标按键直接写在对应数字旁。", ""]
    tokens: list[str] = []
    current_bar: int | None = None
    section_number = 0
    for index, note in enumerate(notes):
        if index in section_set:
            if tokens:
                lines.extend([" ".join(tokens), ""])
                tokens = []
            section_number += 1
            lines.extend(["", f"## 第 {section_number} 段（{_format_time(note.start)}）", ""])
            current_bar = None
        if current_bar is not None and note.bar != current_bar:
            tokens.append("｜")
        if note.bar != current_bar:
            current_bar = note.bar
        tokens.append(continuous_token(note.play_midi))
    if tokens:
        lines.append(" ".join(tokens))
    return "\n".join(lines).strip() + "\n"


def detailed_markdown(title: str, notes: list[QuantizedNote], sections: list[int]) -> str:
    last_bar = max(note.bar for note in notes)
    bars = {bar: ["·"] * 16 for bar in range(1, last_bar + 1)}
    for note in notes:
        start = note.slot - 1
        for offset in range(note.duration_slots):
            absolute = start + offset
            bar = note.bar + absolute // 16
            slot = absolute % 16
            bars.setdefault(bar, ["·"] * 16)[slot] = continuous_token(note.play_midi) if offset == 0 else "—"
    lines = [f"# 《{title}》——详细节奏谱", "", "每拍四格；`—` 为延音，`·` 为休止。", ""]
    section_bars = {notes[index].bar: (number, notes[index].start) for number, index in enumerate(sections, 1)}
    for bar in sorted(bars):
        if bar in section_bars:
            number, start = section_bars[bar]
            lines.extend(["", f"## 第 {number} 段（{_format_time(start)}）", ""])
        groups = [" ".join(bars[bar][index:index + 4]) for index in range(0, 16, 4)]
        lines.append(f"{bar:02d}  | " + " | ".join(groups) + " |")
    return "\n".join(lines) + "\n"


def write_csv(notes: list[QuantizedNote], path: Path) -> None:
    fields = ["start", "end", "source_note", "play_note", "confidence", "bar", "slot", "duration_slots", "game_input"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for note in notes:
            fingering = canonical_fingering(note.play_midi)
            modifiers = ("左+" if fingering.shift < 0 else "右+" if fingering.shift > 0 else "") + ("中+" if fingering.sharp else "")
            writer.writerow({
                "start": f"{note.start:.3f}", "end": f"{note.end:.3f}",
                "source_note": note_name(note.source_midi), "play_note": note_name(note.play_midi),
                "confidence": f"{note.confidence:.3f}", "bar": note.bar, "slot": note.slot,
                "duration_slots": note.duration_slots,
                "game_input": modifiers + KEYS[fingering.key_index],
            })
```

- [ ] **Step 4: 实现共享时间轴上的 MIDI 与 WAV**

```python
def _score_seconds(note: QuantizedNote, grid: TempoGrid) -> tuple[float, float]:
    slot_seconds = 60.0 / grid.bpm / 4.0
    start_slot = (note.bar - 1) * 16 + note.slot - 1
    return start_slot * slot_seconds, (start_slot + note.duration_slots) * slot_seconds


def write_midi(notes: list[QuantizedNote], grid: TempoGrid, path: Path) -> float:
    ticks_per_beat = 480
    ticks_per_slot = ticks_per_beat // 4
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(grid.bpm), time=0))
    track.append(mido.Message("program_change", program=22, time=0))
    events = []
    for note in notes:
        start_slot = (note.bar - 1) * 16 + note.slot - 1
        start_tick = start_slot * ticks_per_slot
        end_tick = (start_slot + note.duration_slots) * ticks_per_slot
        events.extend([(start_tick, 1, note.play_midi), (end_tick, 0, note.play_midi)])
    previous = 0
    for tick, on, pitch in sorted(events, key=lambda item: (item[0], item[1])):
        track.append(mido.Message("note_on" if on else "note_off", note=pitch, velocity=88 if on else 0, time=tick - previous))
        previous = tick
    midi.save(path)
    return midi.length


def write_preview(notes: list[QuantizedNote], grid: TempoGrid, path: Path, sample_rate: int = 22050) -> float:
    duration = max(_score_seconds(note, grid)[1] for note in notes)
    audio = np.zeros(max(1, round(duration * sample_rate)), dtype=np.float64)
    rng = np.random.default_rng(20260916)
    for note in notes:
        start, end = _score_seconds(note, grid)
        left, right = round(start * sample_rate), min(len(audio), round(end * sample_rate))
        time = np.arange(right - left) / sample_rate
        frequency = 440.0 * 2 ** ((note.play_midi - 69) / 12)
        phase = 2 * np.pi * frequency * time
        wave = np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
        envelope = np.ones_like(time)
        attack, release = min(len(time), round(0.02 * sample_rate)), min(len(time), round(0.04 * sample_rate))
        envelope[:attack] *= np.linspace(0, 1, attack)
        envelope[-release:] *= np.linspace(1, 0, release)
        audio[left:right] += 0.2 * envelope * (wave + 0.01 * rng.standard_normal(len(time)))
    peak = float(np.max(np.abs(audio)))
    if peak:
        audio *= 0.92 / peak
    sf.write(path, audio.astype(np.float32), sample_rate, subtype="PCM_16")
    return len(audio) / sample_rate
```

- [ ] **Step 5: 实现报告和统一输出入口**

```python
def render_all(
    title: str,
    notes: list[QuantizedNote],
    grid: TempoGrid,
    transpose: int,
    output: Path,
    sections: list[int],
    analysis: dict[str, object] | None = None,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "continuous": output / f"{title}-连续按键谱.md",
        "detailed": output / f"{title}-详细节奏谱.md",
        "csv": output / f"{title}-音符明细.csv",
        "midi": output / f"{title}-主旋律.mid",
        "wav": output / f"{title}-口琴试听.wav",
        "report": output / f"{title}-分析报告.json",
    }
    paths["continuous"].write_text(continuous_markdown(title, notes, sections), encoding="utf-8")
    paths["detailed"].write_text(detailed_markdown(title, notes, sections), encoding="utf-8")
    write_csv(notes, paths["csv"])
    midi_seconds = write_midi(notes, grid, paths["midi"])
    wav_seconds = write_preview(notes, grid, paths["wav"])
    report: dict[str, object] = {
        "bpm": round(grid.bpm, 3), "grid_source": grid.source,
        "grid_consistency": round(grid.consistency, 3), "transpose": transpose,
        "note_count": len(notes),
        "source_range": [note_name(min(note.source_midi for note in notes)), note_name(max(note.source_midi for note in notes))],
        "play_range": [note_name(min(note.play_midi for note in notes)), note_name(max(note.play_midi for note in notes))],
        "source_midi_range": [min(note.source_midi for note in notes), max(note.source_midi for note in notes)],
        "play_midi_range": [min(note.play_midi for note in notes), max(note.play_midi for note in notes)],
        "median_confidence": round(float(np.median([note.confidence for note in notes])), 3),
        "midi_seconds": round(midi_seconds, 3), "wav_seconds": round(wav_seconds, 3),
    }
    report.update(analysis or {"filtered_frame_count": 0, "filtered_note_count": 0, "warnings": []})
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return list(paths.values())
```

- [ ] **Step 6: 运行渲染测试并提交**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_render -v`

Expected: 3 tests PASS，且 MIDI/WAV 时长差小于 0.1 秒。

```powershell
git add scripts/harmonica/render.py tests/test_render.py
git commit -m "feat: render harmonica scores and preview files"
```

---

### Task 8: 编排完整流程并保证输出事务性

**Files:**
- Create: `scripts/extract_harmonica_score.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: 写参数解析、成功发布、拒绝覆盖和失败保留测试**

```python
# tests/test_cli.py
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("extract_harmonica_score", ROOT / "scripts" / "extract_harmonica_score.py")
assert SPEC and SPEC.loader
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)

from harmonica.models import NoteEvent, TempoGrid


class CliTests(unittest.TestCase):
    @mock.patch.object(cli, "render_all")
    @mock.patch.object(cli, "detect_tempo_grid", return_value=TempoGrid(120, 0, "beats", 0.9))
    @mock.patch.object(
        cli,
        "transcribe_vocals",
        return_value=(
            [NoteEvent(i, i + 0.8, 60, 0.9) for i in range(12)],
            mock.Mock(pitch=np.array([60.0, np.nan, 60.0])),
        ),
    )
    @mock.patch.object(cli, "load_mono", return_value=(np.zeros(22050), 22050))
    @mock.patch.object(cli, "separate_audio")
    def test_success_publishes_named_directory(self, separate: mock.Mock, load: mock.Mock, transcribe: mock.Mock, tempo: mock.Mock, render: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "歌曲.mp3"
            source.write_bytes(b"x")
            stems = root / "prepared"
            stems.mkdir()
            vocals, accompaniment = stems / "vocals.wav", stems / "no_vocals.wav"
            vocals.write_bytes(b"v")
            accompaniment.write_bytes(b"n")
            separate.return_value = vocals, accompaniment
            def fake_render(title, notes, grid, transpose, output, sections, analysis=None):
                path = output / "done.txt"
                path.write_text("ok", encoding="utf-8")
                return [path]

            render.side_effect = fake_render
            output = cli.run_pipeline(source, root / "out", keep_stems=True, force=False)
            self.assertEqual(output, root / "out" / "歌曲")
            self.assertTrue((output / "stems" / "vocals.wav").exists())

    def test_existing_output_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "song.mp3"
            source.write_bytes(b"x")
            (root / "out" / "song").mkdir(parents=True)
            with self.assertRaisesRegex(Exception, "--force"):
                cli.run_pipeline(source, root / "out", False, False)

    @mock.patch.object(cli, "separate_audio", side_effect=RuntimeError("boom"))
    def test_failure_writes_chinese_error_log_without_destroying_old_output(self, separate: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "song.mp3"
            source.write_bytes(b"x")
            old = root / "out" / "song"
            old.mkdir(parents=True)
            (old / "old.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                cli.run_pipeline(source, root / "out", False, True)
            self.assertEqual((old / "old.txt").read_text(encoding="utf-8"), "keep")
            self.assertTrue(list((root / "out").glob("song-失败-*")))

    def test_python_311_is_rejected_with_clear_message(self) -> None:
        with self.assertRaisesRegex(Exception, "Python 3.12"):
            cli.validate_python_version((3, 11))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行并确认入口文件不存在**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_cli -v`

Expected: FAIL，包含找不到 `scripts/extract_harmonica_score.py`。

- [ ] **Step 3: 实现流程编排和阶段信息**

```python
# scripts/extract_harmonica_score.py
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from harmonica.audio import load_mono, separate_audio, validate_input
from harmonica.instrument import choose_transpose
from harmonica.models import HarmonicaError, InputValidationError
from harmonica.render import render_all
from harmonica.rhythm import detect_tempo_grid, quantize_notes, section_starts
from harmonica.transcription import transcribe_vocals


def safe_title(path: Path) -> str:
    invalid = '<>:"/\\|?*'
    title = "".join("_" if character in invalid else character for character in path.stem).strip(" .")
    return title or "未命名歌曲"


def _publish(result: Path, final: Path, force: bool) -> None:
    backup = final.with_name(f".{final.name}.backup")
    if final.exists():
        if not force:
            raise InputValidationError(f"输出目录已存在：{final}；确认覆盖请使用 --force")
        if backup.exists():
            shutil.rmtree(backup)
        final.rename(backup)
    try:
        result.rename(final)
    except Exception:
        if backup.exists() and not final.exists():
            backup.rename(final)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def _save_failure(temp_root: Path, output_root: Path, title: str, error: Exception) -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    failure = output_root / f"{title}-失败-{timestamp}"
    failure.mkdir(parents=True, exist_ok=False)
    (failure / "错误日志.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
    stems = next(temp_root.glob("**/htdemucs/*"), None)
    if stems and stems.is_dir():
        shutil.copytree(stems, failure / "stems")


def run_pipeline(source: Path, output_root: Path, keep_stems: bool, force: bool) -> Path:
    title = safe_title(source)
    validate_input(source, output_root)
    final = output_root / title
    if final.exists() and not force:
        raise InputValidationError(f"输出目录已存在：{final}；确认覆盖请使用 --force")
    temp_root = Path(tempfile.mkdtemp(prefix=f".{title}-", dir=output_root))
    result = temp_root / "result"
    result.mkdir()
    try:
        print("[1/6] 正在分离人声（第一次可能需要下载模型）……", flush=True)
        vocals_path, accompaniment_path = separate_audio(source, temp_root / "separation")
        print("[2/6] 正在分析音高和重复音……", flush=True)
        vocals, sample_rate = load_mono(vocals_path)
        accompaniment, _ = load_mono(accompaniment_path)
        notes, frames = transcribe_vocals(vocals, sample_rate)
        print("[3/6] 正在检测节拍……", flush=True)
        grid = detect_tempo_grid(accompaniment, sample_rate, notes)
        print("[4/6] 正在选择游戏口琴移调……", flush=True)
        transpose = choose_transpose(notes)
        quantized = quantize_notes(notes, grid, transpose)
        sections = section_starts(quantized)
        pitch = frames.pitch
        valid = np.isfinite(pitch)
        raw_segment_starts = valid & np.r_[True, (~valid[:-1]) | (pitch[1:] != pitch[:-1])]
        raw_segment_count = int(np.count_nonzero(raw_segment_starts))
        analysis = {
            "filtered_frame_count": int(np.count_nonzero(~np.isfinite(frames.pitch))),
            "filtered_note_count": max(0, raw_segment_count - len(notes)),
            "warnings": ["伴奏拍点不稳定，节奏使用音符间隔回退网格"] if grid.source == "intervals" else [],
        }
        print("[5/6] 正在生成谱面和试听……", flush=True)
        render_all(title, quantized, grid, transpose, result, sections, analysis=analysis)
        if keep_stems:
            stem_output = result / "stems"
            stem_output.mkdir()
            shutil.copy2(vocals_path, stem_output / "vocals.wav")
            shutil.copy2(accompaniment_path, stem_output / "no_vocals.wav")
        print("[6/6] 正在发布结果……", flush=True)
        _publish(result, final, force)
        return final
    except Exception as error:
        _save_failure(temp_root, output_root, title, error)
        raise
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
```

- [ ] **Step 4: 实现参数解析、中文错误和退出码**

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从音频自动提取《三角洲行动》口琴谱")
    parser.add_argument("audio", type=Path, help="MP3、WAV 或 FLAC 文件")
    parser.add_argument("--output", type=Path, default=Path("output"), help="输出根目录")
    parser.add_argument("--keep-stems", action="store_true", help="保留人声和伴奏分离音轨")
    parser.add_argument("--force", action="store_true", help="成功生成新结果后覆盖旧结果")
    return parser.parse_args(argv)


def validate_python_version(version: tuple[int, int]) -> None:
    if version != (3, 12):
        raise InputValidationError(f"需要 Python 3.12，当前为 Python {version[0]}.{version[1]}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_python_version(sys.version_info[:2])
        output = run_pipeline(args.audio.resolve(), args.output.resolve(), args.keep_stems, args.force)
    except HarmonicaError as error:
        print(f"转谱失败：{error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"转谱意外失败：{error}", file=sys.stderr)
        return 1
    print(f"转谱完成：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 运行 CLI 测试**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_cli -v`

Expected: 4 tests PASS，包括旧结果保护、失败日志和 Python 版本提示。

- [ ] **Step 6: 运行至此的全部快速测试并提交**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest discover -s tests -p 'test_*.py' -v`

Expected: 所有快速测试 PASS；仅缺少本地素材的测试可以 SKIP。

```powershell
git add scripts/extract_harmonica_score.py tests/test_cli.py
git commit -m "feat: orchestrate transactional score extraction"
```

---

### Task 9: 添加拖放批处理和首次环境安装

**Files:**
- Create: `提取口琴谱.bat`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: 写批处理静态行为测试**

向 `tests/test_cli.py` 加入：

```python
class BatchLauncherTests(unittest.TestCase):
    def test_batch_quotes_dragged_path_and_keeps_stems(self) -> None:
        text = (ROOT / "提取口琴谱.bat").read_text(encoding="utf-8")
        self.assertIn('"%~1"', text)
        self.assertIn("--keep-stems", text)
        self.assertIn("requirements-audio.txt", text)
        self.assertIn("py -3.12", text)
        self.assertIn(".audio-deps-ready", text)
```

- [ ] **Step 2: 运行并确认批处理不存在**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_cli.BatchLauncherTests -v`

Expected: FAIL，包含 `FileNotFoundError`。

- [ ] **Step 3: 实现中文拖放入口**

```bat
@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo 请把 MP3、WAV 或 FLAC 文件拖到“提取口琴谱.bat”上。
    pause
    exit /b 2
)

set "VENV_ROOT=%~dp0.tools\audio-venv"
set "VENV_PY=%VENV_ROOT%\Scripts\python.exe"
set "DEPS_READY=%VENV_ROOT%\.audio-deps-ready"
if not exist "%VENV_PY%" (
    echo 首次使用：正在创建 Python 3.12 环境……
    py -3.12 -m venv "%VENV_ROOT%"
    if errorlevel 1 (
        echo 未找到 Python 3.12，请先安装后重试。
        pause
        exit /b 3
    )
)

if not exist "%DEPS_READY%" (
    echo 正在检查并安装音频分析依赖……
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install -r "%~dp0requirements-audio.txt"
    if errorlevel 1 (
        echo 依赖安装失败，请检查网络后重试。
        pause
        exit /b 4
    )
    >"%DEPS_READY%" echo ready
)

echo 第一次转谱会下载 Demucs 模型，可能需要较长时间和网络连接。
"%VENV_PY%" "%~dp0scripts\extract_harmonica_score.py" "%~1" --output "%~dp0output" --keep-stems
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    echo 转谱没有完成，请查看 output 中的失败目录。
    pause
    exit /b %RESULT%
)

start "" explorer.exe "%~dp0output"
echo 已完成，结果位于：%~dp0output
pause
```

- [ ] **Step 4: 运行批处理静态测试并人工空参数冒烟测试**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_cli.BatchLauncherTests -v`

Expected: PASS。

Run: `cmd /c "提取口琴谱.bat < nul"`

Expected: 显示“请把 MP3、WAV 或 FLAC 文件拖到……”并以代码 2 退出；不创建或删除用户文件。

- [ ] **Step 5: 提交拖放入口**

```powershell
git add 提取口琴谱.bat tests/test_cli.py
git commit -m "feat: add Windows drag-and-drop launcher"
```

---

### Task 10: 添加不依赖外部歌曲的合成音频全链路测试

**Files:**
- Create: `tests/test_synthetic_pipeline.py`

- [ ] **Step 1: 写包含稳定音、同音重复、滑音、颤音和静音的合成测试**

```python
# tests/test_synthetic_pipeline.py
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from harmonica.instrument import choose_transpose
from harmonica.rhythm import quantize_notes
from harmonica.models import TempoGrid
from harmonica.transcription import transcribe_vocals


def tone(midi: float, seconds: float, sample_rate: int, pulses: int = 1) -> np.ndarray:
    count = round(seconds * sample_rate)
    time = np.arange(count) / sample_rate
    vibrato_midi = midi + 0.20 * np.sin(2 * np.pi * 5.2 * time)
    frequency = 440.0 * 2 ** ((vibrato_midi - 69) / 12)
    carrier = np.sin(2 * np.pi * np.cumsum(frequency) / sample_rate)
    envelope = np.ones(count)
    for boundary in np.linspace(0, count, pulses + 1, dtype=int)[:-1]:
        attack = min(round(0.025 * sample_rate), count - boundary)
        envelope[boundary:boundary + attack] *= np.linspace(0, 1, attack)
    return 0.3 * carrier * envelope


def glide(start_midi: float, end_midi: float, seconds: float, sample_rate: int) -> np.ndarray:
    count = round(seconds * sample_rate)
    midi = np.linspace(start_midi, end_midi, count)
    frequency = 440.0 * 2 ** ((midi - 69) / 12)
    return 0.22 * np.sin(2 * np.pi * np.cumsum(frequency) / sample_rate)


class SyntheticPipelineTests(unittest.TestCase):
    def test_melody_keeps_three_repeated_notes_and_known_pitch_order(self) -> None:
        sample_rate = 22050
        silence = np.zeros(round(0.08 * sample_rate))
        parts = []
        expected = [60, 60, 60, 62, 64, 65, 67, 69, 67, 65, 64, 62]
        for index, midi in enumerate(expected):
            parts.append(tone(midi, 0.75, sample_rate))
            if index == 2:
                parts.append(glide(60, 62, 0.06, sample_rate))
            parts.append(silence)
        audio = np.concatenate(parts)
        notes, _ = transcribe_vocals(audio, sample_rate)
        detected = [note.midi for note in notes]
        self.assertEqual(detected[:3], [60, 60, 60])
        self.assertEqual(detected, expected)
        transpose = choose_transpose(notes)
        quantized = quantize_notes(notes, TempoGrid(80, 0, "synthetic", 1.0), transpose)
        self.assertEqual(len(quantized), len(expected))
        self.assertTrue(all(note.duration_slots >= 1 for note in quantized))
```

- [ ] **Step 2: 运行合成测试**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_synthetic_pipeline -v`

Expected: PASS，音符顺序与 12 个合成音完全一致，前三个同音保持为三个事件。

- [ ] **Step 3: 运行全部快速测试并提交**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest discover -s tests -p 'test_*.py' -v`

Expected: 全部快速测试 PASS；真实歌曲回归可按环境条件 SKIP。

```powershell
git add tests/test_synthetic_pipeline.py
git commit -m "test: cover synthetic vocal transcription pipeline"
```

---

### Task 11: 用《晴天》执行真实回归并写使用说明

**Files:**
- Create: `tests/test_sunny_regression.py`
- Create: `README.md`

- [ ] **Step 1: 写默认跳过、显式开启的真实歌曲回归**

```python
# tests/test_sunny_regression.py
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_harmonica_score import run_pipeline

SOURCE = Path(r"D:\QQ音乐缓存\周杰伦 - 晴天_L.mp3")


@unittest.skipUnless(os.environ.get("RUN_REAL_AUDIO_REGRESSION") == "1", "set RUN_REAL_AUDIO_REGRESSION=1")
class SunnyDayRegressionTests(unittest.TestCase):
    def test_full_pipeline_matches_known_sunny_day_characteristics(self) -> None:
        if not SOURCE.exists():
            self.skipTest("local Sunny Day source audio is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            output = run_pipeline(SOURCE, Path(directory), keep_stems=True, force=False)
            expected_suffixes = [
                "-连续按键谱.md", "-详细节奏谱.md", "-音符明细.csv",
                "-主旋律.mid", "-口琴试听.wav", "-分析报告.json",
            ]
            for suffix in expected_suffixes:
                self.assertTrue((output / f"{SOURCE.stem}{suffix}").exists(), suffix)
            self.assertTrue((output / "stems" / "vocals.wav").exists())
            self.assertTrue((output / "stems" / "no_vocals.wav").exists())
            report = json.loads((output / f"{SOURCE.stem}-分析报告.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(report["note_count"], 330)
            self.assertLessEqual(report["note_count"], 450)
            self.assertIn(report["source_range"][0], {"C#3", "D3", "D#3"})
            self.assertIn(report["source_range"][1], {"G#4", "A4", "A#4"})
            self.assertEqual(report["transpose"], 5)
            self.assertGreaterEqual(report["play_midi_range"][0], 48)
            self.assertLessEqual(report["play_midi_range"][1], 85)
            self.assertLess(abs(report["midi_seconds"] - report["wav_seconds"]), 0.1)
```

- [ ] **Step 2: 运行真实回归**

Run:

```powershell
$env:RUN_REAL_AUDIO_REGRESSION = '1'
& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_sunny_regression -v
Remove-Item Env:RUN_REAL_AUDIO_REGRESSION
```

Expected: PASS，生成六类正式结果和两条分离音轨；音符数 330–450、原始音域约 D3–A4、MIDI/WAV 时长差小于 0.1 秒。

- [ ] **Step 3: 写用户使用说明**

```markdown
# 三角洲行动口琴自动扒谱

## 最简单的用法

把一个 MP3、WAV 或 FLAC 文件拖到 `提取口琴谱.bat` 上。第一次使用会建立本地环境并下载人声分离模型；完成后会打开 `output` 文件夹。

每首歌会得到连续按键谱、详细节奏谱、CSV、MIDI、口琴试听 WAV、分析报告，以及用于检查的人声/伴奏分离音轨。

连续谱示例：`5(左)` 表示鼠标左键加数字 5，`4(中)` 表示鼠标中键加数字 4，`2(右)` 表示鼠标右键加数字 2，`1(,)` 表示直接按逗号键。

## 命令行

```powershell
& '.\.tools\audio-venv\Scripts\python.exe' scripts\extract_harmonica_score.py 'D:\Music\song.mp3' --keep-stems
```

已有同名结果时，程序不会覆盖；确认覆盖可添加 `--force`。自动扒谱适合单人演唱的有调旋律，合唱、说唱、强和声和自由节拍可能需要人工校对。
```

- [ ] **Step 4: 跑完整验证**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest discover -s tests -p 'test_*.py' -v`

Expected: 快速测试全部 PASS，真实歌曲测试显示 SKIP（因为未设置环境变量）。

Run: `git status --short`

Expected: 只显示本任务尚未提交的 README、真实回归测试和有证据的算法修复；不显示 `.tools/`、`analysis/`、`output/` 或音频文件。

- [ ] **Step 5: 提交真实回归和文档**

```powershell
git add README.md tests/test_sunny_regression.py
git commit -m "docs: add usage guide and real-audio regression"
```

---

### Task 12: 最终验收和干净交付

**Files:**
- Verify only; no planned source changes

- [ ] **Step 1: 验证依赖可从锁定文件解析**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m pip install --dry-run -r requirements-audio.txt`

Expected: 命令成功，所有固定版本均可解析；现有环境中应显示已满足。

- [ ] **Step 2: 验证 CLI 帮助和中文路径**

Run: `& '.\.tools\audio-venv\Scripts\python.exe' scripts\extract_harmonica_score.py --help`

Expected: 帮助中包含 `audio`、`--output`、`--keep-stems` 和 `--force`。

Run: `& '.\.tools\audio-venv\Scripts\python.exe' -m unittest tests.test_cli tests.test_audio tests.test_render -v`

Expected: 全部 PASS，含中文和空格的模拟路径处理正常。

- [ ] **Step 3: 验证 Git 状态和提交历史**

Run: `git status --short`

Expected: 空输出。

Run: `git log --oneline --decorate -12`

Expected: 能看到每个独立阶段的提交，从包骨架、指法、转录、节奏、音频、渲染、编排、拖放、合成测试到回归文档。

- [ ] **Step 4: 向用户交付**

交付信息只包含：拖放入口的可点击路径、一个连续谱输出示例路径、测试总数与结果、真实《晴天》回归的音符数/音域/移调/时长差，以及 Git 最终提交号。不要让用户阅读安装日志才能知道是否成功。
