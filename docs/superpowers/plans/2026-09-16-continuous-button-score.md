# Continuous Button Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a separate, sectioned continuous-button edition of the existing 398-note *Sunny Day* game-harmonica score, with mouse buttons written beside each affected number.

**Architecture:** Extend the existing score generator with one pure MIDI-pitch-to-continuous-token formatter and one Markdown renderer. Reuse the existing parsed `Note` sequence and section/bar metadata, so the new output cannot drift from the detailed score. Add unit and integration checks for modifier spelling, note count, section layout, and preservation of existing artifacts.

**Tech Stack:** Python 3.12, standard-library `unittest`, existing `mido`, existing score generator.

---

### Task 1: Add continuous-token formatting with tests

**Files:**
- Create: `tests/test_continuous_score.py`
- Modify: `scripts/build_game_score.py:102-143`

- [ ] **Step 1: Write the failing token-format tests**

Create `tests/test_continuous_score.py` with these imports and cases:

```python
from __future__ import annotations

import importlib.util
import sys
import unittest
from unittest import mock
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_game_score.py"
SPEC = importlib.util.spec_from_file_location("build_game_score", MODULE_PATH)
assert SPEC and SPEC.loader
score = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = score
SPEC.loader.exec_module(score)


class ContinuousTokenTests(unittest.TestCase):
    def test_plain_middle_note(self) -> None:
        self.assertEqual(score.continuous_token(67), "5")

    def test_low_note_puts_left_button_beside_number(self) -> None:
        self.assertEqual(score.continuous_token(55), "5(左)")

    def test_sharp_puts_middle_button_beside_number(self) -> None:
        self.assertEqual(score.continuous_token(66), "4(中)")

    def test_low_sharp_combines_mouse_buttons(self) -> None:
        self.assertEqual(score.continuous_token(56), "5(左+中)")

    def test_high_two_uses_right_button(self) -> None:
        self.assertEqual(score.continuous_token(74), "2(右)")

    def test_high_one_uses_comma_without_right_button(self) -> None:
        self.assertEqual(score.continuous_token(72), "1(,)")

    def test_high_sharp_one_uses_middle_and_comma(self) -> None:
        self.assertEqual(score.continuous_token(73), "1(中+,)")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the formatter is missing**

Run:

```powershell
& '.\.tools\audio-venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Expected: all seven tests error with `AttributeError: module 'build_game_score' has no attribute 'continuous_token'`.

- [ ] **Step 3: Implement the minimal token formatter**

Add this function immediately after `numbered_symbol` in `scripts/build_game_score.py`:

```python
def continuous_token(midi: int) -> str:
    relative = midi - 60
    octave, pc = divmod(relative, 12)
    degree, sharp = PC_TO_DEGREE[pc]

    if octave == 1 and degree == 1:
        return f"1({'\u4e2d+' if sharp else ''},)"

    modifiers: list[str] = []
    if octave < 0:
        modifiers.extend(["左"] * abs(octave))
    elif octave > 0:
        modifiers.extend(["右"] * octave)
    if sharp:
        modifiers.append("中")
    return f"{degree}({'+'.join(modifiers)})" if modifiers else str(degree)
```

- [ ] **Step 4: Run the token tests and confirm they pass**

Run:

```powershell
& '.\.tools\audio-venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Expected: `Ran 7 tests` and `OK`.

### Task 2: Render, generate, and validate the continuous edition

**Files:**
- Modify: `scripts/build_game_score.py:198-260`
- Modify: `scripts/build_game_score.py:365-384`
- Modify: `tests/test_continuous_score.py`
- Create: `output/晴天-连续按键版.md`

- [ ] **Step 1: Add failing document-rendering tests**

Add these imports and tests to `tests/test_continuous_score.py`:

```python
MIDI_PATH = Path(__file__).parents[1] / "analysis" / "reference" / "hamienet-qing-tian.mid"


class ContinuousDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notes = score.extract_track_notes(MIDI_PATH, "Lead")
        cls.document = score.continuous_markdown(cls.notes)

    def test_document_uses_all_398_notes(self) -> None:
        self.assertEqual(len(self.notes), 398)
        with mock.patch.object(score, "continuous_token", wraps=score.continuous_token) as formatter:
            score.continuous_markdown(self.notes)
        self.assertEqual(formatter.call_count, 398)

    def test_document_has_no_old_pitch_symbols(self) -> None:
        self.assertNotIn("↑", self.document)
        self.assertNotIn("↓", self.document)
        self.assertNotIn("♯", self.document)

    def test_document_keeps_sections_and_bar_separators(self) -> None:
        self.assertIn("## 主歌 A（00:29）", self.document)
        self.assertIn("## 副歌 2（03:03）", self.document)
        self.assertIn("｜", self.document)

    def test_document_explains_rhythm_source(self) -> None:
        self.assertIn("详细节奏谱", self.document)

    def test_document_omits_rest_placeholders(self) -> None:
        self.assertNotIn("·", self.document)
```

- [ ] **Step 2: Run the tests and confirm the renderer is missing**

Run:

```powershell
& '.\.tools\audio-venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Expected: the seven token tests pass and the five document tests error because `continuous_markdown` is not defined.

- [ ] **Step 3: Implement the continuous Markdown renderer**

Add this function after `markdown_score` in `scripts/build_game_score.py`:

```python
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
        tokens = [
            continuous_token(note.play_midi)
            for note in notes_by_bar[bar]
        ]
        if tokens:
            current_bars.append(" ".join(tokens))
    if current_bars:
        lines.extend(["｜".join(current_bars), ""])
    return "\n".join(lines)
```

- [ ] **Step 4: Add the new output without changing existing artifacts**

In `main`, after writing `score_path`, add:

```python
    continuous_path = args.output / "晴天-连续按键版.md"
    continuous_path.write_text(continuous_markdown(notes), encoding="utf-8")
```

Add this final status line after the existing score print:

```python
    print(f"continuous_score={continuous_path}")
```

- [ ] **Step 5: Run all tests**

Run:

```powershell
& '.\.tools\audio-venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Expected: `Ran 12 tests` and `OK`.

- [ ] **Step 6: Generate the continuous score**

Run:

```powershell
& '.\.tools\audio-venv\Scripts\python.exe' '.\scripts\build_game_score.py' '.\analysis\reference\hamienet-qing-tian.mid' '.\output'
```

Expected output includes:

```text
notes=398
continuous_score=output\晴天-连续按键版.md
```

- [ ] **Step 7: Verify artifact preservation and notation constraints**

Run:

```powershell
$continuous = Get-Content -LiteralPath '.\output\晴天-连续按键版.md' -Raw -Encoding UTF8
if ($continuous -match '[↑↓♯]') { throw '连续版仍包含旧音高记号' }
Get-Item '.\output\晴天-三角洲口琴谱.md', '.\output\晴天-口琴主旋律.mid', '.\output\晴天-口琴主旋律试听.wav', '.\output\晴天-音符明细.csv', '.\output\晴天-连续按键版.md' | Select-Object Name,Length
```

Expected: no exception; all five artifacts exist and have nonzero length.

This workspace is not a Git repository, so commit steps are intentionally omitted.
