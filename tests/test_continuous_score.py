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
MIDI_PATH = Path(__file__).parents[1] / "analysis" / "reference" / "hamienet-qing-tian.mid"


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


if __name__ == "__main__":
    unittest.main()
