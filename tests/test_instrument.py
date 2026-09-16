import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.harmonica.instrument import (
    InstrumentRangeError,
    canonical_fingering,
    choose_transpose,
    continuous_token,
    playable_pitches,
)
from scripts.harmonica.models import Fingering, NoteEvent


class InstrumentTests(unittest.TestCase):
    def test_playable_pitches_are_contiguous_c3_to_csharp6(self):
        self.assertEqual(playable_pitches(), set(range(48, 86)))

    def test_canonical_fingering_prefers_plain_z(self):
        self.assertEqual(canonical_fingering(60), Fingering(0, False, 0))

    def test_continuous_token_uses_comma_for_high_octave(self):
        self.assertEqual(continuous_token(72), "1(,)")

    def test_continuous_token_labels_modifiers(self):
        self.assertEqual(continuous_token(55), "5(左)")
        self.assertEqual(continuous_token(66), "4(中)")
        self.assertEqual(continuous_token(56), "5(左+中)")

    def test_canonical_fingering_rejects_below_range(self):
        with self.assertRaisesRegex(InstrumentRangeError, "C3.*C#6"):
            canonical_fingering(47)

    def test_choose_transpose_prefers_plus_five(self):
        notes = [NoteEvent(0, 1, p, 1.0) for p in (55, 57, 59, 60)]
        self.assertEqual(choose_transpose(notes), 5)

    def test_choose_transpose_rejects_melody_too_wide(self):
        notes = [NoteEvent(0, 1, 40, 1.0), NoteEvent(1, 2, 90, 1.0)]
        with self.assertRaises(InstrumentRangeError):
            choose_transpose(notes)


if __name__ == "__main__":
    unittest.main()
