import unittest

import numpy as np

from scripts.harmonica.models import NoteEvent, QuantizedNote, RhythmError, TempoGrid
from scripts.harmonica.rhythm import (
    detect_tempo_grid,
    quantize_notes,
    section_starts,
)


class TempoGridTests(unittest.TestCase):
    def test_detects_120_bpm_from_regular_click_beats(self):
        import librosa

        sample_rate = 22_050
        accompaniment = librosa.clicks(
            times=np.arange(0.5, 8.0, 0.5), sr=sample_rate, length=8 * sample_rate
        )

        grid = detect_tempo_grid(accompaniment, sample_rate, notes=[])

        self.assertAlmostEqual(grid.bpm, 120.0, delta=3.0)
        self.assertEqual(grid.source, "beats")

    def test_falls_back_to_note_intervals_when_accompaniment_has_no_beats(self):
        notes = [NoteEvent(i * 0.15, i * 0.15 + 0.1, 60, 0.9) for i in range(40)]

        grid = detect_tempo_grid(np.zeros(22_050), 22_050, notes)

        self.assertAlmostEqual(grid.bpm, 100.0, delta=2.0)
        self.assertEqual(grid.source, "intervals")

    def test_rejects_unusable_audio_and_insufficient_note_intervals(self):
        with self.assertRaises(RhythmError):
            detect_tempo_grid(
                np.zeros(22_050), 22_050, [NoteEvent(0.0, 0.2, 60, 0.9)]
            )


class QuantizationTests(unittest.TestCase):
    def test_quantizes_notes_to_sixteenth_slots(self):
        notes = [
            NoteEvent(1.01, 1.02, 60, 0.8),
            NoteEvent(1.26, 1.48, 62, 0.9),
        ]

        quantized = quantize_notes(notes, TempoGrid(120.0, 1.0, "beats", 1.0), 0)

        self.assertEqual(
            (
                quantized[0].bar,
                quantized[0].slot,
                quantized[0].duration_slots,
            ),
            (1, 1, 1),
        )
        self.assertEqual((quantized[1].bar, quantized[1].slot), (1, 3))
        self.assertEqual(quantized[1].play_midi, 62)

    def test_rejects_empty_notes_or_invalid_tempo(self):
        with self.assertRaises(RhythmError):
            quantize_notes([], TempoGrid(120.0, 0.0, "beats", 1.0), 0)
        with self.assertRaises(RhythmError):
            quantize_notes(
                [NoteEvent(0.0, 0.1, 60, 0.9)],
                TempoGrid(0.0, 0.0, "beats", 1.0),
                0,
            )


class SectionTests(unittest.TestCase):
    def test_starts_new_section_after_large_time_gap(self):
        notes = [
            NoteEvent(0.0, 0.2, 60, 0.9),
            NoteEvent(0.3, 0.5, 62, 0.9),
            NoteEvent(2.0, 2.2, 64, 0.9),
        ]

        self.assertEqual(section_starts(notes), [0, 2])

    def test_starts_new_section_after_eight_bars(self):
        notes = [
            QuantizedNote(0.0, 0.1, 60, 60, 0.9, 1, 1, 1),
            QuantizedNote(0.2, 0.3, 62, 62, 0.9, 9, 1, 1),
        ]

        self.assertEqual(section_starts(notes, max_bars=8), [0, 1])


if __name__ == "__main__":
    unittest.main()
