import unittest
from unittest import mock

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

    def test_fits_long_beat_timestamps_without_cumulative_slot_drift(self):
        sample_rate = 22_050
        hop_length = 512
        beat_frames = np.rint(
            np.arange(120) * 0.5 * sample_rate / hop_length
        ).astype(int)
        beat_times = beat_frames * hop_length / sample_rate
        notes = [
            NoteEvent(float(beat_times[0]), float(beat_times[0] + 0.1), 60, 0.9),
            NoteEvent(float(beat_times[-1]), float(beat_times[-1] + 0.1), 62, 0.9),
        ]

        with mock.patch(
            "scripts.harmonica.rhythm.librosa.beat.beat_track",
            return_value=(np.array([117.45]), beat_frames),
        ):
            grid = detect_tempo_grid(np.zeros(sample_rate), sample_rate, notes=[])

        self.assertAlmostEqual(grid.bpm, 120.0, delta=0.1)
        quantized = quantize_notes(notes, grid, 0)
        positions = [(note.bar - 1) * 16 + note.slot for note in quantized]
        self.assertEqual(positions[1] - positions[0], 4 * 119)

    def test_accepts_locally_consistent_beats_with_gradual_tempo_drift(self):
        sample_rate = 22_050
        hop_length = 512
        drifting_bpms = np.linspace(159.8, 161.0, 589)
        beat_times = np.concatenate(
            ([0.67], 0.67 + np.cumsum(60.0 / drifting_bpms))
        )
        beat_frames = np.rint(beat_times * sample_rate / hop_length).astype(int)

        with mock.patch(
            "scripts.harmonica.rhythm.librosa.beat.beat_track",
            return_value=(np.array([160.0]), beat_frames),
        ):
            grid = detect_tempo_grid(np.zeros(sample_rate), sample_rate, notes=[])

        self.assertEqual(grid.source, "beats-adaptive")
        self.assertGreaterEqual(grid.consistency, 0.95)
        self.assertAlmostEqual(grid.bpm, 160.4, delta=1.0)
        self.assertEqual(len(grid.beat_times), len(beat_times))

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

    def test_interval_fallback_scores_global_onset_phase(self):
        starts = np.cumsum([0.0] + [0.16, 0.31, 0.46, 0.61] * 10)
        notes = [
            NoteEvent(float(start), float(start + 0.1), 60, 0.9)
            for start in starts
        ]

        with mock.patch(
            "scripts.harmonica.rhythm.librosa.beat.beat_track",
            return_value=(np.array([0.0]), np.array([], dtype=int)),
        ):
            grid = detect_tempo_grid(np.zeros(22_050), 22_050, notes)

        positions = (starts - grid.anchor) / (60.0 / grid.bpm / 4.0)
        phase_consistency = float(
            np.mean(np.abs(positions - np.rint(positions)) <= 0.18)
        )
        self.assertGreaterEqual(phase_consistency, 0.55)
        self.assertAlmostEqual(grid.consistency, phase_consistency)


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

    def test_quantizes_gradual_tempo_drift_against_observed_beats(self):
        drifting_bpms = np.linspace(150.0, 165.0, 99)
        beat_times = np.concatenate(
            ([1.0], 1.0 + np.cumsum(60.0 / drifting_bpms))
        )
        beat_indexes = np.array([0, 25, 50, 75, 99])
        notes = [
            NoteEvent(float(beat_times[index]), float(beat_times[index] + 0.1), 60, 0.9)
            for index in beat_indexes
        ]
        period, _ = np.polyfit(np.arange(len(beat_times)), beat_times, 1)
        grid = TempoGrid(
            60.0 / float(period),
            float(beat_times[0]),
            "beats-adaptive",
            0.95,
            tuple(float(value) for value in beat_times),
        )

        quantized = quantize_notes(notes, grid, 0)

        positions = [
            (note.bar - 1) * 16 + note.slot - 1 for note in quantized
        ]
        self.assertEqual(positions, (beat_indexes * 4).tolist())

    def test_rejects_empty_notes_or_invalid_tempo(self):
        with self.assertRaises(RhythmError):
            quantize_notes([], TempoGrid(120.0, 0.0, "beats", 1.0), 0)
        with self.assertRaises(RhythmError):
            quantize_notes(
                [NoteEvent(0.0, 0.1, 60, 0.9)],
                TempoGrid(0.0, 0.0, "beats", 1.0),
                0,
            )

    def test_keeps_adjacent_short_notes_monophonic_after_quantization(self):
        notes = [
            NoteEvent(0.0, 0.071, 60, 0.9),
            NoteEvent(0.071, 0.142, 61, 0.9),
            NoteEvent(0.142, 0.22, 62, 0.9),
        ]

        quantized = quantize_notes(notes, TempoGrid(120.0, 0.0, "beats", 1.0), 0)

        starts = [(note.bar - 1) * 16 + note.slot - 1 for note in quantized]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(all(later > earlier for earlier, later in zip(starts, starts[1:])))
        self.assertTrue(
            all(
                note.duration_slots <= next_start - start
                for note, start, next_start in zip(quantized, starts, starts[1:])
            )
        )

    def test_rejects_nonfinite_anchor_and_note_times(self):
        valid_note = NoteEvent(0.0, 0.1, 60, 0.9)
        with self.assertRaisesRegex(RhythmError, "anchor"):
            quantize_notes([valid_note], TempoGrid(120.0, np.nan, "beats", 1.0), 0)
        with self.assertRaisesRegex(RhythmError, "finite"):
            quantize_notes(
                [NoteEvent(np.nan, 0.1, 60, 0.9)],
                TempoGrid(120.0, 0.0, "beats", 1.0),
                0,
            )

    def test_rejects_nonchronological_or_backwards_notes(self):
        grid = TempoGrid(120.0, 0.0, "beats", 1.0)
        with self.assertRaisesRegex(RhythmError, "chronological"):
            quantize_notes(
                [
                    NoteEvent(0.2, 0.3, 60, 0.9),
                    NoteEvent(0.1, 0.2, 62, 0.9),
                ],
                grid,
                0,
            )
        with self.assertRaisesRegex(RhythmError, "end"):
            quantize_notes([NoteEvent(0.2, 0.1, 60, 0.9)], grid, 0)


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
