import unittest

import numpy as np

from scripts.harmonica.transcription import (
    bridge_tiny_gaps,
    clean_notes,
    detect_onsets,
    extract_pitch_frames,
    frames_to_notes,
    smooth_quantized_pitch,
    validate_melody,
)
from scripts.harmonica.models import NoteEvent, TranscriptionError


class PitchCleanupTests(unittest.TestCase):
    def test_smooth_quantized_pitch_uses_local_mode(self):
        midi = np.array([60.1, 60.2, 61.0, 60.0, 59.9])
        valid = np.ones(len(midi), dtype=bool)

        result = smooth_quantized_pitch(midi, valid, radius=2)

        np.testing.assert_array_equal(result, np.full(5, 60.0))

    def test_bridge_tiny_gaps_fills_matching_neighbours(self):
        pitch = np.array([60.0, np.nan, np.nan, 60.0, 62.0])

        result = bridge_tiny_gaps(pitch, max_frames=2)

        np.testing.assert_array_equal(result, np.array([60.0, 60.0, 60.0, 60.0, 62.0]))

    def test_bridge_tiny_gaps_leaves_mismatched_neighbours(self):
        pitch = np.array([60.0, np.nan, 62.0])

        result = bridge_tiny_gaps(pitch, max_frames=2)

        self.assertTrue(np.isnan(result[1]))


class PitchExtractionTests(unittest.TestCase):
    def test_extract_pitch_frames_tracks_a4_sine_wave(self):
        sample_rate = 22_050
        seconds = np.arange(sample_rate) / sample_rate
        audio = 0.3 * np.sin(2 * np.pi * 440.0 * seconds)

        frames = extract_pitch_frames(audio, sample_rate)

        finite_pitch = frames.pitch[np.isfinite(frames.pitch)]
        self.assertGreater(len(finite_pitch), 20)
        self.assertAlmostEqual(float(np.median(finite_pitch)), 69.0, delta=0.5)


class NoteSegmentationTests(unittest.TestCase):
    def test_detect_onsets_handles_silence_without_invalid_values(self):
        onset_frames, onset_envelope = detect_onsets(
            np.zeros(2_205), sr=22_050, hop=256
        )

        self.assertEqual(onset_frames.dtype, np.dtype(int))
        self.assertTrue(np.all(np.isfinite(onset_envelope)))
        self.assertTrue(np.all(onset_envelope == 0.0))

    def test_onsets_split_repeated_notes_inside_stable_pitch(self):
        pitch = np.full(100, 60.0)
        times = np.arange(100) * 0.01
        probabilities = np.full(100, 0.9)
        onset_frames = np.array([0, 33, 66])
        onset_envelope = np.zeros(100)
        onset_envelope[onset_frames] = 1.0

        notes = frames_to_notes(
            times,
            pitch,
            probabilities,
            onset_frames,
            onset_envelope,
            hop_seconds=0.01,
        )

        self.assertEqual([note.midi for note in notes], [60, 60, 60])

    def test_short_pitch_flip_between_equal_neighbours_is_folded(self):
        pitch = np.array([60] * 20 + [61] * 5 + [60] * 20, dtype=float)
        times = np.arange(len(pitch)) * 0.01

        notes = frames_to_notes(
            times,
            pitch,
            np.full(len(pitch), 0.9),
            np.array([0]),
            np.ones(len(pitch)),
            hop_seconds=0.01,
        )

        self.assertEqual([note.midi for note in notes], [60])

    def test_short_pitch_flip_does_not_promote_discarded_scores(self):
        notes = [
            NoteEvent(0.0, 0.2, 60, 0.3, 0.2),
            NoteEvent(0.2, 0.25, 61, 1.0, 1.0),
            NoteEvent(0.25, 0.45, 60, 0.4, 0.1),
        ]

        result = clean_notes(notes)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].confidence, 0.4)
        self.assertEqual(result[0].onset_strength, 0.2)

    def test_validate_melody_rejects_fewer_than_twelve_notes(self):
        notes = [NoteEvent(index, index + 1, 60, 0.9) for index in range(11)]

        with self.assertRaisesRegex(TranscriptionError, "12"):
            validate_melody(notes)

    def test_validate_melody_rejects_less_than_eight_seconds(self):
        notes = [
            NoteEvent(index * 0.5, index * 0.5 + 0.5, 60, 0.9)
            for index in range(12)
        ]

        with self.assertRaisesRegex(TranscriptionError, "8"):
            validate_melody(notes)

    def test_validate_melody_rejects_low_median_confidence(self):
        notes = [NoteEvent(index, index + 1, 60, 0.54) for index in range(12)]

        with self.assertRaisesRegex(TranscriptionError, "置信度"):
            validate_melody(notes)


if __name__ == "__main__":
    unittest.main()
