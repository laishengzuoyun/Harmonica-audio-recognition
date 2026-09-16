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
    def test_detect_onsets_finds_separated_synthesized_attacks(self):
        sample_rate = 22_050
        audio = np.zeros(sample_rate)
        attack_times = np.array([0.2, 0.5, 0.8])
        burst_samples = int(0.08 * sample_rate)
        burst_time = np.arange(burst_samples) / sample_rate
        burst = np.sin(2 * np.pi * 440.0 * burst_time) * np.exp(-35.0 * burst_time)
        for attack_time in attack_times:
            start = int(attack_time * sample_rate)
            audio[start : start + burst_samples] += burst

        onset_frames, _ = detect_onsets(audio, sr=sample_rate, hop=256)
        detected_times = onset_frames * 256 / sample_rate

        for attack_time in attack_times:
            self.assertLess(np.min(np.abs(detected_times - attack_time)), 0.05)

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

    def test_short_pitch_artifact_across_rests_does_not_join_notes(self):
        pitch = np.array(
            [60] * 20 + [np.nan] * 20 + [61] * 5 + [np.nan] * 20 + [60] * 20
        )
        times = np.arange(len(pitch)) * 0.01

        notes = frames_to_notes(
            times,
            pitch,
            np.full(len(pitch), 0.9),
            np.array([0]),
            np.ones(len(pitch)),
            hop_seconds=0.01,
        )

        self.assertEqual([note.midi for note in notes], [60, 60])
        self.assertAlmostEqual(notes[0].start, 0.0)
        self.assertAlmostEqual(notes[0].end, 0.2)
        self.assertAlmostEqual(notes[1].start, 0.65)
        self.assertAlmostEqual(notes[1].end, 0.85)

    def test_raw_pitch_variation_reduces_stability_confidence(self):
        pitch = np.full(20, 60.0)
        times = np.arange(len(pitch)) * 0.01
        probabilities = np.full(len(pitch), 0.8)
        onsets = np.array([0])
        onset_envelope = np.zeros(len(pitch))

        stable = frames_to_notes(
            times,
            pitch,
            probabilities,
            onsets,
            onset_envelope,
            hop_seconds=0.01,
            raw_midi=np.full(len(pitch), 60.0),
        )
        variable = frames_to_notes(
            times,
            pitch,
            probabilities,
            onsets,
            onset_envelope,
            hop_seconds=0.01,
            raw_midi=np.tile([59.6, 60.4], len(pitch) // 2),
        )

        self.assertAlmostEqual(stable[0].confidence, 0.77)
        self.assertAlmostEqual(
            variable[0].confidence, 0.65 * 0.8 + 0.25 * (1.0 - 0.4 / 0.7)
        )
        self.assertGreater(stable[0].confidence - variable[0].confidence, 0.1)

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
