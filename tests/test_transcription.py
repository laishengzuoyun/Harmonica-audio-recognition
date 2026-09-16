import unittest

import numpy as np

from scripts.harmonica.transcription import (
    bridge_tiny_gaps,
    extract_pitch_frames,
    smooth_quantized_pitch,
)


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


if __name__ == "__main__":
    unittest.main()
