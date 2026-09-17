import unittest

import numpy as np

from scripts.harmonica.instrument import choose_transpose
from scripts.harmonica.models import TempoGrid
from scripts.harmonica.rhythm import quantize_notes
from scripts.harmonica.transcription import transcribe_vocals


SAMPLE_RATE = 22_050
EXPECTED_MIDI = [60, 60, 60, 62, 64, 65, 67, 69, 67, 65, 64, 62]


def _phase_integrated_tone(midi: int, duration: float) -> np.ndarray:
    sample_count = round(duration * SAMPLE_RATE)
    time = np.arange(sample_count) / SAMPLE_RATE
    vibrato_midi = midi + 0.20 * np.sin(2.0 * np.pi * 5.2 * time)
    frequency = 440.0 * 2.0 ** ((vibrato_midi - 69.0) / 12.0)
    phase = 2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    attack = np.minimum(time / 0.012, 1.0)
    return 0.35 * attack * np.sin(phase)


def _phase_integrated_glide(start_midi: int, end_midi: int, duration: float) -> np.ndarray:
    sample_count = round(duration * SAMPLE_RATE)
    midi = np.linspace(start_midi, end_midi, sample_count, endpoint=False)
    frequency = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    phase = 2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    return 0.35 * np.sin(phase)


def _synthetic_melody() -> np.ndarray:
    silence = np.zeros(round(0.080 * SAMPLE_RATE))
    chunks = []
    for index, midi in enumerate(EXPECTED_MIDI):
        chunks.append(_phase_integrated_tone(midi, 0.75))
        if index == 2:
            chunks.append(_phase_integrated_glide(60, 62, 0.060))
        if index < len(EXPECTED_MIDI) - 1:
            chunks.append(silence)
    return np.concatenate(chunks)


class SyntheticPipelineTests(unittest.TestCase):
    def test_transcribes_and_quantizes_vibrato_melody_with_glide(self):
        notes, _ = transcribe_vocals(_synthetic_melody(), SAMPLE_RATE)

        self.assertEqual([note.midi for note in notes], EXPECTED_MIDI)
        self.assertEqual([note.midi for note in notes[:3]], [60, 60, 60])

        transpose = choose_transpose(notes)
        quantized = quantize_notes(
            notes,
            TempoGrid(80.0, 0.0, "synthetic", 1.0),
            transpose,
        )

        self.assertEqual(len(quantized), 12)
        self.assertTrue(all(note.duration_slots >= 1 for note in quantized))


if __name__ == "__main__":
    unittest.main()
