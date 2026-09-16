import unittest

from scripts.harmonica import NoteEvent, QuantizedNote, TempoGrid


class ModelTests(unittest.TestCase):
    def test_note_event_duration(self):
        event = NoteEvent(1.0, 1.25, 60, 0.8)
        self.assertAlmostEqual(event.duration, 0.25)

    def test_quantized_note_rejects_zero_duration(self):
        with self.assertRaisesRegex(ValueError, "duration_slots"):
            QuantizedNote(0.0, 0.0, 60, 60, 0.8, 1, 1, 0)

    def test_tempo_grid_preserves_source(self):
        grid = TempoGrid(120.0, 0.5, "beats", 0.9)
        self.assertEqual(grid.source, "beats")


if __name__ == "__main__":
    unittest.main()
