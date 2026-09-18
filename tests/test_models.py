import unittest
import importlib
import sys
from pathlib import Path

from scripts.harmonica import NoteEvent, QuantizedNote, TempoGrid
from scripts.harmonica import (
    HarmonicaError,
    InputValidationError,
    InstrumentRangeError,
    RhythmError,
    TranscriptionError,
)
from scripts.harmonica import models


class ModelTests(unittest.TestCase):
    def test_exceptions_have_dedicated_compatible_module(self):
        exceptions = importlib.import_module("scripts.harmonica.exceptions")
        names = (
            "HarmonicaError",
            "InputValidationError",
            "TranscriptionError",
            "RhythmError",
            "InstrumentRangeError",
        )
        for name in names:
            package_class = getattr(exceptions, name)
            self.assertIs(package_class, getattr(models, name))
            self.assertIs(package_class, globals()[name])

        scripts_path = str(Path(__file__).parents[1] / "scripts")
        sys.path.insert(0, scripts_path)
        try:
            path_exceptions = importlib.import_module("harmonica.exceptions")
            for name in names:
                self.assertIs(getattr(exceptions, name), getattr(path_exceptions, name))
        finally:
            sys.path.remove(scripts_path)

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
