import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts import extract_harmonica_score


SOURCE = Path(r"D:\QQ音乐缓存\周杰伦 - 晴天_L.mp3")
RUN_REAL_AUDIO_REGRESSION = os.environ.get("RUN_REAL_AUDIO_REGRESSION") == "1"


@unittest.skipUnless(
    RUN_REAL_AUDIO_REGRESSION,
    "set RUN_REAL_AUDIO_REGRESSION=1 to run the real-audio regression",
)
class SunnyRegressionTests(unittest.TestCase):
    def test_full_pipeline_produces_expected_sunny_score(self):
        if not SOURCE.is_file():
            self.skipTest(f"real-audio fixture is unavailable: {SOURCE}")

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = extract_harmonica_score.run_pipeline(
                SOURCE,
                Path(temporary_directory),
                keep_stems=True,
                force=False,
            )

            suffixes = (
                "连续按键谱.md",
                "详细节奏谱.md",
                "音符明细.csv",
                "主旋律.mid",
                "口琴试听.wav",
                "分析报告.json",
            )
            for suffix in suffixes:
                self.assertTrue(
                    (result / f"{SOURCE.stem}-{suffix}").is_file(),
                    suffix,
                )
            self.assertTrue((result / "stems" / "vocals.wav").is_file())
            self.assertTrue((result / "stems" / "no_vocals.wav").is_file())

            report_path = result / f"{SOURCE.stem}-分析报告.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(report["note_count"], 330)
            self.assertLessEqual(report["note_count"], 450)
            self.assertIn(report["source_range"][0], {"C#3", "D3", "D#3"})
            self.assertIn(report["source_range"][1], {"G#4", "A4", "A#4"})
            self.assertEqual(report["transpose"], 5)
            self.assertTrue(
                all(48 <= midi <= 85 for midi in report["play_midi_range"])
            )
            self.assertLess(
                abs(report["midi_seconds"] - report["wav_seconds"]),
                0.1,
            )


if __name__ == "__main__":
    unittest.main()
