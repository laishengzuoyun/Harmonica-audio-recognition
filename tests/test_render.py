import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from scripts.harmonica.models import HarmonicaError, QuantizedNote, TempoGrid
from scripts.harmonica.render import (
    continuous_markdown,
    detailed_markdown,
    render_all,
    write_csv,
    write_midi,
    write_preview,
)


NOTES = [
    QuantizedNote(0.0, 0.2, 55, 55, 0.9, 1, 1, 2),
    QuantizedNote(0.25, 0.5, 60, 60, 0.8, 1, 3, 2),
    QuantizedNote(2.0, 2.2, 72, 72, 0.85, 2, 1, 1),
]
GRID = TempoGrid(120.0, 0.0, "beats", 0.9)


class MarkdownRenderTests(unittest.TestCase):
    def test_render_all_writes_six_scores_with_sections_and_notation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = render_all("测试歌", NOTES, GRID, 0, Path(directory), [0])

            self.assertEqual(len(paths), 6)
            self.assertTrue(all(path.exists() for path in paths))
            self.assertEqual(
                [path.name for path in paths],
                [
                    "测试歌-连续按键谱.md",
                    "测试歌-详细节奏谱.md",
                    "测试歌-音符明细.csv",
                    "测试歌-主旋律.mid",
                    "测试歌-口琴试听.wav",
                    "测试歌-分析报告.json",
                ],
            )
            continuous = (Path(directory) / "测试歌-连续按键谱.md").read_text(
                encoding="utf-8"
            )
            detailed = (Path(directory) / "测试歌-详细节奏谱.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("5(左)", continuous)
            self.assertIn("1(,)", continuous)
            self.assertIn("｜", continuous)
            self.assertIn("—", detailed)
            self.assertIn("·", detailed)
            for rendered in (continuous, detailed):
                self.assertIn("第 1 段", rendered)
                self.assertIn("00:00.0", rendered)

    def test_markdown_starts_each_requested_section_at_its_note(self):
        continuous = continuous_markdown("测试歌", NOTES, [0, 2])
        detailed = detailed_markdown("测试歌", NOTES, [0, 2])

        for rendered in (continuous, detailed):
            self.assertIn("第 2 段（00:02.0）", rendered)

    def test_markdown_preserves_two_section_starts_in_the_same_bar(self):
        same_bar_notes = [
            QuantizedNote(0.0, 0.2, 55, 55, 0.9, 1, 1, 1),
            QuantizedNote(0.5, 0.7, 72, 72, 0.8, 1, 5, 1),
        ]

        continuous = continuous_markdown("测试歌", same_bar_notes, [0, 1])
        detailed = detailed_markdown("测试歌", same_bar_notes, [0, 1])
        for rendered in (continuous, detailed):
            self.assertEqual(rendered.count("第 1 段（00:00.0）"), 1)
            self.assertEqual(rendered.count("第 2 段（00:00.5）"), 1)
            positions = [
                rendered.index("第 1 段（00:00.0）"),
                rendered.index("5(左)"),
                rendered.index("第 2 段（00:00.5）"),
                rendered.index("1(,)"),
            ]
            self.assertEqual(positions, sorted(positions))
        self.assertIn("01.01–04", detailed)
        self.assertIn("01.05–16", detailed)


class MetadataAndCsvTests(unittest.TestCase):
    def test_report_preserves_analysis_and_describes_pitch_ranges(self):
        analysis = {
            "filtered_frame_count": 7,
            "filtered_note_count": 2,
            "warnings": ["低置信度"],
            "detector": "fixture",
        }
        with tempfile.TemporaryDirectory() as directory:
            render_all("测试歌", NOTES, GRID, 12, Path(directory), [0, 2], analysis)

            with (Path(directory) / "测试歌-音符明细.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            report = json.loads(
                (Path(directory) / "测试歌-分析报告.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(report["note_count"], 3)
            self.assertEqual(report["filtered_frame_count"], 7)
            self.assertEqual(report["filtered_note_count"], 2)
            self.assertEqual(report["warnings"], ["低置信度"])
            self.assertEqual(report["detector"], "fixture")
            self.assertEqual(report["transpose"], 12)
            self.assertEqual(report["source_range"], ["G3", "C5"])
            self.assertEqual(report["play_range"], ["G3", "C5"])
            self.assertEqual(report["source_midi_range"], [55, 72])
            self.assertEqual(report["play_midi_range"], [55, 72])

    def test_csv_uses_bom_and_exact_canonical_game_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "notes.csv"
            write_csv(NOTES, destination)

            self.assertTrue(destination.read_bytes().startswith(b"\xef\xbb\xbf"))
            with destination.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["game_input"] for row in rows], ["左+B", "Z", ","])

    def test_partial_or_missing_analysis_keeps_all_default_diagnostics(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            render_all("测试歌", NOTES, GRID, 0, Path(first), [0], None)
            render_all(
                "测试歌",
                NOTES,
                GRID,
                0,
                Path(second),
                [0],
                {"warnings": ["需复核"], "custom": True},
            )

            reports = [
                json.loads(
                    (Path(root) / "测试歌-分析报告.json").read_text(
                        encoding="utf-8"
                    )
                )
                for root in (first, second)
            ]
            self.assertEqual(reports[0]["filtered_frame_count"], 0)
            self.assertEqual(reports[0]["filtered_note_count"], 0)
            self.assertEqual(reports[0]["warnings"], [])
            self.assertEqual(reports[1]["filtered_frame_count"], 0)
            self.assertEqual(reports[1]["filtered_note_count"], 0)
            self.assertEqual(reports[1]["warnings"], ["需复核"])
            self.assertTrue(reports[1]["custom"])

    def test_analysis_cannot_override_authoritative_report_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "reserved"):
                render_all(
                    "测试歌",
                    NOTES,
                    GRID,
                    0,
                    Path(directory),
                    [0],
                    {"note_count": 999, "bpm": 1, "wav_seconds": 0},
                )


class MediaRenderTests(unittest.TestCase):
    def test_midi_and_pcm_preview_are_readable_aligned_and_non_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            midi_path = Path(directory) / "preview.mid"
            wav_path = Path(directory) / "preview.wav"
            midi_seconds = write_midi(NOTES, GRID, midi_path)
            written_wav_seconds = write_preview(NOTES, GRID, wav_path)

            midi = mido.MidiFile(midi_path)
            note_ons = [
                message.note
                for track in midi.tracks
                for message in track
                if message.type == "note_on" and message.velocity > 0
            ]
            audio, sample_rate = sf.read(wav_path)
            info = sf.info(wav_path)
            decoded_wav_seconds = len(audio) / sample_rate
            self.assertEqual(note_ons, [55, 60, 72])
            self.assertLess(abs(midi.length - decoded_wav_seconds), 0.1)
            self.assertEqual(midi_seconds, midi.length)
            self.assertEqual(written_wav_seconds, decoded_wav_seconds)
            self.assertEqual(info.subtype, "PCM_16")
            self.assertTrue(all(math.isfinite(float(sample)) for sample in audio))
            self.assertGreater(float(np.max(np.abs(audio))), 0.01)


class ValidationTests(unittest.TestCase):
    def test_render_all_rejects_empty_notes_bad_tempo_and_bad_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with self.assertRaisesRegex(HarmonicaError, "empty|\u7a7a"):
                render_all("测试歌", [], GRID, 0, destination, [0])
            with self.assertRaisesRegex(HarmonicaError, "tempo|BPM"):
                render_all(
                    "测试歌",
                    NOTES,
                    TempoGrid(0.0, 0.0, "beats", 0.9),
                    0,
                    destination,
                    [0],
                )
            with self.assertRaisesRegex(HarmonicaError, "section|\u5206段"):
                render_all("测试歌", NOTES, GRID, 0, destination, [1])


if __name__ == "__main__":
    unittest.main()
