import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from scripts.harmonica.audio import (
    InputValidationError,
    load_mono,
    separate_audio,
    validate_input,
)


class InputValidationTests(unittest.TestCase):
    def test_validate_input_rejects_m4a_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "song.m4a"
            source.touch()

            with self.assertRaisesRegex(InputValidationError, "MP3、WAV 或 FLAC"):
                validate_input(source, root / "output")

    def test_validate_input_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing.wav"

            with self.assertRaisesRegex(InputValidationError, "不存在"):
                validate_input(missing, missing.parent / "output")

    def test_validate_input_accepts_case_insensitive_extension(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "song.WaV"
            source.touch()

            validate_input(source, root / "output")

    def test_validate_input_requires_three_gigabytes_free(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "song.wav"
            source.touch()

            with patch(
                "scripts.harmonica.audio.shutil.disk_usage",
                return_value=SimpleNamespace(free=2 * 1024**3),
            ):
                with self.assertRaisesRegex(InputValidationError, "3 GB"):
                    validate_input(source, root / "output")


class SeparationTests(unittest.TestCase):
    def test_separate_audio_uses_safe_demucs_command_and_returns_stems(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "有 空格.mp3"
            source.touch()
            work_root = root / "separated"

            def create_stems(command, **_kwargs):
                stem_directory = work_root / "htdemucs" / source.stem
                stem_directory.mkdir(parents=True)
                (stem_directory / "vocals.wav").touch()
                (stem_directory / "no_vocals.wav").touch()

            with patch(
                "scripts.harmonica.audio.subprocess.run", side_effect=create_stems
            ) as run:
                vocals, accompaniment = separate_audio(source, work_root)

            self.assertEqual(vocals, work_root / "htdemucs" / source.stem / "vocals.wav")
            self.assertEqual(
                accompaniment, work_root / "htdemucs" / source.stem / "no_vocals.wav"
            )
            command = run.call_args.args[0]
            self.assertEqual(
                command,
                [
                    sys.executable,
                    "-m",
                    "demucs",
                    "-n",
                    "htdemucs",
                    "--two-stems=vocals",
                    "-o",
                    str(work_root),
                    str(source),
                ],
            )
            self.assertIsNot(run.call_args.kwargs.get("shell", False), True)
            self.assertTrue(run.call_args.kwargs["check"])

    def test_separate_audio_translates_demucs_exit_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "song.wav"
            source.touch()

            with patch(
                "scripts.harmonica.audio.subprocess.run",
                side_effect=subprocess.CalledProcessError(7, ["demucs"]),
            ):
                with self.assertRaisesRegex(InputValidationError, "Demucs.*7"):
                    separate_audio(source, source.parent / "work")

    def test_separate_audio_translates_demucs_launch_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "song.wav"
            source.touch()

            with patch(
                "scripts.harmonica.audio.subprocess.run",
                side_effect=FileNotFoundError("python missing"),
            ):
                with self.assertRaisesRegex(InputValidationError, "Demucs.*启动"):
                    separate_audio(source, source.parent / "work")


class AudioLoadingTests(unittest.TestCase):
    def test_load_mono_resamples_to_requested_rate_as_float32(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "stereo.wav"
            samples = np.column_stack(
                [
                    np.linspace(-0.2, 0.2, 1_600, dtype=np.float32),
                    np.linspace(0.2, -0.2, 1_600, dtype=np.float32),
                ]
            )
            sf.write(source, samples, 16_000)

            audio, sample_rate = load_mono(source, sample_rate=22_050)

            self.assertEqual(sample_rate, 22_050)
            self.assertEqual(audio.dtype, np.dtype(np.float32))
            self.assertTrue(audio.flags.c_contiguous)
            self.assertEqual(audio.ndim, 1)
            self.assertGreater(len(audio), 0)

    def test_load_mono_rejects_an_empty_wav(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "empty.wav"
            sf.write(source, np.empty(0, dtype=np.float32), 22_050)

            with self.assertRaisesRegex(InputValidationError, "为空"):
                load_mono(source)


if __name__ == "__main__":
    unittest.main()
