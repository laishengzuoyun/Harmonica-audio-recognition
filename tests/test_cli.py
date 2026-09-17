import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.harmonica.models import NoteEvent, TempoGrid


SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_harmonica_score.py"


def load_cli():
    spec = importlib.util.spec_from_file_location("extract_harmonica_score_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._load_runtime_dependencies()
    return module


def valid_notes():
    return [
        NoteEvent(index * 0.75, index * 0.75 + 0.70, 60 + index % 3, 0.9)
        for index in range(12)
    ]


class PipelineTests(unittest.TestCase):
    def test_pipeline_publishes_unicode_score_and_requested_stems(self):
        cli = load_cli()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "歌曲.mp3"
            source.write_bytes(b"audio")
            output = root / "output"
            stems = root / "prepared-stems"
            stems.mkdir()
            vocals = stems / "vocals.wav"
            accompaniment = stems / "no_vocals.wav"
            vocals.write_bytes(b"vocal")
            accompaniment.write_bytes(b"music")
            rendered = {}

            def fake_render(title, notes, grid, transpose, destination, sections, analysis):
                rendered.update(
                    title=title,
                    notes=notes,
                    grid=grid,
                    transpose=transpose,
                    sections=sections,
                    analysis=analysis,
                )
                (Path(destination) / "rendered.txt").write_text("ok", encoding="utf-8")

            frames = SimpleNamespace(pitch=np.array([60.0, np.nan, 60.0]))
            with (
                patch.object(cli, "separate_audio", return_value=(vocals, accompaniment)),
                patch.object(cli, "load_mono", return_value=(np.zeros(32), 22_050)),
                patch.object(cli, "transcribe_vocals", return_value=(valid_notes(), frames)),
                patch.object(cli, "detect_tempo_grid", return_value=TempoGrid(120, 0, "beats", 1)),
                patch.object(cli, "render_all", side_effect=fake_render),
            ):
                final = cli.run_pipeline(source, output, keep_stems=True)

            self.assertEqual(final, output / "歌曲")
            self.assertEqual((final / "rendered.txt").read_text(encoding="utf-8"), "ok")
            self.assertEqual((final / "stems" / "vocals.wav").read_bytes(), b"vocal")
            self.assertEqual((final / "stems" / "no_vocals.wav").read_bytes(), b"music")
            self.assertEqual(len(rendered["notes"]), 12)
            self.assertEqual(rendered["sections"], [0])
            self.assertEqual(rendered["analysis"]["filtered_frame_count"], 1)
            self.assertEqual(rendered["analysis"]["filtered_note_count"], 0)

    def test_existing_output_without_force_is_rejected_without_changes(self):
        cli = load_cli()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "song.mp3"
            source.write_bytes(b"audio")
            final = root / "output" / "song"
            final.mkdir(parents=True)
            marker = final / "keep.txt"
            marker.write_bytes(b"old")

            with self.assertRaisesRegex(cli.InputValidationError, "--force"):
                cli.run_pipeline(source, root / "output")

            self.assertEqual(marker.read_bytes(), b"old")

    def test_failure_preserves_existing_output_and_keeps_completed_stems_in_failure_folder(self):
        cli = load_cli()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / " lead.mp3"
            source.write_bytes(b"audio")
            output = root / "output"
            final = output / "lead"
            final.mkdir(parents=True)
            (final / "old.txt").write_bytes(b"unchanged")

            def completed_separation(_source, work_root):
                stem_dir = Path(work_root) / "htdemucs" / source.stem
                stem_dir.mkdir(parents=True)
                (stem_dir / "vocals.wav").write_bytes(b"vocal")
                (stem_dir / "no_vocals.wav").write_bytes(b"music")
                return stem_dir / "vocals.wav", stem_dir / "no_vocals.wav"

            with (
                patch.object(cli, "separate_audio", side_effect=completed_separation),
                patch.object(cli, "load_mono", side_effect=RuntimeError("external failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "external failed"):
                    cli.run_pipeline(source, output, force=True)

            self.assertEqual((final / "old.txt").read_bytes(), b"unchanged")
            failures = list(output.glob("lead-失败-*"))
            self.assertEqual(len(failures), 1)
            self.assertIn("external failed", (failures[0] / "错误日志.txt").read_text(encoding="utf-8"))
            self.assertEqual((failures[0] / "stems" / "vocals.wav").read_bytes(), b"vocal")
            self.assertEqual((failures[0] / "stems" / "no_vocals.wav").read_bytes(), b"music")

    def test_failure_keeps_vocal_created_before_separation_raises(self):
        cli = load_cli()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "partial.mp3"
            source.write_bytes(b"audio")
            output = root / "output"

            def partial_separation(_source, separation_root):
                stem_dir = Path(separation_root) / "htdemucs" / source.stem
                stem_dir.mkdir(parents=True)
                (stem_dir / "vocals.wav").write_bytes(b"partial vocal")
                raise RuntimeError("separation interrupted")

            with patch.object(cli, "separate_audio", side_effect=partial_separation):
                with self.assertRaisesRegex(RuntimeError, "separation interrupted"):
                    cli.run_pipeline(source, output)

            failures = list(output.glob("partial-失败-*"))
            self.assertEqual(len(failures), 1)
            self.assertEqual(
                (failures[0] / "stems" / "vocals.wav").read_bytes(), b"partial vocal"
            )
            self.assertFalse((failures[0] / "stems" / "no_vocals.wav").exists())

    def test_force_replaces_old_output_and_rolls_back_if_publish_rename_fails(self):
        cli = load_cli()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            final = root / "song"
            final.mkdir()
            (final / "old-only.txt").write_bytes(b"old")
            result = root / "result"
            result.mkdir()
            (result / "new.txt").write_bytes(b"new")
            cli._publish(result, final, force=True)
            self.assertFalse((final / "old-only.txt").exists())
            self.assertEqual((final / "new.txt").read_bytes(), b"new")

            result = root / "result-again"
            result.mkdir()
            (result / "half.txt").write_bytes(b"half")
            original_rename = Path.rename

            def fail_only_new(source, destination):
                if Path(source) == result and Path(destination) == final:
                    raise OSError("publish blocked")
                return original_rename(source, destination)

            with patch.object(Path, "rename", autospec=True, side_effect=fail_only_new):
                with self.assertRaisesRegex(OSError, "publish blocked"):
                    cli._publish(result, final, force=True)

            self.assertEqual((final / "new.txt").read_bytes(), b"new")
            self.assertFalse((final / "half.txt").exists())

    def test_publish_keeps_new_final_when_post_commit_backup_cleanup_fails(self):
        cli = load_cli()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            final = root / "song"
            final.mkdir()
            (final / "old.txt").write_bytes(b"old")
            result = root / "result"
            result.mkdir()
            (result / "new.txt").write_bytes(b"new")
            real_rmtree = cli.shutil.rmtree

            def fail_only_backup(path, *args, **kwargs):
                if Path(path).name.startswith(".song-备份"):
                    raise OSError("backup cleanup blocked")
                return real_rmtree(path, *args, **kwargs)

            with patch.object(cli.shutil, "rmtree", side_effect=fail_only_backup):
                self.assertEqual(cli._publish(result, final, force=True), final)

            self.assertEqual((final / "new.txt").read_bytes(), b"new")
            backups = list(root.glob(".song-备份*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / "old.txt").read_bytes(), b"old")

    @unittest.skipUnless(sys.platform == "win32", "Windows junction behavior only")
    def test_publish_rejects_existing_windows_junction_before_any_mutation(self):
        cli = load_cli()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            external = root / "external"
            external.mkdir()
            marker = external / "keep.txt"
            marker.write_bytes(b"external")
            final = root / "song"
            created = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(final), str(external)],
                capture_output=True,
                text=True,
            )
            if created.returncode != 0:
                self.skipTest("junction creation is unavailable on this Windows host")
            result = root / "result"
            result.mkdir()
            (result / "new.txt").write_bytes(b"new")

            with self.assertRaisesRegex(cli.InputValidationError, "链接|联接|junction"):
                cli._publish(result, final, force=True)

            self.assertEqual(marker.read_bytes(), b"external")
            self.assertEqual((result / "new.txt").read_bytes(), b"new")


class CommandLineTests(unittest.TestCase):
    def test_version_arguments_and_safe_titles(self):
        cli = load_cli()
        with self.assertRaisesRegex(cli.InputValidationError, "Python 3.12"):
            cli.validate_python_version((3, 11))
        cli.validate_python_version((3, 12))
        args = cli.parse_args(["input.mp3", "--output", "scores", "--keep-stems", "--force"])
        self.assertEqual(args.audio, Path("input.mp3"))
        self.assertEqual(args.output, Path("scores"))
        self.assertTrue(args.keep_stems)
        self.assertTrue(args.force)
        self.assertEqual(cli.safe_title(Path(' a<>:"|?*.mp3 ')), "a_______")
        self.assertEqual(cli.safe_title(Path("...")), "未命名歌曲")
        for reserved in ("NUL", "CON", "PRN", "AUX", "CLOCK$"):
            self.assertEqual(cli.safe_title(Path(f".{reserved}.mp3")), f"_{reserved}")
        for prefix in ("COM", "LPT"):
            for number in range(1, 10):
                self.assertEqual(
                    cli.safe_title(Path(f" {prefix.lower()}{number} .mp3")),
                    f"_{prefix}{number}",
                )
        self.assertEqual(cli.safe_title(Path("ordinary song.mp3")), "ordinary song")

    def test_dependency_free_script_bootstrap_fails_cleanly_without_site_packages(self):
        completed = subprocess.run(
            [sys.executable, "-S", str(SCRIPT), "song.mp3"],
            cwd=SCRIPT.parents[1],
            capture_output=True,
            text=True,
        )

        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 2)
        self.assertNotIn("Traceback", output)
        self.assertTrue("Python 3.12" in output or "依赖" in output)

    def test_main_returns_documented_codes_and_chinese_messages(self):
        cli = load_cli()
        stdout = io.StringIO()
        with (
            patch.object(cli.sys, "version_info", (3, 12, 0)),
            patch.object(cli, "run_pipeline", side_effect=cli.HarmonicaError("可预期错误")),
        ):
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(cli.main(["song.mp3"]), 2)
        self.assertIn("错误", stdout.getvalue())

        stdout = io.StringIO()
        with (
            patch.object(cli.sys, "version_info", (3, 12, 0)),
            patch.object(cli, "run_pipeline", side_effect=RuntimeError("boom")),
        ):
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(cli.main(["song.mp3"]), 1)
        self.assertIn("失败", stdout.getvalue())

        stdout = io.StringIO()
        with (
            patch.object(cli.sys, "version_info", (3, 12, 0)),
            patch.object(cli, "run_pipeline", return_value=Path("out/song")),
        ):
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(cli.main(["song.mp3"]), 0)
        self.assertIn("完成", stdout.getvalue())

    def test_main_rejects_wrong_python_before_starting_pipeline(self):
        cli = load_cli()
        stdout = io.StringIO()
        with (
            patch.object(cli.sys, "version_info", (3, 11, 0)),
            patch.object(cli, "run_pipeline") as pipeline,
            contextlib.redirect_stdout(stdout),
        ):
            self.assertEqual(cli.main(["song.mp3"]), 2)

        pipeline.assert_not_called()
        self.assertIn("Python 3.12", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
