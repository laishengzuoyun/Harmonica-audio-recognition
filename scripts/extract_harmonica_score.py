"""Command-line orchestration for producing a publishable harmonica score."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from harmonica.audio import load_mono, separate_audio, validate_input
from harmonica.instrument import choose_transpose
from harmonica.models import HarmonicaError, InputValidationError
from harmonica.render import render_all
from harmonica.rhythm import detect_tempo_grid, quantize_notes, section_starts
from harmonica.transcription import transcribe_vocals


INVALID_TITLE_CHARACTERS = '<>:"/\\|?*'


def safe_title(path: Path) -> str:
    """Return a deterministic Windows-safe score title from an audio filename."""
    title = Path(path).stem
    for character in INVALID_TITLE_CHARACTERS:
        title = title.replace(character, "_")
    title = title.strip(" .")
    return title or "未命名歌曲"


def validate_python_version(version: tuple[int, int] | None = None) -> None:
    """Require the interpreter version supported by the audio dependencies."""
    current = version if version is not None else sys.version_info[:2]
    if tuple(current) != (3, 12):
        raise InputValidationError("本工具需要 Python 3.12，请使用 Python 3.12 运行。")


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _remove_owned_path(path: Path, parent: Path) -> None:
    """Remove only a resolved child of *parent*."""
    if not _is_inside(path, parent):
        raise InputValidationError(f"拒绝删除输出目录之外的路径：{path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _unused_sibling(parent: Path, prefix: str) -> Path:
    """Create a collision-free, not-yet-created sibling path."""
    candidate = parent / prefix
    index = 1
    while candidate.exists():
        candidate = parent / f"{prefix}-{index}"
        index += 1
    if not _is_inside(candidate, parent):
        raise InputValidationError(f"输出路径不安全：{candidate}")
    return candidate


def _publish(result: Path, final: Path, force: bool) -> Path:
    """Atomically publish a completed sibling directory, restoring old output on error."""
    result = Path(result)
    final = Path(final)
    parent = final.parent
    if not result.is_dir():
        raise InputValidationError(f"待发布结果目录不存在：{result}")
    if final.exists() and not force:
        raise InputValidationError(f"输出目录已存在：{final}；如需覆盖请使用 --force。")

    backup: Path | None = None
    if final.exists():
        backup = _unused_sibling(parent, f".{final.name}-备份")
        final.rename(backup)
    try:
        result.rename(final)
    except Exception:
        if backup is not None and backup.exists() and not final.exists():
            backup.rename(final)
        raise

    if backup is not None and backup.exists():
        try:
            _remove_owned_path(backup, parent)
        except OSError as error:
            print(f"警告：新结果已发布，但旧备份暂未清理：{error}")
    return final


def _save_failure(
    output_root: Path,
    title: str,
    stems: tuple[Path, Path] | None,
    error: BaseException,
) -> None:
    """Best-effort failure diagnostics that never replace the original exception."""
    try:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        failure = _unused_sibling(output_root, f"{title}-失败-{stamp}")
        failure.mkdir()
        (failure / "错误日志.txt").write_text(
            f"口琴谱提取失败。\n\n{type(error).__name__}: {error}\n",
            encoding="utf-8",
        )
        if stems is None:
            return
        destination = failure / "stems"
        copied = False
        for source, filename in zip(stems, ("vocals.wav", "no_vocals.wav")):
            source = Path(source)
            if source.is_file():
                destination.mkdir(exist_ok=True)
                shutil.copy2(source, destination / filename)
                copied = True
        if not copied and destination.exists():
            _remove_owned_path(destination, failure)
    except Exception:
        pass


def _pitch_diagnostics(frames: object, note_count: int, grid_source: str) -> dict[str, object]:
    """Summarize filtering without assuming every pitch frame is finite."""
    pitch = np.asarray(getattr(frames, "pitch", []), dtype=float).reshape(-1)
    finite = np.isfinite(pitch)
    starts = 0
    previous: int | None = None
    for value in pitch:
        current = int(np.rint(value)) if np.isfinite(value) else None
        if current is not None and current != previous:
            starts += 1
        previous = current
    warnings: list[str] = []
    if grid_source == "intervals":
        warnings.append("伴奏节拍不稳定，已使用旋律音符间隔估计速度。")
    return {
        "filtered_frame_count": int(len(pitch) - int(np.count_nonzero(finite))),
        "filtered_note_count": max(0, starts - note_count),
        "warnings": warnings,
    }


def run_pipeline(
    audio: Path, output: Path, keep_stems: bool = False, force: bool = False
) -> Path:
    """Extract, render, and transactionally publish one audio score."""
    source = Path(audio)
    output_root = Path(output)
    title = safe_title(source)
    validate_input(source, output_root)
    final = output_root / title
    if final.exists() and not force:
        raise InputValidationError(f"输出目录已存在：{final}；如需覆盖请使用 --force。")

    work_root: Path | None = None
    stems: tuple[Path, Path] | None = None
    try:
        work_root = Path(
            tempfile.mkdtemp(prefix=f".{title}-处理中-", dir=output_root)
        )
        result = work_root / "result"
        result.mkdir()

        print("1/6 分离人声与伴奏…")
        vocals, accompaniment = separate_audio(source, work_root)
        stems = (vocals, accompaniment)
        print("2/6 读取音频…")
        vocal_audio, vocal_rate = load_mono(vocals)
        accompaniment_audio, accompaniment_rate = load_mono(accompaniment)
        print("3/6 提取主旋律…")
        notes, frames = transcribe_vocals(vocal_audio, vocal_rate)
        print("4/6 估计节奏速度…")
        grid = detect_tempo_grid(accompaniment_audio, accompaniment_rate, notes)
        print("5/6 移调并量化节奏…")
        transpose = choose_transpose(notes)
        quantized = quantize_notes(notes, grid, transpose)
        sections = section_starts(quantized)
        analysis = _pitch_diagnostics(frames, len(notes), grid.source)
        print("6/6 生成乐谱文件…")
        render_all(title, quantized, grid, transpose, result, sections, analysis)
        if keep_stems:
            stem_output = result / "stems"
            stem_output.mkdir()
            shutil.copy2(vocals, stem_output / "vocals.wav")
            shutil.copy2(accompaniment, stem_output / "no_vocals.wav")
        published = _publish(result, final, force)
        return published
    except Exception as error:
        _save_failure(output_root, title, stems, error)
        raise
    finally:
        if work_root is not None and work_root.exists() and _is_inside(work_root, output_root):
            shutil.rmtree(work_root, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从音频提取三角洲行动口琴谱")
    parser.add_argument("audio", type=Path, help="输入 MP3、WAV 或 FLAC 音频")
    parser.add_argument("--output", type=Path, default=Path("output"), help="输出根目录")
    parser.add_argument("--keep-stems", action="store_true", help="保留人声和伴奏分轨")
    parser.add_argument("--force", action="store_true", help="成功后替换同名输出")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_python_version(sys.version_info[:2])
        final = run_pipeline(args.audio, args.output, args.keep_stems, args.force)
    except HarmonicaError as error:
        print(f"错误：{error}")
        return 2
    except Exception as error:
        print(f"处理失败：{error}")
        return 1
    print(f"完成：乐谱已保存到 {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
