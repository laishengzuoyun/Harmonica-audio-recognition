"""Command-line orchestration for producing a publishable harmonica score."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from harmonica.exceptions import HarmonicaError, InputValidationError


INVALID_TITLE_CHARACTERS = '<>:"/\\|?*'
WINDOWS_SAFE_PATH_LIMIT = 240
PATH_TOKEN_BUDGET = 12
TITLE_HASH_LENGTH = 10
RESULT_FILE_SUFFIXES = (
    "-连续按键谱.md",
    "-详细节奏谱.md",
    "-音符明细.csv",
    "-主旋律.mid",
    "-口琴试听.wav",
    "-分析报告.json",
)
WINDOWS_RESERVED_TITLES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CLOCK$",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}

_RUNTIME_LOADED = False
np = None
load_mono = None
separate_audio = None
validate_input = None
choose_transpose = None
render_all = None
detect_tempo_grid = None
quantize_notes = None
section_starts = None
transcribe_vocals = None


def _load_runtime_dependencies() -> None:
    """Load optional audio dependencies only when score extraction needs them."""
    global _RUNTIME_LOADED
    global np, load_mono, separate_audio, validate_input, choose_transpose
    global render_all, detect_tempo_grid, quantize_notes, section_starts, transcribe_vocals
    if _RUNTIME_LOADED:
        return
    try:
        import numpy as runtime_numpy

        from harmonica.audio import load_mono as runtime_load_mono
        from harmonica.audio import separate_audio as runtime_separate_audio
        from harmonica.audio import validate_input as runtime_validate_input
        from harmonica.instrument import choose_transpose as runtime_choose_transpose
        from harmonica.render import render_all as runtime_render_all
        from harmonica.rhythm import detect_tempo_grid as runtime_detect_tempo_grid
        from harmonica.rhythm import quantize_notes as runtime_quantize_notes
        from harmonica.rhythm import section_starts as runtime_section_starts
        from harmonica.transcription import transcribe_vocals as runtime_transcribe_vocals
    except (ImportError, OSError) as error:
        raise InputValidationError(
            "缺少音频处理依赖或依赖无法加载，请安装 requirements-audio.txt 中的依赖。"
        ) from error

    np = runtime_numpy
    load_mono = runtime_load_mono
    separate_audio = runtime_separate_audio
    validate_input = runtime_validate_input
    choose_transpose = runtime_choose_transpose
    render_all = runtime_render_all
    detect_tempo_grid = runtime_detect_tempo_grid
    quantize_notes = runtime_quantize_notes
    section_starts = runtime_section_starts
    transcribe_vocals = runtime_transcribe_vocals
    _RUNTIME_LOADED = True


def safe_title(path: Path) -> str:
    """Return the complete, Windows-safe display title from an audio filename."""
    title = Path(path).stem
    for character in INVALID_TITLE_CHARACTERS:
        title = title.replace(character, "_")
    title = title.strip(" .")
    if not title:
        return "未命名歌曲"
    if title.upper() in WINDOWS_RESERVED_TITLES:
        return f"_{title.upper()}"
    return title


def _windows_path_length(path: Path) -> int:
    """Count UTF-16 code units, matching how Windows measures native paths."""
    absolute = str(Path(path).resolve(strict=False))
    return len(absolute.encode("utf-16-le")) // 2


def _title_paths(output_root: Path, title: str) -> list[Path]:
    """Return the longest path shapes the CLI may create for *title*."""
    output_root = Path(output_root).resolve(strict=False)
    token = "x" * PATH_TOKEN_BUDGET
    final = output_root / title
    staging = output_root / f".处理中-{token}" / "result"
    backup = output_root / f".{title}-备份-{token}"
    failure = output_root / f"{title}-失败-20260918-235959-{token}"
    paths = [final, staging, backup, failure]
    for suffix in RESULT_FILE_SUFFIXES:
        filename = f"{title}{suffix}"
        paths.extend((final / filename, staging / filename, backup / filename))
    paths.extend(
        (
            failure / "错误日志.txt",
            failure / "stems" / "vocals.wav",
            failure / "stems" / "no_vocals.wav",
        )
    )
    return paths


def _title_fits(output_root: Path, title: str) -> bool:
    return all(
        _windows_path_length(path) <= WINDOWS_SAFE_PATH_LIMIT
        for path in _title_paths(output_root, title)
    )


def _filesystem_title(display_title: str, output_root: Path) -> str:
    """Fit a stable score slug to every path shape used by the pipeline."""
    if _title_fits(output_root, display_title):
        return display_title

    digest = hashlib.sha256(display_title.encode("utf-8")).hexdigest()[:TITLE_HASH_LENGTH]
    suffix = f"-{digest}"
    for prefix_length in range(len(display_title) - 1, 0, -1):
        prefix = display_title[:prefix_length].rstrip(" .")
        if not prefix:
            continue
        candidate = f"{prefix}{suffix}"
        if _title_fits(output_root, candidate):
            return candidate
    raise InputValidationError(
        f"输出路径过长，无法为歌曲名保留安全的文件名预算：{output_root}"
    )


def _validate_separation_path(source: Path, output_root: Path) -> None:
    """Reject a Demucs stem path that cannot fit before starting Demucs."""
    token = "x" * PATH_TOKEN_BUDGET
    stem = (
        Path(output_root).resolve(strict=False)
        / f".处理中-{token}"
        / "separation"
        / "htdemucs"
        / Path(source).stem
        / "no_vocals.wav"
    )
    if _windows_path_length(stem) > WINDOWS_SAFE_PATH_LIMIT:
        raise InputValidationError(
            f"输出路径过长，Demucs 分轨文件将超出 Windows 安全限制：{output_root}"
        )


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


def _is_link_like(path: Path) -> bool:
    """Identify symbolic links and Windows directory junctions without following them."""
    path = Path(path)
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


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
    if _windows_path_length(candidate) > WINDOWS_SAFE_PATH_LIMIT:
        raise InputValidationError(f"输出路径过长：{candidate}")
    return candidate


def _publish(result: Path, final: Path, force: bool) -> Path:
    """Atomically publish a completed sibling directory, restoring old output on error."""
    result = Path(result)
    final = Path(final)
    parent = final.parent
    if _is_link_like(result):
        raise InputValidationError(f"拒绝发布符号链接或目录联接结果：{result}")
    if not result.is_dir():
        raise InputValidationError(f"待发布结果目录不存在：{result}")
    if final.exists() and _is_link_like(final):
        raise InputValidationError(f"拒绝覆盖符号链接或目录联接输出：{final}")
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
        failure = Path(
            tempfile.mkdtemp(prefix=f"{title}-失败-{stamp}-", dir=output_root)
        )
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
    _load_runtime_dependencies()
    source = Path(audio)
    output_root = Path(output)
    display_title = safe_title(source)
    validate_input(source, output_root)
    title = _filesystem_title(display_title, output_root)
    _validate_separation_path(source, output_root)
    final = output_root / title
    if final.exists() and not force:
        raise InputValidationError(f"输出目录已存在：{final}；如需覆盖请使用 --force。")

    work_root: Path | None = None
    stems: tuple[Path, Path] | None = None
    try:
        work_root = Path(
            tempfile.mkdtemp(prefix=".处理中-", dir=output_root)
        )
        result = work_root / "result"
        result.mkdir()

        print("1/6 分离人声与伴奏…")
        separation_root = work_root / "separation"
        expected_stem_dir = separation_root / "htdemucs" / source.stem
        stems = (
            expected_stem_dir / "vocals.wav",
            expected_stem_dir / "no_vocals.wav",
        )
        vocals, accompaniment = separate_audio(source, separation_root)
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
        render_all(
            display_title,
            quantized,
            grid,
            transpose,
            result,
            sections,
            analysis,
            filesystem_title=title,
        )
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
        _load_runtime_dependencies()
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
