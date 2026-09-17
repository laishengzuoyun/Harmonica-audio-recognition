"""Audio input validation, source separation, and loading helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np

from .models import InputValidationError


SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac"}
MIN_FREE_BYTES = 3 * 1024**3


def validate_input(source: Path, output_root: Path) -> None:
    """Validate a source audio file and ensure enough output disk space."""
    source = Path(source)
    output_root = Path(output_root)

    if not source.is_file():
        raise InputValidationError(f"输入音频文件不存在或不是普通文件：{source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise InputValidationError("仅支持 MP3、WAV 或 FLAC 格式的音频文件。")

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(output_root).free
    except OSError as exc:
        raise InputValidationError(f"无法检查输出目录或磁盘空间：{output_root}（{exc}）") from exc

    if free_bytes < MIN_FREE_BYTES:
        raise InputValidationError("输出磁盘可用空间不足，需要至少 3 GB。")


def separate_audio(source: Path, work_root: Path) -> tuple[Path, Path]:
    """Run Demucs vocal separation and return vocal and accompaniment stems."""
    source = Path(source)
    work_root = Path(work_root)
    command = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        "htdemucs",
        "--two-stems=vocals",
        "--shifts=0",
        "-o",
        str(work_root),
        str(source),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise InputValidationError(f"Demucs 分离失败，退出代码：{exc.returncode}。") from exc
    except OSError as exc:
        raise InputValidationError(f"Demucs 无法启动，请检查依赖安装：{exc}。") from exc

    stem_directory = work_root / "htdemucs" / source.stem
    vocals = stem_directory / "vocals.wav"
    accompaniment = stem_directory / "no_vocals.wav"
    if not vocals.is_file() or not accompaniment.is_file():
        raise InputValidationError("Demucs 未生成完整的 vocals.wav 和 no_vocals.wav 输出。")
    return vocals, accompaniment


def load_mono(path: Path, sample_rate: int = 22_050) -> tuple[np.ndarray, int]:
    """Load audio as a non-empty, contiguous mono float32 waveform."""
    path = Path(path)
    try:
        audio, actual_rate = librosa.load(path, sr=sample_rate, mono=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise InputValidationError(f"无法解码音频文件：{path}（{exc}）") from exc

    audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
    if audio.size == 0:
        raise InputValidationError(f"音频文件为空：{path}")
    return audio, actual_rate
