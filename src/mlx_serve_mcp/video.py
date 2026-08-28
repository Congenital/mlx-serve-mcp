"""Assemble mlx-serve video output into a playable MP4.

mlx-serve's ``/v1/video/generations`` answers with *raw* RGB8 frame bytes and
an optional PCM s16le track — not an encoded file. This module muxes both into
an H.264/AAC MP4 via ffmpeg.

ffmpeg resolution order:
1. system ``ffmpeg`` on ``PATH`` (if present);
2. the static ffmpeg binary shipped with the ``imageio-ffmpeg`` wheel, which is
   a hard dependency of this package — so no separate system install is needed.

These helpers are synchronous (ffmpeg is a blocking subprocess); callers on the
event loop should run them via :func:`asyncio.to_thread`.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path


def get_ffmpeg() -> str:
    """Resolve an ffmpeg executable: system binary first, then imageio-ffmpeg."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError(
            "no ffmpeg available: install ffmpeg on PATH or the imageio-ffmpeg wheel"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_wav(pcm_s16le: bytes, sample_rate: int, channels: int, path: Path) -> Path:
    """Wrap raw little-endian s16 PCM in a RIFF WAV container."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)  # 16-bit
        handle.setframerate(sample_rate)
        handle.writeframes(pcm_s16le)
    return path


def mux_video_mp4(
    rgb_bytes: bytes,
    width: int,
    height: int,
    fps: int,
    out_path: Path,
    audio_pcm_s16le: bytes | None = None,
    audio_sample_rate: int | None = None,
    audio_channels: int | None = None,
) -> Path:
    """Encode raw RGB8 frames (+ optional PCM track) to an H.264 MP4.

    Frames are piped over stdin as ``rawvideo``; when audio rides along it is
    wrapped in a temp WAV (second input) and muxed as AAC. The output file is
    returned on success; a :class:`RuntimeError` is raised if ffmpeg fails or
    produces no output.
    """
    if fps <= 0:
        fps = 24
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0",
    ]

    tmp_wav: Path | None = None
    has_audio = bool(audio_pcm_s16le) and audio_sample_rate and audio_channels
    if has_audio:
        tmp_wav = out_path.with_suffix(".tmp.wav")
        write_wav(audio_pcm_s16le, audio_sample_rate, audio_channels, tmp_wav)
        cmd += ["-i", str(tmp_wav)]

    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += [str(out_path)]

    try:
        proc = subprocess.run(cmd, input=rgb_bytes, capture_output=True)
        if proc.returncode != 0:
            detail = (proc.stderr or b"").decode("utf-8", "replace")
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {detail[:1000]}")
    finally:
        if tmp_wav is not None:
            tmp_wav.unlink(missing_ok=True)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced no output file")
    return out_path