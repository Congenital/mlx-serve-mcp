"""Assemble mlx-serve video output into a playable MP4.

mlx-serve's ``/v1/video/generations`` answers with *raw* RGB8 frame bytes and
an optional PCM s16le track — not an encoded file. This module muxes both into
H.264/AAC MP4 via ffmpeg.

ffmpeg resolution order:
1. system ``ffmpeg`` on PATH (if present);
2. the static ffmpeg binary shipped with the ``imageio-ffmpeg`` wheel, which
   is a hard dependency of this package — so no separate system install is
   ever required.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

_FFMPEG_PATH: str | None = None


def get_ffmpeg() -> str:
    """Locate an ffmpeg executable (cached after first call)."""
    global _FFMPEG_PATH
    if _FFMPEG_PATH:
        return _FFMPEG_PATH
    from_path = shutil.which("ffmpeg")
    if from_path:
        _FFMPEG_PATH = from_path
        return _FFMPEG_PATH
    try:
        import imageio_ffmpeg

        _FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - imageio-ffmpeg is a hard dep
        raise RuntimeError(
            "no ffmpeg available: not on PATH and imageio-ffmpeg failed to "
            f"provide its bundled binary ({exc})"
        ) from exc
    return _FFMPEG_PATH


def write_wav(
    pcm_s16le: bytes,
    sample_rate: int,
    channels: int,
    out_path: Path,
) -> Path:
    """Wrap raw PCM s16le bytes into a WAV file (used for audio-only output)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(max(1, channels))
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_s16le)
    return out_path


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
    wrapped in a temp WAV (second input) and muxed as AAC.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg()

    cmd: list[str] = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",  # frames on stdin
    ]

    tmp_wav: Path | None = None
    try:
        if audio_pcm_s16le and audio_sample_rate and audio_channels:
            tmp_wav = Path(tempfile.mkstemp(suffix=".wav")[1])
            write_wav(audio_pcm_s16le, audio_sample_rate, audio_channels, tmp_wav)
            cmd += [
                "-f", "wav",
                "-i", str(tmp_wav),
            ]
        cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        if tmp_wav is not None:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += [str(out_path)]

        proc = subprocess.run(
            cmd,
            input=rgb_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {detail[:1000]}")
    finally:
        if tmp_wav is not None:
            tmp_wav.unlink(missing_ok=True)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced no output file")
    return out_path
