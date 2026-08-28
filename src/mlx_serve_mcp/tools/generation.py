"""Generation tools: proxy the remote mlx-serve media endpoints.

Each tool calls the matching :class:`~mlx_serve_mcp.client.MlxServeClient`
method, which streams the artifact straight to a file under
``config.output_dir`` and returns the saved path; the tools therefore only
ever return a short text summary (no inline base64 payloads).

Return contract:
* image  -> ``[TextContent]`` with the PNG path
* audio  -> ``[TextContent]`` with the WAV path
* video  -> ``[TextContent]`` with the MP4 path
* 3d     -> ``[TextContent]`` with the GLB path
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

    from ..client import MlxServeClient
    from ..config import Config
    from ..state import State
    from . import Deps


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _out(config: "Config", subdir: str, ext: str) -> Path:
    """Target path under ``config.output_dir`` for a fresh artifact."""
    return Path(config.output_dir) / subdir / f"{_stamp()}{ext}"


def _text(text: str) -> mcp_types.TextContent:
    return mcp_types.TextContent(type="text", text=text)


def register(mcp: "FastMCP", deps: Deps) -> None:
    config: Config = deps.config
    client: MlxServeClient = deps.client
    state: State = deps.state

    # ── image ──────────────────────────────────────────────────────────────

    @mcp.tool()
    async def generate_image(
        prompt: str,
        model: str | None = None,
        size: str = "1024x1024",
        seed: int | None = None,
    ) -> list[Any]:
        """Generate an image from a text prompt via the remote mlx-serve instance.

        Args:
            prompt: What the image should depict.
            model: Model id. Defaults to the configured image model
                (a text-turbo model; use a general model for photo-real work).
            size: Output dimensions, e.g. "1024x1024", "1344x768", "768x1344".
            seed: Optional random seed for reproducibility.
        """
        model = model or config.image_model
        path = await client.generate_image(
            prompt=prompt, model=model, size=size, n=1, seed=seed,
            path=_out(config, "images", ".png"),
        )
        summary = (
            f"Generated image ({model}, {size}) -> {path}\n"
            f"prompt: {prompt}" + (f"\nseed: {seed}" if seed is not None else "")
        )
        return [_text(summary)]

    @mcp.tool()
    async def edit_image(
        image_path: str,
        prompt: str,
        model: str | None = None,
        size: str = "1024x1024",
        strength: float = 0.6,
        seed: int | None = None,
    ) -> list[Any]:
        """Edit an existing image with a text prompt (image-to-image).

        Args:
            image_path: Absolute or ~-relative path to the source image (PNG/JPEG).
            prompt: Description of the desired edit.
            model: Model id. Defaults to the configured image-edit model.
            size: Output dimensions.
            strength: Denoising strength 0..1 (how much to change the image).
            seed: Optional random seed.
        """
        model = model or config.image_edit_model
        src = state.resolve(image_path)
        if not src.exists():
            return [_text(f"error: source image not found: {src}")]
        path = await client.edit_image(
            image_path=src, prompt=prompt, model=model, size=size, strength=strength,
            seed=seed, path=_out(config, "images", ".png"),
        )
        summary = f"Edited image ({model}, strength={strength}) -> {path}\nfrom: {src}\nprompt: {prompt}"
        return [_text(summary)]

    # ── audio ──────────────────────────────────────────────────────────────

    @mcp.tool()
    async def generate_speech(
        text: str,
        model: str | None = None,
        voice: str | None = None,
        speed: str | None = None,
    ) -> list[Any]:
        """Speak a line of text aloud (text-to-speech) via the remote instance.

        Args:
            text: Exactly the words to speak.
            model: TTS model id. Defaults to the configured TTS model.
            voice: Optional voice name.
            speed: Optional speaking-rate hint (e.g. "0.5" slow .. "2.0" fast).
        """
        model = model or config.tts_model
        path = await client.generate_speech(
            text=text, model=model, path=_out(config, "audio", ".wav"),
            voice=voice, speed=float(speed) if speed else None,
        )
        summary = f"Spoke {len(text)} chars ({model}) -> {path}"
        return [_text(summary)]

    @mcp.tool()
    async def generate_music(
        prompt: str,
        model: str | None = None,
        lyrics: str | None = None,
        duration_seconds: int | None = None,
        bpm: int | None = None,
        keyscale: str | None = None,
        time_signature: str | None = None,
        vocal_language: str | None = None,
    ) -> list[Any]:
        """Compose a piece of music from a style description.

        Args:
            prompt: Style: genre, mood, instrumentation, tempo feel. Not lyrics.
            model: Music model id. Defaults to the configured music model.
            lyrics: Optional words to sing (omit for an instrumental).
            duration_seconds: Optional target length in seconds.
            bpm: Optional tempo in beats per minute.
            keyscale: Optional key, e.g. "A minor", "C major".
            time_signature: Optional meter, e.g. "4/4", "3/4".
            vocal_language: Optional language for sung lyrics.
        """
        model = model or config.music_model
        path = await client.generate_music(
            prompt=prompt, model=model, path=_out(config, "audio", ".wav"),
            lyrics=lyrics, duration_seconds=duration_seconds,
            bpm=bpm, keyscale=keyscale, time_signature=time_signature, vocal_language=vocal_language,
        )
        summary = f"Generated music ({model}) -> {path}\nprompt: {prompt}"
        return [_text(summary)]

    # ── video ──────────────────────────────────────────────────────────────

    @mcp.tool()
    async def generate_video(
        prompt: str,
        model: str | None = None,
        size: str | None = None,
        seconds: int | None = None,
    ) -> list[Any]:
        """Generate a short video clip from a text prompt.

        The remote instance returns raw frames (+ optional audio) which are muxed
        into an H.264/AAC MP4 via ffmpeg and written under the output dir.

        Args:
            prompt: What the clip should show — subject, motion, camera, lighting.
            model: Video model id. Defaults to the configured video model.
            size: Optional "WIDTHxHEIGHT" or a ratio like "16:9".
            seconds: Optional clip length in seconds (keep short; it is slow).
        """
        model = model or config.video_model
        out_path = _out(config, "video", ".mp4")
        path = await client.generate_video(
            prompt=prompt, model=model, size=size, seconds=seconds, path=out_path,
        )
        summary = f"Generated video ({model}) -> {path}\nprompt: {prompt}"
        return [_text(summary)]

    # ── 3d ─────────────────────────────────────────────────────────────────

    @mcp.tool()
    async def generate_3d(
        prompt: str,
        model: str | None = None,
        image_path: str | None = None,
    ) -> list[Any]:
        """Generate a 3D mesh (GLB) from an image of the subject.

        Args:
            prompt: Text description of the object (used as the image
                description; the default trellis mesh model is
                image-conditioned and needs a real source image).
            model: 3D model id. Defaults to the configured mesh model.
            image_path: Path to a source image of the object to convert
                into a 3D mesh. Required for the default trellis mesh
                model, which is image-to-3D only; the file is base64'd and
                sent as the required ``image`` field.
        """
        model = model or config.mesh_model
        if not image_path:
            return [_text(
                "error: generate_3d needs an 'image_path' (path to a PNG/JPEG of the "
                "object); the default trellis mesh model is image-to-3D only and "
                "rejects text-only requests."
            )]
        src = Path(image_path).expanduser()
        if not src.is_file():
            return [_text(f"error: source image not found: {src}")]
        path = await client.generate_mesh(
            prompt=prompt, model=model, path=_out(config, "mesh", ".glb"),
            image=base64.b64encode(src.read_bytes()).decode(),
        )
        summary = f"Generated 3D mesh ({model}) -> {path}\nfrom: {src}\nprompt: {prompt}"
        return [_text(summary)]