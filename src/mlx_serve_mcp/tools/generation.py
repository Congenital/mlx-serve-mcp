"""Generation tools: proxy the remote mlx-serve media endpoints.

Each tool calls the matching :class:`~mlx_serve_mcp.client.MlxServeClient`
method, writes the artifact under ``config.output_dir`` (so it survives the
session), and returns both an inline MCP content block (so clients can render
it) and a text summary carrying the saved path and metadata.

Return contract:
* image  -> ``[TextContent, ImageContent]``
* audio  -> ``[TextContent, AudioContent]`` (text-only on SDKs without AudioContent)
* video  -> ``[TextContent]`` with the MP4 path (no standard MCP video block)
* 3d     -> ``[TextContent]`` with the GLB path
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types

try:  # AudioContent is the newest content block; fall back gracefully.
    from mcp.types import AudioContent
except ImportError:  # pragma: no cover - older SDK
    AudioContent = None  # type: ignore

# Imported at runtime (not just for type-checking): generate_video muxes the
# decoded frames through ffmpeg. The video module has no SDK dependency.
from .. import video as video_mod

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

    from ..client import MlxServeClient
    from ..config import Config
    from ..state import State
    from . import Deps


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _text(text: str) -> mcp_types.TextContent:
    return mcp_types.TextContent(type="text", text=text)


def _save(config: "Config", subdir: str, data: bytes, ext: str) -> Path:
    out = Path(config.output_dir) / subdir / f"{_stamp()}{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


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
        pngs = await client.generate_image(prompt=prompt, model=model, size=size, n=1, seed=seed)
        path = _save(config, "images", pngs[0], ".png")
        summary = (
            f"Generated image ({model}, {size}) -> {path}\n"
            f"prompt: {prompt}" + (f"\nseed: {seed}" if seed is not None else "")
        )
        blocks: list[Any] = [_text(summary)]
        blocks.append(
            mcp_types.ImageContent(type="image", data=base64.b64encode(pngs[0]).decode(), mimeType="image/png")
        )
        return blocks

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
        img_b64 = base64.b64encode(src.read_bytes()).decode()
        png = await client.edit_image(
            image_b64=img_b64, prompt=prompt, model=model, size=size, strength=strength, seed=seed
        )
        path = _save(config, "images", png, ".png")
        summary = f"Edited image ({model}, strength={strength}) -> {path}\nfrom: {src}\nprompt: {prompt}"
        return [_text(summary), mcp_types.ImageContent(type="image", data=base64.b64encode(png).decode(), mimeType="image/png")]

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
        wav = await client.generate_speech(text=text, model=model, voice=voice, speed=float(speed) if speed else None)
        path = _save(config, "audio", wav, ".wav")
        summary = f"Spoke {len(text)} chars ({model}) -> {path}"
        blocks: list[Any] = [_text(summary)]
        if AudioContent is not None:
            blocks.append(AudioContent(type="audio", data=base64.b64encode(wav).decode(), mimeType="audio/wav"))
        else:  # pragma: no cover - older SDK without AudioContent
            blocks[0] = _text(summary + "\n(inline audio unavailable on this SDK version)")
        return blocks

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
        wav = await client.generate_music(
            prompt=prompt, model=model, lyrics=lyrics, duration_seconds=duration_seconds,
            bpm=bpm, keyscale=keyscale, time_signature=time_signature, vocal_language=vocal_language,
        )
        path = _save(config, "audio", wav, ".wav")
        summary = f"Generated music ({model}) -> {path}\nprompt: {prompt}"
        blocks: list[Any] = [_text(summary)]
        if AudioContent is not None:
            blocks.append(AudioContent(type="audio", data=base64.b64encode(wav).decode(), mimeType="audio/wav"))
        else:  # pragma: no cover
            blocks[0] = _text(summary + "\n(inline audio unavailable on this SDK version)")
        return blocks

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
        result = await client.generate_video(prompt=prompt, model=model, size=size, seconds=seconds)
        out_path = Path(config.output_dir) / "video" / f"{_stamp()}.mp4"
        try:
            import asyncio

            await asyncio.to_thread(
                video_mod.mux_video_mp4,
                result.rgb_bytes, result.width, result.height, result.fps,
                out_path, result.audio_pcm_s16le, result.audio_sample_rate, result.audio_channels,
            )
        except Exception as exc:  # muxing failed; still report the failure clearly
            return [_text(f"error: video generated but ffmpeg mux failed: {exc}")]
        summary = (
            f"Generated video ({model}) -> {out_path}\n"
            f"{result.width}x{result.height} @ {result.fps}fps, {result.frames} frames"
            + (", with audio" if result.audio_pcm_s16le else ", silent")
            + f"\nprompt: {prompt}"
        )
        return [_text(summary)]

    # ── 3d ─────────────────────────────────────────────────────────────────

    @mcp.tool()
    async def generate_3d(
        prompt: str,
        model: str | None = None,
    ) -> list[Any]:
        """Generate a 3D mesh (GLB) from a text prompt.

        Args:
            prompt: What the object should be.
            model: 3D model id. Defaults to the configured mesh model.
        """
        model = model or config.mesh_model
        glb = await client.generate_mesh(prompt=prompt, model=model)
        path = _save(config, "mesh", glb, ".glb")
        summary = f"Generated 3D mesh ({model}) -> {path} ({len(glb)} bytes)\nprompt: {prompt}"
        return [_text(summary)]