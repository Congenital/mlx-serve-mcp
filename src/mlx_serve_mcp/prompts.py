"""MCP prompts: ready-made recipes for the mlx-serve generation tools.

Exposed through the standard MCP ``prompts/list`` / ``prompts/get`` methods so
clients (Claude Desktop, LobeHub, ...) can render them as one-click prompt
templates. Each names the exact tool to call, the recommended model, and the
argument values that work well in practice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


def register(mcp: "FastMCP") -> None:
    @mcp.prompt()
    def portrait(subject: str, scene: str = "a neutral studio", lighting: str = "") -> str:
        """Create a photorealistic portrait with generate_image.

        Args:
            subject: Who or what is in the portrait.
            scene: The setting, e.g. "a rainy city street at night".
            lighting: Lighting direction (or empty), e.g. "soft window light".
        """
        light = f", {lighting}" if lighting else ""
        return (
            "Use the generate_image tool to create a portrait. "
            f"Prompt: \"Photorealistic portrait of {subject}, {scene}{light}, "
            "sharp facial features, 85mm lens, shallow depth of field\". "
            "Use model 'Runpod/FLUX.2-klein-4B-mflux-4bit' (best for faces) and a "
            "portrait size like '896x1152'."
        )

    @mcp.prompt()
    def poster(headline: str, style: str = "bold minimalist") -> str:
        """Create a text-heavy poster with generate_image.

        Args:
            headline: The main text to render in the image.
            style: Visual style, e.g. "retro 70s", "brutalist", "watercolor".
        """
        return (
            "Use the generate_image tool to create a poster. "
            f"Prompt: \"{style} poster with the text '{headline}', clean typography, "
            "high contrast\". Use the default text-turbo model "
            "('ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit') — it is far more reliable "
            "for in-image text than the general models."
        )

    @mcp.prompt()
    def lofi_track(mood: str = "chill", duration_seconds: int = 30) -> str:
        """Compose a lo-fi hip hop track with generate_music.

        Args:
            mood: The mood, e.g. "chill", "melancholic", "cozy".
            duration_seconds: Target length in seconds.
        """
        return (
            "Use the generate_music tool. Prompt: \"warm lo-fi hip hop, "
            f"{mood}, mellow Rhodes piano, dusty vinyl texture, soft boom-bap drums\". "
            f"Set duration_seconds={duration_seconds}, bpm=85. Omit lyrics (instrumental)."
        )

    @mcp.prompt()
    def song(lyrics: str, genre: str = "indie pop", vocal_language: str = "English") -> str:
        """Write and generate a short song with generate_music.

        Args:
            lyrics: The lyrics, with [Verse] / [Chorus] section tags.
            genre: The genre / style.
            vocal_language: Language for the sung lyrics.
        """
        return (
            "Use the generate_music tool. Prompt: \"" + genre + " song\". "
            f"Lyrics:\n{lyrics}\nSet vocal_language='{vocal_language}'."
        )

    @mcp.prompt()
    def short_video(subject: str, motion: str = "slow cinematic push-in", seconds: int = 2) -> str:
        """Generate a short video clip with generate_video.

        Args:
            subject: What the clip shows.
            motion: Camera / subject motion.
            seconds: Clip length (keep short; generation is slow).
        """
        return (
            "Use the generate_video tool. "
            f"Prompt: \"{subject}, {motion}, cinematic lighting\". "
            f"Set seconds={seconds}."
        )