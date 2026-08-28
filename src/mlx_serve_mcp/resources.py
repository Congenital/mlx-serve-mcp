"""MCP resources: live data sources from the remote mlx-serve instance.

Exposed through the standard MCP ``resources/list`` / ``resources/read`` methods
so clients can attach context (model inventory, server status, model guidance)
without calling a tool.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

    from . import Deps
    from .client import MlxServeClient

MODEL_GUIDANCE = """\
Model guidance (from real-world testing on mlx-serve):
* Text-centric image work (posters, typography, in-image text) is far more
  reliable on 'ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit' — the default for generate_image.
* 'Runpod/FLUX.2-klein-4B-mflux-4bit' is the best general-purpose image model and the
  only one that renders faces well — the default for edit_image, and the pick for
  portraits / photo-real scenes in generate_image.
* Pick TTS / music / video / 3D models from the /models inventory (see the
  'models' resource); the configured defaults are reasonable starting points.
"""


def register(mcp: "FastMCP", deps: Deps) -> None:
    client: MlxServeClient = deps.client

    @mcp.resource("mlx-serve://status")
    async def status() -> str:
        """Live health of the remote mlx-serve instance."""
        try:
            data = await client.health()
        except Exception as exc:  # surface, don't crash the resource
            return f"unreachable: {exc}"
        return json.dumps(data, indent=2)

    @mcp.resource("mlx-serve://models")
    async def models() -> str:
        """The model inventory, with capability flags, from the remote instance."""
        try:
            entries = await client.list_models()
        except Exception as exc:
            return f"unreachable: {exc}"
        lines = []
        for entry in entries:
            mid = entry.get("id", "?")
            caps = []
            for key in ("image", "image_edit", "tts", "music", "video", "mesh", "3d"):
                if entry.get(key):
                    caps.append(key)
            cap_str = f" [{', '.join(caps)}]" if caps else ""
            lines.append(f"- {mid}{cap_str}")
        return "\n".join(lines) if lines else "(no models reported)"

    @mcp.resource("mlx-serve://guidance")
    def guidance() -> str:
        """Static guidance on which model to use for which kind of generation."""
        return MODEL_GUIDANCE