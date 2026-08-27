"""MCP resources: live data sources from the remote mlx-serve instance.

Exposed through the standard MCP ``resources/list`` / ``resources/read``
methods so clients can attach context data (model inventory, server status,
model guidance) without calling a tool.
"""

from __future__ import annotations

from typing import Any

from .client import MlxServeClient, MlxServeError

# ── Static resource definitions ─────────────────────────────────────────

_RESOURCES: list[dict[str, Any]] = [
    {
        "name": "models",
        "uri": "mlx-serve://models",
        "description": (
            "Live model inventory from the remote mlx-serve instance — "
            "model ids, capability flags (chat, vision, image, speech, "
            "music, video, 3D) and load status."
        ),
        "mimeType": "application/json",
    },
    {
        "name": "server_status",
        "uri": "mlx-serve://status",
        "description": (
            "Health and status of the remote mlx-serve instance — "
            "reachable, version, loaded models, GPU info."
        ),
        "mimeType": "application/json",
    },
    {
        "name": "model_guidance",
        "uri": "mlx-serve://guidance",
        "description": (
            "Recommended model choices for each media tool based on "
            "real-world testing (which model for text rendering, faces, "
            "music, video, 3D)."
        ),
        "mimeType": "text/markdown",
    },
]

_MODEL_GUIDANCE_MD = """\
# mlx-serve Model Guidance

Based on real-world testing on mlx-serve (Apple Silicon):

| Tool | Recommended Model | Why |
|------|------------------|-----|
| `generate_image` (text/posters) | `ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit` | Most reliable at rendering text in images |
| `generate_image` (portraits/faces) | `Runpod/FLUX.2-klein-4B-mflux-4bit` | Best face quality, loads reliably |
| `edit_image` | `Runpod/FLUX.2-klein-4B-mflux-4bit` | Facial realism, reliable loading |
| `text_to_speech` | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | Most natural TTS |
| `generate_music` | `ddalcu/MiniMax-Music3-MLX-Serve-8bit` | Best music quality |
| `generate_video` | `ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit` | Best video quality |
| `generate_3d` | `ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit` | Best 3D mesh quality |

## Warnings

- `ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit` can hit `MissingMageFlowWeight` — avoid for `edit_image`.
- `mlx-community/flux2-klein-9b-4bit` has a similar load-failure issue — avoid.
- `ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit` is fast but face quality is below FLUX.2-klein-4B.

## Strategy

- **Text-centric art** (posters, typography, signs) → Mage-Flow-Turbo
- **Portraits / photo-real** → FLUX.2-klein-4B
- **Music / Video / 3D / TTS** → the respective default (best in class)
"""


def list_resources() -> list[dict[str, Any]]:
    """MCP ``resources/list`` result."""
    return list(_RESOURCES)


async def read_resource(uri: str, client: MlxServeClient | None = None) -> dict[str, Any]:
    """MCP ``resources/read`` result for a single resource URI.

    - ``mlx-serve://models`` — calls the live server for model inventory.
    - ``mlx-serve://status`` — calls the live server for health/status.
    - ``mlx-serve://guidance`` — returns static markdown guidance.

    If ``client`` is None or the server is unreachable, falls back to
    static content so the resource is always readable.
    """
    import json

    if uri == "mlx-serve://guidance":
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": _MODEL_GUIDANCE_MD,
                }
            ]
        }

    if uri == "mlx-serve://models":
        if client is None:
            text = json.dumps(
                {"note": "mlx-serve client not configured — start the server and re-read"},
                indent=2,
            )
        else:
            try:
                models = await client.list_models()
                text = json.dumps(models, indent=2, ensure_ascii=False)
            except (MlxServeError, Exception) as exc:
                text = json.dumps(
                    {"error": f"mlx-serve unreachable: {exc}"},
                    indent=2,
                )
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": text,
                }
            ]
        }

    if uri == "mlx-serve://status":
        if client is None:
            text = json.dumps(
                {"status": "unknown", "detail": "No mlx-serve client configured"},
                indent=2,
            )
        else:
            try:
                health = await client.health()
                text = json.dumps(health, indent=2, ensure_ascii=False)
            except (MlxServeError, Exception) as exc:
                text = json.dumps(
                    {"status": "unreachable", "detail": str(exc)},
                    indent=2,
                )
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": text,
                }
            ]
        }

    raise ValueError(f"Unknown resource URI: {uri}")