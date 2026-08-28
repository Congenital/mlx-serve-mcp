"""Runtime configuration: CLI flags layered over ``MLX_SERVE_*`` environment variables.

Resolution order (highest priority first):
    1. explicit CLI flag
    2. ``MLX_SERVE_*`` environment variable
    3. built-in default

Every field is overridable so the same installed server can be pointed at
different mlx-serve instances or tuned per deployment without code changes.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://127.0.0.1:11234"
DEFAULT_TIMEOUT_SECONDS = 1800.0  # video / music generation can run for many minutes

# ── Default model ids (overridable via MLX_SERVE_*_MODEL env vars). ──────────
#
# Model guidance (from real-world testing on mlx-serve):
#   * Text-centric image work (posters, typography, in-image text) is far more
#     reliable on ``ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit`` than on the
#     higher-quality general models, so it is the default for ``generate_image``.
#   * ``Runpod/FLUX.2-klein-4B-mflux-4bit`` is the best general-purpose pick and
#     the only one in this group that renders faces well — it is the default for
#     ``edit_image``, and the recommended ``model`` for portraits and
#     photo-real scenes in ``generate_image``.
DEFAULT_IMAGE_MODEL = "ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit"
DEFAULT_IMAGE_EDIT_MODEL = "Runpod/FLUX.2-klein-4B-mflux-4bit"
DEFAULT_TTS_MODEL = "mlx-community/Kokoro-82M-v1.0-mlx"
DEFAULT_MUSIC_MODEL = "mlx-community/musicgen-small-mlx"
DEFAULT_VIDEO_MODEL = "mlx-community/HunyuanVideo-mlx"
DEFAULT_MESH_MODEL = "mlx-community/trellis-mlx"

# Default on-disk locations.
DEFAULT_OUTPUT_SUBDIR = "output"
DEFAULT_DATA_SUBDIR = ".mlx-serve-mcp"


def normalize_base_url(url: str) -> str:
    """Normalize a base URL: add a scheme if missing, strip a trailing slash."""
    url = (url or "").strip()
    if not url:
        return DEFAULT_BASE_URL
    if not urlparse(url).scheme:
        url = "http://" + url
    return url.rstrip("/")


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def default_data_dir() -> Path:
    return Path.home() / DEFAULT_DATA_SUBDIR


@dataclass
class Config:
    """Resolved runtime configuration for the MCP server process."""

    # Remote mlx-serve instance.
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    # Default model ids (per capability).
    image_model: str = DEFAULT_IMAGE_MODEL
    image_edit_model: str = DEFAULT_IMAGE_EDIT_MODEL
    tts_model: str = DEFAULT_TTS_MODEL
    music_model: str = DEFAULT_MUSIC_MODEL
    video_model: str = DEFAULT_VIDEO_MODEL
    mesh_model: str = DEFAULT_MESH_MODEL

    # Local filesystem roots.
    output_dir: Path = field(default_factory=lambda: Path.cwd() / DEFAULT_OUTPUT_SUBDIR)
    working_dir: Path = field(default_factory=Path.cwd)
    data_dir: Path = field(default_factory=default_data_dir)

    # Transport: "stdio" (default), "sse", or "streamable-http".
    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8765

    @property
    def memory_file(self) -> Path:
        return self.data_dir / "memories.json"

    def ensure_dirs(self) -> None:
        """Create the on-disk directories the server writes to."""
        for d in (self.output_dir, self.data_dir):
            d.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "video").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "mesh").mkdir(parents=True, exist_ok=True)


def load_config(argv: list[str] | None = None) -> Config:
    """Build a :class:`Config` from CLI flags over environment variables.

    Raises ``ValueError`` for an unknown transport so misconfiguration fails
    fast at startup rather than mid-session.
    """
    parser = argparse.ArgumentParser(
        prog="mlx-serve-mcp",
        description="MCP server bridging MCP clients to a remote mlx-serve instance.",
    )
    parser.add_argument("--url", "--base-url", dest="base_url", default=None,
                        help=f"mlx-serve base URL (default: env MLX_SERVE_URL or {DEFAULT_BASE_URL})")
    parser.add_argument("--api-key", default=None,
                        help="Bearer token for the mlx-serve instance (default: env MLX_SERVE_API_KEY)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for generated artifacts (default: env MLX_SERVE_OUTPUT_DIR or ./output)")
    parser.add_argument("--timeout", dest="timeout_seconds", type=float, default=None,
                        help="Per-request timeout in seconds (default: env MLX_SERVE_TIMEOUT or 1800)")
    parser.add_argument("--image-model", default=None, help="Default model for generate_image")
    parser.add_argument("--image-edit-model", default=None, help="Default model for edit_image")
    parser.add_argument("--tts-model", default=None, help="Default model for generate_speech")
    parser.add_argument("--music-model", default=None, help="Default model for generate_music")
    parser.add_argument("--video-model", default=None, help="Default model for generate_video")
    parser.add_argument("--mesh-model", default=None, help="Default model for generate_3d")
    parser.add_argument("--working-dir", default=None,
                        help="Base directory for relative file/shell paths (default: env MLX_SERVE_WORKING_DIR or cwd)")
    parser.add_argument("--data-dir", default=None,
                        help="Directory for persistent state such as memory (default: env MLX_SERVE_DATA_DIR or ~/.mlx-serve-mcp)")
    # No argparse ``choices`` on purpose: an invalid value must surface as a
    # ValueError from load_config (below), not a SystemExit from argparse.
    parser.add_argument("--transport", default=None,
                        help="MCP transport: stdio | sse | streamable-http (default: env MLX_SERVE_TRANSPORT or stdio)")
    parser.add_argument("--host", default=None, help="Bind host for sse/streamable-http (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Bind port for sse/streamable-http (default: 8765)")

    # Allow a partial argv (tests) or the full argv (CLI); ignore the program name.
    args = parser.parse_args(argv)

    base_url = args.base_url or _env("MLX_SERVE_URL", DEFAULT_BASE_URL)
    api_key = args.api_key or os.environ.get("MLX_SERVE_API_KEY") or None
    out_dir = Path(args.output_dir or _env("MLX_SERVE_OUTPUT_DIR", DEFAULT_OUTPUT_SUBDIR))
    timeout = args.timeout_seconds if args.timeout_seconds is not None else float(_env("MLX_SERVE_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))
    working_dir = Path(args.working_dir or _env("MLX_SERVE_WORKING_DIR", "."))
    data_dir = Path(args.data_dir or _env("MLX_SERVE_DATA_DIR", str(default_data_dir())))
    transport = (args.transport or os.environ.get("MLX_SERVE_TRANSPORT") or "stdio").lower()
    host = args.host or os.environ.get("MLX_SERVE_HOST", "127.0.0.1")
    port = args.port if args.port is not None else int(_env("MLX_SERVE_PORT", "8765"))

    if transport not in ("stdio", "sse", "streamable-http"):
        raise ValueError(f"unknown transport: {transport!r} (expected stdio, sse, or streamable-http)")

    return Config(
        base_url=normalize_base_url(base_url),
        api_key=api_key,
        output_dir=out_dir.expanduser().resolve(),
        timeout_seconds=float(timeout),
        image_model=args.image_model or os.environ.get("MLX_SERVE_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL,
        image_edit_model=args.image_edit_model or os.environ.get("MLX_SERVE_IMAGE_EDIT_MODEL") or DEFAULT_IMAGE_EDIT_MODEL,
        tts_model=args.tts_model or os.environ.get("MLX_SERVE_TTS_MODEL") or DEFAULT_TTS_MODEL,
        music_model=args.music_model or os.environ.get("MLX_SERVE_MUSIC_MODEL") or DEFAULT_MUSIC_MODEL,
        video_model=args.video_model or os.environ.get("MLX_SERVE_VIDEO_MODEL") or DEFAULT_VIDEO_MODEL,
        mesh_model=args.mesh_model or os.environ.get("MLX_SERVE_MESH_MODEL") or DEFAULT_MESH_MODEL,
        working_dir=working_dir.expanduser().resolve(),
        data_dir=data_dir.expanduser().resolve(),
        transport=transport,
        host=host,
        port=port,
    )