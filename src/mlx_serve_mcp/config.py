"""Runtime configuration: CLI flags layered over MLX_SERVE_* environment variables."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://127.0.0.1:11234"
DEFAULT_TIMEOUT_SECONDS = 1800.0  # video / music generation can run for many minutes

# Default model ids (overridable via MLX_SERVE_*_MODEL env vars).
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
DEFAULT_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
DEFAULT_MUSIC_MODEL = "ddalcu/MiniMax-Music3-MLX-Serve-8bit"
DEFAULT_VIDEO_MODEL = "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit"
DEFAULT_MESH_MODEL = "ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit"


@dataclass(frozen=True)
class Config:
    """Resolved connection + output + model settings for the MCP server."""

    base_url: str  # normalized, no trailing slash, e.g. "http://192.168.1.10:11234"
    api_key: str | None
    output_dir: Path
    timeout_seconds: float

    # Default model ids — used when a tool is called without an explicit ``model``.
    image_model: str = DEFAULT_IMAGE_MODEL
    image_edit_model: str = DEFAULT_IMAGE_EDIT_MODEL
    tts_model: str = DEFAULT_TTS_MODEL
    music_model: str = DEFAULT_MUSIC_MODEL
    video_model: str = DEFAULT_VIDEO_MODEL
    mesh_model: str = DEFAULT_MESH_MODEL


def normalize_base_url(raw: str) -> str:
    """Normalize a user-supplied address into an http(s) base URL.

    Accepts bare ``ip:port`` (``192.168.1.10:11234``, the common case), full
    URLs (scheme optional), and strips any trailing slash so path joins stay
    predictable.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("mlx-serve URL is empty")
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme {parsed.scheme!r} in {raw!r}")
    if not parsed.netloc:
        raise ValueError(f"URL has no host:port part: {raw!r}")
    return raw.rstrip("/")


def _parse_timeout(raw: str | None) -> float:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid timeout value: {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"timeout must be positive, got {value}")
    return value


def build_config(
    base_url: str,
    api_key: str | None = None,
    output_dir: str | Path | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    image_model: str | None = None,
    image_edit_model: str | None = None,
    tts_model: str | None = None,
    music_model: str | None = None,
    video_model: str | None = None,
    mesh_model: str | None = None,
) -> Config:
    """Assemble a Config with normalization + defaults applied.

    Model parameters fall back to ``MLX_SERVE_*_MODEL`` env vars, then to
    the built-in defaults.
    """
    out_dir = Path(output_dir).expanduser() if output_dir else (
        Path.home() / "Downloads" / "mlx-serve-mcp"
    )
    return Config(
        base_url=normalize_base_url(base_url),
        api_key=(api_key or None) or None,
        output_dir=out_dir.resolve(),
        timeout_seconds=float(timeout_seconds),
        image_model=image_model or os.environ.get("MLX_SERVE_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL,
        image_edit_model=image_edit_model or os.environ.get("MLX_SERVE_IMAGE_EDIT_MODEL") or DEFAULT_IMAGE_EDIT_MODEL,
        tts_model=tts_model or os.environ.get("MLX_SERVE_TTS_MODEL") or DEFAULT_TTS_MODEL,
        music_model=music_model or os.environ.get("MLX_SERVE_MUSIC_MODEL") or DEFAULT_MUSIC_MODEL,
        video_model=video_model or os.environ.get("MLX_SERVE_VIDEO_MODEL") or DEFAULT_VIDEO_MODEL,
        mesh_model=mesh_model or os.environ.get("MLX_SERVE_MESH_MODEL") or DEFAULT_MESH_MODEL,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-serve-mcp",
        description=(
            "MCP server that exposes a remote mlx-serve instance "
            "(image / video / speech / music / 3D generation) as MCP tools. "
            "Point it at the server with --url (ip:port) and wire it into your "
            "MCP client over stdio."
        ),
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("MLX_SERVE_URL"),
        help=f"Base URL of the mlx-serve server, e.g. http://192.168.1.10:11234 or bare ip:port "
        f"(env MLX_SERVE_URL, default {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MLX_SERVE_API_KEY"),
        help="Bearer API key if the server runs with key auth (env MLX_SERVE_API_KEY)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("MLX_SERVE_OUTPUT_DIR"),
        help="Directory where generated media files are written "
        "(env MLX_SERVE_OUTPUT_DIR, default ~/Downloads/mlx-serve-mcp)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="HTTP timeout in seconds for generation requests "
        "(env MLX_SERVE_TIMEOUT, default 1800)",
    )
    return parser


def load_config(argv: list[str] | None = None) -> Config:
    """Parse CLI args (layered over env vars) into a Config."""
    args = make_parser().parse_args(argv)
    base_url = args.url or DEFAULT_BASE_URL
    if args.timeout is not None:
        timeout: float = _require_positive(args.timeout)
    elif os.environ.get("MLX_SERVE_TIMEOUT"):
        timeout = _parse_timeout(os.environ["MLX_SERVE_TIMEOUT"])
    else:
        timeout = DEFAULT_TIMEOUT_SECONDS
    return build_config(
        base_url=base_url,
        api_key=args.api_key,
        output_dir=args.output_dir,
        timeout_seconds=timeout,
        image_model=os.environ.get("MLX_SERVE_IMAGE_MODEL"),
        image_edit_model=os.environ.get("MLX_SERVE_IMAGE_EDIT_MODEL"),
        tts_model=os.environ.get("MLX_SERVE_TTS_MODEL"),
        music_model=os.environ.get("MLX_SERVE_MUSIC_MODEL"),
        video_model=os.environ.get("MLX_SERVE_VIDEO_MODEL"),
        mesh_model=os.environ.get("MLX_SERVE_MESH_MODEL"),
    )


def _require_positive(value: float) -> float:
    if value <= 0:
        raise ValueError(f"timeout must be positive, got {value}")
    return value

