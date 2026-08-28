"""Composition root: build the MCP server from a :class:`Config`.

The server is a standard FastMCP application (official ``mcp`` SDK) speaking
JSON-RPC over stdio (or SSE / streamable-HTTP). It keeps ONE
:class:`~mlx_serve_mcp.client.MlxServeClient` and ONE :class:`State` for the
process lifetime (the stdio transport is one session per process).

Tool groups:
* **generation** — proxy the remote mlx-serve media endpoints (image, image-edit,
  speech, music, video, 3D). This is the only group currently registered.

The other tool modules (files, shell, processes, web, memory, tasks) and the
resources/prompts registrations are intentionally not wired in; see
``register_all`` and the commented-out registrations below.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import prompts as prompts_mod
from . import resources as resources_mod
from .client import MlxServeClient
from .config import Config
from .state import State
from .tools import Deps, register_all

INSTRUCTIONS = """\
You are backed by mlx-serve-mcp, a bridge to a remote mlx-serve inference
server (base URL configured at startup).

Available tools: generate_image, edit_image, generate_speech, generate_music,
generate_video, generate_3d. Artifacts are written under the configured output
dir; tools return only the saved path.
"""


def create_server(config: Config) -> FastMCP:
    """Build and return a fully-populated FastMCP server for ``config``."""
    config.ensure_dirs()

    client = MlxServeClient(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout_seconds=config.timeout_seconds,
    )
    state = State(config)

    try:
        mcp = FastMCP(name="mlx-serve", instructions=INSTRUCTIONS)
    except TypeError:  # older/newer SDK constructor variance
        mcp = FastMCP("mlx-serve")
        try:
            mcp.instructions = INSTRUCTIONS
        except Exception:  # pragma: no cover
            pass
    # Best-effort: expose host/port for the sse / streamable-http transports.
    try:
        mcp.settings.host = config.host
        mcp.settings.port = config.port
    except Exception:  # pragma: no cover - SDK version variance
        pass

    deps = Deps(config=config, client=client, state=state)
    register_all(mcp, deps)
    # Intentionally NOT registered (hidden from the MCP tool list, code kept):
    #   resources_mod.register(mcp, deps)   # mlx-serve://status|models|guidance
    #   prompts_mod.register(mcp)           # portrait / poster / lofi_track / song / short_video
    return mcp