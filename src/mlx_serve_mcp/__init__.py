"""mlx-serve-mcp — MCP server bridging MCP clients to a remote mlx-serve instance.

mlx-serve (https://github.com/ddalcu/mlx-serve) is an OpenAI-compatible local
inference server for Apple Silicon. This package exposes its media generation
endpoints (``/v1/images/*``, ``/v1/audio/*``, ``/v1/video/generations``,
``/v1/3d/generations``) plus model management (``/v1/models``,
``/v1/load-model``, ``/v1/unload-model``) as MCP tools, and additionally exposes
the full local agent toolset (files, shell, web, memory, tasks) so any MCP client
can drive both a remote mlx-serve instance *and* the local machine.

The server speaks the Model Context Protocol over stdio (or SSE / streamable
HTTP) using the official ``mcp`` Python SDK.

``create_server`` is imported lazily so the SDK-independent modules (config,
state, client, video, and the local tool logic) can be imported — and unit
tested — without the ``mcp`` package installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.2.0"

from .config import Config, load_config

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

__all__ = ["Config", "create_server", "load_config", "__version__"]


def __getattr__(name: str):
    """Lazily expose ``create_server`` (pulls in the ``mcp`` SDK on first use)."""
    if name == "create_server":
        from .server import create_server

        return create_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")