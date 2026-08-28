"""Tool modules. Each exposes ``register(mcp, deps)`` which decorates its tool
functions onto the shared :class:`~mcp.server.fastmcp.FastMCP` instance.

Keeping the tool definitions in per-domain modules (generation, files, shell,
processes, web, memory, tasks) keeps ``server.py`` a thin composition root.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

    from ..client import MlxServeClient
    from ..config import Config
    from ..state import State


@dataclass
class Deps:
    """Everything a tool module needs, bundled for easy passing."""

    config: "Config"
    client: "MlxServeClient"
    state: "State"


def register_all(mcp: "FastMCP", deps: Deps) -> None:
    """Register the media-generation tool group onto ``mcp``.

    Only the remote mlx-serve generation tools are exposed. The other tool
    modules (files, shell, processes, web, memory, tasks) stay in the codebase
    but are intentionally NOT registered, so the MCP tool list contains just
    the six generation tools. Re-add a line below to expose a group again.
    """
    from . import generation

    generation.register(mcp, deps)