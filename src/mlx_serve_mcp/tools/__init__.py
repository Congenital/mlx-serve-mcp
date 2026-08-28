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
    """Register every tool module onto ``mcp``."""
    from . import files, generation, memory, processes, shell, tasks, web

    generation.register(mcp, deps)
    files.register(mcp, deps)
    shell.register(mcp, deps)
    processes.register(mcp, deps)
    web.register(mcp, deps)
    memory.register(mcp, deps)
    tasks.register(mcp, deps)