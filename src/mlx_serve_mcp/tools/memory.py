"""Memory tools: a small persistent note store.

Memories are plain strings persisted as JSON under ``config.data_dir`` so they
survive server restarts. This mirrors the assistant's ``saveMemory`` — a place
to stash user preferences, project context, or important facts for future
sessions. Core logic is in standalone ``_`` functions (unit-testable); the
``@mcp.tool`` wrappers are thin adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

    from ..state import State
    from . import Deps


# ── core logic (SDK-independent, unit-testable) ────────────────────────────

def _save_memory(state: "State", memory: str) -> str:
    memories = state.save_memory(memory)
    return f"saved memory (now {len(memories)} total): {memory}"


def _recall_memory(state: "State") -> str:
    memories = state.load_memory()
    if not memories:
        return "(no saved memories)"
    return "\n".join(f"- {m}" for m in memories)


def _clear_memory(state: "State") -> str:
    n = state.clear_memory()
    return f"cleared {n} memor" + ("y" if n == 1 else "ies")


# ── MCP tool adapters ──────────────────────────────────────────────────────

def register(mcp: "FastMCP", deps: Deps) -> None:
    state = deps.state

    @mcp.tool()
    def save_memory(memory: str) -> str:
        """Save a memory for future sessions.

        Use for user preferences, project context, or important facts.

        Args:
            memory: The memory to save.
        """
        return _save_memory(state, memory)

    @mcp.tool()
    def recall_memory() -> str:
        """List all saved memories."""
        return _recall_memory(state)

    @mcp.tool()
    def clear_memory() -> str:
        """Delete all saved memories."""
        return _clear_memory(state)