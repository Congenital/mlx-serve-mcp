"""Process-management tools for background processes started by ``shell``.

These operate on the in-process registry in :class:`State` (handles ``bg1``,
``bg2``, ...). They let a client poll output from, or stop, long-lived
background processes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

    from . import Deps

MAX_OUTPUT = 20000


def register(mcp: "FastMCP", deps: Deps) -> None:
    state = deps.state

    @mcp.tool()
    def list_processes() -> str:
        """List background processes started this session, with handle and status."""
        procs = state.list_processes()
        if not procs:
            return "(no background processes)"
        lines = []
        for p in procs:
            rc = f", exit {p.returncode}" if p.returncode is not None else ""
            lines.append(f"{p.handle}: {p.status}{rc} — {p.command}")
        return "\n".join(lines)

    @mcp.tool()
    def read_process_output(handle: str) -> str:
        """Read the output a background process has produced.

        Args:
            handle: The process handle returned by ``shell`` (e.g. ``bg1``).
        """
        p = state.get_process(handle)
        if p is None:
            return f"error: no such process handle: {handle}"
        out = p.output or "(no output yet)"
        rc = f", exit {p.returncode}" if p.returncode is not None else ""
        return f"[{handle} {p.status}{rc}]\n{out[-MAX_OUTPUT:]}"

    @mcp.tool()
    async def kill_process(handle: str) -> str:
        """Stop a background process started this session.

        Args:
            handle: The process handle to stop (e.g. ``bg1``).
        """
        p = state.get_process(handle)
        if p is None:
            return f"error: no such process handle: {handle}"
        if p.status == "exited":
            return f"{handle} already exited (exit {p.returncode})"
        proc = p.process
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
                try:
                    await asyncio_wait_for(proc.wait(), timeout=5.0)
                except Exception:
                    proc.kill()
            except ProcessLookupError:
                pass
        p.status = "exited"
        if p.returncode is None:
            p.returncode = proc.returncode if proc is not None else None
        return f"stopped {handle} ({p.command})"


async def asyncio_wait_for(coro, timeout: float):
    """Small wrapper so we can import asyncio lazily inside the tool."""
    import asyncio

    return await asyncio.wait_for(coro, timeout=timeout)