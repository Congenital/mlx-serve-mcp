"""Shell tool: run commands on the local machine (foreground or background).

Foreground commands block until they finish and return combined output plus the
exit code. Background commands start an asyncio subprocess, register it in
:class:`State`, and return immediately with a handle (``bg1``, ``bg2``, ...)
that the process-management tools can poll or kill.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

    from ..state import ProcessHandle
    from . import Deps

MAX_OUTPUT = 20000  # cap returned output so a chatty command can't flood the client


def _fmt(result: tuple[int, str, str], command: str) -> str:
    code, out, err = result
    parts = [f"$ {command}\n"]
    if out:
        parts.append(out[-MAX_OUTPUT:])
    if err:
        parts.append(f"[stderr]\n{err[-MAX_OUTPUT // 2:]}")
    parts.append(f"[exit {code}]")
    return "\n".join(p for p in parts if p)


def register(mcp: "FastMCP", deps: Deps) -> None:
    state = deps.state

    @mcp.tool()
    async def shell(command: str, run_in_background: str = "false", cwd: str | None = None) -> str:
        """Run a shell command on the machine hosting the MCP server.

        Use ``run_in_background="true"`` for long-lived processes (servers,
        watchers); it returns a handle (e.g. ``bg1``) you can poll with
        ``read_process_output`` or stop with ``kill_process``.

        Args:
            command: The shell command to execute.
            run_in_background: "true" to start in the background and return a handle; "false" (default) to wait.
            cwd: Optional working directory (defaults to the server's current directory).
        """
        workdir = str(state.resolve(cwd)) if cwd else str(state.cwd)
        env = dict(os.environ)

        if run_in_background.lower() in ("true", "1", "yes"):
            try:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=workdir,
                    env=env,
                )
            except OSError as exc:
                return f"error: could not start background process: {exc}"
            handle = await state.alloc_handle()
            ph = ProcessHandle(handle=handle, command=command, process=proc, status="running")
            state.put_process(handle, ph)
            # Drain output in the background so the pipe never blocks.
            asyncio.get_running_loop().create_task(_drain(proc, ph))
            return f"started background process {handle}: {command}\npoll with read_process_output({handle!r}); stop with kill_process({handle!r})"

        # Foreground: wait for completion.
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=env,
            )
            stdout, stderr = await proc.communicate()
        except OSError as exc:
            return f"error: could not run command: {exc}"
        out = stdout.decode("utf-8", "replace")
        err = stderr.decode("utf-8", "replace")
        return _fmt((proc.returncode or 0, out, err), command)


async def _drain(proc: asyncio.subprocess.Process, ph: "ProcessHandle") -> None:
    """Continuously read a background process's output into its handle."""
    assert proc.stdout is not None
    try:
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            ph.output += chunk.decode("utf-8", "replace")
            if len(ph.output) > MAX_OUTPUT * 4:
                ph.output = ph.output[-MAX_OUTPUT * 4 :]
    except Exception:  # pragma: no cover - defensive
        pass
    await proc.wait()
    ph.status = "exited"
    ph.returncode = proc.returncode