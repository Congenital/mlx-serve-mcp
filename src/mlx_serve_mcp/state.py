"""In-process state shared by the local (non-mlx-serve) tools.

The MCP server is one process per session, so a small amount of mutable state
lives here for the process lifetime:

* ``cwd``            — the current working directory for relative file/shell paths.
* ``processes``      — registry of background processes (``shell`` with ``run_in_background``).
* ``tasks``          — registry of background / scheduled jobs (``create_task``).
* memory             — a small persistent store (JSON on disk) for ``save_memory`` / ``recall_memory``.

Generation tools do not use this state; they only need the :class:`Config`
(roots) and the :class:`~mlx_serve_mcp.client.MlxServeClient`.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config


@dataclass
class ProcessHandle:
    """One background process started by the ``shell`` tool."""

    handle: str
    command: str
    process: asyncio.subprocess.Process | None = None
    output: str = ""
    status: str = "running"  # "running" | "exited"
    returncode: int | None = None
    created: float = field(default_factory=time.time)


@dataclass
class Task:
    """One background / scheduled job created by the ``create_task`` tool."""

    task_id: str
    goal: str
    command: str | None = None
    schedule: str | None = None
    status: str = "running"  # "running" | "done" | "failed" | "scheduled"
    output: str = ""
    returncode: int | None = None
    created: float = field(default_factory=time.time)
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _timer: asyncio.Task | None = field(default=None, repr=False)


class State:
    """Mutable, process-lifetime state for the local tools."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cwd = Path(config.working_dir)
        self.processes: dict[str, ProcessHandle] = {}
        self.tasks: dict[str, Task] = {}
        self._next_handle = 1
        self._next_task = 1
        self._lock = asyncio.Lock()

    # ── path resolution ────────────────────────────────────────────────────

    def resolve(self, path: str | Path) -> Path:
        """Resolve a possibly-relative path against the current working dir."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.cwd / p
        return p

    def set_cwd(self, path: str | Path) -> Path:
        """Change the working directory (must exist). Returns the new absolute cwd."""
        target = self.resolve(path)
        target = target.resolve()
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {target}")
        self.cwd = target
        return target

    # ── background processes ───────────────────────────────────────────────

    async def alloc_handle(self) -> str:
        async with self._lock:
            handle = f"bg{self._next_handle}"
            self._next_handle += 1
            return handle

    def put_process(self, handle: str, proc: ProcessHandle) -> None:
        self.processes[handle] = proc

    def get_process(self, handle: str) -> ProcessHandle | None:
        return self.processes.get(handle)

    def list_processes(self) -> list[ProcessHandle]:
        return list(self.processes.values())

    # ── background / scheduled tasks ───────────────────────────────────────

    async def alloc_task(self) -> str:
        async with self._lock:
            task_id = f"task{self._next_task}"
            self._next_task += 1
            return task_id

    def put_task(self, task_id: str, task: Task) -> None:
        self.tasks[task_id] = task

    def get_task(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        return list(self.tasks.values())

    # ── persistent memory ──────────────────────────────────────────────────

    def load_memory(self) -> list[str]:
        """Load the persistent memory list (empty if the store is absent)."""
        try:
            raw = self.config.memory_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x) for x in data]
        except (FileNotFoundError, ValueError):
            pass
        return []

    def save_memory(self, text: str) -> list[str]:
        """Append a memory and persist the store. Returns the full list."""
        memories = self.load_memory()
        if text not in memories:
            memories.append(text)
        self.config.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.memory_file.write_text(json.dumps(memories, indent=2), encoding="utf-8")
        return memories

    def clear_memory(self) -> int:
        """Drop all memories; returns how many were removed."""
        memories = self.load_memory()
        self.config.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.memory_file.write_text("[]", encoding="utf-8")
        return len(memories)