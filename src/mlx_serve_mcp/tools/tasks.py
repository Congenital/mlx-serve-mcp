"""Task tool: run a background job, optionally on a repeating schedule.

In this server a *task* is a command job: ``goal`` is the shell command to run
(there is no LLM-agent backend inside the MCP server). ``schedule`` is optional
natural language — omit it (or use ``"now"``) to run once, or give a repeating
interval such as ``"every 5m"``, ``"every hour"``, or ``"every day at 9am"``.
Created tasks are tracked in :class:`State` and can be listed via
``list_tasks``.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

    from ..state import Task
    from . import Deps

_UNITS = {"s": 1, "sec": 1, "secs": 1, "m": 60, "min": 60, "mins": 60,
          "h": 3600, "hr": 3600, "hrs": 3600, "d": 86400, "day": 86400, "days": 86400}
_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def parse_schedule(schedule: str | None) -> dict[str, Any] | None:
    """Parse a natural-language schedule into a spec dict, or ``None`` for run-once.

    Returns one of:
      * ``None``                       — run once immediately.
      * ``{"interval": <seconds>}``    — repeat every N seconds.
      * ``{"daily": (hour, minute)}``  — repeat daily at HH:MM (local).
    """
    if not schedule or schedule.strip().lower() in ("now", "once", "immediately"):
        return None
    s = schedule.strip().lower()

    m = re.match(r"^every\s+(\d+(?:\.\d+)?)\s*(s|sec|secs|m|min|mins|h|hr|hrs|d|day|days)?$", s)
    if m:
        value = float(m.group(1))
        unit = m.group(2) or "s"
        return {"interval": value * _UNITS.get(unit, 1)}

    m = re.match(r"^every\s+(?:day|daily)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", s)
    if m:
        hour = int(m.group(1)) % 12
        if m.group(3) == "pm":
            hour += 12
        minute = int(m.group(2) or 0)
        return {"daily": (hour, minute)}

    m = re.match(r"^every\s+((?:mon|tue|wed|thu|fri|sat|sun)(?:day)?)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", s)
    if m:
        weekday = _WEEKDAYS.get(m.group(1)[:3])
        if weekday is not None:
            hour = int(m.group(2)) % 12
            if m.group(4) == "pm":
                hour += 12
            minute = int(m.group(3) or 0)
            return {"weekly": (weekday, hour, minute)}

    # Unrecognized — treat as run-once but surface that the schedule was ignored.
    return {"unrecognized": schedule}


def register(mcp: "FastMCP", deps: Deps) -> None:
    state = deps.state

    async def _run_once(command: str, task: "Task") -> None:
        """Run the task's command to completion, recording output + status."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(state.cwd),
                env=dict(os.environ),
            )
            task._process = proc
            stdout, _ = await proc.communicate()
            task.output = (stdout or b"").decode("utf-8", "replace")[-20000:]
            task.returncode = proc.returncode
            task.status = "done" if proc.returncode == 0 else "failed"
        except OSError as exc:
            task.status = "failed"
            task.output = f"error: {exc}"

    async def _repeat(task: "Task", spec: dict[str, Any]) -> None:
        """Scheduler loop for a repeating task."""
        try:
            if "interval" in spec:
                while True:
                    await _run_once(task.command or task.goal, task)
                    await asyncio.sleep(spec["interval"])
            elif "daily" in spec or "weekly" in spec:
                while True:
                    await _sleep_until_next(spec)
                    await _run_once(task.command or task.goal, task)
                    # after a clock-based run, sleep a day to avoid tight loops
                    await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            task.status = "failed"

    @mcp.tool()
    async def create_task(goal: str, schedule: str | None = None) -> str:
        """Create a background job that runs a shell command, optionally on a schedule.

        In this server a task is a *command job* (no LLM-agent backend is
        bundled). ``goal`` is the shell command to run.

        Args:
            goal: The shell command to run in the background.
            schedule: Optional. Omit or "now" to run once; a natural-language
                interval like "every 5m", "every hour", or "every day at 9am"
                to repeat.
        """
        task_id = await state.alloc_task()
        spec = parse_schedule(schedule)
        task = Task(task_id=task_id, goal=goal, command=goal,
                    schedule=schedule, status="scheduled" if spec else "running")
        state.put_task(task_id, task)

        if spec is None or "unrecognized" in spec:
            # Run once now.
            asyncio.get_running_loop().create_task(_run_once(goal, task))
            note = ""
            if spec and "unrecognized" in spec:
                note = f"\n(note: schedule {schedule!r} not recognized; ran once)"
            return f"created task {task_id} (run-once): {goal}{note}"

        task.status = "scheduled"
        task._timer = asyncio.get_running_loop().create_task(_repeat(task, spec))
        desc = _describe_schedule(spec, schedule)
        return f"created task {task_id} ({desc}): {goal}"

    @mcp.tool()
    def list_tasks() -> str:
        """List background jobs created this session, with id and status."""
        tasks = state.list_tasks()
        if not tasks:
            return "(no tasks)"
        lines = []
        for t in tasks:
            rc = f", exit {t.returncode}" if t.returncode is not None else ""
            sched = f", {t.schedule}" if t.schedule else ""
            lines.append(f"{t.task_id}: {t.status}{rc}{sched} — {t.goal}")
        return "\n".join(lines)

    @mcp.tool()
    async def cancel_task(task_id: str) -> str:
        """Cancel a background job (stops a repeating schedule / kills a running command).

        Args:
            task_id: The task id returned by ``create_task`` (e.g. ``task1``).
        """
        task = state.get_task(task_id)
        if task is None:
            return f"error: no such task: {task_id}"
        if task._timer is not None and not task._timer.done():
            task._timer.cancel()
        if task._process is not None and task._process.returncode is None:
            try:
                task._process.terminate()
            except ProcessLookupError:
                pass
        task.status = "cancelled"
        return f"cancelled {task_id} ({task.goal})"


def _describe_schedule(spec: dict[str, Any], original: str | None) -> str:
    if "interval" in spec:
        return f"every {spec['interval']:.0f}s"
    if "daily" in spec:
        h, m = spec["daily"]
        return f"daily at {h:02d}:{m:02d}"
    if "weekly" in spec:
        wd, h, m = spec["weekly"]
        names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        return f"weekly on {names[wd]} at {h:02d}:{m:02d}"
    return original or "repeating"


async def _sleep_until_next(spec: dict[str, Any]) -> None:
    """Sleep until the next daily/weekly fire time (local time)."""
    import datetime as _dt

    now = _dt.datetime.now()
    if "daily" in spec:
        hour, minute = spec["daily"]
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += _dt.timedelta(days=1)
    elif "weekly" in spec:
        weekday, hour, minute = spec["weekly"]
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        days_ahead = (weekday - now.weekday()) % 7
        target += _dt.timedelta(days=days_ahead)
        if target <= now:
            target += _dt.timedelta(days=7)
    else:
        return
    delay = (target - now).total_seconds()
    if delay > 0:
        await asyncio.sleep(delay)


# keep time import referenced (used by Task.created default elsewhere)
_ = time.time