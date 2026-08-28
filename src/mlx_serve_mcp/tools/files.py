"""Filesystem tools: read / write / edit / search / list files.

All paths are resolved against the server's current working directory
(:attr:`State.cwd`), so relative paths behave like they do in a shell. These
operate on the *local* machine hosting the MCP server.

The core logic lives in standalone ``_`` functions (unit-testable without the
MCP SDK); the ``@mcp.tool`` wrappers are thin adapters over them.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

    from ..state import State
    from . import Deps


# ── core logic (SDK-independent, unit-testable) ────────────────────────────

def _read_file(state: "State", path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    target = state.resolve(path)
    if not target.exists():
        return f"error: no such file: {target}"
    if not target.is_file():
        return f"error: not a file: {target}"
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"error: cannot read {target}: {exc}"
    total = len(lines)
    lo = max(1, start_line or 1)
    hi = min(total, end_line or total)
    if lo > hi:
        return f"error: empty range (start_line={lo} > end_line={hi}) in {target}"
    body = "\n".join(lines[lo - 1 : hi])
    header = f"[{target} | lines {lo}-{hi} of {total}]"
    return f"{header}\n{body}" if body else header


def _write_file(state: "State", path: str, content: str, append: bool = False) -> str:
    target = state.resolve(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        if append and existed:
            with target.open("a", encoding="utf-8") as handle:
                handle.write(content)
        else:
            target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"error: cannot write {target}: {exc}"
    verb = "appended to" if (append and existed) else "wrote"
    return f"{verb} {len(content)} chars to {target}"


def _edit_file(
    state: "State",
    path: str,
    replace: str,
    start_line: int | None = None,
    end_line: int | None = None,
    find: str | None = None,
) -> str:
    target = state.resolve(path)
    if not target.exists():
        return f"error: no such file: {target}"
    try:
        original = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"error: cannot read {target}: {exc}"

    if find is not None:
        if find not in original:
            return f"error: text not found in {target}"
        updated = original.replace(find, replace, 1)
        how = "matched text"
    elif start_line is not None:
        lines = original.splitlines(keepends=True)
        lo = max(1, start_line)
        hi = max(lo, end_line or start_line)
        if hi > len(lines):
            return f"error: end_line {hi} exceeds file length {len(lines)}"
        replacement = replace if replace.endswith("\n") or hi == len(lines) else replace + "\n"
        lines[lo - 1 : hi] = [replacement]
        updated = "".join(lines)
        how = f"lines {start_line}-{end_line or start_line}"
    else:
        return "error: provide either (start_line[, end_line]) or find"

    try:
        target.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return f"error: cannot write {target}: {exc}"
    return f"edited {target} ({how}); file is now {len(updated)} chars"


def _search_files(
    state: "State",
    pattern: str,
    path: str = ".",
    include: str | None = None,
    context: int = 0,
    max_results: int = 100,
) -> str:
    root = state.resolve(path)
    if not root.exists():
        return f"error: no such directory: {root}"
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"error: invalid regex: {exc}"

    def iter_files():
        if root.is_file():
            yield root
            return
        for p in root.rglob("*"):
            if p.is_file() and (include is None or fnmatch.fnmatch(p.name, include)):
                yield p

    results: list[str] = []
    for p in iter_files():
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except (OSError, UnicodeError):
            continue
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                lo = max(1, i - context)
                hi = min(len(lines), i + context)
                results.append(f"{p}:{i}: " + "\n".join(lines[lo - 1 : hi]))
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            break
    if not results:
        return f"no matches for {pattern!r} under {root}"
    suffix = f"\n... (truncated at {max_results} matches)" if len(results) >= max_results else ""
    return "\n".join(results) + suffix


def _list_files(state: "State", path: str = ".", pattern: str | None = None, recursive: bool = False) -> str:
    root = state.resolve(path)
    if not root.exists():
        return f"error: no such directory: {root}"
    if root.is_file():
        return str(root)
    iterator = root.rglob("*") if recursive else root.iterdir()
    entries = []
    for p in sorted(iterator):
        if pattern and not fnmatch.fnmatch(p.name, pattern):
            continue
        entries.append(str(p) + ("/" if p.is_dir() else ""))
    if not entries:
        return f"(no entries in {root}" + (f" matching {pattern}" if pattern else "") + ")"
    return "\n".join(entries)


# ── MCP tool adapters ──────────────────────────────────────────────────────

def register(mcp: "FastMCP", deps: Deps) -> None:
    state = deps.state

    @mcp.tool()
    def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """Read a file's contents, optionally a 1-based inclusive line range.

        Args:
            path: File path (relative paths resolve against the working directory).
            start_line: First line to read (1-based). Defaults to line 1.
            end_line: Last line to read (inclusive). Defaults to end of file.
        """
        return _read_file(state, path, start_line, end_line)

    @mcp.tool()
    def write_file(path: str, content: str, append: bool = False) -> str:
        """Write content to a file, creating parent directories as needed.

        Overwrites by default; pass ``append=True`` to add to the end instead.

        Args:
            path: Destination path (relative paths resolve against the working directory).
            content: The text to write.
            append: If true, append to the file instead of overwriting.
        """
        return _write_file(state, path, content, append)

    @mcp.tool()
    def edit_file(
        path: str,
        replace: str,
        start_line: int | None = None,
        end_line: int | None = None,
        find: str | None = None,
    ) -> str:
        """Edit a file. Two modes:

        * line-based: provide ``start_line``/``end_line`` (1-based, inclusive) + ``replace``.
        * text-based: provide ``find`` (exact text) + ``replace``.

        Args:
            path: File to edit (relative paths resolve against the working directory).
            replace: The replacement text.
            start_line: First line to replace (line-based mode).
            end_line: Last line to replace (line-based mode; defaults to start_line).
            find: Exact text to find (text-based mode).
        """
        return _edit_file(state, path, replace, start_line, end_line, find)

    @mcp.tool()
    def search_files(
        pattern: str,
        path: str = ".",
        include: str | None = None,
        context: int = 0,
        max_results: int = 100,
    ) -> str:
        """Search file contents for a pattern (regex) and return matching lines.

        Args:
            pattern: Regular expression to search for.
            path: Directory to search in (default: the working directory).
            include: Optional glob to filter files (e.g. "*.py").
            context: Number of context lines around each match (0-10).
            max_results: Maximum number of matches to return.
        """
        return _search_files(state, pattern, path, include, context, max_results)

    @mcp.tool()
    def list_files(path: str = ".", pattern: str | None = None, recursive: bool = False) -> str:
        """List files and directories, optionally filtered by a glob pattern.

        Args:
            path: Directory to list (default: the working directory).
            pattern: Optional glob to filter entries (e.g. "*.py").
            recursive: If true, search subdirectories recursively.
        """
        return _list_files(state, path, pattern, recursive)