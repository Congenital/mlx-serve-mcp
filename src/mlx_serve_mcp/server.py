"""Composition root: build the MCP server from a :class:`Config`.

The server is a standard FastMCP application (official ``mcp`` SDK) speaking
JSON-RPC over stdio (or SSE / streamable-HTTP). It keeps ONE
:class:`~mlx_serve_mcp.client.MlxServeClient` and ONE :class:`State` for the
process lifetime (the stdio transport is one session per process).

Tool groups:
* **generation** — proxy the remote mlx-serve media endpoints (image, image-edit,
  speech, music, video, 3D).
* **files**      — read / write / edit / search / list local files.
* **shell**      — run local commands (foreground or background).
* **processes**  — poll / kill background processes.
* **web**        — web search + fetch-based page reading.
* **memory**     — a small persistent note store.
* **tasks**      — background / scheduled command jobs.

Plus live **resources** (models, status, guidance) and one-click **prompts**.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import prompts as prompts_mod
from . import resources as resources_mod
from .client import MlxServeClient
from .config import Config
from .state import State
from .tools import Deps, register_all

INSTRUCTIONS = f"""\
You are backed by mlx-serve-mcp, a bridge to a remote mlx-serve inference server
(base URL configured at startup) plus a local toolset on the machine hosting this
server.

Generation (remote mlx-serve): generate_image, edit_image, generate_speech,
generate_music, generate_video, generate_3d. Artifacts are written under the
configured output dir and returned inline where the MCP content model allows.

Local: read_file, write_file, edit_file, search_files, list_files, shell
(foreground or background), list_processes, read_process_output, kill_process,
web_search, browse, save_memory, recall_memory, clear_memory, create_task,
list_tasks, cancel_task. Relative paths resolve against the server's working
directory.

Inspect the remote model inventory and health via the mlx-serve://models and
mlx-serve://status resources; mlx-serve://guidance explains model selection.
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
    resources_mod.register(mcp, deps)
    prompts_mod.register(mcp)
    return mcp