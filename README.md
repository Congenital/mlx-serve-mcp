# [![MCP Badge](https://lobehub.com/badge/mcp/congenital-mlx-serve-mcp?style=for-the-badge)](https://lobehub.com/mcp/congenital-mlx-serve-mcp)

An MCP (Model Context Protocol) server that gives any MCP client two things at once:

1. **A bridge to a remote [mlx-serve](https://github.com/ddalcu/mlx-serve) instance** — Apple
   Silicon's OpenAI-compatible local inference server — exposing its media generation
   endpoints (image, image-edit, speech, music, video, 3D) and model management as MCP tools.
2. **The full local agent toolset** — files, shell, background processes, web search, memory,
   and scheduled tasks — so the same server can also drive the machine it runs on.

It speaks MCP using the official Python SDK (`mcp.server.fastmcp.FastMCP`) over **stdio**
by default, or **SSE** / **streamable-HTTP** if you prefer a network transport.

> This is a ground-up reimplementation. The previous revision hand-rolled the JSON-RPC
> protocol, which is fragile and drifts from the spec. This version delegates the protocol
> to the official SDK and adds the local toolset on top.

## Install

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync                      # creates .venv and installs deps (add --extra dev for tests)
uv run mlx-serve-mcp       # launch the server
```

> **Important:** after `uv sync` (or `uv pip install`), the `mlx-serve-mcp` console
> script exists only inside the project's `.venv` — start it with `uv run
> mlx-serve-mcp`, **not** a bare `mlx-serve-mcp` (that works only after
> `uv tool install` / `pipx install`, or a `pip install`).

Or with plain pip:

```bash
pip install -e ".[dev]"
mlx-serve-mcp              # or: python -m mlx_serve_mcp
```

Requires Python ≥ 3.10 (`uv run` picks a suitable interpreter automatically).
`ffmpeg` comes from the `imageio-ffmpeg` dependency (a system `ffmpeg` on `PATH`
is used preferentially when present).

## Run

```bash
# stdio (default) — for MCP hosts that spawn the server (Claude Desktop, etc.)
uv run mlx-serve-mcp

# Native deps avoided entirely: `cryptography` is excluded via
# `[tool.uv] override-dependencies` (uv sync will not install it; `pyjwt`
# stays on the pure-Python `2.10` line, and nothing in the code requires it)

# point it at a specific mlx-serve instance
uv run mlx-serve-mcp --url http://127.0.0.1:11234 --api-key "$MLX_SERVE_API_KEY"

# network transport (SSE / streamable-HTTP on 127.0.0.1:8765)
uv run mlx-serve-mcp --transport sse --host 127.0.0.1 --port 8765
```

If you installed with plain `pip`, drop the `uv run` prefix (or use
`python -m mlx_serve_mcp`).

## Configure an MCP host

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(Windows: `%APPDATA%\Claude\claude_desktop_config.json`) and restart Claude:

```json
{
  "mcpServers": {
    "mlx-serve": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/mlx-serve-mcp", "mlx-serve-mcp"],
      "env": {
        "MLX_SERVE_URL": "http://127.0.0.1:11234",
        "MLX_SERVE_API_KEY": "your-key"
      }
    }
  }
}
```

After `uv tool install mlx-serve-mcp` (or `pipx install mlx-serve-mcp`) you can
instead use `"command": "mlx-serve-mcp"` with no `uv run` wrapper.

### LobeHub

The repo ships `lhm.plugin.json` — upload it as a plugin. Its `stdio` deployment
option spawns `mlx-serve-mcp` and reads `MLX_SERVE_URL` / `MLX_SERVE_API_KEY`
from the deployment environment.

## Smoke-test without a host

```bash
uv sync --extra dev
uv run python mcp_client.py --list
uv run python mcp_client.py --call recall_memory
uv run python mcp_client.py --call generate_image --args '{"prompt": "a red fox in the snow"}'
```

`mcp_client.py` spawns the server itself (auto-detecting `mlx-serve-mcp` or
`python -m mlx_serve_mcp`, and adding `src/` to `PYTHONPATH` when running from
a source checkout) and speaks MCP over stdio.

## Toolset

### Generation (proxied to the remote mlx-serve instance)
| Tool | What it does |
| --- | --- |
| `generate_image` | Text → image (saved to disk; returns the saved path) |
| `edit_image` | Image + prompt → edited image (image-to-image, saved to disk) |
| `generate_speech` | Text → spoken audio (TTS, saved as WAV) |
| `generate_music` | Style description (+ optional lyrics) → music (saved as WAV) |
| `generate_video` | Prompt → short video clip (muxed to MP4 via ffmpeg, saved to disk) |
| `generate_3d` | Prompt → 3D mesh (GLB) |

### Local (run on the machine hosting the server)
| Tool | What it does |
| --- | --- |
| `read_file` / `write_file` / `edit_file` | Filesystem read / write / edit |
| `search_files` / `list_files` | Content search (regex) and directory listing (glob) |
| `shell` | Run a command (foreground, or background with a handle) |
| `list_processes` / `read_process_output` / `kill_process` | Manage background processes |
| `web_search` | DuckDuckGo web search |
| `browse` | Fetch-based page reading (navigate / readText / extractText / readHTML) |
| `save_memory` / `recall_memory` / `clear_memory` | A small persistent note store |
| `create_task` / `list_tasks` / `cancel_task` | Background / scheduled command jobs |

### Resources & prompts
* **Resources** — `mlx-serve://status` (live health), `mlx-serve://models` (model
  inventory with capability flags), `mlx-serve://guidance` (model-selection advice).
* **Prompts** — one-click recipes: `portrait`, `poster`, `lofi_track`, `song`, `short_video`.

## Configuration

CLI flags are layered over `MLX_SERVE_*` environment variables (flag wins):

| Flag | Env var | Default |
| --- | --- | --- |
| `--url` / `--base-url` | `MLX_SERVE_URL` | `http://127.0.0.1:11234` |
| `--api-key` | `MLX_SERVE_API_KEY` | *(none)* |
| `--output-dir` | `MLX_SERVE_OUTPUT_DIR` | `./output` |
| `--timeout` | `MLX_SERVE_TIMEOUT` | `1800` (seconds) |
| `--image-model` | `MLX_SERVE_IMAGE_MODEL` | `ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit` |
| `--image-edit-model` | `MLX_SERVE_IMAGE_EDIT_MODEL` | `Runpod/FLUX.2-klein-4B-mflux-4bit` |
| `--tts-model` | `MLX_SERVE_TTS_MODEL` | `mlx-community/Kokoro-82M-v1.0-mlx` |
| `--music-model` | `MLX_SERVE_MUSIC_MODEL` | `mlx-community/musicgen-small-mlx` |
| `--video-model` | `MLX_SERVE_VIDEO_MODEL` | `mlx-community/HunyuanVideo-mlx` |
| `--mesh-model` | `MLX_SERVE_MESH_MODEL` | `mlx-community/trellis-mlx` |
| `--working-dir` | `MLX_SERVE_WORKING_DIR` | current directory |
| `--data-dir` | `MLX_SERVE_DATA_DIR` | `~/.mlx-serve-mcp` |
| `--transport` | `MLX_SERVE_TRANSPORT` | `stdio` |
| `--host` / `--port` | `MLX_SERVE_HOST` / `MLX_SERVE_PORT` | `127.0.0.1` / `8765` |

Generated artifacts land under `--output-dir` (`images/`, `audio/`, `video/`,
`mesh/`); the memory store lives under `--data-dir`.

## Layout

```
src/mlx_serve_mcp/
  server.py        # composition root: builds the FastMCP app
  config.py        # CLI flags over MLX_SERVE_* env vars
  client.py        # async HTTP client for the remote mlx-serve instance
  video.py         # raw frames + PCM -> H.264/AAC MP4 via ffmpeg
  state.py         # process-lifetime state (cwd, processes, tasks, memory)
  prompts.py       # one-click prompt recipes
  resources.py     # live resources (status / models / guidance)
  tools/
    generation.py  # image / edit / speech / music / video / 3d
    files.py       # read / write / edit / search / list
    shell.py       # shell (foreground + background)
    processes.py   # list / read / kill background processes
    web.py         # web_search + browse
    memory.py      # save / recall / clear memory
    tasks.py       # create / list / cancel background jobs
```

## SDK note (v1 pin)

This package pins `mcp>=1.9,<2.0` and uses `mcp.server.fastmcp.FastMCP`. The MCP
Python SDK **v2 removed the bundled `FastMCP`** (it is now `MCPServer`, and the
maintained high-level framework is the standalone
[`fastmcp`](https://gofastmcp.com) package). Upgrading is a one-line import
change — `from fastmcp import FastMCP` — plus repointing the content imports;
the tool / resource / prompt decorators and `run()` are unchanged. The
construction in `server.py` is deliberately defensive so it keeps working
across SDK constructor variance.

## Tests

```bash
uv sync --extra dev
uv run pytest            # unit tests (no network / mlx-serve required)
uv run pytest -m live    # live tests (need a running mlx-serve + ffmpeg)
```
