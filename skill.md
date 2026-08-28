---
name: mlx-serve-mcp
description: Generate images, speech, music, video and 3D models on a remote Apple Silicon mlx-serve instance via MCP tools, plus a full local toolset (files, shell, background processes, web, memory, scheduled tasks). Use when the user wants to create media with MLX models on a Mac server, or to drive the local machine (files, commands, web) through one MCP server.
---

# mlx-serve MCP

An MCP server (official `mcp` SDK / FastMCP, over stdio by default) that gives a client two
capabilities at once:

1. **Remote generation** — a bridge to an [mlx-serve](https://github.com/ddalcu/mlx-serve)
   inference server on Apple Silicon: images, image edits, speech, music, video, and 3D.
2. **Local agent tools** — files, shell, background processes, web search/browse, persistent
   memory, and scheduled tasks, running on the machine that hosts the server.

## When to use

* The user wants to **create media** (image / audio / video / 3D) using MLX models running on a
  Mac — use the `generate_*` / `edit_image` tools.
* The user wants to **read or edit files**, **run commands**, **search the web**, **remember
  facts**, or **schedule a recurring job** on the host — use the local tools.

## Tools

### Generation (remote mlx-serve)
`generate_image`, `edit_image`, `generate_speech`, `generate_music`, `generate_video`,
`generate_3d`. Every tool streams its artifact straight to a file under the
  output dir and returns only a text summary with the saved path (no inline
  base64 payloads).
the configured `--output-dir`.

### Local
`read_file`, `write_file`, `edit_file`, `search_files`, `list_files`, `shell` (foreground or
background), `list_processes`, `read_process_output`, `kill_process`, `web_search`, `browse`,
`save_memory`, `recall_memory`, `clear_memory`, `create_task`, `list_tasks`, `cancel_task`.

Relative paths resolve against the server's working directory (`--working-dir`).

## Resources & prompts

* **Resources** — `mlx-serve://status` (live health), `mlx-serve://models` (inventory +
  capability flags), `mlx-serve://guidance` (model-selection advice).
* **Prompts** — one-click recipes: `portrait`, `poster`, `lofi_track`, `song`, `short_video`.

## Model guidance

* **Text in images** (posters, typography): `ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit` (default for
  `generate_image`) — far more reliable for rendered text.
* **Faces / photo-real**: `Runpod/FLUX.2-klein-4B-mflux-4bit` (default for `edit_image`) — the
  best general-purpose image model and the only one that renders faces well.
* Check the live inventory and capability flags via the `mlx-serve://models` resource before
  picking a non-default model.

## Configuration

CLI flags layer over `MLX_SERVE_*` env vars (flag wins). Key ones: `--url` / `MLX_SERVE_URL`
(default `http://127.0.0.1:11234`), `--api-key` / `MLX_SERVE_API_KEY`, `--output-dir`,
`--timeout`, per-capability `--*-model`, `--working-dir`, `--data-dir`, `--transport`
(`stdio` | `sse` | `streamable-http`).

## Notes

* `browse` is fetch-based: `navigate` / `readText` / `extractText` / `readHTML` work without a
  browser; `click` / `executeJS` / `screenshot` require a real browser backend and return a clear
  notice rather than hanging.
* `create_task` runs a **command job** in the background (optionally on a schedule); it is not an
  LLM agent — there is no model backend inside the MCP server.
* The server pins `mcp>=1.9,<2.0` (the SDK v2 line removed the bundled `FastMCP`); upgrading to
  the standalone `fastmcp` package is a one-line import change.