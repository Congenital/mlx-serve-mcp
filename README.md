# mlx-serve-mcp

**MCP server that turns a remote [mlx-serve](https://github.com/ddalcu/mlx-serve) instance into callable tools** — so any MCP client (Claude Code, Claude Desktop, Cline, ...) on any device can generate images, speech, music, video and 3D meshes through your Mac's `ip:port`.

mlx-serve runs the models natively on Apple Silicon; this bridge speaks MCP on one side and mlx-serve's OpenAI-style media API (`/v1/images`, `/v1/audio`, `/v1/video`, `/v1/3d`) on the other. Nothing is generated locally — your machine only talks HTTP to the server.

```
┌──────────────┐  stdio/MCP   ┌────────────────┐    HTTP     ┌──────────────────┐
│ MCP client   │ ◄──────────► │  mlx-serve-mcp │ ──────────► │ mlx-serve server │
│ (any device) │              │  (this package)│  ip:port    │  (Apple Silicon) │
└──────────────┘              └────────────────┘             └──────────────────┘
```

## Install & run

Requires Python ≥ 3.10. With [uv](https://docs.astral.sh/uv/) installed:

```bash
cd mlx-serve-mcp
uv sync                 # create venv + install deps
uv run mlx-serve-mcp --url 192.168.1.10:11234
```

The URL accepts bare `ip:port` (http is assumed), `host:port`, or a full `http(s)://...` URL.

### Configuration

CLI flags override environment variables:

| Flag | Env var | Default | Meaning |
|------|---------|---------|---------|
| `--url` | `MLX_SERVE_URL` | `http://127.0.0.1:11234` | mlx-serve address |
| `--api-key` | `MLX_SERVE_API_KEY` | *(none)* | Bearer key when the server runs with API-key auth |
| `--output-dir` | `MLX_SERVE_OUTPUT_DIR` | `~/Downloads/mlx-serve-mcp` | Where generated media files are written |
| `--timeout` | `MLX_SERVE_TIMEOUT` | `1800` | HTTP timeout in seconds (video/music can take many minutes) |

## Wire into your MCP client

Claude Code (`.mcp.json` / `claude mcp add`):

```json
{
  "mcpServers": {
    "mlx-serve": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/mlx-serve-mcp",
        "run", "mlx-serve-mcp",
        "--url", "192.168.1.10:11234"
      ]
    }
  }
}
```

Claude Desktop (`claude_desktop_config.json`) uses the same `command`/`args` shape. Add `"env": {"MLX_SERVE_API_KEY": "..."}` if the server requires a key.

## Tools

| Tool | Endpoint | Returns |
|------|----------|---------|
| `health_check` | `GET /health` | reachability text |
| `list_models` | `GET /v1/models` | model ids + capability flags (image/speech/music/video/3d/chat) |
| `load_model(model)` | `POST /v1/load-model` | load into GPU memory (optionally as default) |
| `unload_model(model)` | `POST /v1/unload-model` | free GPU memory |
| `generate_image(prompt, size?, seed?, steps?, model?)` | `POST /v1/images/generations` | inline image + saved PNG path |
| `edit_image(prompt, image_path, mode=edit\|variation, ...)` | same | inline image + saved PNG path |
| `text_to_speech(text, voice?/ref_audio_path?, speed?, seed?)` | `POST /v1/audio/speech` | saved WAV path |
| `generate_music(prompt_style, lyrics?, duration_seconds?, bpm?, task?, src_audio_path?)` | `POST /v1/audio/music-generations` | saved WAV path |
| `generate_video(prompt, num_frames?, width?, height?, turbo?, first_frame_image_path?...)` | `POST /v1/video/generations` | **encoded MP4 path** |
| `generate_3d(image_path, steps?, octree_resolution?, texture?...)` | `POST /v1/3d/generations` | saved GLB path |

Output files land under `<output-dir>/{images,audio,video,mesh}/` with timestamped names; every tool reports absolute paths in its result text.

## Design notes

- **Video**: mlx-serve answers with *raw* RGB8 frame bytes (+ optional PCM s16le track), not an encoded file. This bridge muxes them into H.264/AAC MP4 via ffmpeg — preferring system `ffmpeg`, falling back to the static binary bundled with the [`imageio-ffmpeg`](https://pypi.org/project/imageio-ffmpeg/) dependency, so no separate install is needed.
- **Images** are returned both inline (MCP image content, instant preview) and as saved PNG files.
- **Errors**: mlx-serve's named-400 messages (e.g. `'speed' must be in (0, 5]`) are surfaced verbatim so the calling LLM can self-correct.
- **LoRA fields** are intentionally not exposed: they require `.safetensors` paths on the *server's* disk, which rarely makes sense for remote callers.
- Long generations are just slow HTTP requests here; raise `--timeout` if your clips are ambitious.

## Development

```bash
uv sync
uv run pytest          # unit tests (mocked HTTP, no server required)
uv run mlx-serve-mcp --help
```
