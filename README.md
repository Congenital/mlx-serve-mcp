[![MCP Badge](https://lobehub.com/badge/mcp-full/congenital-mlx-serve-mcp?theme=light)](https://lobehub.com/mcp/congenital-mlx-serve-mcp)

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

### Default models

Each media tool accepts an optional `model` argument. When omitted, the tool
falls back to a configurable default (env var → built-in):

| Env var | Tool | Built-in default |
|---------|------|-----------------|
| `MLX_SERVE_IMAGE_MODEL` | `generate_image` | `Runpod/FLUX.2-klein-4B-mflux-4bit` |
| `MLX_SERVE_IMAGE_EDIT_MODEL` | `edit_image` | `Runpod/FLUX.2-klein-4B-mflux-4bit` |
| `MLX_SERVE_TTS_MODEL` | `text_to_speech` | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` |
| `MLX_SERVE_MUSIC_MODEL` | `generate_music` | `ddalcu/MiniMax-Music3-MLX-Serve-8bit` |
| `MLX_SERVE_VIDEO_MODEL` | `generate_video` | `ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit` |
| `MLX_SERVE_MESH_MODEL` | `generate_3d` | `ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit` |

> **Model recommendation** (based on real-world testing on mlx-serve):
>
> - `ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit` is fast, but its quality is below
>   `Runpod/FLUX.2-klein-4B-mflux-4bit` — in particular, face generation tends
>   to come out distorted. However, it is far more reliable than
>   `mlx-community/flux2-klein-9b-4bit` at rendering text in images, so for
>   text-centric art (posters, typography, signs) rather than portraits,
>   `ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit` is the recommended choice.
> - `ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit` can hit a weight/parameter
>   error on mlx-serve (`Model load failed: MissingMageFlowWeight`), which makes
>   the model unusable.
> - `mlx-community/flux2-klein-9b-4bit` has a similar load-failure issue.
>
> **Bottom line: use `Runpod/FLUX.2-klein-4B-mflux-4bit`** for both
> `generate_image` and `edit_image` — it is the only image model in this group
> that both loads reliably and produces good results (including faces).

Set them in your MCP client config to pin the models you actually have
installed on the server:

```json
{
  "mcpServers": {
    "mlx-serve": {
      "command": "uv",
      "args": ["--directory", "/path/to/mlx-serve-mcp", "run", "mlx-serve-mcp", "--url", "192.168.1.10:11234"],
      "env": {
        "MLX_SERVE_API_KEY": "private",
        "MLX_SERVE_IMAGE_MODEL": "ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit",
        "MLX_SERVE_TTS_MODEL": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
      }
    }
  }
}
```

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
