---
name: mlx-serve-mcp
description: Generate images, speech, music, video and 3D models on a remote Apple Silicon mlx-serve instance via MCP tools. Use when the user wants to create media (images, audio, video, 3D) using locally-running MLX models on a Mac server.
---

# mlx-serve MCP

A stdio bridge that turns a remote **mlx-serve** inference server (running on Apple Silicon) into callable MCP tools for any client (Claude Code, Claude Desktop, Cline, …).

> Nothing is generated locally — your machine only talks HTTP to the mlx-serve server.

## Architecture

```
┌──────────────┐  stdio/MCP  ┌────────────────┐  HTTP  ┌──────────────────┐
│ MCP client   │ ◄──────────► │ mlx-serve-mcp  │ ─────► │ mlx-serve server │
│ (any device) │              │ (this package) │ip:port │ (Apple Silicon)  │
└──────────────┘              └────────────────┘        └──────────────────┘
```

## Installation

Requires Python ≥ 3.10 and `uv`.

```bash
cd mlx-serve-mcp
uv sync   # create venv + install deps
```

Register in your MCP client config:

```json
{
  "mcpServers": {
    "mlx-serve": {
      "command": "uv",
      "args": ["--directory", "/path/to/mlx-serve-mcp", "run", "mlx-serve-mcp", "--url", "192.168.1.10:11234"],
      "env": {
        "MLX_SERVE_URL": "http://192.168.1.10:11234",
        "MLX_SERVE_API_KEY": "your-bearer-token-if-auth-enabled"
      }
    }
  }
}
```

The `--url` flag accepts bare `ip:port` (http is assumed), `host:port`, or a full `http(s)://…` URL.

### Configuration

| Flag | Env var | Default | Meaning |
|------|---------|---------|---------|
| `--url` | `MLX_SERVE_URL` | `http://127.0.0.1:11234` | mlx-serve address |
| `--api-key` | `MLX_SERVE_API_KEY` | *(none)* | Bearer key when the server runs with API-key auth |
| `--output-dir` | `MLX_SERVE_OUTPUT_DIR` | `~/Downloads/mlx-serve-mcp` | Where generated media files are written |
| `--timeout` | `MLX_SERVE_TIMEOUT` | `1800` | HTTP timeout in seconds (video/music can take many minutes) |

## Tools

| Tool | Purpose |
|------|---------|
| `health_check` | Verify the mlx-serve instance is reachable |
| `list_models` | Show all models with capability flags (chat, vision, image, speech, music, video, 3D) |
| `load_model` | Cold-load a model into GPU memory |
| `unload_model` | Free a model's GPU memory |
| `generate_image` | Text → image (PNG) |
| `edit_image` | Image + prompt → edited image |
| `text_to_speech` | Text → speech (WAV) |
| `generate_music` | Style prompt → music (WAV) |
| `generate_video` | Scene prompt → video (MP4) |
| `generate_3d` | Subject image → textured 3D mesh (GLB) |

## Prompts (one-click recipes)

| Prompt | What it does |
|--------|-------------|
| `create_poster` | Text-centric poster/typography (uses Mage-Flow-Turbo) |
| `portrait_photo` | Realistic portrait (uses FLUX.2-klein-4B) |
| `lofi_track` | Lo-fi hip-hop music track |
| `speak_text` | Natural TTS |
| `image_to_3d` | Cutout photo → textured GLB |
| `short_video` | 9-frame preview video (fastest path) |

## Resources (live context)

| Resource | URI | Content |
|----------|-----|---------|
| `models` | `mlx-serve://models` | Live model inventory with capability flags |
| `server_status` | `mlx-serve://status` | Health, version, loaded models |
| `model_guidance` | `mlx-serve://guidance` | Recommended model per tool (based on real-world testing) |

## Model Recommendations

Based on real-world testing on mlx-serve:

| Use case | Model | Why |
|----------|-------|-----|
| Text/posters | `ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit` | Most reliable text rendering |
| Portraits/faces | `Runpod/FLUX.2-klein-4B-mflux-4bit` | Best face quality, loads reliably |
| Image editing | `Runpod/FLUX.2-klein-4B-mflux-4bit` | Facial realism |
| TTS | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | Most natural |
| Music | `ddalcu/MiniMax-Music3-MLX-Serve-8bit` | Best music quality |
| Video | `ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit` | Best video quality |
| 3D | `ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit` | Best mesh quality |

> ⚠️ `Mage-Flow-Edit-Turbo` can hit `MissingMageFlowWeight` — avoid for `edit_image`.
> ⚠️ `mlx-community/flux2-klein-9b-4bit` has a similar load-failure issue — avoid.

## Workflow

1. **Check connectivity**: Call `health_check` first.
2. **Discover models**: Call `list_models` to see what's available.
3. **Generate**: Call the appropriate generation tool with your prompt.
4. **Get the file**: The tool returns an absolute path to the saved file.

## Output Contract

All generated artifacts are written under `output_dir`:

- `images/` — PNG files (also returned inline as MCP image content)
- `audio/` — WAV files
- `video/` — MP4 files (H.264/AAC, muxed via ffmpeg)
- `mesh/` — GLB files

Errors from mlx-serve are surfaced verbatim (e.g. `'speed' must be in (0, 5]`) so the calling LLM can self-correct.

## Development

```bash
uv sync
uv run pytest   # unit tests (mocked HTTP, no server required)
uv run mlx-serve-mcp --help
```