"""MCP server implementation: manual JSON-RPC over stdio.

Every tool maps to one mlx-serve endpoint. The server keeps ONE
:class:`MlxServeClient` for the process lifetime (stdio transport = one session
per process), pointed at the remote mlx-serve instance from the resolved
:class:`~mlx_serve_mcp.config.Config`.

Output contract: every generated artifact is written under ``output_dir``
(``images/``, ``audio/``, ``video/``, ``mesh/`` subfolders) and reported back
as an absolute path so other devices / apps can pick the file up. Images are
additionally returned inline as MCP image content for instant preview.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, get_type_hints

from .client import MlxServeClient, MlxServeError
from .config import Config, DEFAULT_TIMEOUT_SECONDS
from . import prompts as prompts_module
from .video import mux_video_mp4, write_wav

# ── MCP protocol types ─────────────────────────────────────────────────

PROTOCOL_VERSION = "2024-11-05"


class ToolDefinition:
    """A registered MCP tool with its schema and handler."""

    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable[..., Coroutine[Any, Any, str]],
        parameters: dict[str, Any],
    ) -> None:
        self.name = name
        self.description = description
        self.handler = handler
        self.parameters = parameters

    def to_mcp(self) -> dict[str, Any]:
        """Serialize to MCP tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters.get("properties", {}),
                "required": self.parameters.get("required", []),
            },
        }


class McpServer:
    """Minimal MCP server over stdio (JSON-RPC 2.0)."""

    def __init__(self, name: str, instructions: str = "") -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, ToolDefinition] = {}
        self._client: MlxServeClient | None = None
        self._config: Config | None = None

    def init(self, config: Config) -> None:
        """Bind to a config and create the mlx-serve client."""
        config.output_dir.mkdir(parents=True, exist_ok=True)
        self._config = config
        self._client = MlxServeClient(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
        )

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[Callable[..., Coroutine[Any, Any, str]]], Callable[..., Coroutine[Any, Any, str]]]:
        """Decorator to register a tool."""

        def decorator(fn: Callable[..., Coroutine[Any, Any, str]]) -> Callable[..., Coroutine[Any, Any, str]]:
            tool_name = name or fn.__name__
            tool_desc = description or (fn.__doc__ or "").strip()

            # Extract parameters from type hints
            hints = get_type_hints(fn)
            props: dict[str, Any] = {}
            required: list[str] = []
            import inspect

            sig = inspect.signature(fn)
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                ann = hints.get(param_name)
                if ann is str:
                    props[param_name] = {"type": "string"}
                elif ann is int:
                    props[param_name] = {"type": "integer"}
                elif ann is float:
                    props[param_name] = {"type": "number"}
                elif ann is bool:
                    props[param_name] = {"type": "boolean"}
                elif ann is None:
                    props[param_name] = {"type": "null"}
                else:
                    # Union types (e.g. int | None)
                    origin = getattr(ann, "__origin__", None)
                    if origin is not None:
                        args = getattr(ann, "__args__", ())
                        if len(args) == 2 and type(None) in args:
                            non_null = [a for a in args if a is not type(None)][0]
                            if non_null is str:
                                props[param_name] = {"type": "string"}
                            elif non_null is int:
                                props[param_name] = {"type": "integer"}
                            elif non_null is float:
                                props[param_name] = {"type": "number"}
                            elif non_null is bool:
                                props[param_name] = {"type": "boolean"}
                            else:
                                props[param_name] = {}
                        else:
                            props[param_name] = {}
                    else:
                        props[param_name] = {}

                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            self.tools[tool_name] = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                handler=fn,
                parameters={"properties": props, "required": required},
            )
            return fn

        return decorator

    async def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle the MCP initialize request."""
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
                "prompts": {},
            },
            "serverInfo": {
                "name": self.name,
                "version": "0.1.0",
            },
            "instructions": self.instructions,
        }

    async def _handle_tools_list(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """Handle tools/list request."""
        return {
            "tools": [tool.to_mcp() for tool in self.tools.values()],
        }

    async def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool = self.tools[tool_name]
        try:
            result = await tool.handler(**arguments)
            return {
                "content": [{"type": "text", "text": str(result)}],
            }
        except MlxServeError as exc:
            return {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            }
        except Exception as exc:
            return {
                "content": [{"type": "text", "text": f"Error: {type(exc).__name__}: {exc}"}],
                "isError": True,
            }

    async def _handle_prompts_list(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """Handle prompts/list request (ready-made generation recipes)."""
        return {"prompts": prompts_module.list_prompts()}

    async def _handle_prompts_get(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle prompts/get request."""
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str):
            raise ValueError("prompts/get requires a 'name' string")
        return prompts_module.get_prompt(name, arguments)

    async def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch a JSON-RPC request to the appropriate handler."""
        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params")

        if method == "initialize":
            result = await self._handle_initialize(params or {})
        elif method == "tools/list":
            result = await self._handle_tools_list(params)
        elif method == "tools/call":
            result = await self._handle_tools_call(params or {})
        elif method == "prompts/list":
            result = await self._handle_prompts_list(params)
        elif method == "prompts/get":
            result = await self._handle_prompts_get(params or {})
        elif method == "notifications/initialized":
            # Notification: no response needed
            return None
        else:
            # Unknown method: return error
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }

    def run(self) -> None:
        """Run the MCP server over stdio (blocking).

        Reads newline-delimited JSON-RPC messages from stdin and writes
        responses to stdout. Terminates cleanly on EOF (client disconnect)
        or Ctrl-C.
        """
        import asyncio

        async def _run() -> None:
            loop = asyncio.get_running_loop()

            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            await loop.connect_read_pipe(lambda: protocol, sys.stdin)

            while True:
                line = await reader.readline()
                # EOF: the MCP client closed the pipe — exit cleanly.
                if not line:
                    break

                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    request = json.loads(line)
                except json.JSONDecodeError:
                    continue

                try:
                    response = await self._handle_request(request)
                except Exception as exc:  # never let the loop die silently
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {exc}",
                        },
                    }
                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.flush()


# ── module-level server instance ─────────────────────────────────────────

mcp = McpServer(
    "mlx-serve",
    instructions=(
        "Tools for a remote mlx-serve inference server: list/load/unload models and "
        "generate images, speech, music, video and textured 3D meshes. Media files "
        "are saved to a local output directory; paths are returned as absolute paths. "
        "Call health_check first to verify connectivity, then list_models to see which "
        "models are available on this server."
    ),
)


# ── helpers ──────────────────────────────────────────────────────────────


def _get_client() -> MlxServeClient:
    if mcp._client is None:
        raise RuntimeError("mlx-serve-mcp state not initialized")
    return mcp._client


def _get_config() -> Config:
    if mcp._config is None:
        raise RuntimeError("mlx-serve-mcp state not initialized")
    return mcp._config


def _resolve_model(explicit: str | None, cfg_attr: str) -> str | None:
    """Return the explicit model if given, else the configured default.

    ``cfg_attr`` is a ``Config`` field name (e.g. ``"image_model"``).
    """
    if explicit:
        return explicit
    cfg = _get_config()
    return getattr(cfg, cfg_attr)


def _output_path(kind: str, suffix: str) -> Path:
    """Timestamped, collision-safe output path under output_dir/<kind>s/."""
    cfg = _get_config()
    folder = cfg.output_dir / f"{kind}s"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = secrets.token_hex(3)
    return folder / f"{stamp}_{tag}{suffix}"


def _save_file(kind: str, suffix: str, data: bytes) -> Path:
    path = _output_path(kind, suffix)
    path.write_bytes(data)
    return path


def _read_image_file(image_path: str, field: str = "image_path") -> tuple[str, str]:
    """Read a local image/audio file, return (base64, original path)."""
    path = Path(image_path).expanduser()
    if not path.exists():
        raise ValueError(f"{field} file not found: {path}")
    if not path.is_file():
        raise ValueError(f"{field} is not a file: {path}")
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii"), str(path)


def _optional_add(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        payload[key] = value


def _fmt_size(num: float) -> str:
    for unit in ("bytes", "KiB", "MiB", "GiB"):
        if num < 1024 or unit == "GiB":
            return f"{num:.1f} {unit}" if unit != "bytes" else f"{int(num)} {unit}"
        num /= 1024
    return f"{num:.1f} GiB"


def _model_flag(entry: dict[str, Any], key: str) -> bool:
    value = entry.get(key)
    return bool(value) if isinstance(value, bool) else False


# ── management tools ────────────────────────────────────────────────────


@mcp.tool()
async def health_check() -> str:
    """Check connectivity to the remote mlx-serve instance.

    Pings ``GET /health`` on the configured server. Use this first to verify
    that the ip:port is reachable before running any generation.
    """
    client = _get_client()
    cfg = _get_config()
    data = await client.health()
    status = data.get("status", "unknown") if isinstance(data, dict) else "unknown"
    return f"mlx-serve at {cfg.base_url} is reachable (status: {status})"


@mcp.tool()
async def list_models() -> str:
    """List models available on the remote mlx-serve server with their capabilities.

    Returns each model's id plus capability flags: chat/vision, image engine,
    speech (TTS) engine, music backend, video engine, and 3D mesh engine.
    Pick ids from this list for the ``model`` argument of generation tools;
    a media tool without ``model`` uses whatever matching model the server
    has loaded or configured as default.
    """
    entries = await _get_client().list_models()
    if not entries:
        return "No models registered on the server."
    lines: list[str] = []
    for i, entry in enumerate(entries):
        mid = entry.get("id") or entry.get("model") or "?"
        caps_raw = entry.get("capabilities")
        caps = [str(c) for c in caps_raw] if isinstance(caps_raw, list) else []
        # Normalize: a model with both audio+music handles speech AND music.
        if "audio" in caps and "music" in caps:
            caps.remove("audio")
            caps = ["speech+music"] + [c for c in caps if c != "speech"]
        elif "audio" in caps:
            caps = ["speech"] + [c for c in caps if c != "audio"]
        state = entry.get("state")
        loaded = entry.get("loaded")
        status = ""
        if loaded is True or state == "ready":
            status = " [loaded]"
        elif state:
            status = f" [{state}]"
        resident = entry.get("bytes_resident")
        mem = f", {_fmt_size(float(resident))} in memory" if isinstance(resident, (int, float)) and resident > 0 else ""
        cap_str = f" — capabilities: {', '.join(caps)}" if caps else ""
        lines.append(f"{i + 1}. {mid}{status}{mem}{cap_str}")
    header = f"{len(entries)} model(s) on the server:\n"
    return header + "\n".join(lines)


@mcp.tool()
async def load_model(
    model: str,
    make_default: bool = False,
) -> str:
    """Explicitly cold-load a model on the remote mlx-serve server into GPU memory.

    Args:
        model: Model id as returned by list_models (or an absolute path to a
            model directory on the SERVER machine).
        make_default: Also promote it to the server-wide default model. Leave
            false for side-loaded media models so chat traffic keeps its
            current default.
    """
    result = await _get_client().load_model(model, make_default=make_default)
    entry = result.get("model") if isinstance(result, dict) else None
    state = entry.get("state", "loaded") if isinstance(entry, dict) else "loaded"
    resident = entry.get("bytes_resident") if isinstance(entry, dict) else None
    extra = ""
    if isinstance(resident, (int, float)) and resident > 0:
        extra = f", resident {_fmt_size(float(resident))}"
    note = " and set as default" if make_default else ""
    return f"Model '{model}' is now {state}{extra}{note}."


@mcp.tool()
async def unload_model(model: str) -> str:
    """Free a model's GPU memory on the remote mlx-serve server.

    The model stays registered (it can be reloaded later); only its resident
    weights are evicted. Useful after heavy media generation to reclaim
    unified memory.
    """
    await _get_client().unload_model(model)
    return f"Model '{model}' unloaded."


# ── image tools ─────────────────────────────────────────────────────────


@mcp.tool()
async def generate_image(
    prompt: str,
    model: str | None = None,
    size: str | None = None,
    seed: int | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    guidance_scale: float | None = None,
) -> str:
    """Generate an image from a text prompt on the remote server.

    The image is saved as a local PNG file; the returned text includes the
    absolute path of the saved file.

    Args:
        prompt: Text description of the image to generate.
        model: Model id (from list_models). Defaults to MLX_SERVE_IMAGE_MODEL
            (or the built-in default if unset).
        size: Output dimensions, e.g. "512x512" or "1024x1024".
        seed: Optional random seed for reproducibility.
        steps: Sampling steps (backend-specific).
        cfg_scale: Classifier-free guidance scale (SDXL/Flux style).
        guidance_scale: Guidance scale (MAGE-Flow style).
    """
    payload: dict[str, Any] = {"prompt": prompt}
    _optional_add(payload, "model", _resolve_model(model, "image_model"))
    _optional_add(payload, "size", size)
    _optional_add(payload, "seed", seed)
    _optional_add(payload, "steps", steps)
    _optional_add(payload, "cfg_scale", cfg_scale)
    _optional_add(payload, "guidance_scale", guidance_scale)

    items = await _get_client().generate_image(payload)
    if not items:
        return "No image data returned by server."
    b64 = items[0].get("b64_json")
    if not b64:
        return "Server returned no image data."
    png_bytes = base64.b64decode(b64)
    saved = _save_file("image", ".png", png_bytes)
    return f"Image generated: {saved} ({_fmt_size(len(png_bytes))})"


@mcp.tool()
async def edit_image(
    image_path: str,
    prompt: str,
    model: str | None = None,
    size: str | None = None,
    strength: float | None = None,
    seed: int | None = None,
) -> str:
    """Edit an existing image with a text prompt.

    Args:
        image_path: Absolute or ~-relative path to the source image (PNG/JPEG).
        prompt: Description of the desired edit.
        model: Model id. Defaults to MLX_SERVE_IMAGE_EDIT_MODEL.
        size: Output dimensions.
        strength: Denoising strength 0..1 (how much to change the image).
        seed: Optional random seed.
    """
    img_b64, _ = _read_image_file(image_path, field="image_path")
    payload: dict[str, Any] = {"image": img_b64, "prompt": prompt}
    _optional_add(payload, "model", _resolve_model(model, "image_edit_model"))
    _optional_add(payload, "size", size)
    _optional_add(payload, "strength", strength)
    _optional_add(payload, "seed", seed)

    items = await _get_client().generate_image(payload)
    if not items:
        return "No image data returned by server."
    b64 = items[0].get("b64_json")
    if not b64:
        return "Server returned no image data."
    png_bytes = base64.b64decode(b64)
    saved = _save_file("image", ".png", png_bytes)
    return f"Image edited: {saved} ({_fmt_size(len(png_bytes))})"


# ── speech tool ─────────────────────────────────────────────────────────


@mcp.tool()
async def text_to_speech(
    text: str,
    model: str | None = None,
    voice: str | None = None,
    speed: float | None = None,
) -> str:
    """Synthesize speech from text; returns a local WAV file path.

    Args:
        text: The text to speak.
        model: TTS model id. Defaults to MLX_SERVE_TTS_MODEL.
        voice: Voice name or id (backend-specific).
        speed: Playback speed multiplier (0.25..4.0, default 1.0).
    """
    if speed is not None and not (0.25 <= speed <= 4.0):
        raise ValueError(f"speed must be in [0.25, 4.0], got {speed}")
    payload: dict[str, Any] = {"text": text}
    _optional_add(payload, "model", _resolve_model(model, "tts_model"))
    _optional_add(payload, "voice", voice)
    _optional_add(payload, "speed", speed)

    wav_bytes = await _get_client().generate_speech(payload)
    saved = _save_file("speech", ".wav", wav_bytes)
    return f"Speech generated: {saved} ({_fmt_size(len(wav_bytes))})"


# ── music tool ──────────────────────────────────────────────────────────


@mcp.tool()
async def generate_music(
    prompt_style: str,
    model: str | None = None,
    lyrics: str | None = None,
    instrumental: bool | None = None,
    duration_seconds: int | None = None,
    bpm: int | None = None,
    keyscale: str | None = None,
    timesignature: str | None = None,
    vocal_language: str | None = None,
    task: str | None = None,
    src_audio_path: str | None = None,
    cover_strength: float | None = None,
    seed: int | None = None,
    steps: int | None = None,
) -> str:
    """Generate a music track from a style prompt; returns a local WAV path.

    Args:
        prompt_style: Style description, e.g. "lo-fi hip hop, mellow piano".
        model: Music model id. Defaults to MLX_SERVE_MUSIC_MODEL.
        lyrics: Optional lyrics for vocal tracks.
        instrumental: True for no vocals (default true for text2music).
        duration_seconds: Target length 10..600 (default 60).
        bpm: Tempo, e.g. 120.
        keyscale: Musical key, e.g. "C major" / "E minor".
        timesignature: Time signature, e.g. "4/4".
        vocal_language: Vocal language code for sung lyrics, e.g. "en", "zh".
        task: "text2music" (default) | "cover" | "complete".
        src_audio_path: Local WAV (10-600 s) source for cover/complete tasks;
            its length becomes the output length.
        cover_strength: 0..1 how strongly to follow the cover source (default 1).
        seed: Optional seed for reproducibility.
        steps: Optional sampling steps (music3 backend only).
    """
    if duration_seconds is not None and not (10 <= duration_seconds <= 600):
        raise ValueError(f"duration_seconds must be in [10, 600], got {duration_seconds}")
    if task is not None and task not in ("text2music", "cover", "complete"):
        raise ValueError(f"task must be 'text2music', 'cover' or 'complete', got {task!r}")
    payload: dict[str, Any] = {"prompt": prompt_style}
    _optional_add(payload, "model", _resolve_model(model, "music_model"))
    _optional_add(payload, "lyrics", lyrics)
    _optional_add(payload, "instrumental", instrumental)
    _optional_add(payload, "duration_seconds", duration_seconds)
    _optional_add(payload, "bpm", bpm)
    _optional_add(payload, "keyscale", keyscale)
    _optional_add(payload, "timesignature", timesignature)
    _optional_add(payload, "vocal_language", vocal_language)
    _optional_add(payload, "task", task)
    _optional_add(payload, "cover_strength", cover_strength)
    _optional_add(payload, "seed", seed)
    _optional_add(payload, "steps", steps)
    if src_audio_path:
        if task is None:
            payload["task"] = "cover"
        elif task == "text2music":
            raise ValueError("src_audio_path requires task='cover' or 'complete'")
        src_b64, _ = _read_image_file(src_audio_path, field="src_audio_path")
        payload["src_audio"] = src_b64
    wav_bytes = await _get_client().generate_music(payload)
    saved = _save_file("music", ".wav", wav_bytes)
    return f"Music generated ({_fmt_size(len(wav_bytes))} WAV), saved to {saved}"


# ── video tool ──────────────────────────────────────────────────────────


@mcp.tool()
async def generate_video(
    prompt: str,
    model: str | None = None,
    num_frames: int | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    turbo: bool | None = None,
    seed: int | None = None,
    cfg_scale: float | None = None,
    first_frame_image_path: str | None = None,
    last_frame_image_path: str | None = None,
    audio_path: str | None = None,
) -> str:
    """Generate a short video from a text prompt; returns a local MP4 path.

    mlx-serve answers with raw frames (+ optional soundtrack); this tool
    encodes them into an H.264 MP4 (AAC audio when present) locally via
    ffmpeg, so the returned file is playable anywhere. Generation is slow —
    minutes per clip depending on frame count and resolution.

    Args:
        prompt: Scene description for the video.
        model: Video model id. Defaults to MLX_SERVE_VIDEO_MODEL.
        num_frames: Frame count. LTX backends use an 8N+1 ladder (default 9;
            e.g. 9/25/33/49/57/81...); MiniMax-H3 uses 17k+5 (default 56).
        width: Pixel width (defaults: LTX 384, H3 256). Two-stage pipelines
            need both dimensions divisible by 64.
        height: Pixel height (defaults: LTX/H3 256).
        steps: Sampling steps (backend-specific defaults).
        turbo: Use the distilled 4-step turbo path where the model pack
            provides it.
        seed: Optional seed.
        cfg_scale: Guidance scale.
        first_frame_image_path: Optional local image to condition the first frame.
        last_frame_image_path: Optional local image to condition the final frame.
        audio_path: Optional local WAV to mix as soundtrack (must match frame
            duration).
    """
    payload: dict[str, Any] = {"prompt": prompt}
    _optional_add(payload, "model", _resolve_model(model, "video_model"))
    _optional_add(payload, "num_frames", num_frames)
    _optional_add(payload, "width", width)
    _optional_add(payload, "height", height)
    _optional_add(payload, "steps", steps)
    _optional_add(payload, "turbo", turbo)
    _optional_add(payload, "seed", seed)
    _optional_add(payload, "cfg_scale", cfg_scale)
    if first_frame_image_path:
        b64, _ = _read_image_file(first_frame_image_path, field="first_frame_image_path")
        payload["first_frame_image"] = b64
    if last_frame_image_path:
        b64, _ = _read_image_file(last_frame_image_path, field="last_frame_image_path")
        payload["last_frame_image"] = b64
    if audio_path:
        b64, _ = _read_image_file(audio_path, field="audio_path")
        payload["audio"] = b64

    result = await _get_client().generate_video(payload)
    out = _output_path("video", ".mp4")
    mux_video_mp4(
        rgb_bytes=result.rgb_bytes,
        width=result.width,
        height=result.height,
        fps=result.fps,
        out_path=out,
        audio_pcm_s16le=result.audio_pcm_s16le,
        audio_sample_rate=result.audio_sample_rate,
        audio_channels=result.audio_channels,
    )
    track = "with audio" if result.audio_pcm_s16le else "silent"
    return (
        f"Video generated: {result.frames} frames @ {result.fps} fps, "
        f"{result.width}x{result.height}, {track}; MP4 saved to {out}"
    )


# ── 3D mesh tool ────────────────────────────────────────────────────────


@mcp.tool()
async def generate_3d(
    image_path: str,
    model: str | None = None,
    steps: int | None = None,
    octree_resolution: int | None = None,
    guidance_scale: float | None = None,
    texture: bool | None = None,
    texture_steps: int | None = None,
    seed: int | None = None,
) -> str:
    """Generate a textured 3D mesh (GLB) from a subject image on the remote server.

    The image should be a clean cutout of the subject with real alpha
    transparency (an opaque photo also works — the server composites it on
    white). Returns a GLB file path you can open in Blender / three.js /
    Quick Look.

    Args:
        image_path: Absolute or ~-relative path to a local PNG/JPEG of the subject.
        model: 3D mesh model id. Defaults to MLX_SERVE_MESH_MODEL.
        steps: Shape sampling steps (default 30).
        octree_resolution: Mesh grid resolution in [64, 512] (default 256).
        guidance_scale: Shape guidance in [0, 20] (default 5).
        texture: Request the texture-paint stage for a textured GLB. Requires
            the paint weights to be installed server-side, else a named error.
        texture_steps: Texture painting steps in [1, 100] when texture=true.
        seed: Optional seed.
    """
    if octree_resolution is not None and not (64 <= octree_resolution <= 512):
        raise ValueError(f"octree_resolution must be in [64, 512], got {octree_resolution}")
    if guidance_scale is not None and not (0.0 <= guidance_scale <= 20.0):
        raise ValueError(f"guidance_scale must be in [0, 20], got {guidance_scale}")
    if texture_steps is not None and not (1 <= texture_steps <= 100):
        raise ValueError(f"texture_steps must be in [1, 100], got {texture_steps}")
    img_b64, img_path = _read_image_file(image_path, field="image_path")
    payload: dict[str, Any] = {"image": img_b64}
    _optional_add(payload, "model", _resolve_model(model, "mesh_model"))
    _optional_add(payload, "steps", steps)
    _optional_add(payload, "octree_resolution", octree_resolution)
    _optional_add(payload, "guidance_scale", guidance_scale)
    _optional_add(payload, "texture", texture)
    _optional_add(payload, "texture_steps", texture_steps)
    _optional_add(payload, "seed", seed)
    glb_bytes = await _get_client().generate_mesh(payload)
    saved = _save_file("mesh", ".glb", glb_bytes)
    return f"3D mesh generated ({_fmt_size(len(glb_bytes))} GLB) from {img_path}, saved to {saved}"


# ── factory ──────────────────────────────────────────────────────────────


def create_server(config: Config) -> McpServer:
    """Initialize the module-level server against ``config`` and return it."""
    mcp.init(config)
    return mcp
