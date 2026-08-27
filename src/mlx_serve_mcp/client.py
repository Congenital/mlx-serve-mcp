"""Async HTTP client for a remote mlx-serve instance.

Mirrors the wire contract implemented by mlx-serve's ``src/server.zig`` and
``src/gen.zig``:

* ``GET  /health``                    -> ``{"status":"ok"}``
* ``GET  /v1/models``                 -> OpenAI list, entries carry capability flags
* ``POST /v1/load-model``             -> ``{"model":"<id>","default":bool?}``
* ``POST /v1/unload-model``           -> ``{"model":"<id>"}``
* ``POST /v1/images/generations``     -> ``{"created":0,"data":[{"b64_json":...}]}`` (PNG)
* ``POST /v1/audio/speech``           -> raw ``audio/wav`` bytes
* ``POST /v1/audio/music-generations``-> raw ``audio/wav`` bytes
* ``POST /v1/video/generations``      -> ``{"frames","height","width","fps",
    "format":"rgb8","data":"<b64>","audio_sample_rate"?,"audio_channels"?,
    "audio_format":"pcm_s16le","audio_data"?}``
* ``POST /v1/3d/generations``         -> ``{"created":0,"format":"glb","data":"<b64>"}``

Errors arrive as JSON bodies in two shapes (both produced by the server):
``{"error":{"message":"..."}}`` and the OpenAI-style
``{"error":{"type":...,"message":...}}``. Both are unwrapped into
:class:`MlxServeError`.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

import httpx

USER_AGENT = "mlx-serve-mcp"


class MlxServeError(Exception):
    """A request reached the server but was rejected (or the response broke)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class MlxServeConnectionError(MlxServeError):
    """The server could not be reached at all (DNS, refused, timeout)."""


@dataclass(frozen=True)
class VideoResult:
    """Decoded `/v1/video/generations` payload: raw RGB8 frames + optional PCM."""

    frames: int
    width: int
    height: int
    fps: int
    rgb_bytes: bytes  # frames * height * width * 3, row-major per frame
    audio_pcm_s16le: bytes | None
    audio_sample_rate: int | None
    audio_channels: int | None


def extract_error_message(response: httpx.Response) -> str:
    """Pull a human-readable message out of either error body shape."""
    try:
        data = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return text[:500] if text else response.reason_phrase or "request failed"
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or json.dumps(err, ensure_ascii=False))
    if isinstance(err, str) and err:
        return err
    if isinstance(data, dict) and "message" in data:
        return str(data["message"])
    return json.dumps(data, ensure_ascii=False)[:500]


def decode_b64(data: str | bytes) -> bytes:
    if isinstance(data, str):
        data = data.encode("ascii")
    try:
        return base64.b64decode(data, validate=False)
    except Exception as exc:  # binascii.Error subclasses ValueError
        raise MlxServeError(f"server returned invalid base64 payload: {exc}") from exc

class MlxServeClient:
    """Thin async wrapper over one mlx-serve instance."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 1800.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, audio/wav, octet-stream",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # Generation requests block for minutes; only the connect phase is short.
        timeout = httpx.Timeout(timeout_seconds, connect=10.0)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "MlxServeClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ── low-level helpers ────────────────────────────────────────────────

    async def _send(
        self, method: str, path: str, json_body: dict[str, Any] | None
    ) -> httpx.Response:
        try:
            return await self._client.request(method, path, json=json_body)
        except httpx.HTTPError as exc:
            raise MlxServeConnectionError(
                f"cannot reach mlx-serve at {self._client.base_url}{path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    async def _request_json(
        self, method: str, path: str, json_body: dict[str, Any] | None = None
    ) -> Any:
        response = await self._send(method, path, json_body)
        if response.status_code >= 400:
            raise MlxServeError(
                extract_error_message(response), status_code=response.status_code
            )
        try:
            return response.json()
        except ValueError as exc:
            raise MlxServeError(
                f"expected JSON from {method} {path}, got "
                f"{response.headers.get('content-type', 'unknown content type')}",
                status_code=response.status_code,
            ) from exc

    # ── management endpoints ─────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """GET /health — cheap liveness probe (open even when key auth is on)."""
        return await self._request_json("GET", "/health")

    async def list_models(self) -> list[dict[str, Any]]:
        """GET /v1/models — every registry entry with its capability flags."""
        data = await self._request_json("GET", "/v1/models")
        models = data.get("data") if isinstance(data, dict) else None
        if not isinstance(models, list):
            raise MlxServeError("unexpected /v1/models response shape")
        return models

    async def load_model(self, model: str, *, make_default: bool = False) -> dict[str, Any]:
        """POST /v1/load-model — cold-load a model; strict 404 on unknown ids."""
        body: dict[str, Any] = {"model": model}
        if make_default:
            body["default"] = True
        return await self._request_json("POST", "/v1/load-model", body)

    async def unload_model(self, model: str) -> dict[str, Any]:
        """POST /v1/unload-model — free resident GPU state (stub stays registered)."""
        return await self._request_json("POST", "/v1/unload-model", {"model": model})

    # ── media generation ─────────────────────────────────────────────────

    async def generate_image(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """POST /v1/images/generations — returns the parsed ``data`` array.

        Each entry looks like ``{"b64_json": "<png bytes base64>"}``.
        """
        data = await self._request_json("POST", "/v1/images/generations", payload)
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise MlxServeError("unexpected /v1/images/generations response shape")
        return items

    async def generate_speech(self, payload: dict[str, Any]) -> bytes:
        """POST /v1/audio/speech — non-streaming answer is a WAV byte string."""
        return await self._expect_wav("/v1/audio/speech", payload)

    async def generate_music(self, payload: dict[str, Any]) -> bytes:
        """POST /v1/audio/music-generations — same binary contract as speech."""
        return await self._expect_wav("/v1/audio/music-generations", payload)

    async def _expect_wav(self, path: str, payload: dict[str, Any]) -> bytes:
        response = await self._send("POST", path, payload)
        if response.status_code >= 400:
            raise MlxServeError(
                extract_error_message(response), status_code=response.status_code
            )
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("audio/"):
            return response.content
        # Defensive: a 200 that is not audio means the server changed contract.
        raise MlxServeError(
            f"expected audio/wav from {path}, got {content_type or 'no content-type'}",
            status_code=response.status_code,
        )

    async def generate_video(self, payload: dict[str, Any]) -> VideoResult:
        """POST /v1/video/generations — decode RGB8 frames (+ optional PCM track)."""
        data = await self._request_json("POST", "/v1/video/generations", payload)
        if not isinstance(data, dict):
            raise MlxServeError("video response was not a JSON object")
        if data.get("format") != "rgb8":
            raise MlxServeError(
                f"video response format is {data.get('format')!r}, expected 'rgb8'"
            )
        try:
            frames = int(data["frames"])
            width = int(data["width"])
            height = int(data["height"])
            fps = int(data["fps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MlxServeError(f"video response missing geometry fields: {exc}") from exc
        rgb = decode_b64(data.get("data", ""))
        expected = frames * height * width * 3
        if len(rgb) != expected:
            raise MlxServeError(
                f"video payload size mismatch: got {len(rgb)} RGB bytes, "
                f"expected {expected} for {frames}f {width}x{height}"
            )
        audio_pcm: bytes | None = None
        sample_rate = channels = None
        if data.get("audio_data"):
            audio_pcm = decode_b64(data["audio_data"])
            sample_rate = int(data.get("audio_sample_rate") or 48000)
            channels = int(data.get("audio_channels") or 2)
        return VideoResult(
            frames=frames,
            width=width,
            height=height,
            fps=fps,
            rgb_bytes=rgb,
            audio_pcm_s16le=audio_pcm,
            audio_sample_rate=sample_rate,
            audio_channels=channels,
        )

    async def generate_mesh(self, payload: dict[str, Any]) -> bytes:
        """POST /v1/3d/generations — returns decoded GLB bytes."""
        data = await self._request_json("POST", "/v1/3d/generations", payload)
        if not isinstance(data, dict):
            raise MlxServeError("mesh response was not a JSON object")
        fmt = data.get("format", "glb")
        if fmt != "glb":
            raise MlxServeError(f"mesh response format is {fmt!r}, expected 'glb'")
        glb = decode_b64(data.get("data", ""))
        if not glb:
            raise MlxServeError("mesh response carried an empty GLB payload")
        return glb
