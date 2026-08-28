"""Async HTTP client for a remote mlx-serve instance.

Mirrors the wire contract implemented by mlx-serve's ``src/server.zig`` and
``src/gen.zig``:

* ``GET  /health``                     -> ``{"status":"ok"}``
* ``GET  /v1/models``                  -> OpenAI list, entries carry capability flags
* ``POST /v1/load-model``              -> ``{"model":"<id>","default":bool?}``
* ``POST /v1/unload-model``            -> ``{"model":"<id>"}``
* ``POST /v1/images/generations``      -> ``{"created":0,"data":[{"b64_json":...}]}`` (PNG)
* ``POST /v1/images/edits``           -> ``{"created":0,"data":[{"b64_json":...}]}`` (PNG)
* ``POST /v1/audio/speech``           -> raw ``audio/wav`` bytes
* ``POST /v1/audio/music-generations``-> raw ``audio/wav`` bytes
* ``POST /v1/video/generations``      -> ``{"frames","height","width","fps","format":"rgb8",
                                          "data":"<b64>","audio_sample_rate"?,"audio_channels"?,
                                          "audio_format":"pcm_s16le","audio_data"?}``
* ``POST /v1/3d/generations``         -> ``{"created":0,"format":"glb","data":"<b64>"}``

Errors come in two shapes: ``{"error":{"message":...}}`` and
``{"error":{"type":...,"message":...}}``; both are surfaced as
:class:`MlxServeError`.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import httpx

USER_AGENT = "mlx-serve-mcp/0.2"


class MlxServeError(RuntimeError):
    """A non-2xx (or malformed) response from the mlx-serve instance."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class VideoResult:
    """Decoded ``/v1/video/generations`` payload: raw RGB8 frames + optional PCM."""

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
        return text[:500] if text else (response.reason_phrase or "request failed")
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        msg = err.get("message") or err.get("type") or "error"
        typ = err.get("type")
        return f"{typ}: {msg}" if typ and err.get("message") else str(msg)
    if isinstance(err, str):
        return err
    return response.text[:500] or response.reason_phrase or "request failed"


def _b64_to_bytes(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        return value
    return base64.b64decode(value)


class MlxServeClient:
    """A thin async client for one remote mlx-serve instance.

    One instance is kept for the life of the MCP server process (the stdio
    transport is one session per process). Generation requests may block for
    many minutes, so the read timeout is long while the connect timeout stays
    short to fail fast on an unreachable host.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 1800.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, audio/wav, application/octet-stream",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        timeout = httpx.Timeout(timeout_seconds, connect=10.0)
        self._client = client or httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout)
        self._owns_client = client is None

    async def __aenter__(self) -> "MlxServeClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ── low-level helpers ──────────────────────────────────────────────────

    async def _request(self, method: str, path: str, *, json: Any | None = None) -> httpx.Response:
        try:
            response = await self._client.request(method, path, json=json)
        except httpx.HTTPError as exc:
            raise MlxServeError(f"could not reach {self.base_url}{path}: {exc}") from exc
        if response.status_code >= 400:
            raise MlxServeError(extract_error_message(response), status_code=response.status_code)
        return response

    async def _request_json(self, method: str, path: str, *, json: Any | None = None) -> dict[str, Any]:
        response = await self._request(method, path, json=json)
        try:
            return response.json()
        except ValueError as exc:
            raise MlxServeError(f"malformed JSON from {path}: {response.text[:200]!r}") from exc

    async def _request_bytes(self, method: str, path: str, *, json: Any | None = None) -> bytes:
        response = await self._request(method, path, json=json)
        return response.content

    # ── management endpoints ───────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """``GET /health`` — cheap liveness probe (open even when key auth is on)."""
        return await self._request_json("GET", "/health")

    async def list_models(self) -> list[dict[str, Any]]:
        """``GET /v1/models`` — the OpenAI-style model list with capability flags."""
        data = await self._request_json("GET", "/v1/models")
        return data.get("data", []) if isinstance(data, dict) else []

    async def load_model(self, model: str, default: bool = False) -> dict[str, Any]:
        """``POST /v1/load-model`` — load (and optionally set default) a model."""
        return await self._request_json("POST", "/v1/load-model", json={"model": model, "default": default})

    async def unload_model(self, model: str) -> dict[str, Any]:
        """``POST /v1/unload-model`` — free a model's memory."""
        return await self._request_json("POST", "/v1/unload-model", json={"model": model})

    # ── image ──────────────────────────────────────────────────────────────

    async def generate_image(
        self,
        prompt: str,
        model: str,
        size: str = "1024x1024",
        n: int = 1,
        seed: int | None = None,
    ) -> list[bytes]:
        """``POST /v1/images/generations`` -> PNG bytes (one per requested image)."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": n,
            "response_format": "b64_json",
        }
        if seed is not None:
            payload["seed"] = seed
        data = await self._request_json("POST", "/v1/images/generations", json=payload)
        out: list[bytes] = []
        for entry in data.get("data", []):
            b64 = entry.get("b64_json")
            if b64:
                out.append(_b64_to_bytes(b64))
        if not out:
            raise MlxServeError(f"image generation returned no data: {str(data)[:200]}")
        return out

    async def edit_image(
        self,
        image_b64: str,
        prompt: str,
        model: str,
        size: str = "1024x1024",
        strength: float = 0.6,
        seed: int | None = None,
    ) -> bytes:
        """``POST /v1/images/edits`` -> edited PNG bytes."""
        payload: dict[str, Any] = {
            "model": model,
            "image": image_b64,
            "prompt": prompt,
            "size": size,
            "strength": strength,
            "response_format": "b64_json",
        }
        if seed is not None:
            payload["seed"] = seed
        data = await self._request_json("POST", "/v1/images/edits", json=payload)
        for entry in data.get("data", []):
            b64 = entry.get("b64_json")
            if b64:
                return _b64_to_bytes(b64)
        raise MlxServeError(f"image edit returned no data: {str(data)[:200]}")

    # ── audio ──────────────────────────────────────────────────────────────

    async def generate_speech(
        self,
        text: str,
        model: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> bytes:
        """``POST /v1/audio/speech`` -> raw WAV bytes."""
        payload: dict[str, Any] = {"model": model, "input": text, "response_format": "wav"}
        if voice is not None:
            payload["voice"] = voice
        if speed is not None:
            payload["speed"] = speed
        return await self._request_bytes("POST", "/v1/audio/speech", json=payload)

    async def generate_music(
        self,
        prompt: str,
        model: str,
        lyrics: str | None = None,
        duration_seconds: int | None = None,
        bpm: int | None = None,
        keyscale: str | None = None,
        time_signature: str | None = None,
        vocal_language: str | None = None,
    ) -> bytes:
        """``POST /v1/audio/music-generations`` -> raw WAV bytes."""
        payload: dict[str, Any] = {"model": model, "prompt": prompt}
        for key, value in (
            ("lyrics", lyrics),
            ("duration_seconds", duration_seconds),
            ("bpm", bpm),
            ("keyscale", keyscale),
            ("time_signature", time_signature),
            ("vocal_language", vocal_language),
        ):
            if value is not None:
                payload[key] = value
        return await self._request_bytes("POST", "/v1/audio/music-generations", json=payload)

    # ── video ──────────────────────────────────────────────────────────────

    async def generate_video(
        self,
        prompt: str,
        model: str,
        size: str | None = None,
        seconds: int | None = None,
        fps: int | None = None,
    ) -> VideoResult:
        """``POST /v1/video/generations`` -> decoded :class:`VideoResult`."""
        payload: dict[str, Any] = {"model": model, "prompt": prompt}
        for key, value in (("size", size), ("seconds", seconds), ("fps", fps)):
            if value is not None:
                payload[key] = value
        data = await self._request_json("POST", "/v1/video/generations", json=payload)
        return VideoResult(
            frames=int(data.get("frames", 0)),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            fps=int(data.get("fps", 0) or 0),
            rgb_bytes=_b64_to_bytes(data.get("data", "")),
            audio_pcm_s16le=_b64_to_bytes(data["audio_data"]) if data.get("audio_data") else None,
            audio_sample_rate=data.get("audio_sample_rate"),
            audio_channels=data.get("audio_channels"),
        )

    # ── 3d ─────────────────────────────────────────────────────────────────

    async def generate_mesh(
        self,
        prompt: str,
        model: str,
        **extra: Any,
    ) -> bytes:
        """``POST /v1/3d/generations`` -> raw GLB bytes."""
        payload: dict[str, Any] = {"model": model, "prompt": prompt, **extra}
        data = await self._request_json("POST", "/v1/3d/generations", json=payload)
        b64 = data.get("data")
        if not b64:
            raise MlxServeError(f"3d generation returned no data: {str(data)[:200]}")
        return _b64_to_bytes(b64)