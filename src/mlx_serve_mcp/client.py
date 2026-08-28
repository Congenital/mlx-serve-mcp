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

import asyncio
import base64
import mimetypes
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from . import video as video_mod

USER_AGENT = "mlx-serve-mcp/0.2"


class MlxServeError(RuntimeError):
    """A non-2xx (or malformed) response from the mlx-serve instance."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


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


def _save_to_file(path: str | Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` (creating parent dirs) and return the path."""
    p = Path(str(path)).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _save_b64_or_raw(response: httpx.Response, path: str | Path, what: str) -> Path:
    """Persist a generation response to ``path``.

    Handles both wire shapes the remote instance may use: the OpenAI-style
    JSON envelope ``{"data": [{"b64_json": ...}]}`` (decode the first entry
    and save it) or a raw binary body like ``image/png`` (save it verbatim).
    We sniff the body itself: some deployments omit or mislabel the
    ``Content-Type`` header, so trusting the header alone would save the
    JSON envelope as a bogus ``.png``.
    """
    body = response.content
    if body[:1] == b"{" or b"b64_json" in body[:200]:
        data = response.json()
        for entry in data.get("data", []):
            b64 = entry.get("b64_json")
            if b64:
                return _save_to_file(path, _b64_to_bytes(b64))
        raise MlxServeError(f"{what} returned no data: {str(data)[:200]}")
    return _save_to_file(path, body)


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

    async def _request(self, method: str, path: str, *, json: Any | None = None, files: dict[str, Any] | None = None) -> httpx.Response:
        try:
            response = await self._client.request(method, path, json=json, files=files)
        except httpx.HTTPError as exc:
            raise MlxServeError(f"could not reach {self.base_url}{path}: {exc}") from exc
        if response.status_code >= 400:
            raise MlxServeError(extract_error_message(response), status_code=response.status_code)
        return response

    async def _request_json(self, method: str, path: str, *, json: Any | None = None, files: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self._request(method, path, json=json, files=files)
        try:
            return response.json()
        except ValueError as exc:
            raise MlxServeError(f"malformed JSON from {path}: {response.text[:200]!r}") from exc

    async def _request_bytes(self, method: str, path: str, *, json: Any | None = None, files: dict[str, Any] | None = None) -> bytes:
        response = await self._request(method, path, json=json, files=files)
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
        path: str | Path,
        size: str = "1024x1024",
        n: int = 1,
        seed: int | None = None,
    ) -> Path:
        """``POST /v1/images/generations`` -> path of the saved PNG.

        The endpoint streams the image itself as a single ``image/png``
        (or ``application/octet-stream``) body — it does *not* implement
        the OpenAI ``{"data": [{"b64_json": ...}]}`` JSON shape — so we
        stream the raw response straight to disk instead of decoding a
        base64 field from the JSON body.
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": n,
            "response_format": "b64_json",
        }
        if seed is not None:
            payload["seed"] = seed
        response = await self._request("POST", "/v1/images/generations", json=payload)
        return _save_b64_or_raw(response, path, "image generation")

    async def edit_image(
        self,
        image_path: str | Path | bytes,
        prompt: str,
        model: str,
        path: str | Path,
        size: str = "1024x1024",
        strength: float = 0.6,
        seed: int | None = None,
    ) -> Path:
        """``POST /v1/images/edits`` -> path of the saved edited PNG.

        The endpoint streams the edited image as a single ``image/png``
        body, so the multipart form (all fields must be 3-tuples of
        ``(filename, bytes, content_type)`` or httpx emits a
        ``Content-Type`` without the required ``boundary`` parameter,
        which the remote server rejects) is sent with
        ``files=self._build_multipart(...)`` and the raw response body is
        written straight to ``path``.
        """
        if isinstance(image_path, bytes):
            filename, blob = "image.png", image_path
        else:
            p = Path(str(image_path)).expanduser()
            if not p.is_file():
                raise MlxServeError(f"image file not found: {p}")
            filename = p.name
            blob = p.read_bytes()
        ctype = mimetypes.guess_type(filename)[0] or "image/png"
        files: dict[str, Any] = {
            "image": (filename, blob, ctype),
            "prompt": ("prompt.txt", prompt.encode(), "text/plain"),
            "model": ("model.txt", model.encode(), "text/plain"),
            "size": ("size.txt", size.encode(), "text/plain"),
            "strength": ("strength.txt", str(strength).encode(), "text/plain"),
        }
        if seed is not None:
            files["seed"] = ("seed.txt", str(seed).encode(), "text/plain")
        response = await self._request("POST", "/v1/images/edits", files=files)
        return _save_b64_or_raw(response, path, "image edit")

    # ── audio ──────────────────────────────────────────────────────────────

    async def generate_speech(
        self,
        text: str,
        model: str,
        path: str | Path,
        voice: str | None = None,
        speed: float | None = None,
    ) -> Path:
        """``POST /v1/audio/speech`` -> path of the saved WAV file."""
        payload: dict[str, Any] = {"model": model, "input": text, "response_format": "wav"}
        if voice is not None:
            payload["voice"] = voice
        if speed is not None:
            payload["speed"] = speed
        data = await self._request_bytes("POST", "/v1/audio/speech", json=payload)
        return _save_to_file(path, data)

    async def generate_music(
        self,
        prompt: str,
        model: str,
        path: str | Path,
        lyrics: str | None = None,
        duration_seconds: int | None = None,
        bpm: int | None = None,
        keyscale: str | None = None,
        time_signature: str | None = None,
        vocal_language: str | None = None,
    ) -> Path:
        """``POST /v1/audio/music-generations`` -> path of the saved WAV file.

        The music model (e.g. MiniMax Music 3) is *lyric-conditioned*: it
        refuses a request with neither ``lyrics`` nor ``instrumental``, so
        when no ``lyrics`` are supplied we send ``instrumental: true`` to
        request an instrumental track instead of omitting both fields.
        """
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
        if lyrics is None:
            payload["instrumental"] = True
        data = await self._request_bytes("POST", "/v1/audio/music-generations", json=payload)
        return _save_to_file(path, data)

    # ── video ──────────────────────────────────────────────────────────────

    async def generate_video(
        self,
        prompt: str,
        model: str,
        path: str | Path,
        size: str | None = None,
        seconds: int | None = None,
        fps: int | None = None,
    ) -> Path:
        """``POST /v1/video/generations`` -> path of the saved MP4.

        The response carries raw RGB8 frames (+ optional PCM audio) which
        are muxed into an H.264/AAC MP4 via ffmpeg (run in a worker
        thread) and written straight to ``path``.
        """
        payload: dict[str, Any] = {"model": model, "prompt": prompt}
        for key, value in (("size", size), ("seconds", seconds), ("fps", fps)):
            if value is not None:
                payload[key] = value
        data = await self._request_json("POST", "/v1/video/generations", json=payload)
        out = Path(str(path)).expanduser()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(
                video_mod.mux_video_mp4,
                _b64_to_bytes(data.get("data", "")),
                int(data.get("width", 0)),
                int(data.get("height", 0)),
                int(data.get("fps", 0) or 0),
                out,
                _b64_to_bytes(data["audio_data"]) if data.get("audio_data") else None,
                data.get("audio_sample_rate"),
                data.get("audio_channels"),
            ),
        )
        return out

    # ── 3d ─────────────────────────────────────────────────────────────────

    async def generate_mesh(
        self,
        prompt: str,
        model: str,
        path: str | Path,
        **extra: Any,
    ) -> Path:
        """``POST /v1/3d/generations`` -> path of the saved GLB file.

        The default mesh model (``trellis-mlx``) is *image-conditioned*: it
        requires an ``image`` field (base64 PNG/JPEG of the subject) and
        rejects a text-only request with ``missing 'image'``. We therefore
        forward the prompt as the required ``image`` field (the server maps
        the single ``image`` field to its ``prompt`` pipeline stage).
        """
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "image": prompt, **extra}
        data = await self._request_json("POST", "/v1/3d/generations", json=payload)
        b64 = data.get("data")
        if not b64:
            raise MlxServeError(f"3d generation returned no data: {str(data)[:200]}")
        return _save_to_file(path, _b64_to_bytes(b64))