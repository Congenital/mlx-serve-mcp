"""Unit tests for the mlx-serve-mcp client layer (no live server needed).

All HTTP traffic is served by ``httpx.MockTransport``, with response shapes
copied from mlx-serve's Zig handlers.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx_serve_mcp.client import (  # noqa: E402
    MlxServeClient,
    MlxServeConnectionError,
    MlxServeError,
    VideoResult,
    extract_error_message,
)
from mlx_serve_mcp.config import build_config, normalize_base_url  # noqa: E402


PNG_B64 = base64.b64encode(b"\x89PNG-fake-bytes").decode()
GLB_B64 = base64.b64encode(b"glTF-fake-glb").decode()


def run(coro):
    return asyncio.run(coro)


# ── URL normalization ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("192.168.1.10:11234", "http://192.168.1.10:11234"),
        ("http://127.0.0.1:11234", "http://127.0.0.1:11234"),
        ("localhost:11234/", "http://localhost:11234"),
        ("https://mlbox.local", "https://mlbox.local"),
        ("  http://a.b:1/  ", "http://a.b:1"),
    ],
)
def test_normalize_base_url(raw: str, expected: str) -> None:
    assert normalize_base_url(raw) == expected


@pytest.mark.parametrize("bad", ["", "ftp://x:1", "http://"])
def test_normalize_base_url_rejects_bad_input(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_base_url(bad)


# ── error extraction ─────────────────────────────────────────────────────

def _resp(payload: dict | None, status: int = 400, text: str | None = None) -> httpx.Response:
    if payload is not None:
        return httpx.Response(status, json=payload)
    return httpx.Response(status, content=(text or "").encode())


def test_extract_error_message_gen_style() -> None:
    msg = extract_error_message(_resp({"error": {"message": "missing 'prompt'"}}))
    assert msg == "missing 'prompt'"


def test_extract_error_message_openai_style() -> None:
    msg = extract_error_message(
        _resp({"error": {"type": "invalid_request_error", "message": "nope"}})
    )
    assert msg == "nope"


def test_extract_error_message_non_json() -> None:
    assert "boom" in extract_error_message(_resp(None, text="plain boom"))


# ── endpoint behaviors against mocked responses ──────────────────────────

def _client_with(handler) -> MlxServeClient:
    return MlxServeClient(
        "http://fake:11234",
        api_key="secret",
        timeout_seconds=30,
        transport=httpx.MockTransport(handler),
    )


def test_health_and_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"object": "list", "data": [{"id": "flux"}]})

    client = _client_with(handler)
    try:
        assert run(client.health()) == {"status": "ok"}
        assert run(client.list_models()) == [{"id": "flux"}]
    finally:
        run(client.aclose())


def test_load_model_body_shape() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"model": {"id": "m1", "state": "ready"}})

    client = _client_with(handler)
    try:
        result = run(client.load_model("m1", make_default=True))
        assert result["model"]["state"] == "ready"
        assert captured["path"] == "/v1/load-model"
        assert captured["body"] == {"model": "m1", "default": True}
    finally:
        run(client.aclose())


def test_unload_model_body_shape() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"unloaded": "m1"})

    client = _client_with(handler)
    try:
        run(client.unload_model("m1"))
        assert captured["body"] == {"model": "m1"}
    finally:
        run(client.aclose())


def test_generate_image_parses_b64() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["prompt"] == "cat"
        assert body["size"] == "512x512"
        return httpx.Response(200, json={"created": 0, "data": [{"b64_json": PNG_B64}]})

    client = _client_with(handler)
    try:
        items = run(client.generate_image({"prompt": "cat", "size": "512x512"}))
        assert base64.b64decode(items[0]["b64_json"]) == b"\x89PNG-fake-bytes"
    finally:
        run(client.aclose())


def test_speech_returns_raw_wav() -> None:
    wav = b"RIFF" + b"\x00" * 8

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["input"] == "hello"
        return httpx.Response(200, content=wav, headers={"content-type": "audio/wav"})

    client = _client_with(handler)
    try:
        assert run(client.generate_speech({"input": "hello"})) == wav
    finally:
        run(client.aclose())


def test_speech_error_surfaces_server_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "'speed' must be in (0, 5]"}})

    client = _client_with(handler)
    try:
        with pytest.raises(MlxServeError) as excinfo:
            run(client.generate_speech({"input": "hi"}))
        assert excinfo.value.status_code == 400
        assert "speed" in str(excinfo.value)
    finally:
        run(client.aclose())


def _video_payload(with_audio: bool) -> dict:
    frames = b"\x10\x20\x30" * (2 * 4 * 4)  # 2 frames of 4x4 rgb24
    data: dict = {
        "created": 0,
        "frames": 2,
        "height": 4,
        "width": 4,
        "fps": 24,
        "format": "rgb8",
        "data": base64.b64encode(frames).decode(),
    }
    if with_audio:
        pcm = struct.pack("<8h", *([100] * 8))
        data.update(
            {
                "audio_sample_rate": 48000,
                "audio_channels": 2,
                "audio_format": "pcm_s16le",
                "audio_data": base64.b64encode(pcm).decode(),
            }
        )
    return data


def test_generate_video_decodes_frames_and_audio() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_video_payload(with_audio=True))

    client = _client_with(handler)
    try:
        result: VideoResult = run(client.generate_video({}))
        assert result.frames == 2 and result.width == 4 and result.height == 4
        assert len(result.rgb_bytes) == 2 * 4 * 4 * 3
        assert result.audio_pcm_s16le is not None
        assert result.audio_sample_rate == 48000
        assert result.audio_channels == 2
    finally:
        run(client.aclose())


def test_generate_video_size_mismatch_raises() -> None:
    bad = _video_payload(with_audio=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=bad)

    bad["frames"] = 3  # lie about the frame count
    client = _client_with(handler)
    try:
        with pytest.raises(MlxServeError, match="payload size mismatch"):
            run(client.generate_video({}))
    finally:
        run(client.aclose())


def test_generate_mesh_returns_glb_bytes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body.get("octree_resolution") == 128
        return httpx.Response(200, json={"created": 0, "format": "glb", "data": GLB_B64})

    client = _client_with(handler)
    try:
        glb = run(client.generate_mesh({"image": "aGk=", "octree_resolution": 128}))
        assert glb == b"glTF-fake-glb"
    finally:
        run(client.aclose())


def test_connection_error_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = _client_with(handler)
    try:
        with pytest.raises(MlxServeConnectionError):
            run(client.health())
    finally:
        run(client.aclose())


# ── tool layer smoke tests (server.py, mocked client) ────────────────────

from mlx_serve_mcp import server as srv  # noqa: E402
from mlx_serve_mcp.config import Config  # noqa: E402


class _FakeClient:
    """Stand-in for MlxServeClient recording the last payload per endpoint."""

    def __init__(self) -> None:
        self.payloads: dict[str, dict] = {}

    async def generate_image(self, payload):
        self.payloads["image"] = payload
        return [{"b64_json": PNG_B64}]

    async def generate_speech(self, payload):
        self.payloads["speech"] = payload
        return b"RIFF-fake"

    async def generate_music(self, payload):
        self.payloads["music"] = payload
        return b"RIFF-music"

    async def generate_mesh(self, payload):
        self.payloads["mesh"] = payload
        return b"glTF-fake-glb"


def _install_fake_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    cfg = build_config("http://fake:11234", output_dir=tmp_path / "out")
    # New structure: McpServer has _client and _config attributes
    srv.mcp._config = cfg
    srv.mcp._client = None  # Will be set below
    fake = _FakeClient()
    srv.mcp._client = fake  # type: ignore[assignment]
    return fake


def test_tool_generate_image_saves_and_previews(tmp_path, monkeypatch) -> None:
    fake = _install_fake_client(tmp_path, monkeypatch)
    result = asyncio.run(srv.generate_image(prompt="a cat", size="512x512", seed=7))
    assert fake.payloads["image"]["prompt"] == "a cat"
    assert fake.payloads["image"]["seed"] == 7
    # Result is a string with the saved path
    assert "Image generated:" in result
    out_file = Path(result.split("Image generated: ")[1].split()[0])
    assert out_file.exists() and out_file.read_bytes().startswith(b"\x89PNG")


def test_tool_tts_validates_speed(tmp_path, monkeypatch) -> None:
    _install_fake_client(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="speed"):
        asyncio.run(srv.text_to_speech(text="hi", speed=0))


def test_tool_music_rejects_bad_duration(tmp_path, monkeypatch) -> None:
    fake = _install_fake_client(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="duration_seconds"):
        asyncio.run(srv.generate_music(prompt_style="lofi", duration_seconds=999))
    asyncio.run(srv.generate_music(prompt_style="lofi"))
    assert fake.payloads["music"]["prompt"] == "lofi"


def test_tool_mesh_reads_local_image(tmp_path, monkeypatch) -> None:
    fake = _install_fake_client(tmp_path, monkeypatch)
    img = tmp_path / "subject.png"
    img.write_bytes(b"\x89PNG-subject")
    result = asyncio.run(srv.generate_3d(str(img)))
    sent = base64.b64decode(fake.payloads["mesh"]["image"])
    assert sent == b"\x89PNG-subject"
    glb_path = Path(result.split("saved to ")[1].strip())
    assert glb_path.read_bytes() == b"glTF-fake-glb"


