"""Unit tests for the mlx-serve HTTP client (no network: uses httpx.MockTransport)."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from mlx_serve_mcp.client import MlxServeClient, MlxServeError, extract_error_message


def make_client(handler) -> MlxServeClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    return MlxServeClient("http://test", client=http)


def test_extract_error_message_openai_shape():
    resp = httpx.Response(500, request=httpx.Request("POST", "/x"),
                          content=json.dumps({"error": {"message": "boom"}}).encode())
    assert extract_error_message(resp) == "boom"


def test_extract_error_message_typed_shape():
    resp = httpx.Response(400, request=httpx.Request("POST", "/x"),
                          content=json.dumps({"error": {"type": "invalid_request", "message": "bad"}}).encode())
    assert "invalid_request" in extract_error_message(resp) and "bad" in extract_error_message(resp)


def test_extract_error_message_non_json():
    resp = httpx.Response(502, request=httpx.Request("GET", "/x"), content=b"bad gateway")
    assert extract_error_message(resp) == "bad gateway"


@pytest.mark.asyncio
async def test_health_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"status": "ok"}).encode())
    client = make_client(handler)
    assert (await client.health()) == {"status": "ok"}


@pytest.mark.asyncio
async def test_error_raises_mlxserveerror():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=json.dumps({"error": {"message": "nope"}}).encode())
    client = make_client(handler)
    with pytest.raises(MlxServeError) as exc:
        await client.health()
    assert "nope" in str(exc.value)
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_generate_image_saves_file(tmp_path):
    png = base64.b64encode(b"\x89PNG-fake").decode()
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "m" and body["prompt"] == "hi" and body["response_format"] == "b64_json"
        return httpx.Response(200, content=json.dumps({"created": 0, "data": [{"b64_json": png}]}).encode())
    client = make_client(handler)
    out = await client.generate_image(prompt="hi", model="m", path=tmp_path / "img.png")
    assert out == tmp_path / "img.png"
    assert out.read_bytes() == b"\x89PNG-fake"


@pytest.mark.asyncio
async def test_generate_image_saves_raw_png(tmp_path):
    """The real server streams the PNG itself (not a JSON b64_json envelope)."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x89PNG-real", headers={"content-type": "image/png"})
    client = make_client(handler)
    out = await client.generate_image(prompt="hi", model="m", path=tmp_path / "img.png")
    assert out.read_bytes() == b"\x89PNG-real"


@pytest.mark.asyncio
async def test_edit_image_sends_multipart(tmp_path):
    png = base64.b64encode(b"\x89PNG-fake").decode()
    seen: dict = {}
    def handler(request: httpx.Request) -> httpx.Response:
        seen["ctype"] = request.headers.get("content-type", "")
        seen["body"] = request.content
        return httpx.Response(200, content=json.dumps({"created": 0, "data": [{"b64_json": png}]}).encode())
    client = make_client(handler)
    out = await client.edit_image(image_path=b"src-png", prompt="rotate", model="m",
                                  path=tmp_path / "edited.png")
    assert out == tmp_path / "edited.png"
    assert "multipart/form-data" in seen["ctype"]
    assert "boundary=" in seen["ctype"]
    # The image bytes must be sent as a file part, not JSON.
    assert b"src-png" in seen["body"]
    assert b"image.png" in seen["body"]
    assert out.read_bytes() == b"\x89PNG-fake"


@pytest.mark.asyncio
async def test_generate_video_decodes_frames_and_audio(tmp_path):
    rgb = base64.b64encode(b"\x00" * 12)
    pcm = base64.b64encode(b"\x01\x02")
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({
            "frames": 2, "width": 2, "height": 2, "fps": 8, "format": "rgb8",
            "data": rgb.decode(), "audio_sample_rate": 24000, "audio_channels": 1,
            "audio_format": "pcm_s16le", "audio_data": pcm.decode(),
        }).encode())
    client = make_client(handler)
    out = await client.generate_video(prompt="x", model="v",
                                      path=tmp_path / "clip.mp4")
    assert out == tmp_path / "clip.mp4"
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.asyncio
async def test_speech_saves_wav(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["input"] == "hello" and body["response_format"] == "wav"
        return httpx.Response(200, content=b"RIFF-fake-wav")
    client = make_client(handler)
    out = await client.generate_speech(text="hello", model="t", path=tmp_path / "t.wav")
    assert out.read_bytes() == b"RIFF-fake-wav"