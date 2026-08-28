"""Live end-to-end tests against a real mlx-serve instance.

These are skipped unless ``pytest --run-live`` is passed (see conftest.py). They
require a reachable mlx-serve (configured via ``MLX_SERVE_URL`` / ``--url``) and,
for the video test, ffmpeg. They exercise the full path: HTTP client -> decode ->
disk -> (ffmpeg mux for video).
"""

from __future__ import annotations

import os

import pytest

from mlx_serve_mcp.client import MlxServeClient
from mlx_serve_mcp.config import load_config
from mlx_serve_mcp import video as video_mod

pytestmark = pytest.mark.live


@pytest.fixture()
def client():
    cfg = load_config([])
    c = MlxServeClient(cfg.base_url, cfg.api_key, cfg.timeout_seconds)
    yield c
    import asyncio

    asyncio.get_event_loop_policy().new_event_loop()  # best-effort cleanup


@pytest.mark.asyncio
async def test_health(client):
    data = await client.health()
    assert data.get("status") == "ok"


@pytest.mark.asyncio
async def test_list_models(client):
    models = await client.list_models()
    assert isinstance(models, list)


@pytest.mark.asyncio
async def test_generate_image(client, tmp_path):
    cfg = load_config([])
    out = await client.generate_image(prompt="a red fox in the snow", model=cfg.image_model,
                                      size="512x512", path=tmp_path / "fox.png")
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:4] == b"\x89PNG"


@pytest.mark.asyncio
async def test_generate_speech(client, tmp_path):
    cfg = load_config([])
    out = await client.generate_speech(text="hello world", model=cfg.tts_model,
                                       path=tmp_path / "hello.wav")
    assert out.read_bytes()[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_generate_video_and_mux(client, tmp_path):
    cfg = load_config([])
    out = tmp_path / "clip.mp4"
    result = await client.generate_video(prompt="a slow pan across a calm sea", model=cfg.video_model,
                                         seconds=2, path=out)
    assert result.frames > 0 and result.width > 0
    assert out.exists() and out.stat().st_size > 0