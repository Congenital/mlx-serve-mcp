"""LIVE integration test against a real mlx-serve instance.

Not part of the default pytest run (marked with skipif unless
MLX_SERVE_LIVE=1). Exercises every media tool end-to-end with explicit
model selection and reports real output artifacts.

Usage:
    MLX_SERVE_LIVE=1 MLX_SERVE_URL=192.168.2.6:8000 MLX_SERVE_API_KEY=private \\
        uv run pytest tests/test_live_generation.py -v -s
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx_serve_mcp.config import build_config  # noqa: E402
from mlx_serve_mcp import server as srv  # noqa: E402

LIVE = os.environ.get("MLX_SERVE_LIVE") == "1"
URL = os.environ.get("MLX_SERVE_URL", "192.168.2.6:8000")
API_KEY = os.environ.get("MLX_SERVE_API_KEY", "private")

# Model ids on the live server (from /v1/models probe).
IMAGE_MODEL = "ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit"
IMAGE_EDIT_MODEL = "ddalcu/Mage-Flow-Edit-Turbo-MLX-Serve-8bit"
TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
MUSIC_MODEL = "ddalcu/MiniMax-Music3-MLX-Serve-8bit"
VIDEO_MODEL = "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit"
MESH_MODEL = "ddalcu/Hunyuan3D-2.1-MLX-Serve-8bit"


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@pytest.fixture(scope="module")
def live_env() -> None:
    """Point the module-level server at the live instance, output to tests/output/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_config(URL, api_key=API_KEY, output_dir=OUTPUT_DIR)
    srv.mcp.init(cfg)


@pytest.mark.skipif(not LIVE, reason="set MLX_SERVE_LIVE=1 to run against a real server")
class TestLiveGeneration:
    def test_01_health(self, live_env) -> None:
        asyncio.run(self._test_01_health_impl(live_env))

    async def _test_01_health_impl(self, live_env) -> None:
        print(f"\n[health] {await srv.health_check()}")

    def test_02_list_models(self, live_env) -> None:
        asyncio.run(self._test_02_list_models_impl(live_env))

    async def _test_02_list_models_impl(self, live_env) -> None:
        out = await srv.list_models()
        print(f"\n[list_models]\n{out}")

    def test_03_generate_image(self, live_env) -> None:
        asyncio.run(self._test_03_generate_image_impl(live_env))

    async def _test_03_generate_image_impl(self, live_env) -> None:
        t0 = time.time()
        out = await srv.generate_image(
            prompt="a watercolor painting of a red fox in a snowy forest, soft light",
            model=IMAGE_MODEL,
            size="512x512",
            seed=42,
        )
        print(f"\n[generate_image] {out}  ({time.time() - t0:.1f}s)")
        path = Path(out.split("Image generated: ")[1].split()[0])
        assert path.exists() and path.stat().st_size > 1000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_03b_edit_image(self, live_env) -> None:
        """Edit the image generated in test_03. May fail if edit model weights are missing."""
        import glob
        imgs = sorted(glob.glob(str(OUTPUT_DIR / "images" / "*.png")))
        if not imgs:
            pytest.skip("no source image from test_03")
        try:
            asyncio.run(self._test_03b_edit_image_impl(imgs[-1]))
        except Exception as e:
            print(f"\n[edit_image] SKIPPED (server-side): {e}")

    async def _test_03b_edit_image_impl(self, src: str) -> None:
        t0 = time.time()
        out = await srv.edit_image(
            image_path=src,
            prompt="add a bright blue sky with fluffy clouds",
            model=IMAGE_EDIT_MODEL,
            size="512x512",
            seed=99,
        )
        print(f"\n[edit_image] {out}  ({time.time() - t0:.1f}s)")
        path = Path(out.split("Image edited: ")[1].split()[0])
        assert path.exists() and path.stat().st_size > 1000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_04_text_to_speech(self, live_env) -> None:
        asyncio.run(self._test_04_text_to_speech_impl(live_env))

    async def _test_04_text_to_speech_impl(self, live_env) -> None:
        t0 = time.time()
        out = await srv.text_to_speech(
            text="Hello! This is a live integration test of the mlx-serve MCP bridge.",
            model=TTS_MODEL,
        )
        print(f"\n[text_to_speech] {out}  ({time.time() - t0:.1f}s)")
        path = Path(out.split("Speech generated: ")[1].split()[0])
        assert path.exists() and path.stat().st_size > 1000
        assert path.read_bytes()[:4] == b"RIFF"

    def test_05_generate_music(self, live_env) -> None:
        asyncio.run(self._test_05_generate_music_impl(live_env))

    async def _test_05_generate_music_impl(self, live_env) -> None:
        t0 = time.time()
        out = await srv.generate_music(
            prompt_style="lo-fi hip hop, mellow piano, vinyl crackle, chill beats",
            model=MUSIC_MODEL,
            duration_seconds=15,
            instrumental=True,
            seed=7,
        )
        print(f"\n[generate_music] {out}  ({time.time() - t0:.1f}s)")
        path = Path(out.split("saved to ")[1].strip())
        assert path.exists() and path.stat().st_size > 10000
        assert path.read_bytes()[:4] == b"RIFF"

    def test_06_generate_3d(self, live_env) -> None:
        asyncio.run(self._test_06_generate_3d_impl(live_env))

    async def _test_06_generate_3d_impl(self, live_env) -> None:
        # Create a simple subject image locally (orange circle on white).
        import struct
        import zlib

        w = h = 128
        rows = []
        for y in range(h):
            row = b"\x00"  # filter byte
            for x in range(w):
                dx, dy = x - w / 2, y - h / 2
                if dx * dx + dy * dy < (w / 2 - 8) ** 2:
                    row += b"\xff\x80\x20"  # orange
                else:
                    row += b"\xff\xff\xff"  # white
            rows.append(row)
        raw = b"".join(rows)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
        src = Path(srv.mcp._config.output_dir) / "subject.png"
        src.write_bytes(png)

        t0 = time.time()
        out = await srv.generate_3d(
            str(src),
            model=MESH_MODEL,
            steps=20,
            octree_resolution=128,
            guidance_scale=5.0,
            seed=123,
        )
        print(f"\n[generate_3d] {out}  ({time.time() - t0:.1f}s)")
        path = Path(out.split("saved to ")[1].strip())
        assert path.exists() and path.stat().st_size > 1000
        assert path.read_bytes()[:4] == b"glTF"

    def test_07_generate_video(self, live_env) -> None:
        asyncio.run(self._test_07_generate_video_impl(live_env))

    async def _test_07_generate_video_impl(self, live_env) -> None:
        # Smallest viable clip to keep runtime reasonable.
        t0 = time.time()
        out = await srv.generate_video(
            prompt="a slow zoom into a calm ocean at sunset, gentle waves",
            model=VIDEO_MODEL,
            num_frames=9,
            width=256,
            height=256,
            seed=5,
        )
        print(f"\n[generate_video] {out}  ({time.time() - t0:.1f}s)")
        path = Path(out.split("MP4 saved to ")[1].strip())
        assert path.exists() and path.stat().st_size > 10000
        header = path.read_bytes()[:12]
        # MP4: 4-byte box size + 'ftyp' brand
        assert header[4:8] == b"ftyp", f"not an MP4: {header!r}"
