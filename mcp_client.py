#!/usr/bin/env python3
"""MCP client that spawns the mlx-serve-mcp server over stdio and exercises
every tool end-to-end.

Usage:
    python3 mcp_client.py [--url 192.168.2.6:8000] [--api-key private]
                          [--output-dir ./output]

This is a REAL MCP client — it speaks the full MCP protocol (initialize →
initialized → tools/list → tools/call) over stdio, exactly like Claude Code
or Claude Desktop would.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent


def build_server_params(
    url: str,
    api_key: str | None,
    output_dir: str,
) -> StdioServerParameters:
    """Build the StdioServerParameters with the MLX_SERVE_* env vars."""
    env = dict(os.environ)
    env["MLX_SERVE_URL"] = url
    env["MLX_SERVE_OUTPUT_DIR"] = str(Path(output_dir).resolve())
    if api_key:
        env["MLX_SERVE_API_KEY"] = api_key

    return StdioServerParameters(
        command="uv",
        args=[
            "--directory", str(PROJECT_ROOT),
            "run", "mlx-serve-mcp",
        ],
        env=env,
        cwd=str(PROJECT_ROOT),
    )


async def call_tool(
    session: ClientSession,
    name: str,
    arguments: dict,
    timeout: float = 600.0,
) -> str:
    """Call a tool and return the text content."""
    t0 = time.time()
    result = await asyncio.wait_for(
        session.call_tool(name, arguments),
        timeout=timeout,
    )
    elapsed = time.time() - t0

    texts = []
    for block in result.content:
        if block.type == "text":
            texts.append(block.text)

    status = "ERROR" if result.isError else "OK"
    text = "\n".join(texts)
    print(f"  [{status}] {name}  ({elapsed:.1f}s)")
    print(f"    {text[:300]}")
    return text


def _make_subject_png(path: Path) -> None:
    """Create a simple 128x128 PNG (orange circle on white)."""
    import struct
    import zlib

    w = h = 128
    rows = []
    for y in range(h):
        row = b"\x00"
        for x in range(w):
            dx, dy = x - w / 2, y - h / 2
            if dx * dx + dy * dy < (w / 2 - 8) ** 2:
                row += b"\xff\x80\x20"
            else:
                row += b"\xff\xff\xff"
        rows.append(row)
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    path.write_bytes(png)


async def main() -> None:
    parser = argparse.ArgumentParser(description="MCP client for mlx-serve-mcp")
    parser.add_argument("--url", default=os.environ.get("MLX_SERVE_URL", "192.168.2.6:8000"))
    parser.add_argument("--api-key", default=os.environ.get("MLX_SERVE_API_KEY", "private"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = build_server_params(args.url, args.api_key, str(output_dir))

    print(f"=== MCP Client: mlx-serve-mcp ===")
    print(f"Server URL: {args.url}")
    print(f"Output dir: {output_dir}\n")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize (MCP handshake)
            print("── MCP Handshake ──")
            init = await session.initialize()
            print(f"  Server: {init.serverInfo.name} v{init.serverInfo.version}")
            print(f"  Protocol: {init.protocolVersion}")
            print(f"  Instructions: {init.instructions[:120]}...\n")

            # 2. List tools
            print("── Tools ──")
            tools = await session.list_tools()
            for t in tools.tools:
                print(f"  • {t.name}: {t.description.splitlines()[0].strip()}")
            print()

            # 3. Call each tool
            print("── Generation ──\n")

            await call_tool(session, "health_check", {})

            await call_tool(session, "list_models", {})

            await call_tool(
                session, "generate_image",
                {"prompt": "a watercolor painting of a red fox in a snowy forest, soft light",
                 "size": "512x512", "seed": 42},
            )

            imgs = sorted(output_dir.glob("images/*.png"))
            if imgs:
                await call_tool(
                    session, "edit_image",
                    {"image_path": str(imgs[-1]),
                     "prompt": "add a full moon in the night sky"},
                )
            else:
                print("  [SKIP] edit_image (no source image)\n")

            await call_tool(
                session, "text_to_speech",
                {"text": "Hello! This is a test of the mlx-serve MCP bridge."},
            )

            await call_tool(
                session, "generate_music",
                {"prompt_style": "lo-fi hip hop, mellow piano, vinyl crackle, chill beats",
                 "duration_seconds": 15, "instrumental": True, "seed": 7},
                timeout=300,
            )

            subject = output_dir / "subject.png"
            if not subject.exists():
                _make_subject_png(subject)
                print(f"  (created subject image: {subject})")

            await call_tool(
                session, "generate_3d",
                {"image_path": str(subject), "steps": 20,
                 "octree_resolution": 128, "guidance_scale": 5.0, "seed": 123},
                timeout=300,
            )

            await call_tool(
                session, "generate_video",
                {"prompt": "a slow zoom into a calm ocean at sunset, gentle waves",
                 "num_frames": 9, "width": 256, "height": 256, "seed": 5},
                timeout=600,
            )

    # Summary
    print(f"\n── Output files ──")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            if size < 1024:
                sz = f"{size} B"
            elif size < 1024 * 1024:
                sz = f"{size / 1024:.1f} KiB"
            else:
                sz = f"{size / 1024 / 1024:.1f} MiB"
            print(f"  {f.relative_to(output_dir)}  ({sz})")


if __name__ == "__main__":
    asyncio.run(main())