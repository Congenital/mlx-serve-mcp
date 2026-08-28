"""Demo / smoke-test client for the mlx-serve-mcp server.

Speaks MCP over stdio to a server subprocess (using the official ``mcp`` SDK
client), lists the advertised tools, and optionally calls one. Useful for a
quick end-to-end check without a full MCP host.

Examples:
    python mcp_client.py --list
    python mcp_client.py --call recall_memory
    python mcp_client.py --call generate_image --args '{"prompt": "a red fox"}'
    python mcp_client.py --command "python -m mlx_serve_mcp" --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _default_command() -> list[str]:
    """Prefer the installed console script; fall back to ``python -m``."""
    script = "mlx-serve-mcp"
    # If running from a source checkout without an install, use the module.
    if not _on_path(script):
        return [sys.executable, "-m", "mlx_serve_mcp"]
    return [script]


def _on_path(executable: str) -> bool:
    from shutil import which

    return which(executable) is not None


async def run(command: list[str], env: dict[str, str] | None, list_tools: bool, call: str | None, args: str | None) -> int:
    params = StdioServerParameters(command=command[0], args=command[1:], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print(f"connected: {info.serverInfo.name} v{getattr(info.serverInfo, 'version', '?')}")
            if list_tools:
                tools = (await session.list_tools()).tools
                print(f"{len(tools)} tools:")
                for tool in tools:
                    print(f"  - {tool.name}: {(tool.description or '').splitlines()[0] if tool.description else ''}")
            if call:
                kwargs = json.loads(args) if args else {}
                result = await session.call_tool(call, kwargs)
                for block in result.content:
                    text = getattr(block, "text", None)
                    if text is not None:
                        print(text)
                    else:
                        print(f"[{getattr(block, 'type', 'content')} block]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCP client for mlx-serve-mcp")
    parser.add_argument("--command", default=None, help="Server command (default: auto-detect)")
    parser.add_argument("--list", action="store_true", help="List tools and exit")
    parser.add_argument("--call", default=None, help="Tool name to call")
    parser.add_argument("--args", default=None, help="JSON object of tool arguments")
    cli = parser.parse_args(argv)

    command = cli.command.split() if cli.command else _default_command()
    env = dict(os.environ)
    # Make the source checkout importable when not installed.
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "src")
    if os.path.isdir(src) and src not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    try:
        return anyio.run(run, command, env, cli.list, cli.call, cli.args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())