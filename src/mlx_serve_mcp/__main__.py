"""CLI entry point: ``mlx-serve-mcp`` / ``python -m mlx_serve_mcp``."""

from __future__ import annotations

import sys

from .config import load_config
from .server import create_server


def main(argv: list[str] | None = None) -> int:
    try:
        config = load_config(argv)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    app = create_server(config)

    # stdio transport is the default; sse / streamable-http use config.host/port.
    try:
        app.run(transport=config.transport)
    except KeyboardInterrupt:  # graceful shutdown on Ctrl-C
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())