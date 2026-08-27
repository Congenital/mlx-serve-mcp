"""mlx-serve-mcp — MCP server bridging MCP clients to a remote mlx-serve instance.

mlx-serve (https://github.com/ddalcu/mlx-serve) is an OpenAI-compatible local
inference server for Apple Silicon. This package wraps its media generation
endpoints (`/v1/images/*`, `/v1/audio/*`, `/v1/video/generations`,
`/v1/3d/generations`) plus model management (`/v1/models`, `/v1/load-model`,
`/v1/unload-model`) as MCP tools, so any MCP client can drive a remote
mlx-serve instance over the network by pointing it at ``ip:port``.
"""

__version__ = "0.1.0"

# Public API
from mlx_serve_mcp.client import (
    MlxServeClient,
    MlxServeError,
    MlxServeConnectionError,
    VideoResult,
)
from mlx_serve_mcp.config import Config, build_config
from mlx_serve_mcp.server import (
    McpServer,
    ToolDefinition,
    create_server,
    generate_image,
    edit_image,
    generate_video,
    generate_3d,
    generate_music,
    text_to_speech,
    health_check,
    list_models,
    load_model,
    unload_model,
)

__all__ = [
    "__version__",
    "MlxServeClient",
    "MlxServeError",
    "MlxServeConnectionError",
    "VideoResult",
    "Config",
    "build_config",
    "McpServer",
    "ToolDefinition",
    "create_server",
    "generate_image",
    "edit_image",
    "generate_video",
    "generate_3d",
    "generate_music",
    "text_to_speech",
    "health_check",
    "list_models",
    "load_model",
    "unload_model",
]
