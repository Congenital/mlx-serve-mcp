"""MCP prompts: ready-made recipes for the mlx-serve generation tools.

These are exposed through the standard MCP ``prompts/list`` / ``prompts/get``
methods so clients (Claude Desktop, LobeHub, ...) can render them as one-click
prompt templates. Each prompt names the exact tool to call, the recommended
model from the mlx-serve model zoo, and the argument values that work well in
practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptArgument:
    """A template variable for an MCP prompt (``{{name}}`` in the text)."""

    name: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class PromptDefinition:
    """A named prompt template that maps onto one of the generation tools."""

    name: str
    description: str
    arguments: list[PromptArgument]
    text: str
    tool: str
    tool_args: dict[str, Any]


PROMPTS: list[PromptDefinition] = [
    PromptDefinition(
        name="create_poster",
        description=(
            "Design a text-centric poster or typographic artwork (title, tagline, "
            "labels). Uses Mage-Flow-Turbo, which is the most reliable model for "
            "rendering text inside images."
        ),
        arguments=[
            PromptArgument(
                "title", "The main title text to render on the poster"
            ),
            PromptArgument(
                "tagline",
                "Short supporting line, e.g. ', with a tagline under the title' "
                "(or empty)",
                required=False,
            ),
            PromptArgument(
                "style",
                "Art style for the artwork, e.g. 'vintage screen print' or "
                "'neon cyberpunk'",
                required=False,
            ),
        ],
        text=(
            "Use the generate_image tool to create a poster. "
            "Prompt: \"A {style} poster with the bold title '{title}'"
            "{tagline} — clean centered typography, high contrast, "
            "professional layout\". "
            "Use model \"ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit\" (best at rendering "
            "text in images) and size \"1024x1024\"."
        ),
        tool="generate_image",
        tool_args={
            "prompt": (
                "A {style} poster with the bold title '{title}' {tagline_line}, "
                "clean centered typography, high contrast, professional layout"
            ),
            "model": "ddalcu/Mage-Flow-Turbo-MLX-Serve-8bit",
            "size": "1024x1024",
        },
    ),
    PromptDefinition(
        name="portrait_photo",
        description=(
            "Generate a realistic portrait or photo-real scene. Uses FLUX.2-klein-4B, "
            "the best model in the mlx-serve group for faces and photo-real detail."
        ),
        arguments=[
            PromptArgument("subject", "Who or what to depict, e.g. 'a fisherman in a rain hat'"),
            PromptArgument("scene", "Setting and mood, e.g. 'on a foggy harbor at dawn'"),
            PromptArgument(
                "lighting",
                "Lighting direction (or empty), e.g. 'soft window light'",
                required=False,
            ),
        ],
        text=(
            "Use the generate_image tool to create a portrait. "
            "Prompt: \"Photorealistic portrait of {subject}, {scene}"
            "{lighting}, sharp facial features, natural skin texture, "
            "85mm lens, f/1.8\". "
            "Use model \"Runpod/FLUX.2-klein-4B-mflux-4bit\" (best for faces) and "
            "size \"1024x1024\"."
        ),
        tool="generate_image",
        tool_args={
            "prompt": (
                "Photorealistic portrait of {subject}, {scene}{lighting}, "
                "sharp facial features, natural skin texture, 85mm lens, f/1.8"
            ),
            "model": "Runpod/FLUX.2-klein-4B-mflux-4bit",
            "size": "1024x1024",
        },
    ),
    PromptDefinition(
        name="lofi_track",
        description=(
            "Produce a 60-second lo-fi hip hop instrumental with mellow piano, "
            "chill beats and vinyl crackle — a ready-made background-music track."
        ),
        arguments=[],
        text=(
            "Use the generate_music tool with prompt_style \"lo-fi hip hop, mellow "
            "piano, vinyl crackle, chill beats\", duration_seconds 60, "
            "instrumental true. The result is a playable WAV file."
        ),
        tool="generate_music",
        tool_args={
            "prompt_style": "lo-fi hip hop, mellow piano, vinyl crackle, chill beats",
            "duration_seconds": 60,
            "instrumental": True,
        },
    ),
    PromptDefinition(
        name="speak_text",
        description=(
            "Speak a passage of text aloud as a WAV file using the Qwen3-TTS "
            "voice model."
        ),
        arguments=[
            PromptArgument("text", "The passage of text to speak aloud"),
            PromptArgument(
                "speed",
                "Playback speed multiplier 0.25..4.0 (or empty for normal speed)",
                required=False,
            ),
        ],
        text=(
            "Use the text_to_speech tool with the text \"{text}\""
            "{speed}. The result is a WAV file you can play or attach."
        ),
        tool="text_to_speech",
        tool_args={
            "text": "{text}",
        },
    ),
    PromptDefinition(
        name="image_to_3d",
        description=(
            "Turn a clean cutout photo of an object (real alpha transparency works "
            "best) into a textured 3D model file (GLB) for Blender, three.js or "
            "Quick Look."
        ),
        arguments=[
            PromptArgument(
                "image_path",
                "Path to the subject image (PNG with transparency is best)",
            ),
        ],
        text=(
            "Use the generate_3d tool with image_path \"{image_path}\", "
            "texture true, octree_resolution 256 and guidance_scale 5. "
            "The result is a GLB file you can open in Blender, three.js or "
            "Quick Look."
        ),
        tool="generate_3d",
        tool_args={
            "image_path": "{image_path}",
            "texture": True,
            "octree_resolution": 256,
            "guidance_scale": 5.0,
        },
    ),
    PromptDefinition(
        name="short_video",
        description=(
            "Generate a short, low-resolution video clip (9 frames, 256x256) — "
            "the fastest video path on mlx-serve, good for previews and tests."
        ),
        arguments=[
            PromptArgument(
                "prompt", "Scene description, e.g. 'a slow zoom into a calm ocean at sunset'"
            ),
        ],
        text=(
            "Use the generate_video tool with prompt \"{prompt}\", num_frames 9, "
            "width 256, height 256. The result is an encoded MP4 file "
            "(H.264, playable anywhere)."
        ),
        tool="generate_video",
        tool_args={
            "prompt": "{prompt}",
            "num_frames": 9,
            "width": 256,
            "height": 256,
        },
    ),
]


def list_prompts() -> list[dict[str, Any]]:
    """MCP ``prompts/list`` result: name + description + argument names."""
    return [
        {
            "name": p.name,
            "description": p.description,
            "arguments": [
                {
                    "name": a.name,
                    "description": a.description,
                    "required": a.required,
                }
                for a in p.arguments
            ],
        }
        for p in PROMPTS
    ]


def get_prompt(name: str, arguments: dict[str, str] | None = None) -> dict[str, Any]:
    """MCP ``prompts/get`` result for one template, with values substituted.

    ``{name}`` placeholders are replaced from ``arguments``; missing optional
    placeholders are left as empty strings. Unknown prompt names raise
    :class:`ValueError`.
    """
    for p in PROMPTS:
        if p.name != name:
            continue
        provided = arguments or {}
        text = p.text
        for a in p.arguments:
            value = provided.get(a.name, "")
            text = text.replace("{" + a.name + "}", value)
        # Tidy common leftovers from optional placeholders.
        text = text.replace(" ,", ",").replace(" .", ".").replace("  ", " ")
        return {
            "description": p.description,
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": text},
                }
            ],
        }
    raise ValueError(f"Unknown prompt: {name}")