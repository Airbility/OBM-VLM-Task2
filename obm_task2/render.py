"""Render one centered object the same way as the original benchmark.

Geometry, palette, texture spacing, and probe jitter follow
``binding_memory_integrated/core/generate.js`` and ``build_core.js``.
Images are drawn at 2× and downsampled so the stroke edges stay smooth.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from obm_task2.constants import ALL_COLORS, BACKGROUND, IMAGE_SIZE, INK

SUPERSAMPLE = 2


def hash32(text: str) -> int:
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def probe_variant(probe_id: str) -> dict:
    """Match the original deterministic pan and zoom for a probe id."""
    dx = (hash32(probe_id) % 11) - 5
    dy = (hash32(f"{probe_id}|y") % 11) - 5
    scale = 0.974 + (hash32(f"{probe_id}|scale") % 53) / 1000
    if dx == 0 and dy == 0 and abs(scale - 1) < 1e-8:
        scale = 1.017
    return {"dx": dx, "dy": dy, "scale": scale}


def opaque_name(key: str) -> str:
    digest = hashlib.sha256(f"obm-task2-v1|{key}".encode("utf-8")).hexdigest()
    return digest[:24]


def _scale_points(points, scale: int):
    return [(x * scale, y * scale) for x, y in points]


def _regular_polygon(sides: int, radius: float, rotation: float):
    import math

    points = []
    for index in range(sides):
        angle = rotation + index * 2 * math.pi / sides
        points.append((192 + radius * math.cos(angle), 192 + radius * math.sin(angle)))
    return points


def shape_kind(shape: str) -> tuple[str, object]:
    if shape == "circle":
        return "ellipse", (192 - 116, 192 - 116, 192 + 116, 192 + 116)
    if shape == "ellipse":
        return "ellipse", (192 - 128, 192 - 76, 192 + 128, 192 + 76)
    if shape == "square":
        return "polygon", [(76, 76), (308, 76), (308, 308), (76, 308)]
    if shape == "triangle":
        return "polygon", [(192, 57), (331, 311), (53, 311)]
    if shape == "hexagon":
        return "polygon", [(192, 55), (312, 124), (312, 260), (192, 329), (72, 260), (72, 124)]
    if shape == "star":
        return "polygon", [
            (192, 54), (226, 143), (322, 150), (248, 211), (272, 306),
            (192, 252), (112, 306), (136, 211), (62, 150), (158, 143),
        ]
    if shape == "diamond":
        return "polygon", [(192, 53), (331, 192), (192, 331), (53, 192)]
    if shape == "pentagon":
        import math
        return "polygon", _regular_polygon(5, 124, -math.pi / 2)
    raise ValueError(shape)


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, scale: int, **kwargs) -> None:
    kind, geometry = shape_kind(shape)
    if kind == "ellipse":
        box = tuple(value * scale for value in geometry)
        draw.ellipse(box, **kwargs)
        return
    draw.polygon(_scale_points(geometry, scale), **kwargs)


def _quad(start, control, end, steps: int = 8):
    points = []
    for step in range(1, steps + 1):
        t = step / steps
        u = 1 - t
        x = u * u * start[0] + 2 * u * t * control[0] + t * t * end[0]
        y = u * u * start[1] + 2 * u * t * control[1] + t * t * end[1]
        points.append((x, y))
    return points


def _stroke(draw: ImageDraw.ImageDraw, points, width: int) -> None:
    if len(points) < 2:
        return
    draw.line(points, fill=INK + (255,), width=width, joint="curve")


def _draw_texture(shape_mask: Image.Image, texture: str, scale: int) -> Image.Image:
    size = IMAGE_SIZE * scale
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    ink = INK + (255,)
    if texture == "dots":
        radius = 3.5 * scale
        for y in range(14, 371, 20):
            for x in range(14, 371, 20):
                cx, cy = x * scale, y * scale
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=ink)
    elif texture == "diagonal_stripes":
        width = 4 * scale
        for x in range(-400, 600, 25):
            _stroke(draw, [(x * scale, -10 * scale), ((x + 404) * scale, 394 * scale)], width)
    elif texture == "horizontal_stripes":
        width = 4 * scale
        for y in range(0, IMAGE_SIZE + 1, 22):
            _stroke(draw, [(0, y * scale), (size, y * scale)], width)
    elif texture == "grid":
        width = 3 * scale
        for value in range(0, IMAGE_SIZE + 1, 22):
            _stroke(draw, [(value * scale, 0), (value * scale, size)], width)
            _stroke(draw, [(0, value * scale), (size, value * scale)], width)
    elif texture == "checkerboard":
        cell = 24 * scale
        alpha = int(0.58 * 255)
        fill = INK + (alpha,)
        for row in range(16):
            for col in range(16):
                if (row + col) % 2 == 0:
                    draw.rectangle((col * cell, row * cell, (col + 1) * cell, (row + 1) * cell), fill=fill)
    elif texture == "waves":
        width = max(1, round(3.5 * scale))
        for y in range(10, 375, 24):
            cursor = (-20.0, float(y))
            points = [(cursor[0] * scale, cursor[1] * scale)]
            for x in range(-20, 400, 40):
                mid = (x + 20.0, float(y))
                points.extend((px * scale, py * scale) for px, py in _quad(cursor, (x + 10, y - 10), mid))
                cursor = mid
                end = (x + 40.0, float(y))
                points.extend((px * scale, py * scale) for px, py in _quad(cursor, (x + 30, y + 10), end))
                cursor = end
            _stroke(draw, points, width)
    elif texture == "chevrons":
        width = max(1, round(3.5 * scale))
        for y in range(12, 373, 25):
            points = [(-16 * scale, y * scale)]
            x = -16
            while x < 400:
                x += 16
                points.append((x * scale, (y - 10) * scale))
                x += 16
                points.append((x * scale, y * scale))
            _stroke(draw, points, width)
    elif texture == "rings":
        width = 4 * scale
        for radius in range(28, 210, 28):
            box = (
                (192 - radius) * scale,
                (192 - radius) * scale,
                (192 + radius) * scale,
                (192 + radius) * scale,
            )
            draw.ellipse(box, outline=ink, width=width)
    else:
        raise ValueError(texture)
    return Image.composite(layer, Image.new("RGBA", layer.size, (0, 0, 0, 0)), shape_mask)


def _transform(sprite: Image.Image, variant: dict | None) -> Image.Image:
    if not variant:
        return sprite
    scale = SUPERSAMPLE
    zoom = float(variant["scale"])
    resized = max(1, int(round(sprite.width * zoom)))
    scaled = sprite.resize((resized, resized), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    center_x = sprite.width / 2 + variant["dx"] * scale
    center_y = sprite.height / 2 + variant["dy"] * scale
    left = int(round(center_x - resized / 2))
    top = int(round(center_y - resized / 2))
    canvas.paste(scaled, (left, top), scaled)
    return canvas


def render_object(color: str, shape: str, texture: str, variant: dict | None = None) -> Image.Image:
    scale = SUPERSAMPLE
    size = IMAGE_SIZE * scale
    mask = Image.new("L", (size, size), 0)
    _draw_shape(ImageDraw.Draw(mask), shape, scale, fill=255)
    fill = Image.new("RGBA", (size, size), _hex(ALL_COLORS[color]))
    sprite = Image.composite(fill, Image.new("RGBA", (size, size), (0, 0, 0, 0)), mask)
    sprite = Image.alpha_composite(sprite, _draw_texture(mask, texture, scale))
    outline = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    _draw_shape(ImageDraw.Draw(outline), shape, scale, outline=INK + (255,), width=5 * scale)
    sprite = Image.alpha_composite(sprite, outline)
    sprite = _transform(sprite, variant)
    canvas = Image.new("RGB", (size, size), BACKGROUND)
    canvas.paste(sprite, mask=sprite.getchannel("A"))
    return canvas.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)


def save_object(path: Path, color: str, shape: str, texture: str, variant: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    render_object(color, shape, texture, variant).save(path, format="PNG")


def _hex(value: str) -> tuple[int, int, int, int]:
    value = value.removeprefix("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), 255)
