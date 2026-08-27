from __future__ import annotations

import base64
import io
import math
from typing import Any

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from .outline import COMMANDS


class _FlattenPen(BasePen):
    def __init__(self, glyph_set: Any, steps: int = 12):
        super().__init__(glyph_set)
        self.steps = steps
        self.contours: list[list[tuple[float, float]]] = []
        self.current: list[tuple[float, float]] = []
        self.point = (0.0, 0.0)

    def _moveTo(self, point: tuple[float, float]) -> None:
        if self.current:
            self.contours.append(self.current)
        self.current = [point]
        self.point = point

    def _lineTo(self, point: tuple[float, float]) -> None:
        self.current.append(point)
        self.point = point

    def _curveToOne(self, p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> None:
        p0 = self.point
        for index in range(1, self.steps + 1):
            t = index / self.steps
            mt = 1 - t
            self.current.append((
                mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0],
                mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1],
            ))
        self.point = p3

    def _qCurveToOne(self, p1: tuple[float, float], p2: tuple[float, float]) -> None:
        p0 = self.point
        for index in range(1, self.steps + 1):
            t = index / self.steps
            mt = 1 - t
            self.current.append((
                mt**2 * p0[0] + 2 * mt * t * p1[0] + t**2 * p2[0],
                mt**2 * p0[1] + 2 * mt * t * p1[1] + t**2 * p2[1],
            ))
        self.point = p2

    def _closePath(self) -> None:
        if self.current:
            self.contours.append(self.current)
        self.current = []

    def _endPath(self) -> None:
        self._closePath()


def glyph_mask(font: TTFont, character: str, size: int = 64, supersample: int = 4) -> Image.Image | None:
    cmap = font.getBestCmap() or {}
    glyph_name = cmap.get(ord(character))
    if glyph_name is None:
        return None
    glyph_set = font.getGlyphSet()
    pen = _FlattenPen(glyph_set)
    glyph_set[glyph_name].draw(pen)
    pen._endPath()
    canvas_size = size * supersample
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.bool_)
    units_per_em = float(font["head"].unitsPerEm)
    x_min, x_max = -0.18, 1.18
    y_min, y_max = -0.28, 1.08
    for contour in pen.contours:
        if len(contour) < 3:
            continue
        polygon = [
            (
                (x / units_per_em - x_min) / (x_max - x_min) * (canvas_size - 1),
                (y_max - y / units_per_em) / (y_max - y_min) * (canvas_size - 1),
            )
            for x, y in contour
            if math.isfinite(x) and math.isfinite(y)
        ]
        layer = Image.new("1", (canvas_size, canvas_size), 0)
        ImageDraw.Draw(layer).polygon(polygon, fill=1)
        canvas ^= np.asarray(layer, dtype=np.bool_)
    image = Image.fromarray(canvas.astype(np.uint8) * 255, mode="L")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def encode_mask(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def decode_mask(payload: str, size: int) -> np.ndarray:
    image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("L").resize((size, size))
    return np.asarray(image, dtype=np.float32) / 255.0


def signed_distance_field(mask: np.ndarray, maximum_distance: float = 16.0) -> np.ndarray:
    """Return a normalized SDF: positive inside, zero at the outline, negative outside."""
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional glyph mask")
    inside = np.asarray(mask >= 0.5, dtype=np.bool_)
    if not inside.any():
        return np.full(mask.shape, -1.0, dtype=np.float32)
    if inside.all():
        return np.full(mask.shape, 1.0, dtype=np.float32)
    distance_inside = distance_transform_edt(inside)
    distance_outside = distance_transform_edt(~inside)
    sdf = (distance_inside - distance_outside) / max(float(maximum_distance), 1.0)
    return np.clip(sdf, -1.0, 1.0).astype(np.float32)


def outline_mask(
    commands: list[int] | list[str],
    coordinates: list[list[float]],
    size: int = 128,
    supersample: int = 4,
    curve_steps: int = 12,
) -> Image.Image:
    """Rasterize normalized M/L/Q/C/Z training contours using even-odd fill."""
    contours: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    point = (0.0, 0.0)
    for raw_command, values in zip(commands, coordinates, strict=True):
        command = COMMANDS[int(raw_command)] if isinstance(raw_command, int) else raw_command
        if command == "M":
            if current:
                contours.append(current)
            point = (float(values[0]), float(values[1]))
            current = [point]
        elif command == "L" and current:
            point = (float(values[0]), float(values[1]))
            current.append(point)
        elif command == "Q" and current:
            control = (float(values[0]), float(values[1]))
            end = (float(values[2]), float(values[3]))
            start = point
            for index in range(1, curve_steps + 1):
                t = index / curve_steps
                mt = 1 - t
                current.append((
                    mt**2 * start[0] + 2 * mt * t * control[0] + t**2 * end[0],
                    mt**2 * start[1] + 2 * mt * t * control[1] + t**2 * end[1],
                ))
            point = end
        elif command == "C" and current:
            control1 = (float(values[0]), float(values[1]))
            control2 = (float(values[2]), float(values[3]))
            end = (float(values[4]), float(values[5]))
            start = point
            for index in range(1, curve_steps + 1):
                t = index / curve_steps
                mt = 1 - t
                current.append((
                    mt**3 * start[0] + 3 * mt**2 * t * control1[0] + 3 * mt * t**2 * control2[0] + t**3 * end[0],
                    mt**3 * start[1] + 3 * mt**2 * t * control1[1] + 3 * mt * t**2 * control2[1] + t**3 * end[1],
                ))
            point = end
        elif command == "Z" and current:
            contours.append(current)
            current = []
    if current:
        contours.append(current)
    canvas_size = size * supersample
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.bool_)
    x_min, x_max = -0.18, 1.18
    y_min, y_max = -0.28, 1.08
    for contour in contours:
        if len(contour) < 3:
            continue
        polygon = [
            (
                (x - x_min) / (x_max - x_min) * (canvas_size - 1),
                (y_max - y) / (y_max - y_min) * (canvas_size - 1),
            )
            for x, y in contour
            if math.isfinite(x) and math.isfinite(y)
        ]
        layer = Image.new("1", (canvas_size, canvas_size), 0)
        ImageDraw.Draw(layer).polygon(polygon, fill=1)
        canvas ^= np.asarray(layer, dtype=np.bool_)
    image = Image.fromarray(canvas.astype(np.uint8) * 255, mode="L")
    return image.resize((size, size), Image.Resampling.LANCZOS)
