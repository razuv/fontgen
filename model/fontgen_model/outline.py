from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont

COMMANDS = ("PAD", "BOS", "EOS", "M", "L", "Q", "C", "Z")
COMMAND_TO_ID = {name: index for index, name in enumerate(COMMANDS)}
COORDINATE_MASKS = {
    "PAD": 0,
    "BOS": 0,
    "EOS": 0,
    "M": 2,
    "L": 2,
    "Q": 4,
    "C": 6,
    "Z": 0,
}


@dataclass
class EncodedGlyph:
    character: str
    commands: list[int]
    coordinates: list[list[float]]
    advance_width: float
    left_side_bearing: float


def _normalized_point(point: tuple[float, float], units_per_em: int) -> tuple[float, float]:
    # Baseline remains at zero; one em maps to [-1, 1] for stable regression.
    return (float(point[0]) / units_per_em, float(point[1]) / units_per_em)


def _command(name: str, points: Iterable[tuple[float, float]], units_per_em: int) -> tuple[int, list[float]]:
    flattened: list[float] = []
    for point in points:
        flattened.extend(_normalized_point(point, units_per_em))
    return COMMAND_TO_ID[name], (flattened + [0.0] * 6)[:6]


class _OutlinePen(BasePen):
    def __init__(self, glyph_set: Any, units_per_em: int):
        super().__init__(glyph_set)
        self.units_per_em = units_per_em
        self.commands: list[int] = [COMMAND_TO_ID["BOS"]]
        self.coordinates: list[list[float]] = [[0.0] * 6]

    def _append(self, name: str, points: Iterable[tuple[float, float]] = ()) -> None:
        token, coordinates = _command(name, points, self.units_per_em)
        self.commands.append(token); self.coordinates.append(coordinates)

    def _moveTo(self, point: tuple[float, float]) -> None:
        self._append("M", [point])

    def _lineTo(self, point: tuple[float, float]) -> None:
        self._append("L", [point])

    def _curveToOne(self, point1: tuple[float, float], point2: tuple[float, float], point3: tuple[float, float]) -> None:
        self._append("C", [point1, point2, point3])

    def _qCurveToOne(self, point1: tuple[float, float], point2: tuple[float, float]) -> None:
        self._append("Q", [point1, point2])

    def _closePath(self) -> None:
        self._append("Z")

    def _endPath(self) -> None:
        self._append("Z")


def extract_glyph(font: TTFont, character: str, max_commands: int) -> EncodedGlyph | None:
    cmap = font.getBestCmap() or {}
    glyph_name = cmap.get(ord(character))
    if glyph_name is None:
        return None
    glyph_set = font.getGlyphSet()
    units_per_em = int(font["head"].unitsPerEm)
    pen = _OutlinePen(glyph_set, units_per_em)
    glyph_set[glyph_name].draw(pen)
    if len(pen.commands) >= max_commands - 1:
        return None
    pen.commands.append(COMMAND_TO_ID["EOS"]); pen.coordinates.append([0.0] * 6)
    advance, side_bearing = font["hmtx"].metrics[glyph_name]
    return EncodedGlyph(
        character=character,
        commands=pen.commands,
        coordinates=pen.coordinates,
        advance_width=float(advance) / units_per_em,
        left_side_bearing=float(side_bearing) / units_per_em,
    )


def inspect_font(path: Path) -> dict[str, Any]:
    font = TTFont(path, lazy=True)
    names = font["name"]
    family = names.getDebugName(1) or path.stem
    subfamily = names.getDebugName(2) or "Regular"
    os2 = font.get("OS/2")
    return {
        "path": str(path),
        "family": family,
        "subfamily": subfamily,
        "weight": int(getattr(os2, "usWeightClass", 400)),
        "width": int(getattr(os2, "usWidthClass", 5)),
        "italic": bool(getattr(os2, "fsSelection", 0) & 1),
    }


def pad_encoded(glyph: EncodedGlyph, max_commands: int) -> tuple[np.ndarray, np.ndarray]:
    command_array = np.zeros(max_commands, dtype=np.int64)
    coordinate_array = np.zeros((max_commands, 6), dtype=np.float32)
    length = min(max_commands, len(glyph.commands))
    command_array[:length] = glyph.commands[:length]
    coordinate_array[:length] = glyph.coordinates[:length]
    return command_array, coordinate_array
