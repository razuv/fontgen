from __future__ import annotations

import re
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
    family = names.getDebugName(16) or names.getDebugName(1) or path.stem
    subfamily = names.getDebugName(2) or "Regular"
    os2 = font.get("OS/2")
    panose = getattr(os2, "panose", None)
    google_metadata = _google_fonts_metadata(path)
    embedded_license = names.getDebugName(13)
    embedded_license_url = names.getDebugName(14)
    inferred_category = _panose_category(font, panose, family)
    return {
        "path": str(path),
        "family": family,
        "subfamily": subfamily,
        "weight": int(getattr(os2, "usWeightClass", 400)),
        "width": int(getattr(os2, "usWidthClass", 5)),
        "italic": bool(getattr(os2, "fsSelection", 0) & 1),
        "panose_contrast": int(getattr(panose, "bContrast", 0)),
        "panose_letterform": int(getattr(panose, "bLetterForm", 0)),
        **google_metadata,
        "license": google_metadata.get("license") or embedded_license,
        "license_id": google_metadata.get("license") or _embedded_license_id(embedded_license),
        "license_url": embedded_license_url,
        "category": google_metadata.get("category") or inferred_category,
    }


def _embedded_license_id(license_text: str | None) -> str | None:
    normalized = (license_text or "").lower()
    if "open font license" in normalized or "sil ofl" in normalized:
        return "OFL-1.1"
    if "apache license" in normalized:
        return "Apache-2.0"
    if "ubuntu font licence" in normalized:
        return "UFL-1.0"
    return None


def _panose_category(font: TTFont, panose: Any, family: str) -> str:
    post = font.get("post")
    normalized_family = family.casefold()
    if bool(getattr(post, "isFixedPitch", 0)) or any(
        token in normalized_family for token in ("mono", "code", "typewriter")
    ):
        return "MONOSPACE"
    os2 = font.get("OS/2")
    ibm_family_class = (int(getattr(os2, "sFamilyClass", 0)) >> 8) & 0xFF
    if ibm_family_class == 10:
        return "HANDWRITING"
    if ibm_family_class == 9:
        return "DISPLAY"
    if ibm_family_class in {1, 2, 3, 4, 5, 7}:
        return "SERIF"
    if ibm_family_class == 8:
        return "SANS_SERIF"
    family_type = int(getattr(panose, "bFamilyType", 0))
    serif_style = int(getattr(panose, "bSerifStyle", 0))
    if family_type == 3:
        return "HANDWRITING"
    if family_type in {4, 5}:
        return "DISPLAY"
    if family_type == 2 and 2 <= serif_style <= 10:
        return "SERIF"
    if any(token in normalized_family for token in ("script", "hand", "cursive", "brush", "calligraph")):
        return "HANDWRITING"
    if any(token in normalized_family for token in (
        "serif", "antiqua", "garamond", "bodoni", "didot", "baskerville", "clarendon", "slab",
    )):
        return "SERIF"
    if any(token in normalized_family for token in ("display", "poster", "deco")):
        return "DISPLAY"
    return "SANS_SERIF"


def _google_fonts_metadata(path: Path) -> dict[str, Any]:
    """Read the small, stable subset of Google Fonts METADATA.pb we train on."""
    metadata_path = path.parent / "METADATA.pb"
    if not metadata_path.exists():
        return {
            "license": None, "category": None, "subsets": [],
            "classifications": [], "stroke": None,
        }
    source = metadata_path.read_text(encoding="utf-8")

    def scalar(name: str) -> str | None:
        match = re.search(rf'^\s*{name}:\s*"?([^"\n]+)"?\s*$', source, re.MULTILINE)
        return match.group(1).strip() if match else None

    return {
        "license": scalar("license"),
        "category": scalar("category"),
        "subsets": re.findall(r'^\s*subsets:\s*"([^"]+)"', source, re.MULTILINE),
        "classifications": re.findall(r'^\s*classifications:\s*"([^"]+)"', source, re.MULTILINE),
        "stroke": scalar("stroke"),
    }


def pad_encoded(glyph: EncodedGlyph, max_commands: int) -> tuple[np.ndarray, np.ndarray]:
    command_array = np.zeros(max_commands, dtype=np.int64)
    coordinate_array = np.zeros((max_commands, 6), dtype=np.float32)
    length = min(max_commands, len(glyph.commands))
    command_array[:length] = glyph.commands[:length]
    coordinate_array[:length] = glyph.coordinates[:length]
    return command_array, coordinate_array
