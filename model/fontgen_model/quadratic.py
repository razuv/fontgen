from __future__ import annotations

import math
from typing import Any

from fontTools.cu2qu.cu2qu import curve_to_quadratic

QUADRATIC_COMMANDS = ("PAD", "BOS", "EOS", "M", "L", "Q", "Z")
QUADRATIC_COMMAND_TO_ID = {name: index for index, name in enumerate(QUADRATIC_COMMANDS)}
QUADRATIC_COORDINATE_COUNTS = (0, 0, 0, 2, 2, 4, 0)


def canonicalize_quadratic(row: dict[str, Any], max_error: float = 0.001) -> dict[str, Any] | None:
    """Convert mixed line/quadratic/cubic outlines to a quadratic-only grammar."""
    from .outline import COMMANDS

    output_commands = [QUADRATIC_COMMAND_TO_ID["BOS"]]
    output_coordinates = [[0.0] * 4]
    point = (0.0, 0.0)
    for raw_command, raw_coordinates in zip(row["commands"], row["coordinates"], strict=True):
        command = COMMANDS[int(raw_command)]
        coordinates = [float(value) for value in raw_coordinates]
        if command in {"PAD", "BOS", "EOS"}:
            continue
        if command == "M" or command == "L":
            point = _snap_point((coordinates[0], coordinates[1]))
            output_commands.append(QUADRATIC_COMMAND_TO_ID[command])
            output_coordinates.append([point[0], point[1], 0.0, 0.0])
        elif command == "Q":
            control = _snap_point((coordinates[0], coordinates[1]))
            end = _snap_point((coordinates[2], coordinates[3]))
            _append_quadratic(output_commands, output_coordinates, point, control, end)
            point = end
        elif command == "C":
            control1 = (coordinates[0], coordinates[1])
            control2 = (coordinates[2], coordinates[3])
            end = _snap_point((coordinates[4], coordinates[5]))
            spline = curve_to_quadratic([point, control1, control2, end], max_error, all_quadratic=True)
            off_curves = [tuple(map(float, candidate)) for candidate in spline[1:-1]]
            for index, control in enumerate(off_curves):
                segment_end = end if index == len(off_curves) - 1 else (
                    (control[0] + off_curves[index + 1][0]) / 2,
                    (control[1] + off_curves[index + 1][1]) / 2,
                )
                _append_quadratic(
                    output_commands, output_coordinates, point,
                    _snap_point(control), _snap_point(segment_end),
                )
                point = _snap_point(segment_end)
        elif command == "Z":
            output_commands.append(QUADRATIC_COMMAND_TO_ID["Z"])
            output_coordinates.append([0.0] * 4)
    output_commands.append(QUADRATIC_COMMAND_TO_ID["EOS"])
    output_coordinates.append([0.0] * 4)
    converted = dict(row)
    converted["commands"] = output_commands
    converted["coordinates"] = output_coordinates
    converted["representation"] = "quadratic-v1"
    converted["original_command_count"] = len(row["commands"])
    converted.pop("raster", None)
    return converted


def _append_quadratic(
    commands: list[int], coordinates: list[list[float]],
    start: tuple[float, float], control: tuple[float, float], end: tuple[float, float],
) -> None:
    if _distance_to_line(control, start, end) < 0.0008:
        commands.append(QUADRATIC_COMMAND_TO_ID["L"])
        coordinates.append([end[0], end[1], 0.0, 0.0])
    else:
        commands.append(QUADRATIC_COMMAND_TO_ID["Q"])
        coordinates.append([control[0], control[1], end[0], end[1]])


def _snap_point(point: tuple[float, float], precision: int = 5) -> tuple[float, float]:
    return round(float(point[0]), precision), round(float(point[1]), precision)


def _distance_to_line(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-8:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    return abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / length
