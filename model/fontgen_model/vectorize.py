from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage import filters, measure, morphology, transform

X_MIN, X_MAX = -0.18, 1.18
Y_MIN, Y_MAX = -0.28, 1.08


@dataclass
class VectorOutline:
    commands: list[str]
    coordinates: list[list[float]]


def _signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _to_em(contour: np.ndarray, size: int) -> np.ndarray:
    row = contour[:, 0]
    column = contour[:, 1]
    x = X_MIN + column / (size - 1) * (X_MAX - X_MIN)
    y = Y_MAX - row / (size - 1) * (Y_MAX - Y_MIN)
    return np.column_stack((x, y)).astype(np.float32)


def _deduplicate(points: np.ndarray, minimum_distance: float = 0.004) -> np.ndarray:
    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) >= minimum_distance:
            kept.append(point)
    if len(kept) > 2 and np.linalg.norm(kept[0] - kept[-1]) < minimum_distance:
        kept.pop()
    return np.asarray(kept, dtype=np.float32)


def _line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return float(np.linalg.norm(point - start))
    offset = point - start
    return float(abs(direction[0] * offset[1] - direction[1] * offset[0]) / length)


def _straight_segment(
    previous: np.ndarray, start: np.ndarray, end: np.ndarray, following: np.ndarray,
) -> bool:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 0.025:
        return False
    axis_error = float(min(abs(direction[0]), abs(direction[1])) / length)
    long_construction_line = length >= 0.13
    clean_axis = length >= 0.055 and axis_error <= 0.022
    locally_collinear = (
        _line_distance(previous, start, end) <= 0.006
        and _line_distance(following, start, end) <= 0.006
    )
    return long_construction_line or clean_axis or locally_collinear


def _limited_handle(anchor: np.ndarray, candidate: np.ndarray, chord: float) -> np.ndarray:
    offset = candidate - anchor
    length = float(np.linalg.norm(offset))
    maximum = chord * 0.38
    if length <= maximum or length < 1e-6:
        return candidate
    return anchor + offset * (maximum / length)


def vectorize_mask(
    mask: np.ndarray,
    threshold: float = 0.48,
    contrast: float = 0.0,
    roundness: float = 0.0,
) -> VectorOutline:
    """Trace a neural glyph mask into smooth, closed cubic contours.

    Topology is recovered from the filled silhouette, not sampled from an
    autoregressive command stream. Even/odd contour direction is retained by
    marching squares, so counters survive as independent closed paths.
    """
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional glyph mask")
    source_size = int(mask.shape[0])
    clipped = np.clip(mask, 0, 1)
    contrast = float(np.clip(contrast, -1, 1))
    roundness = float(np.clip(roundness, -1, 1))
    # Marching squares on the native 128 px binary mask created stair steps that
    # later became dozens of crooked Bézier segments. Interpolate the probability
    # field first and extract a sub-pixel level set after topology cleanup.
    scale = 4
    size = source_size * scale
    smooth = transform.resize(
        clipped, (size, size), order=3, mode="edge", anti_aliasing=False, preserve_range=True,
    )
    base_sigma = 2.5 + roundness * 0.65
    smooth = filters.gaussian(smooth, sigma=base_sigma, preserve_range=True)
    if contrast:
        broad = filters.gaussian(smooth, sigma=4.2, preserve_range=True)
        smooth = np.clip(smooth + contrast * 0.45 * (smooth - broad), 0, 1)
    binary = smooth >= threshold
    closing_radius = 0 if roundness < -0.35 else 4 if roundness > 0.55 else 2
    if closing_radius:
        binary = morphology.closing(binary, footprint=morphology.disk(closing_radius))
    cleanup_area = max(24, round(size * size * 0.0007))
    binary = morphology.remove_small_objects(binary, max_size=cleanup_area)
    binary = morphology.remove_small_holes(binary, max_size=cleanup_area)
    # Preserve the continuous probability/SDF level set. Morphology only decides
    # which components and counters survive; it must not quantize the boundary.
    cleaned_field = smooth.copy()
    support = morphology.dilation(binary, footprint=morphology.disk(2))
    core = morphology.erosion(binary, footprint=morphology.disk(2))
    cleaned_field[~support] = 0.0
    cleaned_field[core] = np.maximum(cleaned_field[core], threshold + 0.1)
    contours = measure.find_contours(cleaned_field, 0.5, fully_connected="high")
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        if len(contour) < 8:
            continue
        simplified = measure.approximate_polygon(contour, tolerance=1.4 * scale)
        if len(simplified) < 4:
            continue
        if np.linalg.norm(simplified[0] - simplified[-1]) < 1.5:
            simplified = simplified[:-1]
        points = _deduplicate(_to_em(simplified, size))
        area = _signed_area(points)
        if abs(area) < 0.0015:
            continue
        candidates.append((abs(area), points))
    candidates.sort(key=lambda item: item[0], reverse=True)
    commands: list[str] = []
    coordinates: list[list[float]] = []
    for _area, points in candidates[:12]:
        if len(points) > 40:
            stride = max(1, math.ceil(len(points) / 40))
            points = points[::stride]
        commands.append("M"); coordinates.append([float(points[0, 0]), float(points[0, 1]), 0, 0, 0, 0])
        tension = 1 / 6
        count = len(points)
        for index in range(count):
            previous = points[(index - 1) % count]
            start = points[index]
            end = points[(index + 1) % count]
            following = points[(index + 2) % count]
            if _straight_segment(previous, start, end, following):
                commands.append("L")
                coordinates.append([float(end[0]), float(end[1]), 0, 0, 0, 0])
                continue
            chord = float(np.linalg.norm(end - start))
            control1 = _limited_handle(start, start + (end - previous) * tension, chord)
            control2 = _limited_handle(end, end - (following - start) * tension, chord)
            commands.append("C")
            coordinates.append([
                float(control1[0]), float(control1[1]),
                float(control2[0]), float(control2[1]),
                float(end[0]), float(end[1]),
            ])
        commands.append("Z"); coordinates.append([0, 0, 0, 0, 0, 0])
    return VectorOutline(commands, coordinates)
