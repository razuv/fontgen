"""Filter near-duplicate fonts from a JSONL manifest.

Two-stage similarity detection:
  1. Feature-vector cosine similarity (fast, coarse) using geometry features
     computed from glyph outlines.
  2. Raster IoU similarity (slower, perceptual) on canonical characters for
     pairs flagged by stage 1.

Usage:
    python scripts/filter_similar_fonts.py data/train.jsonl --output data/train_filtered.jsonl
    python scripts/filter_similar_fonts.py data/train.jsonl --output data/train_filtered.jsonl \
        --feature-threshold 0.92 --raster-threshold 0.80 --report data/duplicates.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fontgen_model.outline import COMMANDS, COORDINATE_MASKS
from fontgen_model.raster import outline_mask

CANONICAL_CHARACTERS = "AaHoxgp0"


# ---------------------------------------------------------------------------
# Feature extraction (mirrors typography.analyze_face but works from rows)
# ---------------------------------------------------------------------------

def _points_from_row(row: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for command_id, coordinates in zip(row["commands"], row["coordinates"], strict=True):
        count = COORDINATE_MASKS[COMMANDS[int(command_id)]]
        points.extend(
            (float(coordinates[i]), float(coordinates[i + 1]))
            for i in range(0, count, 2)
        )
    return points


def _median(values: list[float], fallback: float = 0.0) -> float:
    if not values:
        return fallback
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def compute_face_features(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_character = {str(r["character"]): r for r in rows}
    widths: list[float] = []
    heights: dict[str, float] = {}
    curve_commands = 0
    drawing_commands = 0
    contours: list[int] = []
    complexities: list[int] = []

    for row in rows:
        points = _points_from_row(row)
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            heights[str(row["character"])] = max(ys) - min(ys)
            if str(row["character"]) in "HOnoxaeАОНохае":
                widths.append(max(xs) - min(xs))
        names = [COMMANDS[int(c)] for c in row["commands"]]
        curve_commands += sum(n in {"Q", "C"} for n in names)
        drawing_commands += sum(n in {"L", "Q", "C"} for n in names)
        contours.append(names.count("Z"))
        complexities.append(sum(n not in {"PAD", "BOS", "EOS"} for n in names))

    cap_height = _median([heights[c] for c in "HOEIАОНП" if c in heights], 0.7)
    x_height = _median([heights[c] for c in "xoeaохеа" if c in heights], cap_height * 0.7)
    descenders: list[float] = []
    for ch in "pqgyфруцд":
        r = by_character.get(ch)
        if r:
            pts = _points_from_row(r)
            if pts:
                descenders.append(min(p[1] for p in pts))

    controls = [float(v) for v in rows[0].get("controls", [0, 0, 0, 0, 0])]
    return {
        "mean_glyph_width": _median(widths, float(rows[0].get("advance_width", 0.6))),
        "x_height": x_height,
        "cap_height": cap_height,
        "x_height_ratio": x_height / max(cap_height, 0.01),
        "descender_depth": abs(min(descenders, default=0.0)),
        "curve_ratio": curve_commands / max(drawing_commands, 1),
        "mean_complexity": _median([float(v) for v in complexities]),
        "mean_contours": _median([float(v) for v in contours]),
        "weight_axis": controls[0],
        "width_axis": controls[1],
        "contrast_axis": controls[2],
        "roundness_axis": controls[3],
        "slant_axis": controls[4],
    }


FEATURE_KEYS = [
    "mean_glyph_width", "x_height", "cap_height", "x_height_ratio",
    "descender_depth", "curve_ratio", "mean_complexity", "mean_contours",
    "weight_axis", "width_axis", "contrast_axis", "roundness_axis", "slant_axis",
]


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([features[k] for k in FEATURE_KEYS], dtype=np.float32)


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. vectors shape: (n, d)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = vectors / norms
    return normalized @ normalized.T


def raster_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two binary raster masks (float, 0-1 range)."""
    a = mask_a >= 0.5
    b = mask_b >= 0.5
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def mean_raster_iou(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    characters: str,
    size: int = 64,
) -> float:
    """Average IoU over canonical characters present in both faces."""
    by_char_a = {str(r["character"]): r for r in rows_a}
    by_char_b = {str(r["character"]): r for r in rows_b}
    ious: list[float] = []
    for ch in characters:
        ra = by_char_a.get(ch)
        rb = by_char_b.get(ch)
        if not ra or not rb:
            continue
        try:
            mask_a = _render_mask(ra, size)
            mask_b = _render_mask(rb, size)
            ious.append(raster_iou(mask_a, mask_b))
        except Exception:
            pass
    if not ious:
        return 0.0
    return sum(ious) / len(ious)


def _render_mask(row: dict[str, Any], size: int) -> np.ndarray:
    """Render a manifest row to a float raster mask."""
    img = outline_mask(row["commands"], row["coordinates"], size=size, supersample=2)
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Clustering (union-find)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_by_face(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    faces: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["family"]), str(row.get("subfamily", "Regular")))
        faces[key].append(row)
    return dict(faces)


def face_quality_score(face_rows: list[dict[str, Any]]) -> float:
    """Heuristic quality: prefer faces with more glyphs and richer features."""
    glyph_count = len(face_rows)
    has_geometry = "geometry_features" in face_rows[0]
    return float(glyph_count) + (100.0 if has_geometry else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter near-duplicate fonts from JSONL manifest")
    parser.add_argument("manifest", type=Path, help="Input JSONL manifest")
    parser.add_argument("--output", type=Path, required=True, help="Filtered output JSONL")
    parser.add_argument("--feature-threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for feature-based flagging (default: 0.85)")
    parser.add_argument("--raster-threshold", type=float, default=0.85,
                        help="Mean IoU threshold for raster-based confirmation (default: 0.85)")
    parser.add_argument("--characters", default=CANONICAL_CHARACTERS,
                        help=f"Characters for raster comparison (default: {CANONICAL_CHARACTERS})")
    parser.add_argument("--raster-size", type=int, default=64,
                        help="Raster size for IoU comparison (default: 64)")
    parser.add_argument("--skip-raster", action="store_true",
                        help="Skip stage 2 raster comparison, use feature similarity only")
    parser.add_argument("--report", type=Path, default=None, help="Write duplicate report JSON")
    args = parser.parse_args()

    print(f"loading {args.manifest}...", file=sys.stderr)
    rows = load_manifest(args.manifest)
    print(f"  {len(rows)} glyph rows", file=sys.stderr)

    faces = group_by_face(rows)
    face_keys = list(faces.keys())
    n_faces = len(face_keys)
    print(f"  {n_faces} faces", file=sys.stderr)

    # --- Stage 1: feature similarity (per-category) ---
    print("stage 1: computing face features...", file=sys.stderr)
    feature_vectors = np.zeros((n_faces, len(FEATURE_KEYS)), dtype=np.float32)
    face_categories: list[str] = []
    for i, key in enumerate(face_keys):
        face_rows = faces[key]
        if "geometry_features" in face_rows[0]:
            features = face_rows[0]["geometry_features"]
        else:
            features = compute_face_features(face_rows)
        feature_vectors[i] = features_to_vector(features)
        face_categories.append(str(face_rows[0].get("category", "UNKNOWN")))

    # Normalize features (z-score) per-category for better discrimination
    normalized = np.zeros_like(feature_vectors)
    for cat in set(face_categories):
        idx = [i for i, c in enumerate(face_categories) if c == cat]
        if len(idx) < 2:
            for i in idx:
                normalized[i] = 0.0
            continue
        cat_vectors = feature_vectors[idx]
        mean = cat_vectors.mean(axis=0)
        std = cat_vectors.std(axis=0)
        std = np.maximum(std, 1e-8)
        for i in idx:
            normalized[i] = (feature_vectors[i] - mean) / std

    print("stage 1: computing pairwise cosine similarity (per-category)...", file=sys.stderr)
    candidate_pairs: list[tuple[int, int, float]] = []
    categories = sorted(set(face_categories))
    for cat in categories:
        idx = [i for i, c in enumerate(face_categories) if c == cat]
        if len(idx) < 2:
            continue
        cat_vectors = normalized[idx]
        sim_matrix = cosine_similarity_matrix(cat_vectors)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                sim = float(sim_matrix[a, b])
                if sim >= args.feature_threshold:
                    candidate_pairs.append((idx[a], idx[b], sim))

    print(f"  {len(candidate_pairs)} pairs above feature threshold {args.feature_threshold}", file=sys.stderr)

    # --- Stage 2: raster IoU confirmation ---
    confirmed_pairs: list[tuple[int, int, float, float]] = []  # (i, j, feature_sim, raster_iou)

    if args.skip_raster:
        confirmed_pairs = [(i, j, sim, 0.0) for i, j, sim in candidate_pairs]
        print("stage 2: skipped (--skip-raster)", file=sys.stderr)
    elif candidate_pairs:
        print(f"stage 2: raster IoU on {len(candidate_pairs)} candidate pairs...", file=sys.stderr)
        for idx, (i, j, sim) in enumerate(candidate_pairs):
            rows_i = faces[face_keys[i]]
            rows_j = faces[face_keys[j]]
            iou = mean_raster_iou(rows_i, rows_j, args.characters, args.raster_size)
            if iou >= args.raster_threshold:
                confirmed_pairs.append((i, j, sim, iou))
            if (idx + 1) % 100 == 0 or (idx + 1) == len(candidate_pairs):
                print(f"\r  [{idx + 1}/{len(candidate_pairs)}] confirmed: {len(confirmed_pairs)}",
                      end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)

    print(f"  {len(confirmed_pairs)} confirmed duplicate pairs", file=sys.stderr)

    # --- Clustering ---
    uf = _UnionFind(n_faces)
    for i, j, *_ in confirmed_pairs:
        uf.union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n_faces):
        clusters[uf.find(i)].append(i)

    # Decide which face to keep per cluster
    removed_faces: set[tuple[str, str]] = set()
    duplicate_groups: list[dict[str, Any]] = []

    for root, members in clusters.items():
        if len(members) <= 1:
            continue
        # Rank: prefer higher quality, then more glyphs, then alphabetical
        ranked = sorted(members, key=lambda m: (-face_quality_score(faces[face_keys[m]]), face_keys[m]))
        keep = ranked[0]
        drop = ranked[1:]
        for d in drop:
            removed_faces.add(face_keys[d])

        group_info: dict[str, Any] = {
            "kept": {"family": face_keys[keep][0], "subfamily": face_keys[keep][1],
                     "glyphs": len(faces[face_keys[keep]])},
            "removed": [],
        }
        for d in drop:
            group_info["removed"].append({
                "family": face_keys[d][0],
                "subfamily": face_keys[d][1],
                "glyphs": len(faces[face_keys[d]]),
            })
        # Attach similarity scores
        for i, j, sim, iou in confirmed_pairs:
            if (i in members) and (j in members):
                group_info["feature_similarity"] = round(sim, 4)
                if iou > 0:
                    group_info["raster_iou"] = round(iou, 4)
                break
        duplicate_groups.append(group_info)

    # --- Write filtered output ---
    kept_rows = [r for r in rows if (str(r["family"]), str(r.get("subfamily", "Regular"))) not in removed_faces]
    removed_rows = len(rows) - len(kept_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"\n=== Filter Results ===", file=sys.stderr)
    print(f"  Input faces:    {n_faces}", file=sys.stderr)
    print(f"  Removed faces:  {len(removed_faces)}", file=sys.stderr)
    print(f"  Kept faces:     {n_faces - len(removed_faces)}", file=sys.stderr)
    print(f"  Input rows:     {len(rows)}", file=sys.stderr)
    print(f"  Removed rows:   {removed_rows}", file=sys.stderr)
    print(f"  Output rows:    {len(kept_rows)}", file=sys.stderr)
    print(f"  Output:         {args.output}", file=sys.stderr)

    # --- Write report ---
    if args.report:
        report = {
            "input_faces": n_faces,
            "removed_faces": len(removed_faces),
            "kept_faces": n_faces - len(removed_faces),
            "input_rows": len(rows),
            "removed_rows": removed_rows,
            "output_rows": len(kept_rows),
            "feature_threshold": args.feature_threshold,
            "raster_threshold": args.raster_threshold,
            "duplicate_groups": duplicate_groups,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Report:         {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
