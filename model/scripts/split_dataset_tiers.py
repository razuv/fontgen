"""Split a manifest into nested quality-ranked tiers by font count.

Reads the full manifest, groups by face, ranks by quality (coverage, glyphs),
and writes 5 tier files: tier_500, tier_1000, tier_2000, tier_5000, tier_10000.

Each tier is a superset of the previous — tier_1000 contains all faces from
tier_500 plus 500 more. This ensures consistent comparison across model sizes.

Usage:
    python scripts/split_dataset_tiers.py data/train.jsonl --output-dir data/tiers
    python scripts/split_dataset_tiers.py data/train.jsonl --output-dir data/tiers --tiers 500 1000 2000 5000 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

TIER_SIZES = [500, 1000, 2000, 5000, 10000]


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


def face_sort_key(face_rows: list[dict[str, Any]]) -> tuple[float, float, str]:
    """Rank faces: higher quality first, then more glyphs, then alphabetical."""
    row0 = face_rows[0]
    # Quality score from audit if available, else heuristic
    quality = float(row0.get("quality_score", 0.0))
    if quality == 0.0:
        # Heuristic: prefer faces with more glyphs and geometry features
        quality = len(face_rows) / 100.0
        if "geometry_features" in row0:
            quality += 0.5
    glyph_count = float(len(face_rows))
    family = str(row0["family"])
    return (-quality, -glyph_count, family)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split manifest into nested quality-ranked tiers")
    parser.add_argument("manifest", type=Path, help="Input JSONL manifest (full dataset)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/tiers"),
                        help="Output directory for tier files")
    parser.add_argument("--tiers", type=int, nargs="+", default=TIER_SIZES,
                        help=f"Tier sizes in font count (default: {TIER_SIZES})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for tie-breaking within quality buckets")
    args = parser.parse_args()

    args.tiers = sorted(args.tiers)

    print(f"loading {args.manifest}...", file=sys.stderr)
    rows = load_manifest(args.manifest)
    print(f"  {len(rows)} glyph rows", file=sys.stderr)

    faces = group_by_face(rows)
    face_keys = list(faces.keys())
    print(f"  {len(face_keys)} unique faces", file=sys.stderr)

    if len(face_keys) < args.tiers[-1]:
        print(f"  WARNING: only {len(face_keys)} faces available, "
              f"largest tier requested is {args.tiers[-1]}", file=sys.stderr)

    # Sort faces by quality
    ranked_keys = sorted(face_keys, key=lambda k: face_sort_key(faces[k]))

    # Print top faces for verification
    print(f"\n  top 10 faces by quality:", file=sys.stderr)
    for i, key in enumerate(ranked_keys[:10]):
        r = faces[key][0]
        q = float(r.get("quality_score", 0))
        print(f"    {i + 1}. {key[0]} / {key[1]}  "
              f"quality={q:.3f}  glyphs={len(faces[key])}", file=sys.stderr)

    # Write tier files
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_name = args.manifest.stem

    for tier_size in args.tiers:
        tier_keys = ranked_keys[:min(tier_size, len(ranked_keys))]
        tier_rows: list[dict[str, Any]] = []
        for key in tier_keys:
            tier_rows.extend(faces[key])

        tier_path = args.output_dir / f"{manifest_name}_tier_{tier_size}.jsonl"
        with tier_path.open("w", encoding="utf-8") as f:
            for row in tier_rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

        families = len({k[0] for k in tier_keys})
        print(f"  tier_{tier_size}: {len(tier_keys)} faces, "
              f"{families} families, {len(tier_rows)} rows -> {tier_path}", file=sys.stderr)

    # Write tier index
    index_path = args.output_dir / "tiers.json"
    index = {
        "source": str(args.manifest),
        "total_faces": len(face_keys),
        "total_rows": len(rows),
        "tiers": [],
    }
    for tier_size in args.tiers:
        actual_faces = min(tier_size, len(face_keys))
        tier_path = args.output_dir / f"{manifest_name}_tier_{tier_size}.jsonl"
        index["tiers"].append({
            "name": f"tier_{tier_size}",
            "target_faces": tier_size,
            "actual_faces": actual_faces,
            "path": str(tier_path),
        })
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\n  index -> {index_path}", file=sys.stderr)
    print("done.", file=sys.stderr)


if __name__ == "__main__":
    main()
