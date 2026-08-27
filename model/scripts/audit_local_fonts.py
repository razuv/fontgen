from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from fontTools.ttLib import TTFont

from fontgen_model.outline import inspect_font

CYRILLIC_PROBE = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯабвгдежзийклмнопрстуфхцчшщыэюя"
OPEN_LICENSE_MARKERS = (
    "open font license", "sil ofl", "apache license", "ubuntu font licence",
)


def license_status(path: Path, metadata: dict[str, object]) -> str:
    haystack = " ".join(
        (path.name, str(metadata.get("license") or ""), str(metadata.get("license_url") or ""))
    ).lower()
    if "trial" in haystack or "demo" in haystack:
        return "trial"
    if any(marker in haystack for marker in OPEN_LICENSE_MARKERS):
        return "open"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local fonts before adding them to a training corpus")
    parser.add_argument("font_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    args = parser.parse_args()
    reference_categories: dict[str, str] = {}
    if args.reference_manifest:
        with args.reference_manifest.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    reference = json.loads(line)
                    reference_categories[str(reference["family"]).casefold()] = str(reference["category"])
    rows: list[dict[str, object]] = []
    paths = sorted([*args.font_dir.rglob("*.ttf"), *args.font_dir.rglob("*.otf")])
    for path in paths:
        try:
            metadata = inspect_font(path)
            metadata["category"] = reference_categories.get(
                str(metadata["family"]).casefold(), str(metadata["category"]),
            )
            cmap = TTFont(path, lazy=True).getBestCmap() or {}
            row = {
                **metadata,
                "path": str(path),
                "license_status": license_status(path, metadata),
                "cyrillic_coverage": round(sum(ord(char) in cmap for char in CYRILLIC_PROBE) / len(CYRILLIC_PROBE), 3),
                "glyph_count": len(cmap),
            }
            rows.append(row)
        except Exception as error:  # noqa: BLE001
            rows.append({"path": str(path), "license_status": "broken", "error": str(error)})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    statuses = Counter(str(row["license_status"]) for row in rows)
    families = {str(row.get("family")) for row in rows if row.get("family")}
    cyrillic = sum(float(row.get("cyrillic_coverage", 0)) >= 0.9 for row in rows)
    categories = Counter(str(row.get("category")) for row in rows if row.get("category"))
    print(f"files={len(rows)} families={len(families)} cyrillic>=90%={cyrillic}")
    print("licenses=" + " ".join(f"{key}:{value}" for key, value in sorted(statuses.items())))
    print("categories=" + " ".join(f"{key}:{value}" for key, value in sorted(categories.items())))


if __name__ == "__main__":
    main()
