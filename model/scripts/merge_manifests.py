from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge outline manifests and remove duplicate face glyphs")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, object]] = []
    duplicates = 0
    for manifest in args.inputs:
        with manifest.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    str(row["family"]).casefold().strip(),
                    str(row.get("subfamily", "regular")).casefold().strip(),
                    str(row["character"]),
                )
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    families = {str(row["family"]) for row in rows}
    faces = {(str(row["family"]), str(row.get("subfamily", "Regular"))) for row in rows}
    categories = Counter(str(row.get("category", "UNKNOWN")) for row in rows)
    sources = Counter(str(row.get("source", "google-fonts")) for row in rows)
    print(
        f"rows={len(rows)} families={len(families)} faces={len(faces)} "
        f"duplicates={duplicates} output={args.output}"
    )
    print("categories=" + " ".join(f"{key}:{value}" for key, value in sorted(categories.items())))
    print("sources=" + " ".join(f"{key}:{value}" for key, value in sorted(sources.items())))


if __name__ == "__main__":
    main()
