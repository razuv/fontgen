from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from fontgen_model.typography import analyze_face, make_tags, prompt_variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Add geometry-derived typography tags and prompt variants")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    with args.manifest.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    faces: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["family"]), str(row.get("subfamily", "Regular")))
        faces[key].append(row)
    prompt_set: set[str] = set()
    tag_counts: Counter[str] = Counter()
    for key, face_rows in faces.items():
        features = analyze_face(face_rows)
        category = str(face_rows[0].get("category") or "SANS_SERIF")
        tags = make_tags(category, features)
        variants = prompt_variants(category, tags, "|".join(key))
        for index, row in enumerate(face_rows):
            row["category"] = category
            row["geometry_features"] = features
            row["auto_tags"] = tags
            row["prompt"] = variants[(ord(str(row["character"])[0]) + index) % len(variants)]
            prompt_set.add(str(row["prompt"]))
            tag_counts.update(tags)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"rows={len(rows)} faces={len(faces)} prompts={len(prompt_set)} output={args.output}")
    for prefix in ("category", "subclass", "weight", "width", "xheight", "contrast", "roundness", "curves", "complexity", "slant"):
        values = {tag: count for tag, count in sorted(tag_counts.items()) if tag.startswith(prefix + ":")}
        print(prefix + "=" + " ".join(f"{tag.split(':', 1)[1]}:{count}" for tag, count in values.items()))


if __name__ == "__main__":
    main()
