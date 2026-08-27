from __future__ import annotations

import argparse
import json
from pathlib import Path

from fontgen_model.raster import encode_mask, outline_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize raster targets for vector-only manifest rows")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = generated = reused = 0
    with args.manifest.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("raster"):
                reused += 1
            else:
                row["raster"] = encode_mask(outline_mask(row["commands"], row["coordinates"], args.size))
                generated += 1
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows += 1
    print(f"rows={rows} generated={generated} reused={reused} output={args.output}")


if __name__ == "__main__":
    main()
