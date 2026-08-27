from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from fontgen_model.quadratic import QUADRATIC_COMMANDS, canonicalize_quadratic


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonicalize a font manifest to quadratic Bézier contours")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-commands", type=int, default=192)
    parser.add_argument("--max-error", type=float, default=0.001)
    args = parser.parse_args()
    written = dropped = 0
    command_counts: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as destination:
        for line in source:
            if not line.strip():
                continue
            row = canonicalize_quadratic(json.loads(line), args.max_error)
            if row is None or len(row["commands"]) > args.max_commands:
                dropped += 1
                continue
            command_counts.update(QUADRATIC_COMMANDS[int(command)] for command in row["commands"])
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    print(f"written={written} dropped={dropped} output={args.output}")
    print("commands=" + " ".join(f"{key}:{value}" for key, value in sorted(command_counts.items())))


if __name__ == "__main__":
    main()
