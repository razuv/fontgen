"""Clean a font directory: remove non-font files only.

Moves non-font files (.txt, .ai, .woff2, images, etc.) to quarantine.
Does NOT remove any .ttf or .otf files — use audit_fonts.py to filter by quality.

Usage:
    python scripts/clean_fonts.py /path/to/fonts --dry-run
    python scripts/clean_fonts.py /path/to/fonts
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

FONT_EXTENSIONS = {".ttf", ".otf"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove non-font files from font directory")
    parser.add_argument("font_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quarantine", default="_quarantine")
    args = parser.parse_args()

    font_dir = args.font_dir.resolve()
    quarantine_dir = font_dir / args.quarantine

    all_entries = list(font_dir.iterdir())
    print(f"scanning {len(all_entries)} entries in {font_dir}", file=sys.stderr)

    moved = 0
    for path in sorted(all_entries):
        if path.is_dir():
            if path.name.startswith("_") or path.name.startswith("."):
                continue
            # Skip directories — they may contain font bundles
            continue
        ext = path.suffix.lower()
        if ext in FONT_EXTENSIONS:
            continue
        # Non-font file — move to quarantine
        if args.dry_run:
            print(f"  WOULD MOVE: {path.name}")
        else:
            quarantine_dir.mkdir(exist_ok=True)
            dest = quarantine_dir / path.name
            if dest.exists():
                dest = quarantine_dir / f"{path.stem}_{moved}{path.suffix}"
            shutil.move(str(path), str(dest))
        moved += 1

    font_count = len([p for p in font_dir.iterdir() if p.suffix.lower() in FONT_EXTENSIONS])
    print(f"\nremoved {moved} non-font files, {font_count} fonts remaining", file=sys.stderr)
    if args.dry_run:
        print("(dry run — no files moved)", file=sys.stderr)


if __name__ == "__main__":
    main()
