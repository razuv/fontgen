"""Interactive dataset tier selection for training scripts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

TIERS_DIR = Path("data/tiers")


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _discover_tiers() -> list[tuple[int, Path]]:
    """Scan TIERS_DIR for train_tier_*.jsonl files, sorted by size."""
    tiers: list[tuple[int, Path]] = []
    if not TIERS_DIR.exists():
        return tiers
    for path in sorted(TIERS_DIR.glob("train_tier_*.jsonl")):
        match = re.match(r"train_tier_(\d+)\.jsonl$", path.name)
        if match:
            tiers.append((int(match.group(1)), path))
    tiers.sort(key=lambda t: t[0])
    return tiers


def resolve_tier(tier: int | None, manifest: Path | None) -> Path:
    """Resolve dataset path from tier number or manifest path.

    If both are None, shows interactive menu and returns selected path.
    """
    if tier is not None:
        path = TIERS_DIR / f"train_tier_{tier}.jsonl"
        if not path.exists():
            print(f"ERROR: tier dataset not found: {path}", file=sys.stderr)
            print(f"Run split_dataset_tiers.py first.", file=sys.stderr)
            sys.exit(1)
        return path

    if manifest is not None:
        return manifest

    # Interactive selection — discover whatever tiers exist
    available = _discover_tiers()

    if not available:
        print("ERROR: no tier datasets found in data/tiers/", file=sys.stderr)
        print("Run build_dataset.sh or split_dataset_tiers.py first.", file=sys.stderr)
        sys.exit(1)

    print("\n=== Select Dataset Tier ===\n", file=sys.stderr)
    for i, (size, path) in enumerate(available):
        rows = _count_rows(path)
        print(f"  [{i + 1}] {size:>6} fonts  ({rows:>7} glyph rows)", file=sys.stderr)
    print(f"\n  [0] custom manifest path", file=sys.stderr)
    print(file=sys.stderr)

    while True:
        try:
            choice = input("select tier [1-{}]: ".format(len(available))).strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            sys.exit(1)

        if choice == "0":
            custom = input("manifest path: ").strip()
            if custom:
                p = Path(custom)
                if p.exists():
                    return p
                print(f"  not found: {p}", file=sys.stderr)
            continue

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                size, path = available[idx]
                print(f"\nselected: {size} fonts -> {path}\n", file=sys.stderr)
                return path
        except ValueError:
            pass

        print(f"  invalid choice, enter 1-{len(available)} or 0", file=sys.stderr)
