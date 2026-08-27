from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_row(row: dict[str, Any]) -> dict[str, Any]:
    font = Path(str(row.get("font", "unknown-font"))).name
    portable = dict(row)
    portable["font"] = f"local-font-library/{font}"
    portable["source"] = "local-font-library"
    return portable


def _write_shard(rows: list[dict[str, Any]], destination: Path) -> None:
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as archive,
    ):
        for row in rows:
            archive.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            archive.write(b"\n")


def pack(source: Path, destination: Path, rows_per_shard: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.glob("local-glyphs-*.jsonl.gz")):
        raise SystemExit(f"Destination already contains shards: {destination}")

    shard_rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    total = 0

    def flush() -> None:
        nonlocal shard_rows
        if not shard_rows:
            return
        path = destination / f"local-glyphs-{len(shards) + 1:04d}.jsonl.gz"
        _write_shard(shard_rows, path)
        shards.append({
            "file": path.name,
            "rows": len(shard_rows),
            "bytes": path.stat().st_size,
            "sha256": _digest(path),
        })
        shard_rows = []

    with source.open(encoding="utf-8") as manifest:
        for line in manifest:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != "local-font-library":
                continue
            shard_rows.append(_portable_row(row))
            total += 1
            if len(shard_rows) >= rows_per_shard:
                flush()
    flush()

    metadata = {
        "format": "fontgen-portable-jsonl-gzip-v1",
        "source": "local-font-library",
        "rows": total,
        "contains_font_binaries": False,
        "self_contained_fields": [
            "commands", "coordinates", "raster", "prompt", "controls", "metrics",
        ],
        "shards": shards,
    }
    (destination / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def _verified_rows(bundle: Path) -> Iterator[bytes]:
    metadata = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for shard in metadata["shards"]:
        path = bundle / shard["file"]
        if path.stat().st_size != shard["bytes"] or _digest(path) != shard["sha256"]:
            raise SystemExit(f"Checksum mismatch: {path}")
        with gzip.open(path, "rb") as archive:
            yield from archive


def restore(bundle: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {destination}")
    with destination.open("wb") as output:
        for line in _verified_rows(bundle):
            output.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack or restore a portable Fontgen glyph corpus")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack")
    pack_parser.add_argument("source", type=Path)
    pack_parser.add_argument("destination", type=Path)
    pack_parser.add_argument("--rows-per-shard", type=int, default=5_000)

    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("bundle", type=Path)
    restore_parser.add_argument("destination", type=Path)

    args = parser.parse_args()
    if args.command == "pack":
        pack(args.source, args.destination, args.rows_per_shard)
    else:
        restore(args.bundle, args.destination)


if __name__ == "__main__":
    main()
