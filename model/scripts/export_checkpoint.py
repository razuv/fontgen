from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove optimizer state from a Fontgen inference checkpoint")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    payload = {key: value for key, value in checkpoint.items() if key != "optimizer"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"exported epoch {payload.get('epoch')} inference checkpoint to {args.output}")


if __name__ == "__main__":
    main()

