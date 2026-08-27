import argparse
import json
from pathlib import Path

import torch

from fontgen_model.v5_text import MultilingualPromptEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache multilingual embeddings for unique corpus prompts")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    prompts: set[str] = set()
    with args.manifest.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                prompts.add(json.loads(line)["prompt"])
    ordered = sorted(prompts)
    embeddings = MultilingualPromptEncoder(device=args.device).encode(ordered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompts": ordered, "embeddings": embeddings.half()}, args.output)
    print(f"Saved {len(ordered)} embeddings with shape {tuple(embeddings.shape)} to {args.output}")


if __name__ == "__main__":
    main()
