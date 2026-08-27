from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fontgen_model.config import ModelConfig
from fontgen_model.network import FontgenNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade a Fontgen v2 checkpoint to the hi-res v3 architecture")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raster-size", type=int, default=128)
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    config_values = {**checkpoint["config"], "raster_size": args.raster_size, "category_count": 5}
    config = ModelConfig(**config_values)
    model = FontgenNet(config)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {
        name for name in incompatible.missing_keys
        if name.startswith((
            "raster_refiner.", "category_head.", "prompt_category_head.",
            "prompt_control_head.", "style_film.",
        ))
    }
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint difference: missing={sorted(unexpected_missing)}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "config": config.to_dict(),
        "epoch": int(checkpoint.get("epoch", 0)), "training_mode": "raster-only",
        "upgraded_from": str(args.source),
    }, args.output)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"upgraded epoch {checkpoint.get('epoch', 0)} to {args.raster_size}px / "
        f"{parameters:,} parameters; initialized {len(allowed_missing)} new tensors"
    )


if __name__ == "__main__":
    main()
