from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from fontgen_model.v5_config import V5Config
from fontgen_model.v5_dataset import VectorOutlineDataset
from fontgen_model.v5_loss import vector_training_loss
from fontgen_model.v5_network import VectorFontNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v5 direct-vector font model")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/fontgen-vector-v5.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--validation-family-fraction", type=float, default=0.12)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-batches", type=int, help="Limit batches per phase for pipeline checks")
    parser.add_argument("--device", help="cpu, cuda, cuda:0 or mps; auto-detected by default")
    parser.add_argument(
        "--balanced-styles", action=argparse.BooleanOptionalAction, default=True,
        help="Use sqrt-balanced category sampling and boost italic/rounded examples",
    )
    args = parser.parse_args()

    resumed = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    config = V5Config(**resumed["config"]) if resumed else (
        V5Config(max_commands=48, d_model=64, heads=4, decoder_layers=2, refiner_layers=1, feedforward=128)
        if args.smoke else V5Config()
    )
    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ))
    with args.manifest.open(encoding="utf-8") as source:
        import json

        families = sorted({json.loads(line)["family"] for line in source if line.strip()})
    random.Random(51).shuffle(families)
    validation_count = max(1, round(len(families) * args.validation_family_fraction))
    validation_families = set(families[:validation_count])
    training_families = set(families[validation_count:])
    training = VectorOutlineDataset(args.manifest, args.embeddings, config, training_families)
    validation = VectorOutlineDataset(args.manifest, args.embeddings, config, validation_families)
    sampler = None
    if args.balanced_styles:
        category_counts: dict[str, int] = {}
        for row in training.rows:
            category = str(row.get("category", "SANS_SERIF"))
            category_counts[category] = category_counts.get(category, 0) + 1
        sample_weights = []
        for row in training.rows:
            category = str(row.get("category", "SANS_SERIF"))
            rounded_boost = 1.4 if abs(float(row["controls"][3])) > 0.5 else 1.0
            italic_boost = 1.2 if "italic" in str(row.get("subfamily", "")).casefold() else 1.0
            sample_weights.append(rounded_boost * italic_boost / math.sqrt(category_counts[category]))
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    training_loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    model = VectorFontNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    best_validation = float("inf")
    if resumed:
        model.load_state_dict(resumed["model"])
        if resumed.get("optimizer"):
            optimizer.load_state_dict(resumed["optimizer"])
        start_epoch = int(resumed.get("epoch", 0))
        best_validation = float(resumed.get("best_validation", best_validation))

    def run(loader: DataLoader, training_mode: bool) -> float:
        model.train(training_mode)
        totals: list[float] = []
        context = torch.enable_grad() if training_mode else torch.inference_mode()
        with context:
            for batch_index, batch in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    output = model(
                        batch["prompt_embedding"], batch["glyph_id"], batch["controls"],
                        batch["commands"], batch["coordinates"],
                    )
                    losses = vector_training_loss(batch, output)
                if training_mode:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                totals.append(float(losses["total"].detach()))
        return sum(totals) / max(1, len(totals))

    for epoch in range(start_epoch, args.epochs):
        train_loss = run(training_loader, True)
        validation_loss = run(validation_loader, False)
        print(
            f"epoch={epoch + 1:03d} train={train_loss:.5f} validation={validation_loss:.5f} "
            f"families={len(training_families)}/{len(validation_families)} device={device}"
        )
        if validation_loss <= best_validation:
            best_validation = validation_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": config.to_dict(), "epoch": epoch + 1,
                    "best_validation": best_validation, "architecture": "vector-v5",
                },
                args.output,
            )
            print(f"saved best validation={best_validation:.5f} -> {args.output}")


if __name__ == "__main__":
    main()
