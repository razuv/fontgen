from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from fontgen_model.config import ModelConfig
from fontgen_model.network import FontgenNet
from fontgen_model.structured_config import StructuredConfig
from fontgen_model.structured_dataset import StructuredOutlineDataset
from fontgen_model.structured_loss import structured_training_loss
from fontgen_model.structured_network import StructuredVectorFontNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Train family→structure→quadratic→refinement font model")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/fontgen-structured-v1.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--teacher-checkpoint", type=Path,
        help="V4.1 checkpoint used only to distill prompt/style knowledge",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--validation-family-fraction", type=float, default=0.12)
    args = parser.parse_args()
    resumed = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    config = StructuredConfig(**resumed["config"]) if resumed else (
        StructuredConfig(
            max_commands=64, d_model=64, heads=4, structure_layers=1,
            geometry_layers=1, refiner_layers=1, feedforward=128, family_dimensions=32,
        ) if args.smoke else StructuredConfig()
    )
    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ))
    teacher = None
    teacher_config = None
    if args.teacher_checkpoint:
        teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=True)
        teacher_config = ModelConfig(**teacher_checkpoint["config"])
        if teacher_config.style_dimensions != config.legacy_style_dimensions:
            raise ValueError(
                f"Teacher style size {teacher_config.style_dimensions} does not match "
                f"student compatibility size {config.legacy_style_dimensions}"
            )
        teacher = FontgenNet(teacher_config).to(device)
        teacher.load_state_dict(teacher_checkpoint["model"])
        teacher.eval().requires_grad_(False)
    with args.manifest.open(encoding="utf-8") as source:
        families = sorted({str(json.loads(line)["family"]) for line in source if line.strip()})
    random.Random(71).shuffle(families)
    validation_count = max(1, round(len(families) * args.validation_family_fraction))
    validation_families = set(families[:validation_count])
    training_families = set(families[validation_count:])
    teacher_prompt_bytes = teacher_config.max_prompt_bytes if teacher_config else None
    training = StructuredOutlineDataset(
        args.manifest, args.embeddings, config, training_families, teacher_prompt_bytes,
    )
    validation = StructuredOutlineDataset(
        args.manifest, args.embeddings, config, validation_families, teacher_prompt_bytes,
    )
    counts: dict[str, int] = {}
    for row in training.rows:
        category = str(row.get("category", "SANS_SERIF"))
        counts[category] = counts.get(category, 0) + 1
    weights = []
    for row in training.rows:
        category = str(row.get("category", "SANS_SERIF"))
        italic = 1.2 if "italic" in str(row.get("subfamily", "")).casefold() else 1.0
        rounded = 1.4 if abs(float(row["controls"][3])) > 0.5 else 1.0
        weights.append(italic * rounded / math.sqrt(counts[category]))
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    training_loader = DataLoader(
        training, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    model = StructuredVectorFontNet(config).to(device)
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

    def run(loader: DataLoader, train: bool) -> float:
        model.train(train)
        totals: list[float] = []
        context = torch.enable_grad() if train else torch.inference_mode()
        with context:
            for batch_index, batch in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    output = model(
                        batch["prompt_embedding"], batch["glyph_id"], batch["controls"], batch["commands"],
                    )
                    teacher_output = None
                    if teacher is not None:
                        with torch.no_grad():
                            context = teacher.encode_prompt_context(batch["teacher_prompt"])
                            style = teacher.style_projection(torch.cat((context, batch["controls"]), dim=-1))
                            teacher_output = {"style": style, "category": teacher.category_head(style)}
                    losses = structured_training_loss(batch, output, teacher_output)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                totals.append(float(losses["total"].detach()))
        return sum(totals) / max(len(totals), 1)

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
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config.to_dict(), "epoch": epoch + 1,
                "best_validation": best_validation, "architecture": "structured-quadratic-v1",
                "teacher_checkpoint": str(args.teacher_checkpoint) if args.teacher_checkpoint else None,
            }, args.output)
            print(f"saved best validation={best_validation:.5f} -> {args.output}")


if __name__ == "__main__":
    main()
