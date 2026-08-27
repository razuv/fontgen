from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from fontgen_model.config import ModelConfig
from fontgen_model.dataset import OutlineDataset
from fontgen_model.network import FontgenNet
from fontgen_model.training import raster_training_loss, training_loss


def family_split(dataset: OutlineDataset, fraction: float, seed: int = 17) -> tuple[Subset, Subset]:
    families = sorted({str(row["family"]) for row in dataset.rows})
    random.Random(seed).shuffle(families)
    validation_count = max(1, round(len(families) * fraction)) if len(families) > 1 else 0
    validation_families = set(families[:validation_count])
    training_indices = [i for i, row in enumerate(dataset.rows) if str(row["family"]) not in validation_families]
    validation_indices = [i for i, row in enumerate(dataset.rows) if str(row["family"]) in validation_families]
    return Subset(dataset, training_indices), Subset(dataset, validation_indices)


def curriculum_weight(row: dict[str, object], stage: str) -> float:
    if stage == "full":
        return 1.0
    category = str(row.get("category", "SANS_SERIF"))
    controls = [float(value) for value in row.get("controls", [0.0] * 5)]
    subfamily = str(row.get("subfamily", "")).casefold()
    core_category = category in {"SANS_SERIF", "SERIF", "MONOSPACE"}
    if stage == "anatomy":
        clean_upright = (
            core_category and abs(controls[0]) <= 0.55 and abs(controls[1]) <= 0.55
            and abs(controls[3]) <= 0.4 and abs(controls[4]) <= 0.2
            and "italic" not in subfamily and "oblique" not in subfamily
        )
        return 5.0 if clean_upright else 0.35
    moderate_axes = core_category and abs(controls[3]) <= 0.75 and abs(controls[4]) <= 0.6
    return 2.5 if moderate_axes else 0.6


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Fontgen prompt-to-outline model")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/fontgen-v0.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--preset", choices=("full", "smoke"), default="full")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--raster-only", action="store_true", help="Train the raster path used by production inference")
    parser.add_argument(
        "--geometry-finetune", action="store_true",
        help="Freeze learned prompt/style/content semantics and optimize only raster/SDF geometry",
    )
    parser.add_argument(
        "--sdf-refiner-only", action="store_true",
        help="Freeze V4.1 completely and train only the bounded local SDF correction",
    )
    parser.add_argument("--validation-family-fraction", type=float, default=0.1)
    parser.add_argument("--balanced-styles", action="store_true", help="Balance categories and boost rounded/italic faces")
    parser.add_argument("--reset-best", action="store_true", help="Reset validation baseline after changing the corpus")
    parser.add_argument("--max-batches", type=int, help="Limit batches per phase for timing or smoke checks")
    parser.add_argument("--samples-per-epoch", type=int, help="Balanced fine-tuning samples drawn each epoch")
    parser.add_argument("--validation-batches", type=int, help="Limit validation batches without limiting training")
    parser.add_argument(
        "--curriculum-stage", choices=("anatomy", "axes", "full"), default="full",
        help="Learn clean upright anatomy before adding extreme styles",
    )
    args = parser.parse_args()
    resumed = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    config = ModelConfig(**resumed["config"]) if resumed else ModelConfig() if args.preset == "full" else ModelConfig(
        max_prompt_bytes=64,
        max_commands=48,
        d_model=64,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feedforward=128,
        style_dimensions=32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dataset = OutlineDataset(args.manifest, config)
    training_set, validation_set = family_split(dataset, args.validation_family_fraction)
    sampler = None
    if args.balanced_styles:
        category_counts: dict[str, int] = {}
        for index in training_set.indices:
            category = str(dataset.rows[index].get("category", "SANS_SERIF"))
            category_counts[category] = category_counts.get(category, 0) + 1
        weights = []
        for index in training_set.indices:
            row = dataset.rows[index]
            category = str(row.get("category", "SANS_SERIF"))
            rarity_boost = 1.8 if abs(float(row["controls"][3])) > 0.5 else 1.0
            italic_boost = 1.35 if "italic" in str(row.get("subfamily", "")).lower() else 1.0
            weights.append(
                rarity_boost * italic_boost * curriculum_weight(row, args.curriculum_stage)
                / category_counts[category]
            )
        sampler = WeightedRandomSampler(
            weights, args.samples_per_epoch or len(weights), replacement=True,
        )
    loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, num_workers=args.workers) if validation_set else None
    model = FontgenNet(config).to(device)
    if args.raster_only:
        for module in (model.command_embedding, model.coordinate_embedding, model.decoder, model.command_head, model.coordinate_head):
            module.requires_grad_(False)
    if args.geometry_finetune:
        model.requires_grad_(False)
        for module in (
            model.raster_seed, model.raster_decoder, model.raster_refiner,
            model.sdf_coordinate_refiner, model.raster_encoder,
            model.recognition_head, model.raster_to_model,
        ):
            module.requires_grad_(True)
    if args.sdf_refiner_only:
        model.requires_grad_(False)
        model.sdf_coordinate_refiner.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.05,
    )
    start_epoch = 0
    best_validation = (
        float("inf") if args.reset_best else float(resumed.get("best_validation", "inf"))
    ) if resumed else float("inf")
    if resumed:
        model.load_compatible_state_dict(resumed["model"])
        start_epoch = int(resumed.get("epoch", 0))
        resumed_mode = resumed.get("training_mode", "full")
        current_mode = (
            "sdf-refiner-only" if args.sdf_refiner_only else
            "geometry-sdf" if args.geometry_finetune else
            "raster-only" if args.raster_only else "full"
        )
        if resumed.get("optimizer") and resumed_mode == current_mode:
            try:
                optimizer.load_state_dict(resumed["optimizer"])
            except ValueError:
                print("optimizer state is incompatible with SDF refiner; starting a fresh optimizer")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = []
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            output = (
                model.condition(batch["prompt"], batch["glyph_id"], batch["controls"])
                if args.raster_only else
                model(batch["prompt"], batch["glyph_id"], batch["controls"], batch["commands"], batch["coordinates"])
            )
            losses = raster_training_loss(batch, output) if args.raster_only else training_loss(batch, output)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(losses["total"].detach()))
        validation_totals = []
        model.eval()
        with torch.inference_mode():
            for batch_index, batch in enumerate(validation_loader or []):
                validation_limit = args.validation_batches if args.validation_batches is not None else args.max_batches
                if validation_limit is not None and batch_index >= validation_limit:
                    break
                batch = {key: value.to(device) for key, value in batch.items()}
                output = (
                    model.condition(batch["prompt"], batch["glyph_id"], batch["controls"])
                    if args.raster_only else
                    model(batch["prompt"], batch["glyph_id"], batch["controls"], batch["commands"], batch["coordinates"])
                )
                losses = raster_training_loss(batch, output) if args.raster_only else training_loss(batch, output)
                validation_totals.append(float(losses["total"]))
        train_loss = sum(totals) / max(1, len(totals))
        validation_loss = sum(validation_totals) / max(1, len(validation_totals))
        print(
            f"epoch={epoch + 1:03d} train={train_loss:.5f} validation={validation_loss:.5f} "
            f"families={len({dataset.rows[i]['family'] for i in training_set.indices})}/"
            f"{len({dataset.rows[i]['family'] for i in validation_set.indices})} device={device}"
        )
        if validation_loss <= best_validation:
            best_validation = validation_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config.to_dict(), "epoch": epoch + 1,
                "best_validation": best_validation,
                "training_mode": (
                    "sdf-refiner-only" if args.sdf_refiner_only else
                    "geometry-sdf" if args.geometry_finetune else
                    "raster-only" if args.raster_only else "full"
                ),
                "representation": "sdf-v1", "curriculum_stage": args.curriculum_stage,
            }, args.output)
            print(f"saved best validation={best_validation:.5f} -> {args.output}")


if __name__ == "__main__":
    main()
