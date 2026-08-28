from __future__ import annotations

import argparse
import copy
import math
import random
from collections import deque
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from fontgen_model.config import ModelConfig
from fontgen_model.dataset import OutlineDataset
from fontgen_model.network import FontgenNet, PatchDiscriminator
from fontgen_model.training import (
    PerceptualLoss,
    UncertaintyWeights,
    _contour_smoothness_loss,
    _discriminator_r1_penalty,
    discriminator_loss,
    generator_adversarial_loss,
    raster_training_loss,
    training_loss,
)
from scripts.select_tier import resolve_tier


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


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(value, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class HardExampleMiner:
    """Track per-sample loss and oversample hard examples."""

    def __init__(self, window: int = 64, oversample_ratio: float = 0.2):
        self.loss_history: dict[int, deque[float]] = {}
        self.window = window
        self.oversample_ratio = oversample_ratio

    def update(self, indices: list[int], losses: list[float]) -> None:
        for idx, loss in zip(indices, losses):
            if idx not in self.loss_history:
                self.loss_history[idx] = deque(maxlen=self.window)
            self.loss_history[idx].append(loss)

    def get_hard_indices(self, all_indices: list[int], count: int) -> list[int]:
        scored = []
        for idx in all_indices:
            if idx in self.loss_history and len(self.loss_history[idx]) > 0:
                avg_loss = sum(self.loss_history[idx]) / len(self.loss_history[idx])
                scored.append((avg_loss, idx))
        scored.sort(reverse=True)
        return [idx for _, idx in scored[:count]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Fontgen prompt-to-outline model")
    parser.add_argument("manifest", type=Path, nargs="?", default=None,
                        help="Path to JSONL manifest (omit if using --dataset-tier)")
    parser.add_argument("--dataset-tier", type=int, default=None,
                        help="Use pre-built tier dataset (500, 1000, 2000, 5000, 10000)")
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
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate (0 to disable)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Fraction of total steps for LR warmup")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1, help="Minimum LR as fraction of peak LR")
    parser.add_argument("--discriminator", action="store_true", help="Enable GAN discriminator training")
    parser.add_argument("--disc-lr", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--r1-weight", type=float, default=10.0, help="R1 gradient penalty weight")
    parser.add_argument("--r1-every", type=int, default=16, help="Apply R1 penalty every N steps")
    parser.add_argument("--perceptual-loss", action="store_true", help="Enable perceptual loss")
    parser.add_argument("--uncertainty-weighting", action="store_true", help="Use learned loss weights")
    parser.add_argument("--hard-mining", action="store_true", help="Enable hard example mining")
    parser.add_argument("--hard-mining-ratio", type=float, default=0.2, help="Fraction of batch from hard examples")
    parser.add_argument("--contour-smoothness-weight", type=float, default=0.0, help="Weight for contour smoothness loss")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation (control jitter)")
    parser.add_argument("--control-jitter", type=float, default=0.1, help="Std dev of control perturbation")
    parser.add_argument("--cfg-dropout", type=float, default=0.0, help="CFG prompt dropout probability")
    args = parser.parse_args()
    args.manifest = resolve_tier(args.dataset_tier, args.manifest)

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
    dataset = OutlineDataset(
        args.manifest, config,
        augment=args.augment,
        control_jitter=args.control_jitter,
        cfg_dropout=args.cfg_dropout,
    )
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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.05,
    )

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    ema: EMA | None = None
    if args.ema_decay > 0:
        ema = EMA(model, decay=args.ema_decay)

    perceptual: PerceptualLoss | None = None
    if args.perceptual_loss:
        perceptual = PerceptualLoss().to(device)

    uncertainty_weights: UncertaintyWeights | None = None
    if args.uncertainty_weighting:
        uncertainty_weights = UncertaintyWeights(num_losses=18).to(device)
        optimizer.add_param_group({"params": uncertainty_weights.parameters(), "lr": args.learning_rate * 0.1})

    total_steps = args.epochs * max(1, len(loader))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, args.min_lr_ratio)

    discriminator: PatchDiscriminator | None = None
    disc_optimizer: torch.optim.AdamW | None = None
    if args.discriminator and args.raster_only:
        discriminator = PatchDiscriminator().to(device)
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=args.disc_lr, betas=(0.0, 0.999), weight_decay=0.0,
        )

    hard_miner: HardExampleMiner | None = None
    if args.hard_mining:
        hard_miner = HardExampleMiner(window=64, oversample_ratio=args.hard_mining_ratio)
        print(f"hard example mining enabled: ratio={args.hard_mining_ratio}")

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

    global_step = start_epoch * max(1, len(loader))
    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = []
        loss_component_sums: dict[str, float] = {}
        loss_component_counts: dict[str, int] = {}
        batch_iter = tqdm(loader, desc=f"epoch {epoch + 1:03d}", unit="batch", leave=False)
        for batch_index, batch in enumerate(batch_iter):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = (
                    model.condition(batch["prompt"], batch["glyph_id"], batch["controls"])
                    if args.raster_only else
                    model(batch["prompt"], batch["glyph_id"], batch["controls"], batch["commands"], batch["coordinates"])
                )
                losses = raster_training_loss(batch, output, perceptual=perceptual) if args.raster_only else training_loss(batch, output)

                if args.contour_smoothness_weight > 0 and not args.raster_only:
                    smooth_loss = _contour_smoothness_loss(output["coordinates"], batch["commands"])
                    losses["total"] = losses["total"] + args.contour_smoothness_weight * smooth_loss
                    losses["contour_smoothness"] = smooth_loss

                if uncertainty_weights is not None:
                    component_list = [
                        losses.get(k, torch.tensor(0.0, device=device))
                        for k in [
                            "metrics", "raster", "dice", "recognition", "category",
                            "edges", "prompt_category", "prompt_controls", "style_variance",
                            "entropy", "sdf", "normals", "eikonal", "curvature",
                            "multiscale", "sdf_occupancy", "sdf_multiscale", "perceptual",
                        ]
                    ]
                    losses["total"] = uncertainty_weights(component_list)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scaler.update()
            scheduler.step()
            global_step += 1

            if ema is not None:
                ema.update(model)

            if discriminator is not None and disc_optimizer is not None and args.raster_only:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    real_logits = discriminator(batch["raster"])
                    fake_raster = torch.sigmoid(output["raster"].detach())
                    fake_logits = discriminator(fake_raster)
                    d_loss = discriminator_loss(real_logits, fake_logits)
                disc_optimizer.zero_grad(set_to_none=True)
                scaler.scale(d_loss).backward()
                scaler.unscale_(disc_optimizer)
                disc_optimizer.step()
                scaler.update()

                if global_step % args.r1_every == 0:
                    r1 = _discriminator_r1_penalty(batch["raster"], discriminator)
                    disc_optimizer.zero_grad(set_to_none=True)
                    (r1 * args.r1_weight * args.r1_every).backward()
                    disc_optimizer.step()

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    g_output = model.condition(batch["prompt"], batch["glyph_id"], batch["controls"])
                    fake_logits_for_g = discriminator(torch.sigmoid(g_output["raster"]))
                    g_adv = generator_adversarial_loss(fake_logits_for_g)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(g_adv * 0.1).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scaler.update()

            batch_loss = float(losses["total"].detach())
            totals.append(batch_loss)
            batch_iter.set_postfix(loss=f"{batch_loss:.4f}")
            if hard_miner is not None and "glyph_id" in batch:
                glyph_ids = batch["glyph_id"].tolist() if isinstance(batch["glyph_id"], torch.Tensor) else batch["glyph_id"]
                batch_loss = float(losses["total"].detach())
                hard_miner.update(glyph_ids, [batch_loss] * len(glyph_ids))
            for key, value in losses.items():
                if key != "total" and isinstance(value, torch.Tensor):
                    loss_component_sums[key] = loss_component_sums.get(key, 0.0) + float(value.detach())
                    loss_component_counts[key] = loss_component_counts.get(key, 0) + 1

        validation_totals = []
        model_to_eval = model
        if ema is not None:
            ema_model = copy.deepcopy(model)
            ema.apply(ema_model)
            ema_model.eval()
            model_to_eval = ema_model
        else:
            model.eval()

        with torch.inference_mode():
            val_iter = tqdm(validation_loader or [], desc=f"val   {epoch + 1:03d}", unit="batch", leave=False)
            for batch_index, batch in enumerate(val_iter):
                validation_limit = args.validation_batches if args.validation_batches is not None else args.max_batches
                if validation_limit is not None and batch_index >= validation_limit:
                    break
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                output = (
                    model_to_eval.condition(batch["prompt"], batch["glyph_id"], batch["controls"])
                    if args.raster_only else
                    model_to_eval(batch["prompt"], batch["glyph_id"], batch["controls"], batch["commands"], batch["coordinates"])
                )
                losses = raster_training_loss(batch, output, perceptual=perceptual) if args.raster_only else training_loss(batch, output)
                val_loss = float(losses["total"])
                validation_totals.append(val_loss)
                val_iter.set_postfix(loss=f"{val_loss:.4f}")

        train_loss = sum(totals) / max(1, len(totals))
        validation_loss = sum(validation_totals) / max(1, len(validation_totals))
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch + 1:03d} train={train_loss:.5f} validation={validation_loss:.5f} "
            f"lr={lr_now:.2e} families={len({dataset.rows[i]['family'] for i in training_set.indices})}/"
            f"{len({dataset.rows[i]['family'] for i in validation_set.indices})} device={device}"
        )
        if loss_component_counts:
            components_str = " ".join(
                f"{k}={loss_component_sums[k] / loss_component_counts[k]:.4f}"
                for k in sorted(loss_component_counts)
            )
            print(f"  components: {components_str}")
        if uncertainty_weights is not None:
            print(f"  loss_weights: {uncertainty_weights.log_vars.detach().cpu().tolist()}")

        if validation_loss <= best_validation:
            best_validation = validation_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_model = model
            if ema is not None:
                ema_model_save = copy.deepcopy(model)
                ema.apply(ema_model_save)
                save_model = ema_model_save
            torch.save({
                "model": save_model.state_dict(), "optimizer": optimizer.state_dict(),
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
