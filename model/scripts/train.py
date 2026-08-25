from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from fontgen_model.config import ModelConfig
from fontgen_model.dataset import OutlineDataset
from fontgen_model.network import FontgenNet
from fontgen_model.training import training_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Fontgen prompt-to-outline model")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/fontgen-v0.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--preset", choices=("full", "smoke"), default="full")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    config = ModelConfig() if args.preset == "full" else ModelConfig(
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
    validation_size = max(1, len(dataset) // 20)
    training_size = len(dataset) - validation_size
    training_set, _ = random_split(dataset, [training_size, validation_size], generator=torch.Generator().manual_seed(17))
    loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    model = FontgenNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.05)
    model.train()
    for epoch in range(args.epochs):
        totals = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(batch["prompt"], batch["glyph_id"], batch["controls"], batch["commands"], batch["coordinates"])
            losses = training_loss(batch, output)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(losses["total"].detach()))
        print(f"epoch={epoch + 1:03d} loss={sum(totals) / max(1, len(totals)):.5f} device={device}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": config.to_dict(), "epoch": epoch + 1}, args.output)


if __name__ == "__main__":
    main()
