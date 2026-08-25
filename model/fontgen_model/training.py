from __future__ import annotations

import torch
from torch.nn import functional

from .outline import COMMANDS, COORDINATE_MASKS


def training_loss(batch: dict[str, torch.Tensor], output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    targets = batch["commands"][:, 1:]
    command_logits = output["commands"][:, :-1]
    command_loss = functional.cross_entropy(
        command_logits.reshape(-1, command_logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=0,
        label_smoothing=0.02,
    )
    coordinate_targets = batch["coordinates"][:, 1:]
    coordinate_predictions = output["coordinates"][:, :-1]
    counts = torch.tensor(
        [COORDINATE_MASKS[name] for name in COMMANDS],
        device=targets.device,
    )[targets]
    coordinate_indices = torch.arange(6, device=targets.device).view(1, 1, 6)
    coordinate_mask = coordinate_indices < counts.unsqueeze(-1)
    coordinate_error = functional.smooth_l1_loss(
        coordinate_predictions, coordinate_targets, reduction="none", beta=0.02,
    )
    coordinate_loss = (coordinate_error * coordinate_mask).sum() / coordinate_mask.sum().clamp_min(1)
    metrics_loss = functional.smooth_l1_loss(output["metrics"], batch["metrics"], beta=0.02)
    total = command_loss + coordinate_loss * 8.0 + metrics_loss * 2.0
    return {"total": total, "commands": command_loss, "coordinates": coordinate_loss, "metrics": metrics_loss}

