from __future__ import annotations

import torch
from torch.nn import functional

from .outline import COMMAND_TO_ID, COMMANDS, COORDINATE_MASKS


def _coordinate_mask(commands: torch.Tensor) -> torch.Tensor:
    counts = torch.tensor([COORDINATE_MASKS[name] for name in COMMANDS], device=commands.device)[commands]
    return torch.arange(6, device=commands.device).view(1, 1, 6) < counts.unsqueeze(-1)


def _outline_point_loss(
    predicted: torch.Tensor, target: torch.Tensor, commands: torch.Tensor
) -> torch.Tensor:
    """Compare sampled line/quadratic/cubic geometry, not just raw controls."""
    batch, length, _ = predicted.shape
    previous_predicted = predicted.new_zeros((batch, 2))
    previous_target = target.new_zeros((batch, 2))
    losses: list[torch.Tensor] = []
    samples = torch.tensor((0.25, 0.5, 0.75, 1.0), device=predicted.device).view(1, 4, 1)
    for index in range(length):
        command = commands[:, index]
        current_predicted = predicted[:, index]
        current_target = target[:, index]
        for command_name in ("L", "Q", "C"):
            active = command.eq(COMMAND_TO_ID[command_name])
            if not active.any():
                continue
            t = samples
            if command_name == "L":
                predicted_curve = (1 - t) * previous_predicted[:, None] + t * current_predicted[:, None, :2]
                target_curve = (1 - t) * previous_target[:, None] + t * current_target[:, None, :2]
            elif command_name == "Q":
                predicted_curve = (
                    (1 - t) ** 2 * previous_predicted[:, None]
                    + 2 * (1 - t) * t * current_predicted[:, None, :2]
                    + t**2 * current_predicted[:, None, 2:4]
                )
                target_curve = (
                    (1 - t) ** 2 * previous_target[:, None]
                    + 2 * (1 - t) * t * current_target[:, None, :2]
                    + t**2 * current_target[:, None, 2:4]
                )
            else:
                predicted_curve = (
                    (1 - t) ** 3 * previous_predicted[:, None]
                    + 3 * (1 - t) ** 2 * t * current_predicted[:, None, :2]
                    + 3 * (1 - t) * t**2 * current_predicted[:, None, 2:4]
                    + t**3 * current_predicted[:, None, 4:6]
                )
                target_curve = (
                    (1 - t) ** 3 * previous_target[:, None]
                    + 3 * (1 - t) ** 2 * t * current_target[:, None, :2]
                    + 3 * (1 - t) * t**2 * current_target[:, None, 2:4]
                    + t**3 * current_target[:, None, 4:6]
                )
            losses.append(functional.smooth_l1_loss(predicted_curve[active], target_curve[active], beta=0.01))
        endpoint = torch.where(
            command.eq(COMMAND_TO_ID["C"]).unsqueeze(-1), current_predicted[:, 4:6],
            torch.where(command.eq(COMMAND_TO_ID["Q"]).unsqueeze(-1), current_predicted[:, 2:4], current_predicted[:, :2]),
        )
        target_endpoint = torch.where(
            command.eq(COMMAND_TO_ID["C"]).unsqueeze(-1), current_target[:, 4:6],
            torch.where(command.eq(COMMAND_TO_ID["Q"]).unsqueeze(-1), current_target[:, 2:4], current_target[:, :2]),
        )
        drawable = command.ge(COMMAND_TO_ID["M"]) & command.le(COMMAND_TO_ID["C"])
        previous_predicted = torch.where(drawable.unsqueeze(-1), endpoint, previous_predicted)
        previous_target = torch.where(drawable.unsqueeze(-1), target_endpoint, previous_target)
    return torch.stack(losses).mean() if losses else predicted.sum() * 0


def vector_training_loss(batch: dict[str, torch.Tensor], output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    targets = batch["commands"][:, 1:]
    coordinates = batch["coordinates"][:, 1:]
    mask = _coordinate_mask(targets)

    def pass_loss(command_logits: torch.Tensor, coordinate_predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        command_logits = command_logits[:, :-1]
        coordinate_predictions = coordinate_predictions[:, :-1]
        command_loss = functional.cross_entropy(
            command_logits.reshape(-1, command_logits.shape[-1]), targets.reshape(-1),
            ignore_index=COMMAND_TO_ID["PAD"], label_smoothing=0.01,
        )
        errors = functional.smooth_l1_loss(coordinate_predictions, coordinates, reduction="none", beta=0.01)
        coordinate_loss = (errors * mask).sum() / mask.sum().clamp_min(1)
        geometry_loss = _outline_point_loss(coordinate_predictions, coordinates, targets)
        return command_loss, coordinate_loss, geometry_loss

    initial = pass_loss(output["initial_commands"], output["initial_coordinates"])
    refined = pass_loss(output["commands"], output["coordinates"])
    metrics_loss = functional.smooth_l1_loss(output["metrics"], batch["metrics"], beta=0.01)
    category_loss = functional.cross_entropy(output["category"], batch["category_id"])
    typography_error = functional.smooth_l1_loss(
        output["typography"], batch["typography"], reduction="none", beta=0.02,
    ).mean(dim=-1)
    typography_loss = (
        typography_error * batch["typography_mask"]
    ).sum() / batch["typography_mask"].sum().clamp_min(1)
    total = (
        0.35 * initial[0] + 2.0 * initial[1] + initial[2]
        + refined[0] + 8.0 * refined[1] + 3.0 * refined[2]
        + 2.0 * metrics_loss + 0.5 * category_loss + typography_loss
    )
    return {
        "total": total,
        "commands": refined[0],
        "coordinates": refined[1],
        "geometry": refined[2],
        "initial_commands": initial[0],
        "initial_coordinates": initial[1],
        "metrics": metrics_loss,
        "category": category_loss,
        "typography": typography_loss,
    }
