from __future__ import annotations

import torch
from torch.nn import functional

from .quadratic import QUADRATIC_COMMAND_TO_ID, QUADRATIC_COORDINATE_COUNTS


def _geometry_losses(
    predicted: torch.Tensor, target: torch.Tensor, commands: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = torch.tensor(QUADRATIC_COORDINATE_COUNTS, device=commands.device)[commands]
    coordinate_mask = torch.arange(4, device=commands.device).view(1, 1, 4) < counts.unsqueeze(-1)
    error = functional.smooth_l1_loss(predicted, target, reduction="none", beta=0.008)
    coordinate_loss = (error * coordinate_mask).sum() / coordinate_mask.sum().clamp_min(1)

    batch, length, _ = predicted.shape
    previous_predicted = predicted.new_zeros((batch, 2))
    previous_target = target.new_zeros((batch, 2))
    curve_losses: list[torch.Tensor] = []
    axis_losses: list[torch.Tensor] = []
    curvature_losses: list[torch.Tensor] = []
    samples = torch.tensor((0.2, 0.4, 0.6, 0.8, 1.0), device=predicted.device).view(1, 5, 1)
    for index in range(length):
        command = commands[:, index]
        predicted_values = predicted[:, index]
        target_values = target[:, index]
        is_line = command.eq(QUADRATIC_COMMAND_TO_ID["L"])
        is_quad = command.eq(QUADRATIC_COMMAND_TO_ID["Q"])
        if is_line.any():
            predicted_curve = (
                (1 - samples) * previous_predicted[:, None] + samples * predicted_values[:, None, :2]
            )
            target_curve = (1 - samples) * previous_target[:, None] + samples * target_values[:, None, :2]
            curve_losses.append(functional.smooth_l1_loss(
                predicted_curve[is_line], target_curve[is_line], beta=0.006,
            ))
            target_delta = target_values[:, :2] - previous_target
            predicted_delta = predicted_values[:, :2] - previous_predicted
            vertical = is_line & target_delta[:, 0].abs().lt(0.001)
            horizontal = is_line & target_delta[:, 1].abs().lt(0.001)
            if vertical.any():
                axis_losses.append(predicted_delta[vertical, 0].abs().mean())
            if horizontal.any():
                axis_losses.append(predicted_delta[horizontal, 1].abs().mean())
        if is_quad.any():
            control_predicted, end_predicted = predicted_values[:, :2], predicted_values[:, 2:4]
            control_target, end_target = target_values[:, :2], target_values[:, 2:4]
            predicted_curve = (
                (1 - samples) ** 2 * previous_predicted[:, None]
                + 2 * (1 - samples) * samples * control_predicted[:, None]
                + samples**2 * end_predicted[:, None]
            )
            target_curve = (
                (1 - samples) ** 2 * previous_target[:, None]
                + 2 * (1 - samples) * samples * control_target[:, None]
                + samples**2 * end_target[:, None]
            )
            curve_losses.append(functional.smooth_l1_loss(
                predicted_curve[is_quad], target_curve[is_quad], beta=0.006,
            ))
            predicted_curvature = previous_predicted - 2 * control_predicted + end_predicted
            target_curvature = previous_target - 2 * control_target + end_target
            curvature_losses.append(functional.smooth_l1_loss(
                predicted_curvature[is_quad], target_curvature[is_quad], beta=0.008,
            ))
        is_move = command.eq(QUADRATIC_COMMAND_TO_ID["M"])
        endpoint_predicted = torch.where(is_quad.unsqueeze(-1), predicted_values[:, 2:4], predicted_values[:, :2])
        endpoint_target = torch.where(is_quad.unsqueeze(-1), target_values[:, 2:4], target_values[:, :2])
        drawable = is_move | is_line | is_quad
        previous_predicted = torch.where(drawable.unsqueeze(-1), endpoint_predicted, previous_predicted)
        previous_target = torch.where(drawable.unsqueeze(-1), endpoint_target, previous_target)

    zero = predicted.sum() * 0
    curve_loss = torch.stack(curve_losses).mean() if curve_losses else zero
    axis_loss = torch.stack(axis_losses).mean() if axis_losses else zero
    curvature_loss = torch.stack(curvature_losses).mean() if curvature_losses else zero
    return coordinate_loss, curve_loss, axis_loss, curvature_loss


def structured_training_loss(
    batch: dict[str, torch.Tensor], output: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    structure_targets = batch["commands"][:, 1:]
    structure_logits = output["structure_logits"][:, :-1]
    structure_loss = functional.cross_entropy(
        structure_logits.reshape(-1, structure_logits.shape[-1]), structure_targets.reshape(-1),
        ignore_index=QUADRATIC_COMMAND_TO_ID["PAD"], label_smoothing=0.01,
    )
    initial = _geometry_losses(output["initial_coordinates"], batch["coordinates"], batch["commands"])
    refined = _geometry_losses(output["coordinates"], batch["coordinates"], batch["commands"])
    metrics_loss = functional.smooth_l1_loss(output["metrics"], batch["metrics"], beta=0.01)
    category_loss = functional.cross_entropy(output["category"], batch["category_id"])
    typography_loss = functional.smooth_l1_loss(output["typography"], batch["typography"], beta=0.02)
    family_std = torch.sqrt(output["family"].var(dim=0, unbiased=False) + 1e-4)
    family_variance_loss = functional.relu(0.12 - family_std).mean()
    controls_loss = functional.smooth_l1_loss(
        output["reconstructed_controls"], batch["controls"], beta=0.02,
    )
    zero = output["family"].sum() * 0
    style_distillation = zero
    category_distillation = zero
    if teacher is not None:
        direction = torch.ones(output["legacy_style"].shape[0], device=output["legacy_style"].device)
        style_distillation = functional.cosine_embedding_loss(
            output["legacy_style"], teacher["style"], direction,
        )
        temperature = 2.0
        category_distillation = functional.kl_div(
            functional.log_softmax(output["category"] / temperature, dim=-1),
            functional.softmax(teacher["category"] / temperature, dim=-1),
            reduction="batchmean",
        ) * temperature**2
    total = (
        structure_loss + 1.5 * initial[0] + initial[1] + 0.5 * initial[2] + 0.5 * initial[3]
        + 7.0 * refined[0] + 3.0 * refined[1] + 2.0 * refined[2] + 1.5 * refined[3]
        + 2.0 * metrics_loss + 0.5 * category_loss + typography_loss + 0.25 * family_variance_loss
        + 0.5 * controls_loss + 0.75 * style_distillation + 0.35 * category_distillation
    )
    return {
        "total": total, "structure": structure_loss, "coordinates": refined[0],
        "curves": refined[1], "straight_lines": refined[2], "curvature": refined[3],
        "metrics": metrics_loss, "category": category_loss, "typography": typography_loss,
        "family_variance": family_variance_loss,
        "controls": controls_loss, "style_distillation": style_distillation,
        "category_distillation": category_distillation,
    }
