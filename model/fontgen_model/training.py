from __future__ import annotations

import torch
from torch.nn import functional

from .outline import COMMANDS, COORDINATE_MASKS


def _sdf_geometry_losses(
    predicted: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boundary_weight = 1.0 + 5.0 * torch.exp(-target.abs() * 10.0)
    sdf_error = functional.smooth_l1_loss(predicted, target, reduction="none", beta=0.025)
    sdf_loss = (sdf_error * boundary_weight).sum() / boundary_weight.sum().clamp_min(1)

    sobel_x = predicted.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    ).view(1, 1, 3, 3) / 8
    sobel_y = sobel_x.transpose(2, 3)
    predicted_x = functional.conv2d(predicted, sobel_x, padding=1)
    predicted_y = functional.conv2d(predicted, sobel_y, padding=1)
    target_x = functional.conv2d(target, sobel_x, padding=1)
    target_y = functional.conv2d(target, sobel_y, padding=1)
    predicted_magnitude = torch.sqrt(predicted_x.square() + predicted_y.square() + 1e-6)
    target_magnitude = torch.sqrt(target_x.square() + target_y.square() + 1e-6)
    boundary = target.abs().lt(0.3).float()
    normal_similarity = (
        predicted_x * target_x + predicted_y * target_y
    ) / (predicted_magnitude * target_magnitude).clamp_min(1e-5)
    normal_loss = ((1 - normal_similarity) * boundary).sum() / boundary.sum().clamp_min(1)
    eikonal_loss = (
        (predicted_magnitude - target_magnitude).abs() * boundary
    ).sum() / boundary.sum().clamp_min(1)

    laplacian = predicted.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    ).view(1, 1, 3, 3)
    predicted_curvature = functional.conv2d(predicted, laplacian, padding=1)
    target_curvature = functional.conv2d(target, laplacian, padding=1)
    curvature_loss = (
        functional.smooth_l1_loss(
            predicted_curvature, target_curvature, reduction="none", beta=0.01,
        ) * boundary
    ).sum() / boundary.sum().clamp_min(1)
    return sdf_loss, normal_loss, eikonal_loss, curvature_loss


def _multiscale_occupancy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for size in (32, 64):
        predicted_level = functional.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)
        target_level = functional.interpolate(target, size=(size, size), mode="area")
        losses.append(functional.binary_cross_entropy_with_logits(predicted_level, target_level))
    return torch.stack(losses).mean()


def raster_training_loss(batch: dict[str, torch.Tensor], output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    metrics_loss = functional.smooth_l1_loss(output["metrics"], batch["metrics"], beta=0.02)
    raster_loss = functional.binary_cross_entropy_with_logits(output["raster"], batch["raster"])
    probability = torch.sigmoid(output["raster"])
    intersection = (probability * batch["raster"]).sum(dim=(1, 2, 3))
    dice_loss = 1 - (
        (2 * intersection + 1)
        / (probability.sum(dim=(1, 2, 3)) + batch["raster"].sum(dim=(1, 2, 3)) + 1)
    ).mean()
    recognition_loss = functional.cross_entropy(output["recognition"], batch["glyph_id"])
    category_loss = functional.cross_entropy(output["category"], batch["category_id"])
    prompt_category_loss = functional.cross_entropy(output["prompt_category"], batch["category_id"])
    prompt_control_loss = functional.smooth_l1_loss(output["prompt_controls"], batch["controls"], beta=0.1)
    style_standard_deviation = torch.sqrt(output["style"].var(dim=0, unbiased=False) + 1e-4)
    style_variance_loss = functional.relu(0.18 - style_standard_deviation).mean()
    entropy_loss = -(
        probability.clamp(1e-5, 1 - 1e-5) * probability.clamp(1e-5, 1 - 1e-5).log()
        + (1 - probability).clamp(1e-5, 1 - 1e-5) * (1 - probability).clamp(1e-5, 1 - 1e-5).log()
    ).mean()
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=probability.device,
    ).view(1, 1, 3, 3) / 4
    sobel_y = sobel_x.transpose(2, 3)
    predicted_edges = torch.cat([
        functional.conv2d(probability, sobel_x, padding=1),
        functional.conv2d(probability, sobel_y, padding=1),
    ], dim=1)
    target_edges = torch.cat([
        functional.conv2d(batch["raster"], sobel_x, padding=1),
        functional.conv2d(batch["raster"], sobel_y, padding=1),
    ], dim=1)
    edge_loss = functional.smooth_l1_loss(predicted_edges, target_edges, beta=0.03)
    sdf_loss, normal_loss, eikonal_loss, curvature_loss = _sdf_geometry_losses(
        output["sdf"], batch["sdf"],
    )
    multiscale_loss = _multiscale_occupancy_loss(output["raster"], batch["raster"])
    total = (
        metrics_loss * 2.0 + raster_loss * 2.0 + dice_loss * 2.0
        + recognition_loss + category_loss * 0.4 + prompt_category_loss
        + prompt_control_loss * 1.5 + style_variance_loss * 0.5
        + edge_loss + entropy_loss * 0.05 + sdf_loss * 6.0
        + normal_loss * 0.8 + eikonal_loss * 1.5 + curvature_loss * 0.75
        + multiscale_loss * 1.5
    )
    return {
        "total": total, "metrics": metrics_loss, "raster": raster_loss,
        "dice": dice_loss, "recognition": recognition_loss,
        "category": category_loss, "edges": edge_loss,
        "prompt_category": prompt_category_loss, "prompt_controls": prompt_control_loss,
        "style_variance": style_variance_loss, "entropy": entropy_loss,
        "sdf": sdf_loss, "normals": normal_loss, "eikonal": eikonal_loss,
        "curvature": curvature_loss, "multiscale": multiscale_loss,
    }


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
    raster_losses = raster_training_loss(batch, output)
    metrics_loss = raster_losses["metrics"]
    raster_loss = raster_losses["raster"]
    dice_loss = raster_losses["dice"]
    recognition_loss = raster_losses["recognition"]
    total = (
        command_loss + coordinate_loss * 8.0 + metrics_loss * 2.0
        + raster_loss * 4.0 + dice_loss * 3.0 + recognition_loss * 0.5
    )
    return {
        "total": total, "commands": command_loss, "coordinates": coordinate_loss,
        "metrics": metrics_loss, "raster": raster_loss, "dice": dice_loss,
        "recognition": recognition_loss,
    }
