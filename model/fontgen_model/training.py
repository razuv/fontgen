from __future__ import annotations

import torch
from torch import nn
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


class PerceptualLoss(nn.Module):
    """Lightweight perceptual loss using a small frozen feature extractor."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
        )
        for param in self.features.parameters():
            param.requires_grad_(False)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self.features(predicted)
        target_features = self.features(target)
        return functional.smooth_l1_loss(pred_features, target_features, beta=0.01)


class UncertaintyWeights(nn.Module):
    """Learnable loss weights via homoscedastic uncertainty (Kendall et al.)."""

    def __init__(self, num_losses: int) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total = torch.zeros(1, device=losses[0].device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total


def _family_consistency_loss(styles: torch.Tensor, families: list[str]) -> torch.Tensor:
    """Pull style vectors of the same family together."""
    if len(families) < 2:
        return styles.sum() * 0
    unique_families = list(dict.fromkeys(families))
    losses = []
    for family in unique_families:
        indices = [i for i, f in enumerate(families) if f == family]
        if len(indices) < 2:
            continue
        family_styles = styles[indices]
        centroid = family_styles.mean(dim=0, keepdim=True)
        losses.append(functional.mse_loss(family_styles, centroid.expand_as(family_styles)))
    return torch.stack(losses).mean() if losses else styles.sum() * 0


def _discriminator_r1_penalty(real: torch.Tensor, discriminator: nn.Module) -> torch.Tensor:
    """R1 gradient penalty for discriminator stability."""
    real.requires_grad_(True)
    pred = discriminator(real)
    grad = torch.autograd.grad(pred.sum(), real, create_graph=True)[0]
    return grad.flatten(1).square().sum(dim=1).mean()


def _contour_smoothness_loss(coordinates: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
    """Penalize high-frequency jitter in generated outlines."""
    drawable = commands.ge(3) & commands.le(6)
    if not drawable.any():
        return coordinates.sum() * 0
    diffs = coordinates[:, 1:] - coordinates[:, :-1]
    return diffs.square().mean()


def raster_training_loss(
    batch: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    *,
    perceptual: PerceptualLoss | None = None,
) -> dict[str, torch.Tensor]:
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
    sdf_occupancy_loss = functional.binary_cross_entropy_with_logits(
        output["sdf_logits"] * 4.0, batch["raster"],
    )
    multiscale_loss = _multiscale_occupancy_loss(output["raster"], batch["raster"])
    sdf_multiscale_loss = _multiscale_occupancy_loss(output["sdf_logits"] * 4.0, batch["raster"])

    perceptual_loss = torch.tensor(0.0, device=probability.device)
    if perceptual is not None:
        perceptual_loss = perceptual(probability, batch["raster"])

    family_loss = torch.tensor(0.0, device=probability.device)
    if "family" in batch and isinstance(batch["family"], (list, tuple)):
        family_loss = _family_consistency_loss(output["style"], batch["family"])

    total = (
        metrics_loss * 2.0 + raster_loss * 2.0 + dice_loss * 2.0
        + recognition_loss + category_loss * 0.4 + prompt_category_loss
        + prompt_control_loss * 1.5 + style_variance_loss * 0.5
        + edge_loss + entropy_loss * 0.05 + sdf_loss * 3.0
        + normal_loss * 0.6 + eikonal_loss + curvature_loss * 0.5
        + multiscale_loss * 1.5 + sdf_occupancy_loss * 3.0 + sdf_multiscale_loss
        + perceptual_loss * 0.5 + family_loss * 0.3
    )
    return {
        "total": total, "metrics": metrics_loss, "raster": raster_loss,
        "dice": dice_loss, "recognition": recognition_loss,
        "category": category_loss, "edges": edge_loss,
        "prompt_category": prompt_category_loss, "prompt_controls": prompt_control_loss,
        "style_variance": style_variance_loss, "entropy": entropy_loss,
        "sdf": sdf_loss, "normals": normal_loss, "eikonal": eikonal_loss,
        "curvature": curvature_loss, "multiscale": multiscale_loss,
        "sdf_occupancy": sdf_occupancy_loss, "sdf_multiscale": sdf_multiscale_loss,
        "perceptual": perceptual_loss, "family_consistency": family_loss,
    }


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """Non-saturating logistic GAN discriminator loss."""
    real_loss = functional.softplus(-real_logits).mean()
    fake_loss = functional.softplus(fake_logits).mean()
    return (real_loss + fake_loss) * 0.5


def generator_adversarial_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating logistic GAN generator loss."""
    return functional.softplus(-fake_logits).mean()


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
