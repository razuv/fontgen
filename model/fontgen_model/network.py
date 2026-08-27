from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .text import BYTE_VOCABULARY


class FontgenNet(nn.Module):
    """Prompt + glyph conditioned autoregressive Bézier command generator.

    It never receives a source glyph during inference. A shared style vector is
    derived from the prompt and controls, while the character embedding supplies
    content. The decoder directly predicts a new ordered outline sequence.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.prompt_embedding = nn.Embedding(BYTE_VOCABULARY, d, padding_idx=0)
        self.prompt_positions = nn.Parameter(torch.randn(config.max_prompt_bytes, d) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.heads, dim_feedforward=config.feedforward,
            dropout=config.dropout, batch_first=True, norm_first=True,
        )
        self.prompt_encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)
        self.style_projection = nn.Sequential(
            nn.Linear(d + config.control_dimensions, d), nn.SiLU(),
            nn.Linear(d, config.style_dimensions),
        )
        self.glyph_embedding = nn.Embedding(config.glyph_buckets, d)
        self.style_to_model = nn.Linear(config.style_dimensions, d)
        self.raster_seed = nn.Linear(d, 128 * 4 * 4)
        self.raster_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
        self.raster_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(32, 64, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(64, d, 5, 2, 2), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.recognition_head = nn.Linear(d, config.glyph_buckets)
        self.category_head = nn.Linear(config.style_dimensions, config.category_count)
        self.prompt_category_head = nn.Linear(d, config.category_count)
        self.prompt_control_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, config.control_dimensions), nn.Tanh())
        self.raster_to_model = nn.Linear(d, d)
        self.style_film = nn.Linear(config.style_dimensions, 48)
        self.raster_refiner = nn.Sequential(
            nn.Conv2d(1, 24, 5, 1, 2), nn.GroupNorm(6, 24), nn.SiLU(),
            nn.Conv2d(24, 24, 3, 1, 1), nn.GroupNorm(6, 24), nn.SiLU(),
            nn.Conv2d(24, 1, 3, 1, 1),
        )
        self.sdf_coordinate_refiner = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 16, 3, 1, 1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 1, 3, 1, 1),
        )
        nn.init.zeros_(self.sdf_coordinate_refiner[-1].weight)
        nn.init.zeros_(self.sdf_coordinate_refiner[-1].bias)
        axis = torch.linspace(-1, 1, config.raster_size)
        y_grid, x_grid = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("sdf_coordinate_grid", torch.stack((x_grid, y_grid)), persistent=False)
        self.command_embedding = nn.Embedding(config.command_count, d)
        self.coordinate_embedding = nn.Linear(config.coordinate_count, d)
        self.decoder_positions = nn.Parameter(torch.randn(config.max_commands, d) * 0.01)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=config.heads, dim_feedforward=config.feedforward,
            dropout=config.dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.command_head = nn.Linear(d, config.command_count)
        self.coordinate_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 6), nn.Tanh())
        self.metrics_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 2))

    def load_compatible_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state_dict, strict=False)
        missing = [
            name for name in incompatible.missing_keys
            if not name.startswith("sdf_coordinate_refiner.")
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Incompatible V4 checkpoint: missing={missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    def encode_prompt_context(self, prompts: torch.Tensor) -> torch.Tensor:
        length = prompts.shape[1]
        embedded = self.prompt_embedding(prompts) + self.prompt_positions[:length]
        padding_mask = prompts.eq(0)
        encoded = self.prompt_encoder(embedded, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return pooled

    def encode_style(self, prompts: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        pooled = self.encode_prompt_context(prompts)
        return self.style_projection(torch.cat([pooled, controls], dim=-1))

    def forward(
        self,
        prompts: torch.Tensor,
        glyph_ids: torch.Tensor,
        controls: torch.Tensor,
        input_commands: torch.Tensor,
        input_coordinates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        conditioning = self.condition(prompts, glyph_ids, controls)
        decoded = self.decode(conditioning["memory"], input_commands, input_coordinates)
        return {**conditioning, **decoded}

    def condition(
        self,
        prompts: torch.Tensor,
        glyph_ids: torch.Tensor,
        controls: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prompt_context = self.encode_prompt_context(prompts)
        style = self.style_projection(torch.cat([prompt_context, controls], dim=-1))
        content_token = self.style_to_model(style) + self.glyph_embedding(glyph_ids)
        raster_logits = self.raster_decoder(self.raster_seed(content_token).view(-1, 128, 4, 4))
        if raster_logits.shape[-1] != self.config.raster_size:
            raster_logits = nn.functional.interpolate(
                raster_logits, size=(self.config.raster_size, self.config.raster_size),
                mode="bilinear", align_corners=False,
            )
        refined = self.raster_refiner[1](self.raster_refiner[0](raster_logits))
        scale, bias = self.style_film(style).chunk(2, dim=-1)
        refined = refined * (1 + torch.tanh(scale).unsqueeze(-1).unsqueeze(-1))
        refined = refined + bias.unsqueeze(-1).unsqueeze(-1)
        refined = self.raster_refiner[2](refined)
        refined = self.raster_refiner[5](self.raster_refiner[4](self.raster_refiner[3](refined)))
        raster_logits = raster_logits + self.raster_refiner[6](refined)
        coordinate_grid = self.sdf_coordinate_grid.unsqueeze(0).expand(raster_logits.shape[0], -1, -1, -1)
        sdf_features = torch.cat((
            torch.tanh(raster_logits / self.config.sdf_logit_scale), coordinate_grid,
        ), dim=1)
        raster_logits = raster_logits + self.sdf_coordinate_refiner(sdf_features)
        raster_features = self.raster_encoder(torch.sigmoid(raster_logits))
        memory_token = content_token + self.raster_to_model(raster_features)
        memory = memory_token.unsqueeze(1)
        return {
            "memory": memory,
            "metrics": self.metrics_head(memory_token),
            "style": style,
            "raster": raster_logits,
            "sdf": torch.tanh(raster_logits / self.config.sdf_logit_scale),
            "recognition": self.recognition_head(raster_features),
            "category": self.category_head(style),
            "prompt_category": self.prompt_category_head(prompt_context),
            "prompt_controls": self.prompt_control_head(prompt_context),
        }

    def decode(
        self,
        memory: torch.Tensor,
        input_commands: torch.Tensor,
        input_coordinates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        length = input_commands.shape[1]
        target = (
            self.command_embedding(input_commands)
            + self.coordinate_embedding(input_coordinates)
            + self.decoder_positions[:length]
        )
        causal_mask = nn.Transformer.generate_square_subsequent_mask(length, device=target.device)
        decoded = self.decoder(target, memory, tgt_mask=causal_mask, tgt_is_causal=True)
        return {
            "commands": self.command_head(decoded),
            "coordinates": self.coordinate_head(decoded),
        }
