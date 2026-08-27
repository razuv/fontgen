from __future__ import annotations

import torch
from torch import nn

from .v5_config import V5Config


class VectorFontNet(nn.Module):
    """Prompt-conditioned direct outline generator with a second-pass refiner.

    Prompt embeddings come from a frozen multilingual language model. The
    network never consumes a reference glyph at inference time and never
    rasterizes its output: both passes predict native outline commands and
    Bézier control points.
    """

    def __init__(self, config: V5Config):
        super().__init__()
        self.config = config
        d = config.d_model
        self.prompt_projection = nn.Sequential(
            nn.Linear(config.prompt_dimensions, d), nn.LayerNorm(d), nn.SiLU(),
        )
        self.control_projection = nn.Sequential(
            nn.Linear(config.control_dimensions, d), nn.SiLU(), nn.Linear(d, d),
        )
        self.glyph_embedding = nn.Embedding(config.glyph_buckets, d)
        self.memory_type = nn.Parameter(torch.randn(3, d) * 0.01)
        self.category_head = nn.Linear(d, config.category_count)

        self.command_embedding = nn.Embedding(config.command_count, d)
        self.coordinate_embedding = nn.Sequential(nn.Linear(6, d), nn.LayerNorm(d))
        self.positions = nn.Parameter(torch.randn(config.max_commands, d) * 0.01)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.initial_norm = nn.LayerNorm(d)
        self.initial_command_head = nn.Linear(d, config.command_count)
        self.initial_coordinate_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 6), nn.Tanh())

        refiner_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.refiner = nn.TransformerEncoder(refiner_layer, config.refiner_layers)
        self.refiner_norm = nn.LayerNorm(d)
        self.refined_command_head = nn.Linear(d, config.command_count)
        self.coordinate_delta_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 6), nn.Tanh())
        self.metrics_head = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 2))
        self.typography_head = nn.Sequential(
            nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, config.typography_dimensions),
        )

    def condition(
        self,
        prompt_embeddings: torch.Tensor,
        glyph_ids: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.stack(
            (
                self.prompt_projection(prompt_embeddings),
                self.glyph_embedding(glyph_ids),
                self.control_projection(controls),
            ),
            dim=1,
        )
        return tokens + self.memory_type.unsqueeze(0)

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
            + self.positions[:length]
        )
        causal_mask = nn.Transformer.generate_square_subsequent_mask(length, device=target.device)
        hidden = self.initial_norm(
            self.decoder(target, memory, tgt_mask=causal_mask, tgt_is_causal=True)
        )
        initial_commands = self.initial_command_head(hidden)
        initial_coordinates = self.initial_coordinate_head(hidden)

        # The second pass sees the entire outline, which lets it repair joins,
        # repeated points and long-range contour consistency.
        refinement_input = (
            hidden
            + initial_commands.softmax(-1) @ self.command_embedding.weight
            + self.coordinate_embedding(initial_coordinates)
        )
        refined_hidden = self.refiner_norm(self.refiner(refinement_input))
        return {
            "initial_commands": initial_commands,
            "initial_coordinates": initial_coordinates,
            "commands": self.refined_command_head(refined_hidden) + initial_commands,
            "coordinates": (initial_coordinates + 0.25 * self.coordinate_delta_head(refined_hidden)).clamp(-1, 1),
        }

    def forward(
        self,
        prompt_embeddings: torch.Tensor,
        glyph_ids: torch.Tensor,
        controls: torch.Tensor,
        input_commands: torch.Tensor,
        input_coordinates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory = self.condition(prompt_embeddings, glyph_ids, controls)
        output = self.decode(memory, input_commands, input_coordinates)
        output["metrics"] = self.metrics_head(torch.cat((memory[:, 0], memory[:, 1]), dim=-1))
        output["category"] = self.category_head(memory[:, 0])
        output["typography"] = self.typography_head(torch.cat((memory[:, 0], memory[:, 2]), dim=-1))
        return output
