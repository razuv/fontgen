from __future__ import annotations

import torch
from torch import nn

from .structured_config import StructuredConfig


class StructuredVectorFontNet(nn.Module):
    """Family style, glyph topology and quadratic geometry are separate stages."""

    def __init__(self, config: StructuredConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.family_encoder = nn.Sequential(
            nn.Linear(config.prompt_dimensions + config.control_dimensions, d),
            nn.LayerNorm(d), nn.GELU(), nn.Linear(d, config.family_dimensions), nn.LayerNorm(config.family_dimensions),
        )
        self.family_to_model = nn.Linear(config.family_dimensions, d)
        self.glyph_embedding = nn.Embedding(config.glyph_buckets, d)
        self.memory_types = nn.Parameter(torch.randn(2, d) * 0.01)
        self.command_embedding = nn.Embedding(config.command_count, d)
        self.positions = nn.Parameter(torch.randn(config.max_commands, d) * 0.01)

        structure_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=config.heads, dim_feedforward=config.feedforward,
            dropout=config.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.structure_decoder = nn.TransformerDecoder(structure_layer, config.structure_layers)
        self.structure_norm = nn.LayerNorm(d)
        self.structure_head = nn.Linear(d, config.command_count)

        geometry_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.heads, dim_feedforward=config.feedforward,
            dropout=config.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.geometry_decoder = nn.TransformerEncoder(geometry_layer, config.geometry_layers)
        self.geometry_norm = nn.LayerNorm(d)
        self.initial_coordinate_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4), nn.Tanh())

        self.coordinate_embedding = nn.Sequential(nn.Linear(4, d), nn.LayerNorm(d))
        refiner_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.heads, dim_feedforward=config.feedforward,
            dropout=config.dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.vector_refiner = nn.TransformerEncoder(refiner_layer, config.refiner_layers)
        self.refiner_norm = nn.LayerNorm(d)
        self.coordinate_delta_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4), nn.Tanh())

        self.category_head = nn.Linear(config.family_dimensions, config.category_count)
        self.typography_head = nn.Sequential(
            nn.Linear(config.family_dimensions, d), nn.GELU(), nn.Linear(d, config.typography_dimensions),
        )
        # Training-only compatibility target that transfers V4.1 prompt/style knowledge
        # without copying its raster or contour decoders into the new vector path.
        self.legacy_style_head = nn.Linear(config.family_dimensions, config.legacy_style_dimensions)
        self.control_head = nn.Sequential(
            nn.Linear(config.family_dimensions, d), nn.GELU(), nn.Linear(d, config.control_dimensions), nn.Tanh(),
        )
        self.metrics_head = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 2))

    def encode_family(self, prompt_embeddings: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        return self.family_encoder(torch.cat((prompt_embeddings, controls), dim=-1))

    def condition(
        self, prompt_embeddings: torch.Tensor, glyph_ids: torch.Tensor, controls: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        family = self.encode_family(prompt_embeddings, controls)
        family_token = self.family_to_model(family)
        glyph_token = self.glyph_embedding(glyph_ids)
        memory = torch.stack((family_token, glyph_token), dim=1) + self.memory_types.unsqueeze(0)
        return {"family": family, "family_token": family_token, "glyph_token": glyph_token, "memory": memory}

    def decode_structure(self, memory: torch.Tensor, commands: torch.Tensor) -> dict[str, torch.Tensor]:
        length = commands.shape[1]
        target = self.command_embedding(commands) + self.positions[:length]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(length, device=commands.device)
        hidden = self.structure_norm(
            self.structure_decoder(target, memory, tgt_mask=causal_mask, tgt_is_causal=True)
        )
        return {"structure_hidden": hidden, "structure_logits": self.structure_head(hidden)}

    def decode_geometry(
        self,
        family_token: torch.Tensor,
        glyph_token: torch.Tensor,
        commands: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        length = commands.shape[1]
        condition = family_token + glyph_token
        tokens = self.command_embedding(commands) + self.positions[:length] + condition.unsqueeze(1)
        geometry_hidden = self.geometry_norm(
            self.geometry_decoder(tokens, src_key_padding_mask=padding_mask)
        )
        initial = self.initial_coordinate_head(geometry_hidden)
        refinement_tokens = geometry_hidden + self.coordinate_embedding(initial)
        refined_hidden = self.refiner_norm(
            self.vector_refiner(refinement_tokens, src_key_padding_mask=padding_mask)
        )
        refined = (initial + 0.15 * self.coordinate_delta_head(refined_hidden)).clamp(-1, 1)
        return {
            "geometry_hidden": geometry_hidden,
            "initial_coordinates": initial,
            "coordinates": refined,
        }

    def forward(
        self,
        prompt_embeddings: torch.Tensor,
        glyph_ids: torch.Tensor,
        controls: torch.Tensor,
        commands: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        conditioned = self.condition(prompt_embeddings, glyph_ids, controls)
        structure = self.decode_structure(conditioned["memory"], commands)
        geometry = self.decode_geometry(
            conditioned["family_token"], conditioned["glyph_token"], commands, commands.eq(0),
        )
        family = conditioned["family"]
        metrics_input = torch.cat((conditioned["family_token"], conditioned["glyph_token"]), dim=-1)
        return {
            **conditioned, **structure, **geometry,
            "category": self.category_head(family),
            "typography": self.typography_head(family),
            "legacy_style": self.legacy_style_head(family),
            "reconstructed_controls": self.control_head(family),
            "metrics": self.metrics_head(metrics_input),
        }
