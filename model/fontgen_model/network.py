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

    def encode_style(self, prompts: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        length = prompts.shape[1]
        embedded = self.prompt_embedding(prompts) + self.prompt_positions[:length]
        padding_mask = prompts.eq(0)
        encoded = self.prompt_encoder(embedded, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.style_projection(torch.cat([pooled, controls], dim=-1))

    def forward(
        self,
        prompts: torch.Tensor,
        glyph_ids: torch.Tensor,
        controls: torch.Tensor,
        input_commands: torch.Tensor,
        input_coordinates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        style = self.encode_style(prompts, controls)
        memory_token = self.style_to_model(style) + self.glyph_embedding(glyph_ids)
        memory = memory_token.unsqueeze(1)
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
            "metrics": self.metrics_head(memory_token),
            "style": style,
        }

