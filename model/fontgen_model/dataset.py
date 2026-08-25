from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .config import ModelConfig
from .text import encode_prompt, glyph_bucket


class OutlineDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, manifest: Path, config: ModelConfig):
        self.config = config
        with manifest.open(encoding="utf-8") as source:
            self.rows = [json.loads(line) for line in source if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row: dict[str, Any] = self.rows[index]
        commands = torch.zeros(self.config.max_commands, dtype=torch.long)
        coordinates = torch.zeros((self.config.max_commands, 6), dtype=torch.float32)
        length = min(len(row["commands"]), self.config.max_commands)
        commands[:length] = torch.tensor(row["commands"][:length], dtype=torch.long)
        coordinates[:length] = torch.tensor(row["coordinates"][:length], dtype=torch.float32)
        return {
            "prompt": encode_prompt(row["prompt"], self.config.max_prompt_bytes),
            "glyph_id": torch.tensor(glyph_bucket(row["character"], self.config.glyph_buckets)),
            "controls": torch.tensor(row["controls"], dtype=torch.float32),
            "commands": commands,
            "coordinates": coordinates,
            "metrics": torch.tensor([row["advance_width"], row["left_side_bearing"]], dtype=torch.float32),
        }

