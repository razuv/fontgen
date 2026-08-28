from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ModelConfig
from .raster import decode_mask, outline_mask, signed_distance_field
from .text import encode_prompt, glyph_bucket

CATEGORY_TO_ID = {name: index for index, name in enumerate(("SANS_SERIF", "SERIF", "DISPLAY", "HANDWRITING", "MONOSPACE"))}


class OutlineDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        manifest: Path,
        config: ModelConfig,
        *,
        augment: bool = False,
        control_jitter: float = 0.1,
        cfg_dropout: float = 0.0,
    ):
        self.config = config
        self.augment = augment
        self.control_jitter = control_jitter
        self.cfg_dropout = cfg_dropout
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
        raster = (
            decode_mask(row["raster"], self.config.raster_size)
            if row.get("raster")
            else np.asarray(
                outline_mask(row["commands"], row["coordinates"], self.config.raster_size),
                dtype=np.float32,
            ) / 255.0
        )
        controls = list(row["controls"])
        if self.augment:
            controls = [
                max(-1.0, min(1.0, c + random.gauss(0, self.control_jitter)))
                for c in controls
            ]
        prompt_text = row["prompt"]
        if self.cfg_dropout > 0 and random.random() < self.cfg_dropout:
            prompt_text = ""
        return {
            "prompt": encode_prompt(prompt_text, self.config.max_prompt_bytes),
            "glyph_id": torch.tensor(glyph_bucket(row["character"], self.config.glyph_buckets)),
            "category_id": torch.tensor(CATEGORY_TO_ID.get(row.get("category"), 0)),
            "controls": torch.tensor(controls, dtype=torch.float32),
            "commands": commands,
            "coordinates": coordinates,
            "metrics": torch.tensor([row["advance_width"], row["left_side_bearing"]], dtype=torch.float32),
            "raster": torch.from_numpy(raster).unsqueeze(0),
            "sdf": torch.from_numpy(signed_distance_field(raster)).unsqueeze(0),
            "family": str(row.get("family", "")),
        }
