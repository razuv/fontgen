from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .dataset import CATEGORY_TO_ID
from .structured_config import StructuredConfig
from .text import encode_prompt, glyph_bucket


class StructuredOutlineDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self, manifest: Path, embeddings_path: Path, config: StructuredConfig,
        families: set[str] | None = None, teacher_prompt_bytes: int | None = None,
    ):
        self.config = config
        with manifest.open(encoding="utf-8") as source:
            self.rows = []
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if families is None or row["family"] in families:
                    self.rows.append(row)
        if any(row.get("representation") != "quadratic-v1" for row in self.rows):
            raise ValueError("Structured model requires a quadratic-v1 manifest")
        cache = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        self.embeddings = cache["embeddings"]
        self.prompt_ids = {prompt: index for index, prompt in enumerate(cache["prompts"])}
        self.teacher_prompt_bytes = teacher_prompt_bytes

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row: dict[str, Any] = self.rows[index]
        commands = torch.zeros(self.config.max_commands, dtype=torch.long)
        coordinates = torch.zeros((self.config.max_commands, 4), dtype=torch.float32)
        length = min(len(row["commands"]), self.config.max_commands)
        commands[:length] = torch.tensor(row["commands"][:length], dtype=torch.long)
        coordinates[:length] = torch.tensor(row["coordinates"][:length], dtype=torch.float32)
        geometry = row.get("geometry_features") or {}
        typography = [
            float(geometry.get("mean_glyph_width", 0.0)), float(geometry.get("x_height_ratio", 0.0)),
            float(geometry.get("descender_depth", 0.0)), float(geometry.get("curve_ratio", 0.0)),
            float(geometry.get("mean_complexity", 0.0)) / 64.0,
            float(geometry.get("mean_contours", 0.0)) / 4.0,
        ]
        item = {
            "prompt_embedding": self.embeddings[self.prompt_ids[row["prompt"]]].float(),
            "glyph_id": torch.tensor(glyph_bucket(row["character"], self.config.glyph_buckets)),
            "category_id": torch.tensor(CATEGORY_TO_ID.get(row.get("category"), 0)),
            "controls": torch.tensor(row["controls"], dtype=torch.float32),
            "commands": commands,
            "coordinates": coordinates,
            "metrics": torch.tensor([row["advance_width"], row["left_side_bearing"]], dtype=torch.float32),
            "typography": torch.tensor(typography, dtype=torch.float32),
        }
        if self.teacher_prompt_bytes is not None:
            item["teacher_prompt"] = encode_prompt(row["prompt"], self.teacher_prompt_bytes)
        return item
