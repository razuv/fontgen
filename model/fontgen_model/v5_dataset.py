from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .dataset import CATEGORY_TO_ID
from .text import glyph_bucket
from .v5_config import V5Config


class VectorOutlineDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        manifest: Path,
        embeddings_path: Path,
        config: V5Config,
        families: set[str] | None = None,
    ):
        self.config = config
        with manifest.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        self.rows = rows if families is None else [row for row in rows if row["family"] in families]
        payload = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        self.prompts: list[str] = payload["prompts"]
        self.embeddings: torch.Tensor = payload["embeddings"]
        self.prompt_ids = {prompt: index for index, prompt in enumerate(self.prompts)}
        missing = {row["prompt"] for row in self.rows}.difference(self.prompt_ids)
        if missing:
            raise ValueError(f"Embedding cache misses {len(missing)} corpus prompts")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row: dict[str, Any] = self.rows[index]
        commands = torch.zeros(self.config.max_commands, dtype=torch.long)
        coordinates = torch.zeros((self.config.max_commands, 6), dtype=torch.float32)
        length = min(len(row["commands"]), self.config.max_commands)
        commands[:length] = torch.tensor(row["commands"][:length], dtype=torch.long)
        coordinates[:length] = torch.tensor(row["coordinates"][:length], dtype=torch.float32)
        geometry = row.get("geometry_features")
        typography = [
            float(geometry.get("mean_glyph_width", 0.0)),
            float(geometry.get("x_height_ratio", 0.0)),
            float(geometry.get("descender_depth", 0.0)),
            float(geometry.get("curve_ratio", 0.0)),
            float(geometry.get("mean_complexity", 0.0)) / 64.0,
            float(geometry.get("mean_contours", 0.0)) / 4.0,
        ] if geometry else [0.0] * self.config.typography_dimensions
        return {
            "prompt_embedding": self.embeddings[self.prompt_ids[row["prompt"]]].float(),
            "glyph_id": torch.tensor(glyph_bucket(row["character"], self.config.glyph_buckets)),
            "category_id": torch.tensor(CATEGORY_TO_ID.get(row.get("category"), 0)),
            "controls": torch.tensor(row["controls"], dtype=torch.float32),
            "commands": commands,
            "coordinates": coordinates,
            "metrics": torch.tensor([row["advance_width"], row["left_side_bearing"]], dtype=torch.float32),
            "typography": torch.tensor(typography, dtype=torch.float32),
            "typography_mask": torch.tensor(1.0 if geometry else 0.0),
        }
