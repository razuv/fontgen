from __future__ import annotations

from pathlib import Path

import torch

DEFAULT_TEXT_MODEL = Path(__file__).parents[1] / "models" / "paraphrase-multilingual-MiniLM-L12-v2"


class MultilingualPromptEncoder:
    """Frozen semantic prompt encoder shared by corpus preparation and inference."""

    def __init__(self, model_path: Path = DEFAULT_TEXT_MODEL, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer

        if not model_path.exists():
            raise FileNotFoundError(
                f"Prompt encoder is missing at {model_path}. Run scripts/download_v5_text_model.py first."
            )
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, fix_mistral_regex=True,
        )
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, prompts: list[str], batch_size: int = 32) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        for start in range(0, len(prompts), batch_size):
            encoded = self.tokenizer(
                prompts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            embeddings.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(embeddings) if embeddings else torch.empty((0, 384))
