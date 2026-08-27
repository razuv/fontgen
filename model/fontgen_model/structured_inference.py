from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .inference import GeneratedGlyph
from .quadratic import QUADRATIC_COMMAND_TO_ID, QUADRATIC_COMMANDS
from .structured_config import StructuredConfig
from .structured_network import StructuredVectorFontNet
from .text import glyph_bucket
from .v5_text import MultilingualPromptEncoder


class StructuredFontGenerator:
    """Inference for family latent → topology → quadratic geometry → refinement."""

    architecture = "structured-quadratic-v1"

    def __init__(self, checkpoint_path: Path, device: str | None = None):
        selected = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.device = torch.device(selected)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.config = StructuredConfig(**checkpoint["config"])
        self.model = StructuredVectorFontNet(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.text_encoder = MultilingualPromptEncoder(device=selected)
        self.checkpoint_id = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:12]
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    @staticmethod
    def _legal_logits(logits: torch.Tensor, previous: int, segments: int) -> torch.Tensor:
        legal = torch.full_like(logits, -torch.inf)
        if previous == QUADRATIC_COMMAND_TO_ID["BOS"]:
            legal[QUADRATIC_COMMAND_TO_ID["M"]] = logits[QUADRATIC_COMMAND_TO_ID["M"]]
        elif previous == QUADRATIC_COMMAND_TO_ID["Z"]:
            legal[QUADRATIC_COMMAND_TO_ID["M"]] = logits[QUADRATIC_COMMAND_TO_ID["M"]]
            legal[QUADRATIC_COMMAND_TO_ID["EOS"]] = logits[QUADRATIC_COMMAND_TO_ID["EOS"]]
        else:
            legal[QUADRATIC_COMMAND_TO_ID["L"]] = logits[QUADRATIC_COMMAND_TO_ID["L"]]
            legal[QUADRATIC_COMMAND_TO_ID["Q"]] = logits[QUADRATIC_COMMAND_TO_ID["Q"]]
            if segments >= 2:
                legal[QUADRATIC_COMMAND_TO_ID["Z"]] = logits[QUADRATIC_COMMAND_TO_ID["Z"]]
        return legal

    @torch.inference_mode()
    def generate_family(
        self, prompt: str, characters: str, controls: list[float], seed: int, temperature: float = 0.65,
    ) -> list[GeneratedGlyph]:
        prompt_embedding = self.text_encoder.encode([prompt]).to(self.device)
        random = torch.Generator(device="cpu").manual_seed(seed)
        output_glyphs: list[GeneratedGlyph] = []
        for character in dict.fromkeys(characters):
            if character.isspace():
                output_glyphs.append(GeneratedGlyph(character, [], [], 0.33, 0.0))
                continue
            glyph_id = torch.tensor([glyph_bucket(character, self.config.glyph_buckets)], device=self.device)
            control_tensor = torch.tensor([controls], dtype=torch.float32, device=self.device)
            conditioned = self.model.condition(prompt_embedding, glyph_id, control_tensor)
            command_tokens = [QUADRATIC_COMMAND_TO_ID["BOS"]]
            contour_segments = 0
            for _ in range(self.config.max_commands - 2):
                commands = torch.tensor([command_tokens], device=self.device)
                structure = self.model.decode_structure(conditioned["memory"], commands)
                logits = structure["structure_logits"][0, -1] / max(temperature, 0.1)
                logits = self._legal_logits(logits, command_tokens[-1], contour_segments)
                next_command = int(torch.multinomial(
                    torch.softmax(logits, dim=-1).cpu(), 1, generator=random,
                ).item())
                if next_command == QUADRATIC_COMMAND_TO_ID["EOS"]:
                    break
                command_tokens.append(next_command)
                if next_command in {QUADRATIC_COMMAND_TO_ID["M"], QUADRATIC_COMMAND_TO_ID["Z"]}:
                    contour_segments = 0
                else:
                    contour_segments += 1
            if command_tokens[-1] != QUADRATIC_COMMAND_TO_ID["Z"]:
                command_tokens.append(QUADRATIC_COMMAND_TO_ID["Z"])
            commands = torch.tensor([command_tokens], device=self.device)
            geometry = self.model.decode_geometry(
                conditioned["family_token"], conditioned["glyph_token"], commands,
            )
            metrics = self.model.metrics_head(torch.cat((
                conditioned["family_token"], conditioned["glyph_token"],
            ), dim=-1))[0].tolist()
            coordinate_rows = geometry["coordinates"][0].tolist()[1:]
            output_glyphs.append(GeneratedGlyph(
                character=character,
                commands=[QUADRATIC_COMMANDS[token] for token in command_tokens[1:]],
                coordinates=[coordinates + [0.0, 0.0] for coordinates in coordinate_rows],
                advance_width=max(0.2, min(2.0, float(metrics[0]))),
                left_side_bearing=max(-0.5, min(0.8, float(metrics[1]))),
            ))
        return output_glyphs
