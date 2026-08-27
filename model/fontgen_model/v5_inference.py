from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .inference import GeneratedGlyph
from .outline import COMMAND_TO_ID, COMMANDS
from .text import glyph_bucket
from .v5_config import V5Config
from .v5_network import VectorFontNet
from .v5_text import MultilingualPromptEncoder


class VectorFontGenerator:
    """Direct Bézier inference path for v5; no rasterization or tracing."""

    def __init__(self, checkpoint_path: Path, device: str | None = None):
        selected = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.device = torch.device(selected)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.config = V5Config(**checkpoint["config"])
        self.model = VectorFontNet(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.text_encoder = MultilingualPromptEncoder(device=selected)
        self.checkpoint_id = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:12]
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    @staticmethod
    def _legal_logits(logits: torch.Tensor, previous: int, contour_segments: int) -> torch.Tensor:
        legal = torch.full_like(logits, -torch.inf)
        if previous == COMMAND_TO_ID["BOS"]:
            legal[COMMAND_TO_ID["M"]] = logits[COMMAND_TO_ID["M"]]
        elif previous == COMMAND_TO_ID["Z"]:
            legal[COMMAND_TO_ID["M"]] = logits[COMMAND_TO_ID["M"]]
            legal[COMMAND_TO_ID["EOS"]] = logits[COMMAND_TO_ID["EOS"]]
        else:
            for name in ("L", "Q", "C"):
                legal[COMMAND_TO_ID[name]] = logits[COMMAND_TO_ID[name]]
            if contour_segments >= 2:
                legal[COMMAND_TO_ID["Z"]] = logits[COMMAND_TO_ID["Z"]]
        return legal

    @torch.inference_mode()
    def generate_family(
        self,
        prompt: str,
        characters: str,
        controls: list[float],
        seed: int,
        temperature: float = 0.65,
    ) -> list[GeneratedGlyph]:
        prompt_embedding = self.text_encoder.encode([prompt]).to(self.device)
        generated: list[GeneratedGlyph] = []
        random = torch.Generator(device="cpu").manual_seed(seed)
        for character in dict.fromkeys(characters):
            if character.isspace():
                generated.append(GeneratedGlyph(character, [], [], 0.33, 0.0))
                continue
            glyph_id = torch.tensor([glyph_bucket(character, self.config.glyph_buckets)], device=self.device)
            control_tensor = torch.tensor([controls], dtype=torch.float32, device=self.device)
            memory = self.model.condition(prompt_embedding, glyph_id, control_tensor)
            command_tokens = [COMMAND_TO_ID["BOS"]]
            coordinate_tokens = [[0.0] * 6]
            contour_segments = 0
            for _ in range(self.config.max_commands - 2):
                commands = torch.tensor([command_tokens], device=self.device)
                coordinates = torch.tensor([coordinate_tokens], dtype=torch.float32, device=self.device)
                output = self.model.decode(memory, commands, coordinates)
                logits = output["commands"][0, -1] / max(0.1, temperature)
                logits = self._legal_logits(logits, command_tokens[-1], contour_segments)
                probabilities = torch.softmax(logits, dim=-1)
                next_command = int(torch.multinomial(probabilities.cpu(), 1, generator=random).item())
                if next_command == COMMAND_TO_ID["EOS"]:
                    break
                next_coordinates = output["coordinates"][0, -1].tolist()
                command_tokens.append(next_command)
                coordinate_tokens.append(next_coordinates)
                if next_command == COMMAND_TO_ID["M"] or next_command == COMMAND_TO_ID["Z"]:
                    contour_segments = 0
                else:
                    contour_segments += 1
            if command_tokens[-1] != COMMAND_TO_ID["Z"]:
                command_tokens.append(COMMAND_TO_ID["Z"])
                coordinate_tokens.append([0.0] * 6)

            # Re-run the complete outline and use the non-causal refiner's
            # coordinates while preserving grammar-constrained commands.
            final = self.model.decode(
                memory,
                torch.tensor([command_tokens], device=self.device),
                torch.tensor([coordinate_tokens], dtype=torch.float32, device=self.device),
            )
            refined_coordinates = final["coordinates"][0, :-1].tolist()
            coordinate_tokens[1:] = refined_coordinates[: len(coordinate_tokens) - 1]
            metrics = self.model.metrics_head(torch.cat((memory[:, 0], memory[:, 1]), dim=-1))[0].tolist()
            generated.append(
                GeneratedGlyph(
                    character=character,
                    commands=[COMMANDS[token] for token in command_tokens[1:]],
                    coordinates=coordinate_tokens[1:],
                    advance_width=max(0.2, min(2.0, float(metrics[0]))),
                    left_side_bearing=max(-0.5, min(0.8, float(metrics[1]))),
                )
            )
        return generated
