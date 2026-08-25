from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import ModelConfig
from .network import FontgenNet
from .outline import COMMAND_TO_ID, COMMANDS
from .text import encode_prompt, glyph_bucket


@dataclass
class GeneratedGlyph:
    character: str
    commands: list[str]
    coordinates: list[list[float]]
    advance_width: float
    left_side_bearing: float


class FontgenGenerator:
    def __init__(self, checkpoint_path: Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.config = ModelConfig(**checkpoint["config"])
        self.model = FontgenNet(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.checkpoint_id = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:12]

    @torch.inference_mode()
    def generate_glyph(
        self,
        prompt: str,
        character: str,
        controls: list[float],
        temperature: float = 0.75,
        generator: torch.Generator | None = None,
    ) -> GeneratedGlyph:
        prompts = encode_prompt(prompt, self.config.max_prompt_bytes).unsqueeze(0).to(self.device)
        glyph_ids = torch.tensor([glyph_bucket(character, self.config.glyph_buckets)], device=self.device)
        control_tensor = torch.tensor([controls], dtype=torch.float32, device=self.device)
        command_tokens = [COMMAND_TO_ID["BOS"]]
        coordinate_tokens = [[0.0] * 6]
        for step in range(self.config.max_commands - 1):
            commands = torch.tensor([command_tokens], device=self.device)
            coordinates = torch.tensor([coordinate_tokens], dtype=torch.float32, device=self.device)
            output = self.model(prompts, glyph_ids, control_tensor, commands, coordinates)
            logits = output["commands"][0, -1] / max(0.05, temperature)
            # A valid outline begins with a move. PAD/BOS are never generated.
            logits[COMMAND_TO_ID["PAD"]] = -torch.inf
            logits[COMMAND_TO_ID["BOS"]] = -torch.inf
            if step == 0:
                next_command = COMMAND_TO_ID["M"]
            else:
                if command_tokens[-1] == COMMAND_TO_ID["Z"]:
                    allowed = torch.full_like(logits, -torch.inf)
                    allowed[COMMAND_TO_ID["M"]] = logits[COMMAND_TO_ID["M"]]
                    allowed[COMMAND_TO_ID["EOS"]] = logits[COMMAND_TO_ID["EOS"]]
                    logits = allowed
                else:
                    logits[COMMAND_TO_ID["M"]] = -torch.inf
                    logits[COMMAND_TO_ID["EOS"]] = -torch.inf
                    last_move = max(index for index, token in enumerate(command_tokens) if token == COMMAND_TO_ID["M"])
                    if len(command_tokens) - last_move - 1 < 2:
                        logits[COMMAND_TO_ID["Z"]] = -torch.inf
                next_command = int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).item())
            next_coordinates = output["coordinates"][0, -1].tolist()
            if next_command == COMMAND_TO_ID["EOS"]:
                break
            command_tokens.append(next_command)
            coordinate_tokens.append(next_coordinates)
        if command_tokens[-1] != COMMAND_TO_ID["Z"]:
            if len(command_tokens) >= self.config.max_commands:
                command_tokens[-1] = COMMAND_TO_ID["Z"]
                coordinate_tokens[-1] = [0.0] * 6
            else:
                command_tokens.append(COMMAND_TO_ID["Z"])
                coordinate_tokens.append([0.0] * 6)
        final = self.model(
            prompts,
            glyph_ids,
            control_tensor,
            torch.tensor([command_tokens], device=self.device),
            torch.tensor([coordinate_tokens], dtype=torch.float32, device=self.device),
        )
        metrics = final["metrics"][0].tolist()
        return GeneratedGlyph(
            character=character,
            commands=[COMMANDS[token] for token in command_tokens[1:]],
            coordinates=coordinate_tokens[1:],
            advance_width=max(0.2, min(2.0, float(metrics[0]))),
            left_side_bearing=max(-0.5, min(0.8, float(metrics[1]))),
        )

    def generate_family(
        self,
        prompt: str,
        characters: str,
        controls: list[float],
        seed: int,
    ) -> list[GeneratedGlyph]:
        # One seeded sampler plus one prompt-derived style latent makes repeats
        # deterministic and keeps the whole family in the same design space.
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return [self.generate_glyph(prompt, character, controls, generator=generator) for character in dict.fromkeys(characters)]
