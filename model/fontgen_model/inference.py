from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import ModelConfig
from .network import FontgenNet
from .outline import COMMAND_TO_ID, COMMANDS
from .text import condition_v41_prompt, encode_prompt, glyph_bucket
from .vectorize import topology_safe_field, vectorize_mask

POINTS_PER_COMMAND = {"M": 1, "L": 1, "Q": 2, "C": 3, "Z": 0}


@dataclass
class GeneratedGlyph:
    character: str
    commands: list[str]
    coordinates: list[list[float]]
    advance_width: float
    left_side_bearing: float


class FontgenGenerator:
    architecture = "fontgen-style-film-vector-v4.1-local-refined"

    def __init__(self, checkpoint_path: Path | str, device: str | None = None):
        checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.config = ModelConfig(**checkpoint["config"])
        self.model = FontgenNet(self.config).to(self.device)
        self.model.load_compatible_state_dict(checkpoint["model"])
        self.model.eval()
        self.checkpoint_id = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:12]
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

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
        *,
        cfg_scale: float = 1.0,
    ) -> list[GeneratedGlyph]:
        del seed  # v1 raster generator is deterministic for a prompt and control vector.
        prompt, controls = condition_v41_prompt(prompt, controls)
        unique_characters = list(dict.fromkeys(characters))
        drawable = [character for character in unique_characters if not character.isspace()]
        if not drawable:
            return [GeneratedGlyph(character, [], [], 0.33, 0.0) for character in unique_characters]
        batch_size = len(drawable)
        prompts = encode_prompt(prompt, self.config.max_prompt_bytes).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        glyph_ids = torch.tensor(
            [glyph_bucket(character, self.config.glyph_buckets) for character in drawable],
            device=self.device,
        )
        control_tensor = torch.tensor([controls], dtype=torch.float32, device=self.device).repeat(batch_size, 1)
        conditioned = self.model.condition(prompts, glyph_ids, control_tensor)

        if cfg_scale > 1.0:
            null_prompts = torch.zeros_like(prompts)
            uncond = self.model.condition(null_prompts, glyph_ids, control_tensor)
            raster_logits = uncond["raster"] + cfg_scale * (conditioned["raster"] - uncond["raster"])
            sdf_logits = uncond["sdf_logits"] + cfg_scale * (conditioned["sdf_logits"] - uncond["sdf_logits"])
            base_fields = torch.sigmoid(raster_logits).detach().cpu().numpy()[:, 0]
            refined_fields = ((torch.tanh(sdf_logits) + 1) * 0.5).detach().cpu().numpy()[:, 0]
        else:
            base_fields = torch.sigmoid(conditioned["raster"]).detach().cpu().numpy()[:, 0]
            refined_fields = ((conditioned["sdf"] + 1) * 0.5).detach().cpu().numpy()[:, 0]

        metrics_batch = conditioned["metrics"].detach().cpu().numpy()
        generated: dict[str, GeneratedGlyph] = {}
        for index, character in enumerate(drawable):
            field = topology_safe_field(base_fields[index], refined_fields[index])
            outline = vectorize_mask(field, threshold=0.5, contrast=controls[2], roundness=controls[3])
            if controls[4]:
                shear = math.tan(math.radians(float(controls[4]) * 18.0))
                for coordinates in outline.coordinates:
                    for point_index in range(0, 6, 2):
                        coordinates[point_index] += coordinates[point_index + 1] * shear
            metrics = metrics_batch[index]
            x_coordinates = [
                coordinates[point_index]
                for command, coordinates in zip(outline.commands, outline.coordinates, strict=True)
                for point_index in range(0, POINTS_PER_COMMAND[command] * 2, 2)
            ]
            x_min = min(x_coordinates, default=0.0)
            x_max = max(x_coordinates, default=0.55)
            geometry_advance = max(x_max, x_max - min(0.0, x_min)) + 0.08
            generated[character] = GeneratedGlyph(
                character=character,
                commands=outline.commands,
                coordinates=outline.coordinates,
                advance_width=max(0.28, min(2.0, max(float(metrics[0]), geometry_advance))),
                left_side_bearing=max(-0.3, min(0.5, x_min)),
            )
        return [
            generated.get(character, GeneratedGlyph(character, [], [], 0.33, 0.0))
            for character in unique_characters
        ]
