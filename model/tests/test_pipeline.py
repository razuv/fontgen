from pathlib import Path

import torch
from fontTools.ttLib import TTFont

from fontgen_model.config import ModelConfig
from fontgen_model.network import FontgenNet
from fontgen_model.outline import COMMAND_TO_ID, extract_glyph
from fontgen_model.training import training_loss


def tiny_config() -> ModelConfig:
    return ModelConfig(
        max_prompt_bytes=32,
        max_commands=24,
        d_model=64,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feedforward=128,
        style_dimensions=32,
    )


def test_outline_has_separate_contour_and_glyph_end_tokens() -> None:
    font_path = Path(__file__).parents[2] / "public" / "fonts" / "inter.ttf"
    glyph = extract_glyph(TTFont(font_path), "B", 192)
    assert glyph is not None
    assert glyph.commands[0] == COMMAND_TO_ID["BOS"]
    assert glyph.commands[-1] == COMMAND_TO_ID["EOS"]
    assert glyph.commands.count(COMMAND_TO_ID["Z"]) >= 2


def test_model_forward_and_loss_are_finite() -> None:
    config = tiny_config()
    model = FontgenNet(config)
    batch_size = 2
    commands = torch.zeros((batch_size, config.max_commands), dtype=torch.long)
    commands[:, 0] = COMMAND_TO_ID["BOS"]
    commands[:, 1] = COMMAND_TO_ID["M"]
    commands[:, 2] = COMMAND_TO_ID["L"]
    commands[:, 3] = COMMAND_TO_ID["L"]
    commands[:, 4] = COMMAND_TO_ID["Z"]
    commands[:, 5] = COMMAND_TO_ID["EOS"]
    coordinates = torch.rand((batch_size, config.max_commands, 6)) * 0.5
    batch = {
        "prompt": torch.randint(1, 255, (batch_size, config.max_prompt_bytes)),
        "glyph_id": torch.tensor([65, 1040]),
        "controls": torch.zeros((batch_size, config.control_dimensions)),
        "commands": commands,
        "coordinates": coordinates,
        "metrics": torch.tensor([[0.6, 0.05], [0.65, 0.04]]),
    }
    output = model(batch["prompt"], batch["glyph_id"], batch["controls"], commands, coordinates)
    losses = training_loss(batch, output)
    assert output["commands"].shape == (batch_size, config.max_commands, config.command_count)
    assert torch.isfinite(losses["total"])

