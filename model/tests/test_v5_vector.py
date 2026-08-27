import torch

from fontgen_model.outline import COMMAND_TO_ID
from fontgen_model.v5_config import V5Config
from fontgen_model.v5_loss import vector_training_loss
from fontgen_model.v5_network import VectorFontNet


def tiny_config() -> V5Config:
    return V5Config(
        max_commands=16, d_model=64, heads=4, decoder_layers=2,
        refiner_layers=1, feedforward=128,
    )


def test_v5_is_vector_native_and_loss_is_finite() -> None:
    config = tiny_config()
    model = VectorFontNet(config)
    commands = torch.zeros((2, config.max_commands), dtype=torch.long)
    commands[:, :7] = torch.tensor([
        COMMAND_TO_ID["BOS"], COMMAND_TO_ID["M"], COMMAND_TO_ID["L"],
        COMMAND_TO_ID["Q"], COMMAND_TO_ID["C"], COMMAND_TO_ID["Z"], COMMAND_TO_ID["EOS"],
    ])
    coordinates = torch.rand((2, config.max_commands, 6)) * 0.7
    batch = {
        "commands": commands,
        "coordinates": coordinates,
        "metrics": torch.tensor([[0.6, 0.04], [0.55, 0.03]]),
        "category_id": torch.tensor([0, 1]),
        "typography": torch.rand((2, config.typography_dimensions)),
        "typography_mask": torch.ones(2),
    }
    output = model(
        torch.rand((2, 384)), torch.tensor([65, 66]), torch.zeros((2, 5)), commands, coordinates,
    )
    losses = vector_training_loss(batch, output)
    assert output["commands"].shape == (2, config.max_commands, config.command_count)
    assert output["coordinates"].shape == (2, config.max_commands, 6)
    assert "raster" not in output
    assert torch.isfinite(losses["total"])


def test_v5_decoder_is_large_enough_for_the_quality_step() -> None:
    parameter_count = sum(parameter.numel() for parameter in VectorFontNet(V5Config()).parameters())
    assert 20_000_000 < parameter_count < 30_000_000
