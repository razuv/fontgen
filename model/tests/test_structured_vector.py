import torch

from fontgen_model.quadratic import QUADRATIC_COMMAND_TO_ID, canonicalize_quadratic
from fontgen_model.structured_config import StructuredConfig
from fontgen_model.structured_inference import StructuredFontGenerator
from fontgen_model.structured_loss import structured_training_loss
from fontgen_model.structured_network import StructuredVectorFontNet


def tiny_config() -> StructuredConfig:
    return StructuredConfig(
        max_commands=16, d_model=64, heads=4, structure_layers=1,
        geometry_layers=1, refiner_layers=1, feedforward=128, family_dimensions=32,
    )


def test_cubic_is_canonicalized_to_quadratic_segments() -> None:
    row = {
        "commands": [1, 3, 6, 7, 2],
        "coordinates": [
            [0.0] * 6, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.5, 0.4, 0.7, 0.6, 0.0], [0.0] * 6, [0.0] * 6,
        ],
    }
    converted = canonicalize_quadratic(row)
    assert converted is not None
    assert converted["representation"] == "quadratic-v1"
    assert all(command != 7 for command in converted["commands"])
    assert QUADRATIC_COMMAND_TO_ID["Q"] in converted["commands"]
    assert all(len(coordinates) == 4 for coordinates in converted["coordinates"])


def test_structured_model_separates_topology_and_geometry() -> None:
    config = tiny_config()
    model = StructuredVectorFontNet(config)
    commands = torch.zeros((2, config.max_commands), dtype=torch.long)
    commands[:, :7] = torch.tensor([
        QUADRATIC_COMMAND_TO_ID["BOS"], QUADRATIC_COMMAND_TO_ID["M"],
        QUADRATIC_COMMAND_TO_ID["L"], QUADRATIC_COMMAND_TO_ID["Q"],
        QUADRATIC_COMMAND_TO_ID["L"], QUADRATIC_COMMAND_TO_ID["Z"],
        QUADRATIC_COMMAND_TO_ID["EOS"],
    ])
    coordinates = torch.rand((2, config.max_commands, 4)) * 0.6
    batch = {
        "commands": commands, "coordinates": coordinates,
        "metrics": torch.rand((2, 2)), "category_id": torch.tensor([0, 1]),
        "controls": torch.zeros((2, config.control_dimensions)),
        "typography": torch.rand((2, config.typography_dimensions)),
    }
    output = model(
        torch.rand((2, config.prompt_dimensions)), torch.tensor([65, 66]),
        torch.zeros((2, config.control_dimensions)), commands,
    )
    losses = structured_training_loss(batch, output)
    assert output["family"].shape == (2, config.family_dimensions)
    assert output["structure_logits"].shape == (2, config.max_commands, config.command_count)
    assert output["coordinates"].shape == (2, config.max_commands, 4)
    assert torch.isfinite(losses["total"])


def test_structured_model_has_capacity_without_raster_decoder() -> None:
    model = StructuredVectorFontNet(StructuredConfig())
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 20_000_000 < count < 50_000_000
    assert not any("raster" in name for name, _ in model.named_parameters())


def test_api_selects_structured_checkpoint(tmp_path) -> None:
    from fontgen_model.api import load_generator

    config = tiny_config()
    checkpoint = tmp_path / "structured.pt"
    torch.save({
        "architecture": "structured-quadratic-v1",
        "config": config.__dict__,
        "model": StructuredVectorFontNet(config).state_dict(),
    }, checkpoint)
    generator = load_generator(checkpoint)
    assert generator.architecture == "structured-quadratic-v1"


def test_structure_grammar_forces_move_and_closed_contours() -> None:
    logits = torch.zeros(7)
    first = StructuredFontGenerator._legal_logits(logits, QUADRATIC_COMMAND_TO_ID["BOS"], 0)
    assert torch.isfinite(first).nonzero().flatten().tolist() == [QUADRATIC_COMMAND_TO_ID["M"]]
    after_close = StructuredFontGenerator._legal_logits(logits, QUADRATIC_COMMAND_TO_ID["Z"], 0)
    assert torch.isfinite(after_close).nonzero().flatten().tolist() == [
        QUADRATIC_COMMAND_TO_ID["EOS"], QUADRATIC_COMMAND_TO_ID["M"],
    ]
