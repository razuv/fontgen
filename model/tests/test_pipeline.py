from pathlib import Path

import numpy as np
import torch
from fontTools.ttLib import TTFont

from fontgen_model.api import GenerationRequest
from fontgen_model.config import ModelConfig
from fontgen_model.network import FontgenNet
from fontgen_model.outline import COMMAND_TO_ID, extract_glyph
from fontgen_model.raster import glyph_mask, outline_mask, signed_distance_field
from fontgen_model.text import SUPPORTED_CHARACTERS, condition_v41_prompt, glyph_bucket
from fontgen_model.training import training_loss
from fontgen_model.vectorize import topology_safe_field, vectorize_mask


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
        "glyph_id": torch.tensor([glyph_bucket("A", config.glyph_buckets), glyph_bucket("А", config.glyph_buckets)]),
        "category_id": torch.tensor([0, 1]),
        "controls": torch.zeros((batch_size, config.control_dimensions)),
        "commands": commands,
        "coordinates": coordinates,
        "metrics": torch.tensor([[0.6, 0.05], [0.65, 0.04]]),
        "raster": torch.rand((batch_size, 1, config.raster_size, config.raster_size)),
        "sdf": torch.rand((batch_size, 1, config.raster_size, config.raster_size)) * 2 - 1,
    }
    output = model(batch["prompt"], batch["glyph_id"], batch["controls"], commands, coordinates)
    losses = training_loss(batch, output)
    assert output["commands"].shape == (batch_size, config.max_commands, config.command_count)
    assert output["raster"].shape == (batch_size, 1, config.raster_size, config.raster_size)
    assert torch.isfinite(losses["total"])


def test_prompt_hash_seed_accepts_full_uint32_range() -> None:
    request = GenerationRequest(prompt="геометрический гротеск", seed=2**32 - 1)
    assert request.seed == 4_294_967_295


def test_compact_model_stays_below_five_million_parameters() -> None:
    compact = ModelConfig(
        max_prompt_bytes=192, max_commands=128, d_model=176, heads=8,
        encoder_layers=3, decoder_layers=4, feedforward=704, style_dimensions=112,
    )
    count = sum(parameter.numel() for parameter in FontgenNet(compact).parameters())
    assert 4_000_000 < count <= 5_000_000


def test_scaled_model_stays_below_fifteen_million_parameters() -> None:
    count = sum(parameter.numel() for parameter in FontgenNet(ModelConfig()).parameters())
    assert 8_000_000 < count <= 15_000_000


def test_supported_latin_and_cyrillic_characters_have_unique_ids() -> None:
    ids = [glyph_bucket(character, ModelConfig().glyph_buckets) for character in dict.fromkeys(SUPPORTED_CHARACTERS)]
    assert len(ids) == len(set(ids))


def test_signed_distance_field_has_stable_inside_and_outside_signs() -> None:
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 10:22] = 1
    sdf = signed_distance_field(mask, maximum_distance=8)
    assert sdf[16, 16] > 0
    assert sdf[0, 0] < 0
    assert sdf.max() <= 1 and sdf.min() >= -1


def test_v41_checkpoint_without_sdf_refiner_loads_compatibly() -> None:
    model = FontgenNet(tiny_config())
    old_state = {
        name: value for name, value in model.state_dict().items()
        if not name.startswith("sdf_coordinate_refiner.")
    }
    restored = FontgenNet(tiny_config())
    restored.load_compatible_state_dict(old_state)
    assert all(parameter.isfinite().all() for parameter in restored.parameters())


def test_raster_tracing_produces_closed_cubic_contours() -> None:
    font_path = Path(__file__).parents[2] / "public" / "fonts" / "inter.ttf"
    mask = glyph_mask(TTFont(font_path), "B")
    assert mask is not None
    outline = vectorize_mask(np.asarray(mask, dtype=np.float32) / 255.0)
    assert outline.commands[0] == "M"
    assert outline.commands[-1] == "Z"
    assert outline.commands.count("Z") >= 2
    assert "L" in outline.commands
    assert "C" in outline.commands


def test_v41_prompt_conditioning_recognizes_free_typographic_language() -> None:
    prompt, controls = condition_v41_prompt(
        "Узкая антиква с высоким контрастом и мягкими кривыми", [0.0] * 5,
    )
    assert "антиква с засечками" in prompt
    assert controls[1] < 0
    assert controls[2] > 0
    assert controls[3] > 0


def test_topology_guard_rejects_a_broken_counter() -> None:
    base = np.zeros((32, 32), dtype=np.float32)
    base[5:27, 5:27] = 1
    base[11:21, 11:21] = 0
    broken = base.copy()
    broken[5:16, 15:17] = 0
    assert topology_safe_field(base, broken) is base


def test_topology_guard_accepts_a_small_boundary_correction() -> None:
    base = np.zeros((32, 32), dtype=np.float32)
    base[6:26, 6:26] = 1
    refined = base.copy()
    refined[6, 6] = 0
    assert topology_safe_field(base, refined) is refined


def test_vector_manifest_outline_raster_matches_source_glyph() -> None:
    font_path = Path(__file__).parents[2] / "public" / "fonts" / "inter.ttf"
    font = TTFont(font_path)
    glyph = extract_glyph(font, "B", 192)
    source = glyph_mask(font, "B", size=128)
    assert glyph is not None and source is not None
    rebuilt = outline_mask(glyph.commands, glyph.coordinates, size=128)
    source_mask = np.asarray(source, dtype=np.float32) / 255.0
    rebuilt_mask = np.asarray(rebuilt, dtype=np.float32) / 255.0
    intersection = np.minimum(source_mask, rebuilt_mask).sum()
    union = np.maximum(source_mask, rebuilt_mask).sum()
    assert intersection / union > 0.95
