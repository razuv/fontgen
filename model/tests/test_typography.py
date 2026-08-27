from fontgen_model.outline import COMMAND_TO_ID
from fontgen_model.typography import analyze_face, make_tags, prompt_variants


def sample_row(character: str, height: float, controls: list[float]) -> dict[str, object]:
    return {
        "character": character,
        "commands": [
            COMMAND_TO_ID["BOS"], COMMAND_TO_ID["M"], COMMAND_TO_ID["L"],
            COMMAND_TO_ID["Q"], COMMAND_TO_ID["L"], COMMAND_TO_ID["Z"], COMMAND_TO_ID["EOS"],
        ],
        "coordinates": [
            [0.0] * 6, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6, height / 2, 0.5, height, 0.0, 0.0],
            [0.0, height, 0.0, 0.0, 0.0, 0.0], [0.0] * 6, [0.0] * 6,
        ],
        "controls": controls,
        "advance_width": 0.6,
    }


def test_geometry_tags_and_prompts_are_deterministic() -> None:
    rows = [
        sample_row("H", 0.7, [0.6, -0.4, 0.7, 0.8, 1.0]),
        sample_row("x", 0.56, [0.6, -0.4, 0.7, 0.8, 1.0]),
    ]
    features = analyze_face(rows)
    tags = make_tags("SERIF", features)
    prompts = prompt_variants("SERIF", tags, "Example|Bold Italic")
    assert "weight:bold" in tags
    assert "width:condensed" in tags
    assert "xheight:high" in tags
    assert "contrast:high" in tags
    assert "roundness:rounded" in tags
    assert "slant:italic" in tags
    assert "subclass:didone" in tags
    assert len(prompts) == 20
    assert any("антиква" in prompt or "serif" in prompt for prompt in prompts)
    assert prompts == prompt_variants("SERIF", tags, "Example|Bold Italic")
