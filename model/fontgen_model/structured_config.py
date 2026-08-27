from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StructuredConfig:
    prompt_dimensions: int = 384
    max_commands: int = 192
    command_count: int = 7
    coordinate_count: int = 4
    glyph_buckets: int = 512
    control_dimensions: int = 5
    category_count: int = 5
    typography_dimensions: int = 6
    legacy_style_dimensions: int = 112
    family_dimensions: int = 256
    d_model: int = 384
    heads: int = 8
    structure_layers: int = 4
    geometry_layers: int = 4
    refiner_layers: int = 4
    feedforward: int = 1536
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
