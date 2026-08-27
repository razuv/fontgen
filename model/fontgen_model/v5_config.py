from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class V5Config:
    """Configuration for the vector-native, unrestricted-size model."""

    prompt_dimensions: int = 384
    max_commands: int = 128
    command_count: int = 8
    coordinate_count: int = 6
    glyph_buckets: int = 512
    control_dimensions: int = 5
    category_count: int = 5
    typography_dimensions: int = 6
    d_model: int = 384
    heads: int = 8
    decoder_layers: int = 6
    refiner_layers: int = 4
    feedforward: int = 1536
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
