from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    max_prompt_bytes: int = 256
    max_commands: int = 192
    command_count: int = 8
    coordinate_count: int = 6
    glyph_buckets: int = 4096
    d_model: int = 384
    heads: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 8
    feedforward: int = 1536
    dropout: float = 0.1
    style_dimensions: int = 192
    control_dimensions: int = 5

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
