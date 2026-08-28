from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    max_prompt_bytes: int = 192
    max_commands: int = 128
    command_count: int = 8
    coordinate_count: int = 6
    glyph_buckets: int = 256
    d_model: int = 256
    heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 6
    feedforward: int = 1024
    dropout: float = 0.1
    style_dimensions: int = 128
    control_dimensions: int = 5
    raster_size: int = 128
    sdf_logit_scale: float = 4.0
    category_count: int = 5
    cfg_dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
