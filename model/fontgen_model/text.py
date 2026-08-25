from __future__ import annotations

import torch

PAD_BYTE = 0
BOS_BYTE = 1
BYTE_OFFSET = 2
BYTE_VOCABULARY = 258


def encode_prompt(prompt: str, max_length: int) -> torch.Tensor:
    payload = list(prompt.strip().encode("utf-8"))[: max_length - 1]
    values = [BOS_BYTE] + [value + BYTE_OFFSET for value in payload]
    values.extend([PAD_BYTE] * (max_length - len(values)))
    return torch.tensor(values, dtype=torch.long)


def glyph_bucket(character: str, buckets: int) -> int:
    value = ord(character[0]) if character else 0
    value ^= value >> 16
    value *= 0x7FEB352D
    value ^= value >> 15
    return value % buckets

