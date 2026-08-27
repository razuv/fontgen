from __future__ import annotations

import torch

PAD_BYTE = 0
BOS_BYTE = 1
BYTE_OFFSET = 2
BYTE_VOCABULARY = 258
SUPPORTED_CHARACTERS = (
    " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "0123456789.,:;!?—-()«»@&%+="
)
CHARACTER_TO_ID = {character: index + 1 for index, character in enumerate(dict.fromkeys(SUPPORTED_CHARACTERS))}

PROMPT_AXES = (
    (("тонк", "лёгк", "light", "thin"), 0, -0.75, "лёгкий"),
    (("жирн", "плотн", "bold", "heavy", "black"), 0, 0.8, "плотный"),
    (("узк", "condensed", "narrow"), 1, -0.75, "узкий"),
    (("широк", "wide", "extended"), 1, 0.75, "широкий"),
    (("высокий контраст", "высоким контраст", "сильный контраст", "high contrast", "didone"), 2, 0.85, "сильный контраст штрихов"),
    (("низкий контраст", "ровный штрих", "low contrast", "monoline"), 2, -0.75, "ровный штрих"),
    (("скруг", "округл", "мягк", "rounded", "soft curves"), 3, 0.8, "мягкая пластика кривых"),
    (("углов", "геометрич", "остр", "angular", "sharp"), 3, -0.65, "угловатая пластика"),
    (("курсив", "наклон", "italic", "oblique"), 4, 0.8, "курсивный"),
)

PROMPT_CATEGORIES = (
    (("антикв", "засеч", "serif"), "антиква с засечками"),
    (("гротеск", "рублен", "sans"), "современный рубленый шрифт"),
    (("моношир", "моносп", "monospace", "fixed-pitch"), "технический моноспейс"),
    (("рукопис", "леттеринг", "handwrit", "script"), "пластичная рукописная гарнитура"),
    (("акцидент", "заголов", "display", "poster"), "выразительный акцидентный шрифт"),
)


def encode_prompt(prompt: str, max_length: int) -> torch.Tensor:
    payload = list(prompt.strip().encode("utf-8"))[: max_length - 1]
    values = [BOS_BYTE] + [value + BYTE_OFFSET for value in payload]
    values.extend([PAD_BYTE] * (max_length - len(values)))
    return torch.tensor(values, dtype=torch.long)


def condition_v41_prompt(prompt: str, controls: list[float]) -> tuple[str, list[float]]:
    """Map free wording to phrases and axes seen during V4.1 training."""
    folded = prompt.casefold().replace("ё", "е")
    conditioned = [float(value) for value in controls]
    descriptors: list[str] = []
    for terms, axis, target, descriptor in PROMPT_AXES:
        if any(term.replace("ё", "е") in folded for term in terms):
            conditioned[axis] = max(-1.0, min(1.0, conditioned[axis] + target * 0.35))
            descriptors.append(descriptor)
    for terms, descriptor in PROMPT_CATEGORIES:
        if any(term in folded for term in terms):
            descriptors.append(descriptor)
            break
    enriched = prompt.strip()
    if descriptors:
        enriched = f"{', '.join(dict.fromkeys(descriptors))}. {enriched}"
    return enriched, conditioned


def glyph_bucket(character: str, buckets: int) -> int:
    value = CHARACTER_TO_ID.get(character[0] if character else "")
    if value is not None:
        return value
    reserved = len(CHARACTER_TO_ID) + 1
    if buckets <= reserved:
        raise ValueError(f"glyph_buckets must be greater than {reserved}")
    value = ord(character[0]) if character else 0
    value ^= value >> 16
    value *= 0x7FEB352D
    value ^= value >> 15
    return reserved + value % (buckets - reserved)
