from __future__ import annotations

import hashlib
import statistics
from collections import Counter
from typing import Any

from .outline import COMMANDS, COORDINATE_MASKS


def _median(values: list[float], fallback: float = 0.0) -> float:
    return float(statistics.median(values)) if values else fallback


def _points(row: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for command_id, coordinates in zip(row["commands"], row["coordinates"], strict=True):
        count = COORDINATE_MASKS[COMMANDS[int(command_id)]]
        points.extend((float(coordinates[index]), float(coordinates[index + 1])) for index in range(0, count, 2))
    return points


def analyze_face(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_character = {str(row["character"]): row for row in rows}
    widths: list[float] = []
    heights: dict[str, float] = {}
    curve_commands = 0
    drawing_commands = 0
    contours: list[int] = []
    complexities: list[int] = []
    for row in rows:
        points = _points(row)
        if points:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            heights[str(row["character"])] = max(ys) - min(ys)
            if str(row["character"]) in "HOnoxaeАОНохае":
                widths.append(max(xs) - min(xs))
        names = [COMMANDS[int(command)] for command in row["commands"]]
        curve_commands += sum(name in {"Q", "C"} for name in names)
        drawing_commands += sum(name in {"L", "Q", "C"} for name in names)
        contours.append(names.count("Z"))
        complexities.append(sum(name not in {"PAD", "BOS", "EOS"} for name in names))

    cap_height = _median([heights[c] for c in "HOEIАОНП" if c in heights], 0.7)
    x_height = _median([heights[c] for c in "xoeaохеа" if c in heights], cap_height * 0.7)
    descenders = []
    for character in "pqgyфруцд":
        row = by_character.get(character)
        if row:
            points = _points(row)
            if points:
                descenders.append(min(point[1] for point in points))
    controls = [float(value) for value in rows[0].get("controls", [0, 0, 0, 0, 0])]
    return {
        "mean_glyph_width": round(_median(widths, float(rows[0].get("advance_width", 0.6))), 4),
        "x_height": round(x_height, 4),
        "cap_height": round(cap_height, 4),
        "x_height_ratio": round(x_height / max(cap_height, 0.01), 4),
        "descender_depth": round(abs(min(descenders, default=0.0)), 4),
        "curve_ratio": round(curve_commands / max(drawing_commands, 1), 4),
        "mean_complexity": round(_median([float(value) for value in complexities]), 2),
        "mean_contours": round(_median([float(value) for value in contours]), 2),
        "weight_axis": round(controls[0], 4),
        "width_axis": round(controls[1], 4),
        "contrast_axis": round(controls[2], 4),
        "roundness_axis": round(controls[3], 4),
        "slant_axis": round(controls[4], 4),
    }


def make_tags(category: str, features: dict[str, float]) -> list[str]:
    weight = features["weight_axis"]
    width = features["width_axis"]
    contrast = features["contrast_axis"]
    roundness = features["roundness_axis"]
    x_ratio = features["x_height_ratio"]
    curves = features["curve_ratio"]
    complexity = features["mean_complexity"]
    tags = [f"category:{category.casefold()}"]
    tags.append("weight:light" if weight < -0.25 else "weight:bold" if weight > 0.3 else "weight:regular")
    tags.append("width:condensed" if width < -0.2 else "width:wide" if width > 0.2 else "width:normal")
    tags.append("xheight:low" if x_ratio < 0.66 else "xheight:high" if x_ratio > 0.76 else "xheight:medium")
    tags.append("contrast:low" if contrast < -0.2 else "contrast:high" if contrast > 0.2 else "contrast:moderate")
    tags.append("roundness:rounded" if roundness > 0.25 else "roundness:angular" if roundness < -0.25 else "roundness:neutral")
    tags.append("curves:soft" if curves > 0.58 else "curves:angular" if curves < 0.28 else "curves:balanced")
    tags.append("complexity:minimal" if complexity < 18 else "complexity:ornate" if complexity > 34 else "complexity:moderate")
    tags.append("slant:italic" if features["slant_axis"] > 0.5 else "slant:upright")
    tags.append(_subclass_tag(category, features))
    return tags


def _subclass_tag(category: str, features: dict[str, float]) -> str:
    if category == "SERIF":
        if features["contrast_axis"] > 0.45:
            return "subclass:didone"
        if features["contrast_axis"] < -0.2 or features["weight_axis"] > 0.45:
            return "subclass:slab-serif"
        if features["x_height_ratio"] < 0.68:
            return "subclass:old-style-serif"
        return "subclass:transitional-serif"
    if category == "SANS_SERIF":
        if features["roundness_axis"] > 0.25:
            return "subclass:rounded-sans"
        if features["curve_ratio"] > 0.58 and features["contrast_axis"] <= 0.2:
            return "subclass:geometric-sans"
        if features["contrast_axis"] > 0.2:
            return "subclass:humanist-sans"
        return "subclass:neo-grotesque"
    return f"subclass:{category.casefold()}"


RU = {
    "SANS_SERIF": ("гротеск без засечек", "современный рубленый шрифт", "функциональный sans-serif"),
    "SERIF": ("антиква с засечками", "книжный serif", "редакционная антиква"),
    "DISPLAY": ("акцидентный шрифт", "выразительный display", "характерный заголовочный шрифт"),
    "HANDWRITING": ("рукописный шрифт", "живой леттеринг", "пластичная рукописная гарнитура"),
    "MONOSPACE": ("моноширинный шрифт", "технический моноспейс", "гарнитура с равным шагом"),
}
EN = {
    "SANS_SERIF": ("sans-serif typeface", "modern grotesque", "functional sans"),
    "SERIF": ("serif typeface", "editorial serif", "book typeface with serifs"),
    "DISPLAY": ("display typeface", "expressive headline face", "distinctive poster font"),
    "HANDWRITING": ("handwritten typeface", "natural lettering face", "fluid script font"),
    "MONOSPACE": ("monospaced typeface", "technical monospace", "fixed-pitch font"),
}


def prompt_variants(category: str, tags: list[str], identity: str) -> list[str]:
    values = {tag.split(":", 1)[0]: tag.split(":", 1)[1] for tag in tags}
    seed = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16)
    ru_category = RU.get(category, RU["DISPLAY"])
    en_category = EN.get(category, EN["DISPLAY"])
    ru = {
        "weight": {"light": "лёгкий", "regular": "средней насыщенности", "bold": "плотный"}[values["weight"]],
        "width": {"condensed": "узкий", "normal": "нормальной ширины", "wide": "широкий"}[values["width"]],
        "xheight": {"low": "с низкой строчной", "medium": "со средней строчной", "high": "с высокой строчной"}[values["xheight"]],
        "contrast": {"low": "с ровным штрихом", "moderate": "с умеренным контрастом", "high": "с сильным контрастом штрихов"}[values["contrast"]],
        "curves": {"soft": "с мягкой пластикой кривых", "balanced": "со сбалансированными кривыми", "angular": "с угловатой пластикой"}[values["curves"]],
        "complexity": {"minimal": "лаконичный", "moderate": "сдержанный", "ornate": "детализированный"}[values["complexity"]],
        "slant": "курсивный" if values["slant"] == "italic" else "прямой",
    }
    en = {
        "weight": {"light": "light", "regular": "regular-weight", "bold": "bold"}[values["weight"]],
        "width": {"condensed": "condensed", "normal": "normal-width", "wide": "wide"}[values["width"]],
        "xheight": {"low": "low x-height", "medium": "medium x-height", "high": "high x-height"}[values["xheight"]],
        "contrast": {"low": "low stroke contrast", "moderate": "moderate stroke contrast", "high": "high stroke contrast"}[values["contrast"]],
        "curves": {"soft": "soft curves", "balanced": "balanced curves", "angular": "angular construction"}[values["curves"]],
        "complexity": {"minimal": "minimal", "moderate": "restrained", "ornate": "detailed"}[values["complexity"]],
        "slant": "italic" if values["slant"] == "italic" else "upright",
    }
    uses_ru = ("для интерфейса", "для длинного чтения", "для айдентики", "для навигации", "для заголовков", "для экранов и печати")
    uses_en = ("for interfaces", "for long-form reading", "for identity design", "for wayfinding", "for headlines", "for screen and print")
    prompts: list[str] = []
    ru_templates = (
        "{category}: {descriptors}, {use}",
        "создай {descriptors} {category}, {use}",
        "нужен {category} — {descriptors}; {use}",
        "{use}: {descriptors} {category}",
    )
    for index in range(12):
        category_phrase = ru_category[(seed + index) % len(ru_category)]
        use = uses_ru[(seed // 7 + index) % len(uses_ru)]
        orders = (
            ("weight", "width", "xheight", "contrast", "curves"),
            ("complexity", "slant", "width", "contrast", "xheight"),
            ("slant", "weight", "curves", "xheight", "complexity"),
        )
        descriptors = ", ".join(ru[key] for key in orders[index % len(orders)])
        prompts.append(ru_templates[index % len(ru_templates)].format(
            category=category_phrase, descriptors=descriptors, use=use,
        ))
    en_templates = (
        "{descriptors} {category} {use}",
        "design a {category}: {descriptors}, {use}",
        "create a {descriptors} {category} {use}",
        "{category} {use}, with {descriptors}",
    )
    for index in range(8):
        category_phrase = en_category[(seed + index) % len(en_category)]
        use = uses_en[(seed // 11 + index) % len(uses_en)]
        order = ("weight", "width", "xheight", "contrast", "curves", "slant")
        descriptors = ", ".join(en[key] for key in order[index % 3 :] + order[: index % 3])
        prompts.append(en_templates[index % len(en_templates)].format(
            category=category_phrase, descriptors=descriptors, use=use,
        ))
    return list(dict.fromkeys(prompts))


def tag_histogram(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(tag for row in rows for tag in row.get("auto_tags", []))
