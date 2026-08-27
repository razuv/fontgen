from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from fontTools.ttLib import TTFont

from fontgen_model.config import ModelConfig
from fontgen_model.outline import extract_glyph, inspect_font
from fontgen_model.raster import encode_mask, glyph_mask

DEFAULT_CHARSET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "0123456789.,:;!?—-()[]«»@&%+="
)


CATEGORY_RU = {
    "SANS_SERIF": "современный гротеск без засечек",
    "SERIF": "книжная антиква с выразительными засечками",
    "DISPLAY": "характерный акцидентный шрифт для крупных заголовков",
    "HANDWRITING": "живой рукописный шрифт с естественным ритмом",
    "MONOSPACE": "моноширинный технический шрифт",
}
CATEGORY_EN = {
    "SANS_SERIF": "modern sans serif",
    "SERIF": "book serif typeface",
    "DISPLAY": "expressive display typeface for large headlines",
    "HANDWRITING": "natural handwritten typeface",
    "MONOSPACE": "technical monospaced typeface",
}
CATEGORY_RU_VARIANTS = {
    "SANS_SERIF": [
        "современный гротеск без засечек", "нейтральный рубленый шрифт",
        "чистый sans-serif для интерфейса", "функциональный шрифт без засечек",
    ],
    "SERIF": [
        "книжная антиква с засечками", "редакционный шрифт для длинного чтения",
        "выразительная антиква", "литературный serif с ясным ритмом",
    ],
    "DISPLAY": [
        "акцидентный заголовочный шрифт", "брутальный плакатный шрифт для афиш",
        "выразительный display для крупных заголовков", "экспериментальный шрифт для постеров",
    ],
    "HANDWRITING": [
        "живой рукописный шрифт", "неформальный леттеринг",
        "плавный шрифт ручной работы", "эмоциональная рукописная гарнитура",
    ],
    "MONOSPACE": [
        "моноширинный технический шрифт", "кодовый шрифт с равным шагом",
        "утилитарный моноспейс", "четкий шрифт для терминала и данных",
    ],
}
USAGE_RU = [
    "с читабельной кириллицей", "для набора текста", "для айдентики и навигации",
    "с цельным ритмом букв", "для экранов и печати", "с узнаваемым характером",
]
ROUNDED_FAMILY_TOKENS = (
    "bagel", "baloo", "bowlby", "cherry bomb", "comfortaa", "concert one",
    "dosis", "fredoka", "lilita", "marmelad", "m plus rounded", "nunito",
    "rowdies", "sniglet", "varela round",
)
CLASSIFICATION_RU = {
    "HUMANIST": "гуманистический",
    "GEOMETRIC": "геометрический",
    "NEO_GROTESQUE": "нейтральный неогротеск",
    "GROTESQUE": "классический гротеск",
    "SLAB_SERIF": "брусковый с массивными засечками",
    "OLD_STYLE_SERIF": "старостильная антиква",
    "TRANSITIONAL_SERIF": "переходная антиква",
    "DIDONE": "контрастная антиква дидона",
    "DISPLAY": "декоративный акцидентный",
    "MONOSPACE": "моноширинный",
}


def captions(metadata: dict[str, object]) -> list[str]:
    weight = int(metadata["weight"])
    width = int(metadata["width"])
    weight_ru = "тонкий и легкий" if weight < 300 else "полужирный" if weight > 550 else "средней насыщенности"
    weight_en = "light" if weight < 300 else "bold" if weight > 550 else "regular weight"
    width_ru = "узкий и компактный" if width < 5 else "широкий" if width > 5 else "нормальной ширины"
    width_en = "condensed" if width < 5 else "wide" if width > 5 else "normal width"
    italic_ru = "курсивный" if metadata["italic"] else "прямой"
    italic_en = "italic" if metadata["italic"] else "upright"
    category = str(metadata.get("category") or "SANS_SERIF")
    category_ru = CATEGORY_RU.get(category, "выразительный шрифт")
    category_en = CATEGORY_EN.get(category, "expressive typeface")
    classifications = [
        CLASSIFICATION_RU[item] for item in metadata.get("classifications", [])
        if item in CLASSIFICATION_RU
    ]
    classification = classifications[0] if classifications else category_ru
    contrast_value = panose_contrast(metadata)
    roundness_value = panose_roundness(metadata)
    stroke_ru = "с контрастными штрихами" if contrast_value > 0.25 else "с ровным штрихом"
    form_ru = "с округлыми мягкими формами" if roundness_value > 0.25 else "с угловатыми четкими формами" if roundness_value < -0.25 else "со сбалансированными формами"
    captions = [
        f"{weight_ru} {width_ru} {italic_ru} {category_ru}",
        f"{classification}, {stroke_ru}, {width_ru}, {italic_ru}",
        f"читабельный {category_ru}, {weight_ru}, для текста и интерфейсов",
        f"{weight_en} {width_en} {italic_en} {category_en}",
        f"coherent readable {category_en}, {weight_en}, balanced proportions",
    ]
    variants = CATEGORY_RU_VARIANTS.get(category, [category_ru])
    for index, variant in enumerate(variants):
        usage = USAGE_RU[index % len(USAGE_RU)]
        captions.extend([
            f"{variant}, {width_ru}, {weight_ru}, {usage}",
            f"создай {italic_ru} {variant} — {usage}",
            f"нужен {weight_ru} {variant}, {width_ru}, {stroke_ru}",
            f"{usage}: {classification}, {variant}, {width_ru}, {form_ru}",
        ])
    return captions


def controls(metadata: dict[str, object]) -> list[float]:
    return [
        (int(metadata["weight"]) - 400) / 500,
        (int(metadata["width"]) - 5) / 4,
        panose_contrast(metadata),
        panose_roundness(metadata),
        1.0 if metadata["italic"] else 0.0,
    ]


def panose_contrast(metadata: dict[str, object]) -> float:
    value = int(metadata.get("panose_contrast", 0))
    return {0: 0.0, 1: 0.0, 2: -0.8, 3: -0.6, 4: -0.35, 5: 0.0, 6: 0.3, 7: 0.55, 8: 0.8, 9: 1.0}.get(value, 0.0)


def panose_roundness(metadata: dict[str, object]) -> float:
    family = str(metadata.get("family", "")).lower()
    if any(token in family for token in ROUNDED_FAMILY_TOKENS):
        return 0.9
    value = int(metadata.get("panose_letterform", 0))
    if value in {6, 13}:
        return 0.8
    if value in {4, 11}:
        return -0.4
    if value in {8, 15}:
        return -0.75
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Fontgen outline JSONL from licensed font files")
    parser.add_argument("font_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--charset", default=DEFAULT_CHARSET)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--faces-per-family", type=int, default=0, help="0 keeps all faces")
    parser.add_argument("--require-subset", help="Only include Google Fonts families declaring this subset")
    parser.add_argument("--require-license", default="OFL", help="Expected metadata license; empty disables the check")
    parser.add_argument("--audit", type=Path, help="Only use paths marked open by audit_local_fonts.py")
    parser.add_argument("--min-cyrillic-coverage", type=float, default=0.9)
    parser.add_argument("--vector-only", action="store_true", help="Omit raster masks for v5 training")
    parser.add_argument("--exclude-trial", action="store_true", help="Skip files or font names marked Trial/Demo")
    parser.add_argument("--source-label", default="google-fonts", help="Provenance label stored in each row")
    args = parser.parse_args()
    config = ModelConfig()
    randomizer = random.Random(args.seed)
    discovered = sorted([*args.font_dir.rglob("*.ttf"), *args.font_dir.rglob("*.otf")])
    if args.audit:
        audit_rows = json.loads(args.audit.read_text(encoding="utf-8"))
        allowed_paths = {
            str(row["path"])
            for row in audit_rows
            if row.get("license_status") == "open"
            and float(row.get("cyrillic_coverage", 0)) >= args.min_cyrillic_coverage
        }
        discovered = [path for path in discovered if str(path) in allowed_paths]
    candidates: list[tuple[Path, dict[str, object]]] = []
    for path in discovered:
        try:
            metadata = inspect_font(path)
            if args.exclude_trial:
                trial_haystack = " ".join(
                    (path.name, str(metadata.get("family", "")), str(metadata.get("subfamily", "")))
                ).casefold()
                if "trial" in trial_haystack or "demo" in trial_haystack:
                    continue
            if args.require_license and metadata.get("license") != args.require_license:
                continue
            if args.require_subset and args.require_subset not in metadata.get("subsets", []):
                continue
            candidates.append((path, metadata))
        except Exception as error:  # noqa: BLE001
            print(f"skip metadata {path}: {error}")
    if args.faces_per_family:
        grouped: dict[str, list[tuple[Path, dict[str, object]]]] = {}
        for candidate in candidates:
            grouped.setdefault(str(candidate[1]["family"]), []).append(candidate)
        candidates = []
        for family_candidates in grouped.values():
            candidates.extend(_select_family_faces(family_candidates, args.faces_per_family))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as destination:
        for path, metadata in candidates:
            try:
                font = TTFont(path)
                prompts = captions(metadata)
                for character in dict.fromkeys(args.charset):
                    glyph = extract_glyph(font, character, config.max_commands)
                    raster = None if args.vector_only else glyph_mask(font, character, config.raster_size)
                    if glyph is None or (not args.vector_only and raster is None):
                        continue
                    row = {
                        "font": str(path), "family": metadata["family"],
                        "subfamily": metadata["subfamily"], "license": metadata.get("license_id") or metadata.get("license"),
                        "source": args.source_label,
                        "category": metadata.get("category"), "subsets": metadata.get("subsets", []),
                        "classifications": metadata.get("classifications", []),
                        "character": character, "prompt": randomizer.choice(prompts),
                        "controls": controls(metadata), "commands": glyph.commands,
                        "coordinates": glyph.coordinates, "advance_width": glyph.advance_width,
                        "left_side_bearing": glyph.left_side_bearing,
                    }
                    if raster is not None:
                        row["raster"] = encode_mask(raster)
                    destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
            except Exception as error:  # noqa: BLE001 - one broken font must not abort the corpus
                print(f"skip {path}: {error}")
    families = len({str(metadata["family"]) for _, metadata in candidates})
    print(f"wrote {written} glyph examples from {len(candidates)} files / {families} families to {args.output}")


def _face_priority(candidate: tuple[Path, dict[str, object]]) -> tuple[int, int, int, str]:
    path, metadata = candidate
    filename = path.name.lower()
    return (
        1 if metadata["italic"] else 0,
        0 if "regular" in filename or "[" in filename else 1,
        abs(int(metadata["weight"]) - 400),
        filename,
    )


def _select_family_faces(
    candidates: list[tuple[Path, dict[str, object]]], limit: int,
) -> list[tuple[Path, dict[str, object]]]:
    upright = sorted((item for item in candidates if not item[1]["italic"]), key=_face_priority)
    italic = sorted((item for item in candidates if item[1]["italic"]), key=_face_priority)
    selected = upright[:1] + italic[:1]
    remaining = sorted((item for item in candidates if item not in selected), key=_face_priority)
    return (selected + remaining)[:limit]


if __name__ == "__main__":
    main()
