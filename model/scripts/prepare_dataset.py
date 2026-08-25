from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from fontTools.ttLib import TTFont

from fontgen_model.config import ModelConfig
from fontgen_model.outline import extract_glyph, inspect_font

DEFAULT_CHARSET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "0123456789.,:;!?—-()[]«»@&%+="
)


def captions(metadata: dict[str, object]) -> list[str]:
    weight = int(metadata["weight"])
    width = int(metadata["width"])
    weight_ru = "тонкий" if weight < 350 else "жирный" if weight > 650 else "средний"
    width_ru = "узкий" if width < 5 else "широкий" if width > 5 else "нормальной ширины"
    italic_ru = "курсивный" if metadata["italic"] else "прямой"
    return [
        f"{weight_ru} {width_ru} {italic_ru} шрифт для набора текста",
        f"typeface with {weight} weight and width class {width}, {'italic' if metadata['italic'] else 'upright'}",
        f"гарнитура {metadata['family']}, начертание {metadata['subfamily']}",
    ]


def controls(metadata: dict[str, object]) -> list[float]:
    return [
        (int(metadata["weight"]) - 400) / 500,
        (int(metadata["width"]) - 5) / 4,
        0.0,
        0.0,
        1.0 if metadata["italic"] else 0.0,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Fontgen outline JSONL from licensed font files")
    parser.add_argument("font_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--charset", default=DEFAULT_CHARSET)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    config = ModelConfig()
    randomizer = random.Random(args.seed)
    paths = sorted([*args.font_dir.rglob("*.ttf"), *args.font_dir.rglob("*.otf")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as destination:
        for path in paths:
            try:
                metadata = inspect_font(path)
                font = TTFont(path)
                prompts = captions(metadata)
                for character in dict.fromkeys(args.charset):
                    glyph = extract_glyph(font, character, config.max_commands)
                    if glyph is None:
                        continue
                    row = {
                        "font": str(path), "family": metadata["family"],
                        "character": character, "prompt": randomizer.choice(prompts),
                        "controls": controls(metadata), "commands": glyph.commands,
                        "coordinates": glyph.coordinates, "advance_width": glyph.advance_width,
                        "left_side_bearing": glyph.left_side_bearing,
                    }
                    destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
            except Exception as error:  # noqa: BLE001 - one broken font must not abort the corpus
                print(f"skip {path}: {error}")
    print(f"wrote {written} glyph examples from {len(paths)} files to {args.output}")


if __name__ == "__main__":
    main()
