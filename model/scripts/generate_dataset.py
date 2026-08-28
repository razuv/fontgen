"""Generate training dataset from audited fonts (parallel version).

Reads the audit JSON and creates a JSONL manifest for training.

Usage:
    python scripts/generate_dataset.py data/font_audit.json data/train.jsonl
    python scripts/generate_dataset.py data/font_audit.json data/train.jsonl --min-score 0.6 --max-faces-per-family 8
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from fontTools.ttLib import TTFont

from fontgen_model.config import ModelConfig
from fontgen_model.outline import extract_glyph
from fontgen_model.raster import encode_mask, glyph_mask

logging.getLogger("fontTools").setLevel(logging.ERROR)


DEFAULT_CHARSET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "0123456789.,:;!?—-()[]«»@&%+="
)

CATEGORY_PROMPTS_RU = {
    "SANS_SERIF": [
        "современный гротеск без засечек", "нейтральный рубленый шрифт",
        "чистый sans-serif для интерфейса", "функциональный шрифт без засечек",
    ],
    "SERIF": [
        "книжная антиква с засечками", "редакционный шрифт для длинного чтения",
        "выразительная антиква", "литературный serif с ясным ритмом",
    ],
    "DISPLAY": [
        "акцидентный заголовочный шрифт", "брутальный плакатный шрифт",
        "выразительный display для крупных заголовков", "декоративный акцидентный шрифт",
    ],
    "HANDWRITING": [
        "живой рукописный шрифт", "неформальный леттеринг",
        "плавный шрифт ручной работы", "эмоциональная рукописная гарнитура",
    ],
    "MONOSPACE": [
        "моноширинный технический шрифт", "кодовый шрифт с равным шагом",
        "утилитарный моноспейс", "четкий шрифт для терминала",
    ],
}

CATEGORY_PROMPTS_EN = {
    "SANS_SERIF": ["modern sans serif", "clean grotesque", "neutral sans-serif"],
    "SERIF": ["book serif typeface", "editorial serif", "classic serif"],
    "DISPLAY": ["expressive display typeface", "poster headline font", "decorative display"],
    "HANDWRITING": ["natural handwritten typeface", "casual lettering", "script font"],
    "MONOSPACE": ["technical monospaced font", "code typeface", "monospace for terminal"],
}


def _weight_word(weight: int, lang: str) -> str:
    if lang == "ru":
        return "тонкий" if weight < 300 else "средний" if weight < 500 else "полужирный" if weight < 700 else "жирный"
    return "light" if weight < 300 else "regular" if weight < 500 else "bold" if weight < 700 else "heavy"


def _width_word(width: int, lang: str) -> str:
    if lang == "ru":
        return "узкий" if width < 4 else "компактный" if width < 6 else "широкий" if width > 7 else "нормальный"
    return "condensed" if width < 4 else "narrow" if width < 6 else "wide" if width > 7 else "normal"


def _italic_word(italic: bool, lang: str) -> str:
    return ("курсивный" if italic else "прямой") if lang == "ru" else ("italic" if italic else "upright")


def generate_prompts(audit: dict) -> list[str]:
    category = audit.get("category", "SANS_SERIF")
    weight = audit.get("weight", 400)
    width = audit.get("width", 5)
    italic = audit.get("italic", False)
    prompts = []
    for lang in ("ru", "en"):
        w = _weight_word(weight, lang)
        wd = _width_word(width, lang)
        it = _italic_word(italic, lang)
        bases = CATEGORY_PROMPTS_RU.get(category, CATEGORY_PROMPTS_RU["SANS_SERIF"]) if lang == "ru" else CATEGORY_PROMPTS_EN.get(category, CATEGORY_PROMPTS_EN["SANS_SERIF"])
        for base in bases:
            prompts.append(f"{w} {wd} {it} {base}")
            prompts.append(f"{base}, {w}, {wd}")
    return prompts


def controls_from_audit(audit: dict) -> list[float]:
    weight = audit.get("weight", 400)
    width = audit.get("width", 5)
    contrast = audit.get("panose_contrast", 0)
    letterform = audit.get("panose_letterform", 0)
    italic = audit.get("italic", False)
    contrast_map = {0: 0.0, 1: 0.0, 2: -0.8, 3: -0.6, 4: -0.35, 5: 0.0, 6: 0.3, 7: 0.55, 8: 0.8, 9: 1.0}
    return [
        max(-1.0, min(1.0, (weight - 400) / 500)),
        max(-1.0, min(1.0, (width - 5) / 4)),
        max(-1.0, min(1.0, contrast_map.get(contrast, 0.0))),
        max(-1.0, min(1.0, 0.8 if letterform in {6, 13} else -0.4 if letterform in {4, 11} else -0.75 if letterform in {8, 15} else 0.0)),
        1.0 if italic else 0.0,
    ]


def _process_font(font_audit: dict, charset: str, seed: int) -> list[dict]:
    """Process a single font — runs in a worker process."""
    path = Path(font_audit["path"])
    try:
        font = TTFont(path)
    except Exception:
        return []

    config = ModelConfig()
    rng = random.Random(seed)
    prompts = generate_prompts(font_audit)
    ctrl = controls_from_audit(font_audit)
    rows = []

    for character in charset:
        try:
            glyph = extract_glyph(font, character, config.max_commands)
        except Exception:
            continue
        if glyph is None:
            continue
        try:
            raster = glyph_mask(font, character, config.raster_size)
        except Exception:
            continue
        if raster is None:
            continue

        rows.append({
            "font": str(path),
            "family": font_audit["family"],
            "subfamily": font_audit["subfamily"],
            "source": "company-library",
            "category": font_audit["category"],
            "subclass": font_audit["subclass"],
            "character": character,
            "prompt": rng.choice(prompts),
            "controls": ctrl,
            "commands": glyph.commands,
            "coordinates": glyph.coordinates,
            "advance_width": glyph.advance_width,
            "left_side_bearing": glyph.left_side_bearing,
            "raster": encode_mask(raster),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training dataset from font audit")
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--charset", default=DEFAULT_CHARSET)
    parser.add_argument("--min-score", type=float, default=0.6)
    parser.add_argument("--min-latin", type=float, default=0.9)
    parser.add_argument("--min-cyrillic", type=float, default=0.0)
    parser.add_argument("--max-faces-per-family", type=int, default=8)
    parser.add_argument("--balance-categories", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=0, help="0 = auto (CPU count)")
    args = parser.parse_args()

    audit_data = json.loads(args.audit.read_text(encoding="utf-8"))
    fonts = audit_data["fonts"]

    # Filter
    filtered = []
    for font in fonts:
        if font["quality_score"] < args.min_score:
            continue
        if font["latin_coverage"] < args.min_latin:
            continue
        if font["cyrillic_coverage"] < args.min_cyrillic:
            continue
        filtered.append(font)
    print(f"filtered: {len(filtered)} / {len(fonts)} fonts", file=sys.stderr)

    # Group by family, limit faces
    if args.max_faces_per_family > 0:
        families: dict[str, list[dict]] = {}
        for font in filtered:
            families.setdefault(font["family"], []).append(font)
        selected = []
        for faces in families.values():
            faces.sort(key=lambda f: (
                0 if "regular" in f["subfamily"].lower() else 1,
                0 if not f["italic"] else 1,
                abs(f["weight"] - 400),
            ))
            selected.extend(faces[:args.max_faces_per_family])
        filtered = selected
        print(f"limited to {len(filtered)} faces ({args.max_faces_per_family} per family)", file=sys.stderr)

    # Balance (optional)
    if args.balance_categories:
        category_counts = Counter(f["category"] for f in filtered)
        min_count = min(category_counts.values())
        balanced = []
        by_category: dict[str, list[dict]] = {}
        for font in filtered:
            by_category.setdefault(font["category"], []).append(font)
        rng = random.Random(args.seed)
        for cat, cat_fonts in by_category.items():
            balanced.extend(rng.sample(cat_fonts, min(min_count, len(cat_fonts))))
        filtered = balanced
        print(f"balanced to {len(filtered)} fonts", file=sys.stderr)

    # Parallel processing
    workers = args.workers or 8
    total = len(filtered)
    print(f"processing {total} fonts with {workers} workers...", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    done = 0

    with args.output.open("w", encoding="utf-8") as out:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_font, fa, args.charset, args.seed + i): fa
                for i, fa in enumerate(filtered)
            }
            for future in as_completed(futures):
                done += 1
                try:
                    rows = future.result()
                    if rows:
                        for row in rows:
                            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        written += len(rows)
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
                if done % 100 == 0 or done == total:
                    pct = done * 100 // total
                    print(f"\r  [{pct:3d}%] {done}/{total} fonts | {written} glyphs | {skipped} skipped", end="", file=sys.stderr, flush=True)

    print(file=sys.stderr)
    families_used = len({f["family"] for f in filtered})
    print(f"\nwrote {written} glyphs from {total - skipped} fonts / {families_used} families to {args.output}", file=sys.stderr)
    cats = Counter(f["category"] for f in filtered)
    for cat, count in cats.most_common():
        print(f"  {cat:15s} {count:5d}", file=sys.stderr)


if __name__ == "__main__":
    main()
