"""Audit a font directory: extract metadata, check quality, categorize.

Outputs a JSON report with per-font metadata, quality scores, and category.
Use the output to decide which fonts to include in training.

Usage:
    python scripts/audit_fonts.py /path/to/fonts -o audit.json
    python scripts/audit_fonts.py /path/to/fonts -o audit.json --min-score 0.6
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

@dataclass
class FontAudit:
    path: str
    filename: str
    family: str
    subfamily: str
    weight: int
    width: int
    italic: bool
    category: str
    subclass: str
    license: str | None
    panose_family_type: int
    panose_serif_style: int
    panose_contrast: int
    panose_letterform: int
    panose_midline: int
    panose_x_height: int
    units_per_em: int
    ascender: int
    descender: int
    glyph_count: int
    latin_coverage: float
    cyrillic_coverage: float
    digit_coverage: float
    punctuation_coverage: float
    outline_complexity: float
    has_kerning: bool
    has_hinting: bool
    is_variable: bool
    quality_score: float
    issues: list[str] = field(default_factory=list)


LATIN_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LATIN_LOWER = "abcdefghijklmnopqrstuvwxyz"
CYRILLIC_UPPER = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
CYRILLIC_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
DIGITS = "0123456789"
PUNCTUATION = ".,:;!?-()[]«»@&%+="


def _coverage(font: TTFont, characters: str) -> float:
    cmap = font.getBestCmap() or {}
    if not characters:
        return 1.0
    covered = sum(1 for ch in characters if ord(ch) in cmap)
    return covered / len(characters)


def _outline_complexity(font: TTFont, sample_chars: str = "ABDEHORabdegopq") -> float:
    """Average commands per glyph (normalized 0-1). Higher = more complex."""
    cmap = font.getBestCmap() or {}
    glyph_set = font.getGlyphSet()
    total_commands = 0
    counted = 0
    for ch in sample_chars:
        name = cmap.get(ord(ch))
        if not name:
            continue
        try:
            pen = _CountPen(glyph_set)
            glyph_set[name].draw(pen)
            total_commands += pen.command_count
            counted += 1
        except Exception:
            pass
    if counted == 0:
        return 0.0
    avg = total_commands / counted
    return min(1.0, avg / 80.0)


class _CountPen(BasePen):
    def __init__(self, glyph_set: Any):
        super().__init__(glyph_set)
        self.command_count = 0

    def _moveTo(self, point: Any) -> None:
        self.command_count += 1

    def _lineTo(self, point: Any) -> None:
        self.command_count += 1

    def _curveToOne(self, *args: Any) -> None:
        self.command_count += 1

    def _qCurveToOne(self, *args: Any) -> None:
        self.command_count += 1

    def _closePath(self) -> None:
        pass

    def _endPath(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------

def _categorize(font: TTFont, family: str) -> tuple[str, str]:
    """Return (category, subclass)."""
    os2 = font.get("OS/2")
    panose = getattr(os2, "panose", None)
    post = font.get("post")
    normalized = family.casefold()

    # Monospace
    if bool(getattr(post, "isFixedPitch", 0)) or any(
        t in normalized for t in ("mono", "code", "typewriter", "courier", "consol")
    ):
        return "MONOSPACE", "MONOSPACE"

    # Handwriting
    ibm_class = (int(getattr(os2, "sFamilyClass", 0)) >> 8) & 0xFF
    family_type = int(getattr(panose, "bFamilyType", 0)) if panose else 0
    if ibm_class == 10 or family_type == 3:
        return "HANDWRITING", "HANDWRITING"
    if any(t in normalized for t in ("script", "hand", "cursive", "brush", "calligraph", "letter")):
        return "HANDWRITING", "HANDWRITING"

    # Display
    if ibm_class == 9 or family_type in {4, 5}:
        return "DISPLAY", "DISPLAY"
    if any(t in normalized for t in ("display", "poster", "deco", "ornament", "dingbat", "symbol", "icon")):
        return "DISPLAY", "DISPLAY"

    # Serif
    serif_style = int(getattr(panose, "bSerifStyle", 0)) if panose else 0
    if ibm_class in {1, 2, 3, 4, 5, 7} or (family_type == 2 and 2 <= serif_style <= 10):
        subclass = _serif_subclass(panose, normalized)
        return "SERIF", subclass
    if any(t in normalized for t in (
        "serif", "antiqua", "garamond", "bodoni", "didot", "baskerville",
        "clarendon", "slab", "times", "georgia", "palatino",
    )):
        subclass = _serif_subclass(panose, normalized)
        return "SERIF", subclass

    # Sans-serif (default)
    subclass = _sans_subclass(panose, normalized)
    return "SANS_SERIF", subclass


def _serif_subclass(panose: Any, family: str) -> str:
    if not panose:
        return "OLD_STYLE_SERIF"
    contrast = int(getattr(panose, "bContrast", 0))
    if "slab" in family or "clarendon" in family:
        return "SLAB_SERIF"
    if contrast >= 8:
        return "DIDONE"
    if contrast >= 6:
        return "TRANSITIONAL_SERIF"
    return "OLD_STYLE_SERIF"


def _sans_subclass(panose: Any, family: str) -> str:
    if not panose:
        return "NEO_GROTESQUE"
    contrast = int(getattr(panose, "bContrast", 0))
    letterform = int(getattr(panose, "bLetterForm", 0))
    if any(t in family for t in ("futura", "avenir", "geometric", "circular")):
        return "GEOMETRIC"
    if letterform in {6, 7, 8, 9} or any(t in family for t in ("gill", "optima", "humanist")):
        return "HUMANIST"
    if contrast <= 2:
        return "GEOMETRIC"
    return "NEO_GROTESQUE"


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def _quality_score(
    latin: float, cyrillic: float, digits: float, punct: float,
    complexity: float, glyph_count: int, issues: list[str],
) -> float:
    score = 0.0
    # Coverage (40%)
    score += latin * 0.15
    score += cyrillic * 0.10
    score += digits * 0.05
    score += punct * 0.10
    # Glyph count (20%)
    count_score = min(1.0, glyph_count / 200)
    score += count_score * 0.20
    # Complexity (20%) — prefer moderate complexity
    complexity_score = 1.0 - abs(complexity - 0.35) * 2
    score += max(0.0, complexity_score) * 0.20
    # Penalties (20%)
    penalty = min(1.0, len(issues) * 0.15)
    score += (1.0 - penalty) * 0.20
    return round(max(0.0, min(1.0, score)), 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def audit_font(path: Path) -> FontAudit | None:
    try:
        font = TTFont(path, fontNumber=0)
    except Exception:
        return None

    try:
        names = font["name"]
        family = names.getDebugName(16) or names.getDebugName(1) or path.stem
        subfamily = names.getDebugName(2) or "Regular"
        os2 = font.get("OS/2")
        panose = getattr(os2, "panose", None)
        post = font.get("post")

        weight = int(getattr(os2, "usWeightClass", 400))
        width = int(getattr(os2, "usWidthClass", 5))
        italic = bool(getattr(os2, "fsSelection", 0) & 1)
        units_per_em = int(font["head"].unitsPerEm)
        ascender = int(getattr(os2, "sTypoAscender", 800))
        descender = int(getattr(os2, "sTypoDescender", -200))

        category, subclass = _categorize(font, family)

        latin_cov = (_coverage(font, LATIN_UPPER) + _coverage(font, LATIN_LOWER)) / 2
        cyrillic_cov = (_coverage(font, CYRILLIC_UPPER) + _coverage(font, CYRILLIC_LOWER)) / 2
        digit_cov = _coverage(font, DIGITS)
        punct_cov = _coverage(font, PUNCTUATION)
        complexity = _outline_complexity(font)

        cmap = font.getBestCmap() or {}
        glyph_count = len(cmap)

        has_kerning = "kern" in font
        has_hinting = "fpgm" in font
        is_variable = "fvar" in font

        issues: list[str] = []
        if latin_cov < 0.9:
            issues.append(f"latin_coverage={latin_cov:.0%}")
        if cyrillic_cov < 0.5:
            issues.append(f"cyrillic_coverage={cyrillic_cov:.0%}")
        if digit_cov < 1.0:
            issues.append(f"digit_coverage={digit_cov:.0%}")
        if glyph_count < 50:
            issues.append(f"low_glyph_count={glyph_count}")
        if complexity > 0.8:
            issues.append(f"high_complexity={complexity:.2f}")
        if is_variable:
            issues.append("variable_font")

        score = _quality_score(
            latin_cov, cyrillic_cov, digit_cov, punct_cov,
            complexity, glyph_count, issues,
        )

        return FontAudit(
            path=str(path),
            filename=path.name,
            family=family,
            subfamily=subfamily,
            weight=weight,
            width=width,
            italic=italic,
            category=category,
            subclass=subclass,
            license=None,
            panose_family_type=int(getattr(panose, "bFamilyType", 0)) if panose else 0,
            panose_serif_style=int(getattr(panose, "bSerifStyle", 0)) if panose else 0,
            panose_contrast=int(getattr(panose, "bContrast", 0)) if panose else 0,
            panose_letterform=int(getattr(panose, "bLetterForm", 0)) if panose else 0,
            panose_midline=int(getattr(panose, "bMidLine", 0)) if panose else 0,
            panose_x_height=int(getattr(panose, "bXHeight", 0)) if panose else 0,
            units_per_em=units_per_em,
            ascender=ascender,
            descender=descender,
            glyph_count=glyph_count,
            latin_coverage=round(latin_cov, 3),
            cyrillic_coverage=round(cyrillic_cov, 3),
            digit_coverage=round(digit_cov, 3),
            punctuation_coverage=round(punct_cov, 3),
            outline_complexity=round(complexity, 3),
            has_kerning=has_kerning,
            has_hinting=has_hinting,
            is_variable=is_variable,
            quality_score=score,
            issues=issues,
        )
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit font directory for training suitability")
    parser.add_argument("font_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("font_audit.json"))
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum quality score to include")
    parser.add_argument("--min-latin", type=float, default=0.8, help="Minimum Latin coverage")
    parser.add_argument("--min-cyrillic", type=float, default=0.0, help="Minimum Cyrillic coverage")
    parser.add_argument("--exclude-variable", action="store_true", help="Exclude variable fonts")
    parser.add_argument("--exclude-trial", action="store_true", help="Exclude trial/demo fonts")
    parser.add_argument("--summary", action="store_true", help="Print summary to stdout")
    args = parser.parse_args()

    font_files = sorted([
        *args.font_dir.rglob("*.ttf"),
        *args.font_dir.rglob("*.otf"),
    ])
    print(f"scanning {len(font_files)} font files in {args.font_dir}", file=sys.stderr)

    audits: list[FontAudit] = []
    skipped = 0
    for path in font_files:
        if args.exclude_trial:
            name = path.stem.casefold()
            if "trial" in name or "demo" in name:
                skipped += 1
                continue
        audit = audit_font(path)
        if audit is None:
            skipped += 1
            continue
        if args.exclude_variable and audit.is_variable:
            skipped += 1
            continue
        if audit.quality_score < args.min_score:
            skipped += 1
            continue
        if audit.latin_coverage < args.min_latin:
            skipped += 1
            continue
        if audit.cyrillic_coverage < args.min_cyrillic:
            skipped += 1
            continue
        audits.append(audit)

    # Group by family
    families: dict[str, list[FontAudit]] = {}
    for audit in audits:
        families.setdefault(audit.family, []).append(audit)

    # Stats
    category_counts = Counter(a.category for a in audits)
    subclass_counts = Counter(a.subclass for a in audits)
    avg_score = sum(a.quality_score for a in audits) / max(1, len(audits))
    avg_latin = sum(a.latin_coverage for a in audits) / max(1, len(audits))
    avg_cyrillic = sum(a.cyrillic_coverage for a in audits) / max(1, len(audits))

    report = {
        "total_fonts": len(font_files),
        "audited": len(audits),
        "skipped": skipped,
        "families": len(families),
        "avg_quality_score": round(avg_score, 3),
        "avg_latin_coverage": round(avg_latin, 3),
        "avg_cyrillic_coverage": round(avg_cyrillic, 3),
        "categories": dict(category_counts),
        "subclasses": dict(subclass_counts),
        "fonts": [asdict(a) for a in audits],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(audits)} fonts ({len(families)} families) to {args.output}", file=sys.stderr)

    if args.summary:
        print(f"\n{'='*60}")
        print(f"FONT AUDIT SUMMARY")
        print(f"{'='*60}")
        print(f"Total scanned:  {len(font_files)}")
        print(f"Audited:        {len(audits)}")
        print(f"Skipped:        {skipped}")
        print(f"Families:       {len(families)}")
        print(f"Avg quality:    {avg_score:.3f}")
        print(f"Avg Latin:      {avg_latin:.1%}")
        print(f"Avg Cyrillic:   {avg_cyrillic:.1%}")
        print(f"\nCategories:")
        for cat, count in category_counts.most_common():
            print(f"  {cat:15s} {count:5d}")
        print(f"\nSubclasses:")
        for sub, count in subclass_counts.most_common():
            print(f"  {sub:20s} {count:5d}")
        print(f"\nTop 10 families by face count:")
        for family, faces in sorted(families.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  {family:30s} {len(faces)} faces")
        print(f"\nQuality distribution:")
        buckets = {"excellent (0.8+)": 0, "good (0.6-0.8)": 0, "fair (0.4-0.6)": 0, "poor (<0.4)": 0}
        for a in audits:
            if a.quality_score >= 0.8:
                buckets["excellent (0.8+)"] += 1
            elif a.quality_score >= 0.6:
                buckets["good (0.6-0.8)"] += 1
            elif a.quality_score >= 0.4:
                buckets["fair (0.4-0.6)"] += 1
            else:
                buckets["poor (<0.4)"] += 1
        for label, count in buckets.items():
            print(f"  {label:20s} {count:5d}")


if __name__ == "__main__":
    main()
