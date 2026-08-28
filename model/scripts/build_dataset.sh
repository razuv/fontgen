#!/bin/bash
# Generate training dataset from audited fonts
# Builds all 5 tier datasets (500, 1000, 2000, 5000, 10000 fonts)
#
# Usage: ./scripts/build_dataset.sh [workers] [max_faces_per_family]
#
# Examples:
#   ./scripts/build_dataset.sh              # defaults: 16 workers, 8 faces/family
#   ./scripts/build_dataset.sh 8 4          # 8 workers, 4 faces/family

set -e
cd "$(dirname "$0")/.."

WORKERS=${1:-16}
MAX_FACES=${2:-8}
AUDIT="data/font_audit.json"
RAW="data/train_raw.jsonl"
ENRICHED="data/train_enriched.jsonl"
OUTPUT="data/train.jsonl"
REPORT="data/duplicates.json"
TIERS_DIR="data/tiers"

if [ ! -f "$AUDIT" ]; then
    echo "ERROR: $AUDIT not found. Run audit first:"
    echo "  .venv/bin/python scripts/audit_fonts.py ../public/fonts -o $AUDIT --summary"
    exit 1
fi

echo "=== Font Dataset Generator ==="
echo "Audit:    $AUDIT"
echo "Output:   $OUTPUT"
echo "Workers:  $WORKERS"
echo "Faces:    $MAX_FACES per family"
echo ""

# Step 1: Generate raw manifest from audited fonts
echo "--- Step 1/3: Generate raw manifest ---"
.venv/bin/python scripts/generate_dataset.py "$AUDIT" "$RAW" \
    --min-score 0.6 \
    --min-latin 0.9 \
    --min-cyrillic 0.0 \
    --max-faces-per-family "$MAX_FACES" \
    --workers "$WORKERS"

# Step 2: Enrich with geometry features, auto-tags, and prompt variants
echo ""
echo "--- Step 2/3: Enrich manifest ---"
.venv/bin/python scripts/enrich_manifest.py "$RAW" "$ENRICHED"

# Step 3: Filter near-duplicate fonts
echo ""
echo "--- Step 3/3: Filter similar fonts ---"
.venv/bin/python scripts/filter_similar_fonts.py "$ENRICHED" \
    --output "$OUTPUT" \
    --feature-threshold 0.85 \
    --raster-threshold 0.85 \
    --report "$REPORT"

# Cleanup intermediate files
rm -f "$RAW" "$ENRICHED"

# Step 4: Build all tier datasets
echo ""
echo "--- Step 4/4: Build all tier datasets ---"
.venv/bin/python scripts/split_dataset_tiers.py "$OUTPUT" --output-dir "$TIERS_DIR"

echo ""
echo "=== Done ==="
echo "Full dataset: $OUTPUT"
wc -l "$OUTPUT" | awk '{print $1 " glyph rows"}'
du -sh "$OUTPUT" | awk '{print "Size: " $1}'
echo ""
echo "Tier datasets:"
ls -lh "$TIERS_DIR"/*.jsonl 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
echo ""
if [ -f "$REPORT" ]; then
    echo "Duplicates report: $REPORT"
fi
