#!/bin/bash
# Full pipeline: audit fonts → generate dataset → train
# Prompts for dataset tier before training.
#
# Usage: ./scripts/full_pipeline.sh [epochs] [batch_size]
#
# Examples:
#   ./scripts/full_pipeline.sh              # defaults: 40 epochs, batch 32
#   ./scripts/full_pipeline.sh 100 64       # 100 epochs, batch 64

set -e
cd "$(dirname "$0")/.."

EPOCHS=${1:-40}
BATCH=${2:-32}
FONT_DIR="../public/fonts"
AUDIT="data/font_audit.json"
DATASET="data/train.jsonl"
TIERS_DIR="data/tiers"
CHECKPOINT="checkpoints/fontgen-v4.3-tier${TIER_SIZE}.pt"

echo "=========================================="
echo "  FONTGEN FULL PIPELINE"
echo "=========================================="
echo ""

# Step 1: Audit
if [ ! -f "$AUDIT" ]; then
    echo "[1/4] Auditing fonts in $FONT_DIR..."
    .venv/bin/python scripts/audit_fonts.py "$FONT_DIR" -o "$AUDIT" --summary
else
    echo "[1/4] Audit exists: $AUDIT (delete to re-audit)"
fi
echo ""

# Step 2: Dataset
if [ ! -f "$DATASET" ]; then
    echo "[2/4] Generating dataset..."
    .venv/bin/python scripts/generate_dataset.py "$AUDIT" "$DATASET" \
        --min-score 0.6 --min-latin 0.9 --min-cyrillic 0.0 \
        --max-faces-per-family 8 --workers 16
else
    ROWS=$(wc -l < "$DATASET")
    echo "[2/4] Dataset exists: $DATASET ($ROWS rows, delete to regenerate)"
fi
echo ""

# Step 3: Build tiers if not present
TIER_COUNT=$(ls "$TIERS_DIR"/train_tier_*.jsonl 2>/dev/null | wc -l)
if [ "$TIER_COUNT" -lt 5 ]; then
    echo "[3/4] Building tier datasets..."
    .venv/bin/python scripts/split_dataset_tiers.py "$DATASET" --output-dir "$TIERS_DIR"
else
    echo "[3/4] Tiers exist ($TIER_COUNT datasets in $TIERS_DIR/)"
fi
echo ""

# Step 4: Select tier
echo "=== Select Dataset Tier ==="
echo ""

TIER_OPTIONS=()
TIER_INDEX=1
for f in "$TIERS_DIR"/train_tier_*.jsonl; do
    [ -f "$f" ] || continue
    BASENAME=$(basename "$f" .jsonl)
    SIZE=${BASENAME#train_tier_}
    ROWS=$(wc -l < "$f")
    TIER_OPTIONS+=("$SIZE|$f")
    printf "  [%d] %6s fonts  (%s rows)\n" "$TIER_INDEX" "$SIZE" "$ROWS"
    TIER_INDEX=$((TIER_INDEX + 1))
done

if [ ${#TIER_OPTIONS[@]} -eq 0 ]; then
    echo "ERROR: no tier datasets found in $TIERS_DIR/"
    exit 1
fi

echo ""
read -rp "select tier [1-${#TIER_OPTIONS[@]}]: " TIER_CHOICE

if [ "$TIER_CHOICE" -lt 1 ] || [ "$TIER_CHOICE" -gt "${#TIER_OPTIONS[@]}" ] 2>/dev/null; then
    echo "invalid choice"; exit 1
fi

SELECTED="${TIER_OPTIONS[$((TIER_CHOICE - 1))]}"
TIER_SIZE="${SELECTED%%|*}"
TIER_DATASET="${SELECTED#*|}"

if [ ! -f "$TIER_DATASET" ]; then
    echo "ERROR: $TIER_DATASET not found"
    exit 1
fi

TIER_ROWS=$(wc -l < "$TIER_DATASET")
CHECKPOINT="checkpoints/fontgen-v4.3-tier${TIER_SIZE}.pt"
echo ""
echo "Selected: $TIER_SIZE fonts ($TIER_ROWS glyph rows)"
echo ""

# Step 5: Train
echo "[4/4] Training for $EPOCHS epochs, batch size $BATCH..."
echo "  Dataset:    $TIER_DATASET"
echo "  Checkpoint: $CHECKPOINT"
echo ""

.venv/bin/python scripts/train.py "$TIER_DATASET" \
    --raster-only \
    --balanced-styles \
    --batch-size "$BATCH" \
    --augment --control-jitter 0.1 \
    --cfg-dropout 0.1 \
    --ema-decay 0.999 \
    --warmup-ratio 0.05 --min-lr-ratio 0.1 \
    --perceptual-loss \
    --uncertainty-weighting \
    --discriminator --disc-lr 1e-4 --r1-weight 10 \
    --epochs "$EPOCHS" \
    --output "$CHECKPOINT"

echo ""
echo "=========================================="
echo "  TRAINING COMPLETE"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""
echo "Export for inference:"
echo "  .venv/bin/python scripts/export_checkpoint.py $CHECKPOINT ${CHECKPOINT%.pt}-inference.pt"
