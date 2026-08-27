# Fontgen Model

This directory is the beginning of a real trainable prompt-to-vector model. It is deliberately separate from the Vite app: GitHub Pages can host the editor, but not a multi-hundred-million-operation PyTorch inference process.

## What “from scratch” means here

At inference time the model receives only:

- a UTF-8 text prompt;
- a Unicode character id;
- requested weight, width, contrast, roundness and slant controls.

It does **not** receive a Google Font glyph or a source outline. `FontgenNet` encodes one family-level style vector, predicts a 128×128 glyph boundary, and verifies the requested Unicode through a recognition head. A category head separates serif, sans, display, handwriting and monospace style latents. A topology-aware tracer removes small components and converts the new boundary into compact closed Bézier contours. Google Fonts are training examples, not templates used during generation.

## Model v0

```text
UTF-8 prompt -> byte Transformer -> family style latent --------+
                                                               |-> raster decoder -> glyph mask
Unicode id -----------------------> glyph embedding -----------+                    |
trained controls -----------------> style conditioning --------+                    +-> recognition loss
                                                                                    |
                                                               topology-aware tracer -> Bézier sequence
```

The checkpoint targets Latin, Russian Cyrillic, digits and punctuation. Family consistency comes from reusing the same style latent for every glyph. Raster BCE/Dice, boundary edge alignment, Unicode recognition and style-category losses are active.

The compact production configuration has **4,649,201 trainable parameters**. v4 applies the prompt style latent through FiLM inside the hi-res refiner instead of adding it only once to the glyph embedding. Prompt-only category/control heads and variance regularization make ignoring or collapsing the prompt costly. Bézier outlines are recovered with deterministic topology-aware tracing.

## Local setup

PyTorch does not yet support the repository machine's system Python 3.14. Use Python 3.11–3.13:

```bash
cd model
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Prepare a licensed font corpus (do not commit the corpus or checkpoints):

```bash
python scripts/prepare_dataset.py /path/to/ofl-font-files data/train.jsonl
python scripts/train.py data/train.jsonl --raster-only --output checkpoints/fontgen-v0.pt
```

The hi-res v3 checkpoint uses 346 faces from 216 OFL families, 39,130 glyph examples, 147 characters, and 384 natural Russian/English prompt variants. It includes 121 actual Italic faces and a targeted rounded-font subset. Validation holds out complete families:

```bash
python scripts/prepare_dataset.py data/google-fonts-repo/ofl data/google-hires.jsonl --faces-per-family 2 --require-license OFL
python scripts/upgrade_checkpoint.py checkpoints/fontgen-google-expanded-inference.pt checkpoints/fontgen-hires-init.pt
python scripts/train.py data/google-hires.jsonl --raster-only --balanced-styles --batch-size 64 --resume checkpoints/fontgen-hires-init.pt --output checkpoints/fontgen-style-v4.pt
python scripts/export_checkpoint.py checkpoints/fontgen-style-v4.pt checkpoints/fontgen-style-v4-inference.pt
```

For a wiring check only (this checkpoint has no visual quality):

```bash
python scripts/train.py data/raster-smoke.jsonl --preset smoke --epochs 1 --batch-size 8 --output checkpoints/raster-smoke.pt
```

Run the inference service after a checkpoint exists:

```bash
FONTGEN_CHECKPOINT=checkpoints/fontgen-style-v4-inference.pt uvicorn fontgen_model.api:app --host 0.0.0.0 --port 8000
```

`GET /health` reports whether the checkpoint loaded. `POST /v1/generate` accepts a prompt, charset, seed and typography controls and returns normalized Bézier sequences. It intentionally returns `503` without a trained checkpoint instead of silently substituting an existing font.

The fonts bundled in `public/fonts` remain UI fallbacks only. Production training uses the ignored Google Fonts checkout and a family-held-out split. Weight, width, contrast, roundness and slant use OS/2/PANOSE or face metadata. A geometric advance-width floor prevents generated outlines from overlapping. Slant receives a final affine vector transform, preserving contour topology even for aggressive oblique values.

## Vector-native v5 (quality track)

V5 removes the five-million-parameter ceiling and the raster/tracing bridge. A frozen local
`paraphrase-multilingual-MiniLM-L12-v2` encoder maps Russian and English prompts to semantic
384-dimensional vectors. A 22.4M-parameter Transformer then generates `M/L/Q/C/Z` commands and
Bézier points directly, followed by a non-causal outline refinement pass. Training supervises
commands, coordinates, sampled curve geometry, metrics and prompt category. Validation families
are held out in their entirety.

Prepare the local text encoder and prompt cache:

```bash
pip install -e '.[v5]'
python scripts/download_v5_text_model.py
python scripts/prepare_v5_embeddings.py \
  data/google-hires.jsonl data/v5-prompt-embeddings.pt
```

Run a pipeline check (the resulting checkpoint is not suitable for the UI):

```bash
python scripts/train_v5.py data/google-hires.jsonl data/v5-prompt-embeddings.pt \
  --smoke --epochs 1 --max-batches 2
```

V5 is currently paused as an experimental track. The active local path is V4.1 plus sub-pixel
vector cleanup; smoke and probe checkpoints must not replace the working API checkpoint.

### V4.1 auto-tagged fine-tune

V4 can reuse the expanded vector corpus after its missing local masks are materialized:

```bash
python scripts/materialize_rasters.py \
  data/fontgen-v5-autotagged.jsonl data/fontgen-v4-autotagged-raster.jsonl
python scripts/train.py data/fontgen-v4-autotagged-raster.jsonl \
  --raster-only --balanced-styles --batch-size 64 \
  --resume checkpoints/fontgen-style-v4.pt --reset-best \
  --epochs 49 --samples-per-epoch 8192 --validation-batches 32 \
  --output checkpoints/fontgen-style-v4.1-autotagged.pt
```

The completed epochs 47–49 improved the new validation loss from 3.46524 to 3.02412. The exported
inference checkpoint is `checkpoints/fontgen-style-v4.1-autotagged-inference.pt` (id
`f910e032b6d9`). The original v4 checkpoints are retained for rollback.

### V4.1 SDF geometry fine-tune

V4.1 can now continue from the existing checkpoint with a signed-distance target. The zero level
set gives a sub-pixel boundary, while normal, Eikonal, curvature and multi-scale losses penalize
crooked stems and local blobs. A zero-initialized coordinate-aware residual refiner keeps old
checkpoints load-compatible. Train the curriculum in order and reset the validation baseline when
the stage changes:

```bash
python scripts/train.py data/fontgen-v4-autotagged-raster.jsonl \
  --raster-only --balanced-styles --curriculum-stage anatomy \
  --resume checkpoints/fontgen-style-v4.1-autotagged-inference.pt --reset-best \
  --learning-rate 5e-5 --batch-size 16 --samples-per-epoch 8192 \
  --validation-batches 32 --epochs 53 --output checkpoints/fontgen-v4.1-sdf-anatomy.pt

python scripts/train.py data/fontgen-v4-autotagged-raster.jsonl \
  --raster-only --balanced-styles --curriculum-stage axes \
  --resume checkpoints/fontgen-v4.1-sdf-anatomy.pt --reset-best \
  --learning-rate 5e-5 --batch-size 16 --samples-per-epoch 8192 \
  --validation-batches 32 --epochs 57 --output checkpoints/fontgen-v4.1-sdf-axes.pt

python scripts/train.py data/fontgen-v4-autotagged-raster.jsonl \
  --raster-only --balanced-styles --curriculum-stage full \
  --resume checkpoints/fontgen-v4.1-sdf-axes.pt --reset-best \
  --learning-rate 3e-5 --batch-size 16 --samples-per-epoch 8192 \
  --validation-batches 32 --epochs 61 --output checkpoints/fontgen-v4.1-sdf.pt
```

`--epochs` is the absolute final epoch stored in the checkpoint. The upgraded network has
4,652,178 parameters and remains below the five-million V4 limit.

### Expanded local corpus

The local macOS font library can be added without copying font binaries into the repository. Files
marked `Trial` or `Demo` are excluded, vector commands are extracted directly, and provenance is
stored per row:

```bash
python scripts/prepare_dataset.py /Users/alexeyrazuvaev/Library/Fonts \
  data/local-fonts-v5.jsonl --faces-per-family 2 --require-license '' \
  --vector-only --exclude-trial --source-label local-font-library
python scripts/merge_manifests.py data/google-hires.jsonl data/local-fonts-v5.jsonl \
  --output data/fontgen-v5-expanded.jsonl
python scripts/prepare_v5_embeddings.py \
  data/fontgen-v5-expanded.jsonl data/v5-expanded-prompt-embeddings.pt
```

Current expanded manifest: 97,347 glyphs, 496 families and 767 faces. Of these, 58,217 unique
rows come from the local font library and 39,130 from Google Fonts. V5 uses sqrt-balanced style
sampling by default so that the larger sans-serif share does not drown out serif, display,
handwriting and monospace examples.

Enrich the merged vectors with measured typography and regenerate semantic embeddings:

```bash
python scripts/enrich_manifest.py \
  data/fontgen-v5-expanded.jsonl data/fontgen-v5-autotagged.jsonl
python scripts/prepare_v5_embeddings.py \
  data/fontgen-v5-autotagged.jsonl data/v5-autotagged-prompt-embeddings.pt
```

The enriched corpus has 10,103 unique prompts (12 Russian and 8 English templates per face) and
ten normalized tag groups: category, subclass, weight, width, x-height, contrast, roundness,
curve construction, outline complexity and slant. Numeric geometry includes mean glyph width,
x-height/cap-height ratio, descender depth, curve ratio, command complexity and contour count.
V5 predicts these measurements through an auxiliary typography head, making prompt grounding an
explicit training objective instead of relying only on outline reconstruction.

## Structured quadratic track

The quality-oriented successor removes both raster tracing and autoregressive coordinate
accumulation. It follows this hierarchy:

```text
multilingual prompt + controls
        ↓
shared family latent ← V4.1 prompt/style distillation
        ↓
glyph id → structure decoder (M/L/Q/Z)
        ↓
parallel quadratic geometry decoder
        ↓
whole-glyph vector refinement
```

All mixed cubic/quadratic outlines are canonicalized to quadratic Bézier segments with a maximum
conversion error of 0.001 em. Straight cubic/quadratic segments are reduced to real `L` commands.
The geometry loss samples points along each curve and separately penalizes curvature plus deviation
from target horizontal and vertical lines.

```bash
python scripts/canonicalize_quadratic.py \
  data/fontgen-v5-autotagged.jsonl data/fontgen-structured-quadratic.jsonl \
  --max-commands 192 --max-error 0.001
```

V4.1 acts only as a frozen teacher for family style and category distributions. Its raster and
cubic contour decoders are deliberately not copied; structure and quadratic geometry learn from
the canonical vector targets. The structured model has 25,118,353 trainable parameters. The canonical corpus retains all 97,347
glyphs: 935,965 true line segments and 1,423,158 quadratic segments. This training track is paused;
V4.1 remains the active UI model and is improved locally at inference time.

## Definition of done for v0

- no source outline is passed during inference;
- one prompt produces a coherent Latin/Cyrillic family, not unrelated glyphs;
- generated contours pass topology validation and compile to OTF/TTF;
- prompt changes are visible in blind A/B tests;
- nearest-neighbor checks reject near-copies of training glyphs;
- the model service returns generated outline JSON that the browser editor can render and export.
