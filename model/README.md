# Fontgen Model

This directory is the beginning of a real trainable prompt-to-vector model. It is deliberately separate from the Vite app: GitHub Pages can host the editor, but not a multi-hundred-million-operation PyTorch inference process.

## What “from scratch” means here

At inference time the model receives only:

- a UTF-8 text prompt;
- a Unicode character id;
- requested weight, width, contrast, roundness and slant controls.

It does **not** receive a Google Font glyph or a source outline. `FontgenNet` encodes one family-level style vector and autoregressively predicts `M/L/Q/C/Z` commands plus Bézier coordinates for each character. Google Fonts are training examples, not templates used during generation.

## Model v0

```text
UTF-8 prompt -> byte Transformer -> family style latent ----+
                                                           |-> causal vector decoder -> Bézier sequence
Unicode id -----------------------> glyph embedding -------+
design controls ------------------> style conditioning ----+
```

The first useful checkpoint should target Latin, basic Cyrillic, digits and punctuation. Family consistency comes from reusing the same style latent for every glyph. A later v1 should add raster supervision with differentiable rendering and a prompt/style contrastive loss.

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
python scripts/train.py data/train.jsonl --output checkpoints/fontgen-v0.pt
```

For a wiring check only (this checkpoint has no visual quality):

```bash
python scripts/train.py data/smoke.jsonl --preset smoke --epochs 1 --batch-size 8 --output checkpoints/smoke.pt
```

Run the inference service after a checkpoint exists:

```bash
FONTGEN_CHECKPOINT=checkpoints/fontgen-v0.pt uvicorn fontgen_model.api:app --host 0.0.0.0 --port 8000
```

`GET /health` reports whether the checkpoint loaded. `POST /v1/generate` accepts a prompt, charset, seed and typography controls and returns normalized Bézier sequences. It intentionally returns `503` without a trained checkpoint instead of silently substituting an existing font.

The 12 fonts currently bundled in `public/fonts` are enough only for a pipeline smoke test. A useful model needs a much broader OFL corpus and a held-out family split, so the validation set measures novel-family generation instead of memorization.

## Definition of done for v0

- no source outline is passed during inference;
- one prompt produces a coherent Latin/Cyrillic family, not unrelated glyphs;
- generated contours pass topology validation and compile to OTF/TTF;
- prompt changes are visible in blind A/B tests;
- nearest-neighbor checks reject near-copies of training glyphs;
- the model service returns generated outline JSON that the browser editor can render and export.
