from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .inference import FontgenGenerator
from .structured_inference import StructuredFontGenerator
from .text import SUPPORTED_CHARACTERS

DEFAULT_CHARSET = SUPPORTED_CHARACTERS


class Controls(BaseModel):
    weight: int = Field(400, ge=100, le=900)
    width: int = Field(100, ge=60, le=135)
    contrast: int = Field(50, ge=0, le=100)
    roundness: int = Field(50, ge=0, le=100)
    slant: int = Field(0, ge=-18, le=18)

    def vector(self) -> list[float]:
        return [
            (self.weight - 400) / 500,
            (self.width - 100) / 40,
            (self.contrast - 50) / 50,
            (self.roundness - 50) / 50,
            self.slant / 18,
        ]


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    family_name: str = Field(default="Untitled Fontgen", min_length=1, max_length=64)
    characters: str = Field(default=DEFAULT_CHARSET, min_length=1, max_length=320)
    controls: Controls = Field(default_factory=Controls)
    seed: int = Field(default=17, ge=0, le=2**32 - 1)
    cfg_scale: float = Field(default=1.0, ge=1.0, le=5.0, description="Classifier-free guidance scale")

    @field_validator("characters")
    @classmethod
    def unique_characters(cls, value: str) -> str:
        return "".join(dict.fromkeys(value))


class GenerationResponse(BaseModel):
    family_name: str
    units_per_em: int = 1000
    ascender: int = 800
    descender: int = -200
    checkpoint: str
    architecture: str = "fontgen-style-film-vector-v4.1-local-refined"
    parameter_count: int
    glyphs: list[dict[str, object]]


class Generator(Protocol):
    checkpoint_id: str
    parameter_count: int
    architecture: str

    def generate_family(
        self, prompt: str, characters: str, controls: list[float], seed: int,
        *, cfg_scale: float = 1.0,
    ) -> list[object]: ...


def load_generator(checkpoint: Path) -> Generator:
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    architecture = metadata.get("architecture", "fontgen-style-film-vector-v4")
    if architecture == "structured-quadratic-v1":
        return StructuredFontGenerator(checkpoint)
    return FontgenGenerator(checkpoint)


generator: Generator | None = None
checkpoint_error: str | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global generator, checkpoint_error
    checkpoint = Path(os.environ.get("FONTGEN_CHECKPOINT", "checkpoints/fontgen-v0.pt"))
    try:
        generator = load_generator(checkpoint)
    except Exception as error:  # noqa: BLE001 - health must expose any checkpoint incompatibility
        checkpoint_error = f"{type(error).__name__}: {error}"
    yield


app = FastAPI(title="Fontgen Model API", version="0.1.0", lifespan=lifespan)
origins = [value.strip() for value in os.environ.get("FONTGEN_CORS_ORIGINS", "http://localhost:5173").split(",") if value.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "model_loaded": generator is not None,
        "checkpoint_error": checkpoint_error,
        "checkpoint": generator.checkpoint_id if generator is not None else None,
        "architecture": generator.architecture if generator is not None else None,
        "parameter_count": generator.parameter_count if generator is not None else None,
    }


@app.post("/v1/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest) -> GenerationResponse:
    if generator is None:
        raise HTTPException(status_code=503, detail="Fontgen checkpoint is not loaded. Train a checkpoint and set FONTGEN_CHECKPOINT.")
    glyphs = generator.generate_family(
        request.prompt, request.characters, request.controls.vector(), request.seed,
        cfg_scale=request.cfg_scale,
    )
    return GenerationResponse(
        family_name=request.family_name,
        checkpoint=generator.checkpoint_id,
        architecture=generator.architecture,
        parameter_count=generator.parameter_count,
        glyphs=[glyph.__dict__ for glyph in glyphs],
    )
