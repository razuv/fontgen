import opentype, { type Font } from "opentype.js";

type ModelCommand = "M" | "L" | "Q" | "C" | "Z";
type ModelGlyph = {
  character: string;
  commands: ModelCommand[];
  coordinates: number[][];
  advance_width: number;
  left_side_bearing: number;
};
type ModelResponse = {
  family_name: string;
  units_per_em: number;
  ascender: number;
  descender: number;
  checkpoint: string;
  glyphs: ModelGlyph[];
};

export type ModelControls = { weight: number; width: number; contrast: number; roundness: number; slant: number };
export const MODEL_CHARSET = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789.,:;!?—-()«»@&%+=";

export const modelApiUrl = (import.meta.env.VITE_MODEL_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

function pathFromGlyph(glyph: ModelGlyph, unitsPerEm: number) {
  const path = new opentype.Path();
  glyph.commands.forEach((command, index) => {
    const values = glyph.coordinates[index]?.map((value) => value * unitsPerEm) ?? [];
    if (command === "M") path.moveTo(values[0], values[1]);
    else if (command === "L") path.lineTo(values[0], values[1]);
    else if (command === "Q") path.quadTo(values[0], values[1], values[2], values[3]);
    else if (command === "C") path.curveTo(values[0], values[1], values[2], values[3], values[4], values[5]);
    else if (command === "Z") path.close();
  });
  return path;
}

function fontFromResponse(response: ModelResponse): Font {
  const notdef = new opentype.Glyph({ name: ".notdef", unicode: 0, advanceWidth: 600, path: new opentype.Path() });
  const glyphs = response.glyphs.map((glyph) => new opentype.Glyph({
    name: glyph.character === " " ? "space" : undefined,
    unicode: glyph.character.codePointAt(0),
    advanceWidth: Math.round(glyph.advance_width * response.units_per_em),
    leftSideBearing: Math.round(glyph.left_side_bearing * response.units_per_em),
    path: glyph.character === " " ? new opentype.Path() : pathFromGlyph(glyph, response.units_per_em),
  }));
  return new opentype.Font({
    familyName: response.family_name,
    styleName: "Regular",
    unitsPerEm: response.units_per_em,
    ascender: response.ascender,
    descender: response.descender,
    glyphs: [notdef, ...glyphs],
  });
}

export async function generateWithModel(input: {
  prompt: string;
  familyName: string;
  characters?: string;
  controls: ModelControls;
  seed: number;
}): Promise<{ font: Font; checkpoint: string }> {
  if (!modelApiUrl) throw new Error("MODEL_API_NOT_CONFIGURED");
  const response = await fetch(`${modelApiUrl}/v1/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt: input.prompt,
      family_name: input.familyName,
      characters: Array.from(new Set(Array.from(`${MODEL_CHARSET}${input.characters ?? ""}`))).join(""),
      controls: input.controls,
      seed: input.seed,
    }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(payload?.detail || `MODEL_API_${response.status}`);
  }
  const payload = await response.json() as ModelResponse;
  return { font: fontFromResponse(payload), checkpoint: payload.checkpoint };
}

