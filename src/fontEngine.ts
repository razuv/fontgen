import opentype, { type Font as OTFont, type Glyph as OTGlyph, type Path as OTPath } from "opentype.js";

export type FontStyle = { id: string; name: string; weight: number; italic: boolean };
export type TransformSettings = {
  familyName: string;
  width: number;
  slant: number;
  tracking: number;
  style: FontStyle;
  kerning: Record<string, number>;
};

export async function loadFont(url: string): Promise<OTFont> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Font loading failed: ${response.status}`);
  return opentype.parse(await response.arrayBuffer());
}

function transformedPath(source: OTPath, xScale: number, slant: number, weight: number) {
  const path = new opentype.Path();
  const shear = Math.tan((slant * Math.PI) / 180);
  const embolden = (weight - 400) / 1000;
  for (const command of source.commands) {
    const next = { ...command } as typeof command;
    const points: ("x" | "x1" | "x2")[] = ["x", "x1", "x2"];
    for (const key of points) {
      const yKey = key === "x" ? "y" : key === "x1" ? "y1" : "y2";
      const x = (command as unknown as Record<string, number>)[key];
      const y = (command as unknown as Record<string, number>)[yKey];
      if (Number.isFinite(x) && Number.isFinite(y)) {
        (next as unknown as Record<string, number>)[key] = x * xScale + y * shear + x * embolden * 0.12;
      }
    }
    path.commands.push(next);
  }
  return path;
}

export function buildFont(source: OTFont, settings: TransformSettings): OTFont {
  const xScale = settings.width / 100;
  const styleSlant = settings.slant + (settings.style.italic ? -10 : 0);
  const glyphs: OTGlyph[] = [];

  for (let index = 0; index < source.glyphs.length; index += 1) {
    const original = source.glyphs.get(index);
    const glyph = new opentype.Glyph({
      name: original.name ?? undefined,
      unicode: original.unicode,
      unicodes: original.unicodes,
      advanceWidth: Math.max(0, Math.round((original.advanceWidth ?? source.unitsPerEm) * xScale + settings.tracking)),
      path: transformedPath(original.getPath(0, 0, source.unitsPerEm), xScale, styleSlant, settings.style.weight),
    });
    glyphs.push(glyph);
  }

  const result = new opentype.Font({
    familyName: settings.familyName,
    styleName: settings.style.name,
    unitsPerEm: source.unitsPerEm,
    ascender: source.ascender,
    descender: source.descender,
    glyphs,
  });

  const pairs: Record<string, number> = {};
  for (const [pair, value] of Object.entries(settings.kerning)) {
    const [left, right] = Array.from(pair);
    if (!left || !right) continue;
    const leftGlyph = result.charToGlyph(left);
    const rightGlyph = result.charToGlyph(right);
    if (leftGlyph && rightGlyph) pairs[`${leftGlyph.index},${rightGlyph.index}`] = value;
  }
  result.kerningPairs = pairs;
  return result;
}

function download(data: BlobPart, name: string, type: string) {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function exportFont(source: OTFont, settings: TransformSettings, type: "otf" | "ttf" | "svg") {
  const generated = buildFont(source, settings);
  const otf = generated.toArrayBuffer();
  const slug = `${settings.familyName}-${settings.style.name}`.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "");
  if (type === "otf") {
    download(otf, `${slug}.otf`, "font/otf");
    return;
  }

  const { createFont } = await import("fonteditor-core");
  const editorFont = createFont(otf, { type: "otf", compound2simple: true, kerning: true });
  const output = editorFont.write({ type, hinting: false });
  download(output as BlobPart, `${slug}.${type}`, type === "ttf" ? "font/ttf" : "image/svg+xml");
}

export function drawPreview(
  canvas: HTMLCanvasElement,
  font: OTFont,
  text: string,
  settings: Omit<TransformSettings, "familyName">,
  fontSize: number,
  showGuides: boolean,
) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const scale = fontSize / font.unitsPerEm;
  const widthScale = settings.width / 100;
  const slant = settings.slant + (settings.style.italic ? -10 : 0);
  const shear = Math.tan((slant * Math.PI) / 180);
  const baseline = rect.height / 2 + fontSize * 0.32;
  const glyphs = Array.from(text).map((char) => font.charToGlyph(char));
  const total = glyphs.reduce((sum, glyph, index) => {
    const next = glyphs[index + 1];
    const kern = next ? (settings.kerning[`${text[index]}${text[index + 1]}`] ?? font.getKerningValue(glyph, next)) : 0;
    return sum + ((glyph.advanceWidth ?? font.unitsPerEm) * widthScale + settings.tracking + kern) * scale;
  }, 0);
  let x = Math.max(30, (rect.width - total) / 2);

  if (showGuides) {
    ctx.save();
    ctx.setLineDash([4, 6]);
    ctx.lineWidth = 1;
    const guides = [
      [baseline, "BASELINE"],
      [baseline - fontSize * 0.52, "X-HEIGHT"],
      [baseline - fontSize * 0.72, "CAP HEIGHT"],
    ] as const;
    ctx.font = "9px JetBrains Mono, monospace";
    for (const [y, label] of guides) {
      ctx.strokeStyle = label === "BASELINE" ? "rgba(220,255,82,.35)" : "rgba(255,255,255,.1)";
      ctx.beginPath(); ctx.moveTo(18, y); ctx.lineTo(rect.width - 18, y); ctx.stroke();
      ctx.fillStyle = label === "BASELINE" ? "#b8d83f" : "#626262";
      ctx.fillText(label, 24, y - 5);
    }
    ctx.restore();
  }

  ctx.fillStyle = "#f1f0eb";
  for (let index = 0; index < glyphs.length; index += 1) {
    const glyph = glyphs[index];
    const path = glyph.getPath(0, 0, font.unitsPerEm);
    ctx.save();
    ctx.translate(x, baseline);
    ctx.scale(scale, -scale);
    ctx.transform(widthScale, 0, shear, 1, 0, 0);
    ctx.beginPath();
    for (const command of path.commands) {
      if (command.type === "M") ctx.moveTo(command.x, command.y);
      else if (command.type === "L") ctx.lineTo(command.x, command.y);
      else if (command.type === "C") ctx.bezierCurveTo(command.x1, command.y1, command.x2, command.y2, command.x, command.y);
      else if (command.type === "Q") ctx.quadraticCurveTo(command.x1, command.y1, command.x, command.y);
      else if (command.type === "Z") ctx.closePath();
    }
    ctx.fill();
    ctx.restore();
    const next = glyphs[index + 1];
    const pair = `${text[index] ?? ""}${text[index + 1] ?? ""}`;
    const kern = next ? (settings.kerning[pair] ?? font.getKerningValue(glyph, next)) : 0;
    x += ((glyph.advanceWidth ?? font.unitsPerEm) * widthScale + settings.tracking + kern) * scale;
  }
}
