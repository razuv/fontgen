import opentype, { type Font as OTFont, type Glyph as OTGlyph, type Path as OTPath } from "opentype.js";

export type FontStyle = { id: string; name: string; weight: number; italic: boolean };
export type TransformSettings = {
  familyName: string;
  width: number;
  slant: number;
  contrast: number;
  roundness: number;
  tracking: number;
  morphSeed: number;
  style: FontStyle;
  kerning: Record<string, number>;
};

type PointKey = "x" | "x1" | "x2";
type NumericCommand = Record<string, number | string>;

export async function loadFont(url: string): Promise<OTFont> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Font loading failed: ${response.status}`);
  return opentype.parse(await response.arrayBuffer());
}

function commandPoint(command: NumericCommand) {
  const x = Number(command.x);
  const y = Number(command.y);
  return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
}

function smoothingDeltas(source: OTPath, amount: number) {
  const deltas = new Map<number, { x: number; y: number }>();
  if (amount <= 0) return deltas;
  const contours: number[][] = [];
  let contour: number[] = [];

  source.commands.forEach((raw, index) => {
    const command = raw as unknown as NumericCommand;
    if (command.type === "M" && contour.length) { contours.push(contour); contour = []; }
    if (command.type !== "Z" && commandPoint(command)) contour.push(index);
    if (command.type === "Z" && contour.length) { contours.push(contour); contour = []; }
  });
  if (contour.length) contours.push(contour);

  for (const indices of contours) {
    if (indices.length < 3) continue;
    indices.forEach((index, position) => {
      const previous = commandPoint(source.commands[indices[(position - 1 + indices.length) % indices.length]] as unknown as NumericCommand)!;
      const current = commandPoint(source.commands[index] as unknown as NumericCommand)!;
      const next = commandPoint(source.commands[indices[(position + 1) % indices.length]] as unknown as NumericCommand)!;
      const targetX = (previous.x + current.x * 2 + next.x) / 4;
      const targetY = (previous.y + current.y * 2 + next.y) / 4;
      deltas.set(index, { x: (targetX - current.x) * amount, y: (targetY - current.y) * amount });
    });
  }
  return deltas;
}

export function synthesizePath(source: OTPath, settings: Omit<TransformSettings, "familyName" | "tracking" | "kerning">) {
  const path = new opentype.Path();
  const bounds = source.getBoundingBox();
  const width = Math.max(1, bounds.x2 - bounds.x1);
  const height = Math.max(1, bounds.y2 - bounds.y1);
  const centerX = (bounds.x1 + bounds.x2) / 2;
  const centerY = (bounds.y1 + bounds.y2) / 2;
  const xScale = settings.width / 100;
  const styleSlant = settings.slant + (settings.style.italic ? -10 : 0);
  const shear = Math.tan((-styleSlant * Math.PI) / 180);
  const weightGain = (settings.style.weight - 400) / 500 * .075;
  const contrastGain = (settings.contrast - 50) / 50 * .2;
  const roundAmount = Math.pow(settings.roundness / 100, 1.35) * .72;
  const deltas = smoothingDeltas(source, roundAmount);
  const phase = (settings.morphSeed % 997) / 997 * Math.PI * 2;
  const organic = 2 + (settings.morphSeed % 7);

  const transform = (x: number, y: number) => {
    const normalizedY = (y - bounds.y1) / height;
    const stress = Math.cos(normalizedY * Math.PI * 2 + phase * .17);
    const localWidth = xScale * (1 + contrastGain * stress);
    const promptWarp = Math.sin((x - bounds.x1) / width * Math.PI * 3 + normalizedY * 2 + phase) * organic;
    return {
      x: centerX * xScale + (x - centerX) * (localWidth + weightGain) + y * shear + promptWarp,
      y: centerY + (y - centerY) * (1 + weightGain * .16) + Math.cos(normalizedY * Math.PI * 3 + phase) * organic * .35,
    };
  };

  source.commands.forEach((raw, index) => {
    const command = raw as unknown as NumericCommand;
    const next = { ...raw } as typeof raw;
    const endpointDelta = deltas.get(index) ?? { x: 0, y: 0 };
    for (const key of ["x", "x1", "x2"] as PointKey[]) {
      const yKey = key === "x" ? "y" : key === "x1" ? "y1" : "y2";
      const x = Number(command[key]);
      const y = Number(command[yKey]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const deltaFactor = key === "x" ? 1 : key === "x1" ? .35 : .7;
      const point = transform(x + endpointDelta.x * deltaFactor, y + endpointDelta.y * deltaFactor);
      (next as unknown as NumericCommand)[key] = point.x;
      (next as unknown as NumericCommand)[yKey] = point.y;
    }
    path.commands.push(next);
  });
  return path;
}

export function buildFont(source: OTFont, settings: TransformSettings): OTFont {
  const glyphs: OTGlyph[] = [];
  for (let index = 0; index < source.glyphs.length; index += 1) {
    const original = source.glyphs.get(index);
    const glyphName = original.unicode === 0 ? ".null" : (original.name ?? undefined);
    glyphs.push(new opentype.Glyph({
      name: glyphName,
      unicode: original.unicode,
      unicodes: original.unicodes,
      advanceWidth: Math.max(0, Math.round((original.advanceWidth ?? source.unitsPerEm) * settings.width / 100 + settings.tracking + (settings.style.weight - 400) * .08)),
      path: synthesizePath(original.path, settings),
    }));
  }

  const result = new opentype.Font({
    familyName: settings.familyName.trim() || "Untitled Fontgen",
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

function safeName(value: string) {
  return value.trim().replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-|-$/g, "") || "Fontgen";
}

function download(data: BlobPart, name: string, type: string) {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

export async function exportFont(source: OTFont, settings: TransformSettings, type: "otf" | "ttf") {
  const generated = buildFont(source, settings);
  const otf = generated.toArrayBuffer();
  const filename = `${safeName(settings.familyName)}-${safeName(settings.style.name)}`;
  if (type === "otf") {
    download(otf, `${filename}.otf`, "font/otf");
    return;
  }
  const { createFont } = await import("fonteditor-core");
  const editorFont = createFont(otf, { type: "otf", compound2simple: true, kerning: true });
  download(editorFont.write({ type: "ttf", hinting: false }) as BlobPart, `${filename}.ttf`, "font/ttf");
}

export function createTextSvg(source: OTFont, settings: TransformSettings, text: string) {
  const generated = buildFont(source, settings);
  const content = text || "Fontgen";
  const path = generated.getPath(content, 0, 0, generated.unitsPerEm, { kerning: true });
  const box = path.getBoundingBox();
  const padding = Math.round(generated.unitsPerEm * .06);
  const width = Math.max(1, Math.ceil(box.x2 - box.x1 + padding * 2));
  const height = Math.max(1, Math.ceil(box.y2 - box.y1 + padding * 2));
  const viewX = Math.floor(box.x1 - padding);
  const viewY = Math.floor(box.y1 - padding);
  const title = `${settings.familyName} — ${content}`.replace(/[<>&"]/g, "");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewX} ${viewY} ${width} ${height}" width="${width}" height="${height}"><title>${title}</title><path fill="#000" d="${path.toPathData(2)}"/></svg>`;
  return { svg, filename: `${safeName(settings.familyName)}-${safeName(content.slice(0, 28))}.svg` };
}

export function exportTextSvg(source: OTFont, settings: TransformSettings, text: string) {
  const { svg, filename } = createTextSvg(source, settings, text);
  download(svg, filename, "image/svg+xml;charset=utf-8");
}

export function drawPreview(canvas: HTMLCanvasElement, font: OTFont, text: string, settings: Omit<TransformSettings, "familyName">, fontSize: number, showGuides: boolean) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const scale = fontSize / font.unitsPerEm;
  const baseline = rect.height / 2 + fontSize * .32;
  const characters = Array.from(text || "Fontgen");
  const glyphs = characters.map((char) => font.charToGlyph(char));
  const advances = glyphs.map((glyph, index) => {
    const next = glyphs[index + 1];
    const pair = `${characters[index] ?? ""}${characters[index + 1] ?? ""}`;
    const kern = next ? (settings.kerning[pair] ?? font.getKerningValue(glyph, next)) : 0;
    return ((glyph.advanceWidth ?? font.unitsPerEm) * settings.width / 100 + settings.tracking + (settings.style.weight - 400) * .08 + kern) * scale;
  });
  const total = advances.reduce((sum, value) => sum + value, 0);
  let x = Math.max(30, (rect.width - total) / 2);

  if (showGuides) {
    ctx.save(); ctx.setLineDash([4, 6]); ctx.lineWidth = 1; ctx.font = "9px JetBrains Mono, monospace";
    const guides = [[baseline, "BASELINE"], [baseline - fontSize * .52, "X-HEIGHT"], [baseline - fontSize * .72, "CAP HEIGHT"]] as const;
    for (const [y, label] of guides) {
      ctx.strokeStyle = label === "BASELINE" ? "rgba(220,255,82,.35)" : "rgba(255,255,255,.1)";
      ctx.beginPath(); ctx.moveTo(18, y); ctx.lineTo(rect.width - 18, y); ctx.stroke();
      ctx.fillStyle = label === "BASELINE" ? "#b8d83f" : "#626262"; ctx.fillText(label, 24, y - 5);
    }
    ctx.restore();
  }

  ctx.fillStyle = "#f1f0eb";
  glyphs.forEach((glyph, index) => {
    const path = synthesizePath(glyph.path, settings);
    ctx.save(); ctx.translate(x, baseline); ctx.scale(scale, -scale); ctx.beginPath();
    for (const command of path.commands) {
      if (command.type === "M") ctx.moveTo(command.x, command.y);
      else if (command.type === "L") ctx.lineTo(command.x, command.y);
      else if (command.type === "C") ctx.bezierCurveTo(command.x1, command.y1, command.x2, command.y2, command.x, command.y);
      else if (command.type === "Q") ctx.quadraticCurveTo(command.x1, command.y1, command.x, command.y);
      else if (command.type === "Z") ctx.closePath();
    }
    ctx.fill(); ctx.restore(); x += advances[index];
  });
}
