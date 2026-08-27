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
  sourceKind?: "prototype" | "model";
};

type PointKey = "x" | "x1" | "x2";
type NumericCommand = Record<string, number | string>;

export async function loadFont(url: string): Promise<OTFont> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Font loading failed: ${response.status}`);
  return opentype.parse(await response.arrayBuffer());
}

function roundLinearContours(source: OTPath, roundness: number) {
  if (roundness <= 0) return source;
  const output = new opentype.Path();
  let contour: typeof source.commands = [];
  const flush = (closed: boolean) => {
    if (!contour.length) return;
    const isLinear = closed && contour.length >= 3 && contour.every((command) => command.type === "M" || command.type === "L");
    if (!isLinear) {
      output.commands.push(...contour.map((command) => ({ ...command })));
      if (closed) output.commands.push({ type: "Z" });
      contour = [];
      return;
    }
    const points = contour.map((command) => ({ x: Number((command as unknown as NumericCommand).x), y: Number((command as unknown as NumericCommand).y) }));
    const fraction = .015 + Math.pow(roundness / 100, 1.2) * .17;
    const corners = points.map((point, index) => {
      const previous = points[(index - 1 + points.length) % points.length];
      const next = points[(index + 1) % points.length];
      return {
        point,
        incoming: { x: point.x + (previous.x - point.x) * fraction, y: point.y + (previous.y - point.y) * fraction },
        outgoing: { x: point.x + (next.x - point.x) * fraction, y: point.y + (next.y - point.y) * fraction },
      };
    });
    output.commands.push({ type: "M", x: corners[0].outgoing.x, y: corners[0].outgoing.y });
    for (let index = 1; index < corners.length; index += 1) {
      const corner = corners[index];
      output.commands.push({ type: "L", x: corner.incoming.x, y: corner.incoming.y });
      output.commands.push({ type: "Q", x1: corner.point.x, y1: corner.point.y, x: corner.outgoing.x, y: corner.outgoing.y });
    }
    output.commands.push({ type: "L", x: corners[0].incoming.x, y: corners[0].incoming.y });
    output.commands.push({ type: "Q", x1: corners[0].point.x, y1: corners[0].point.y, x: corners[0].outgoing.x, y: corners[0].outgoing.y });
    output.commands.push({ type: "Z" });
    contour = [];
  };
  for (const command of source.commands) {
    if (command.type === "M" && contour.length) flush(false);
    if (command.type === "Z") flush(true);
    else contour.push(command);
  }
  flush(false);
  return output;
}

type Profile = { min: number[]; max: number[] };
const PROFILE_BINS = 48;

function pathProfile(path: OTPath): Profile {
  const bounds = path.getBoundingBox();
  const width = Math.max(1, bounds.x2 - bounds.x1);
  const height = Math.max(1, bounds.y2 - bounds.y1);
  const min = Array(PROFILE_BINS).fill(Number.POSITIVE_INFINITY) as number[];
  const max = Array(PROFILE_BINS).fill(Number.NEGATIVE_INFINITY) as number[];
  for (const raw of path.commands) {
    const command = raw as unknown as NumericCommand;
    for (const key of ["x", "x1", "x2"] as PointKey[]) {
      const yKey = key === "x" ? "y" : key === "x1" ? "y1" : "y2";
      const x = Number(command[key]);
      const y = Number(command[yKey]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const normalizedY = Math.max(0, Math.min(.9999, (y - bounds.y1) / height));
      const bin = Math.floor(normalizedY * PROFILE_BINS);
      const normalizedX = (x - bounds.x1) / width;
      min[bin] = Math.min(min[bin], normalizedX);
      max[bin] = Math.max(max[bin], normalizedX);
    }
  }
  for (let index = 0; index < PROFILE_BINS; index += 1) {
    if (Number.isFinite(min[index])) continue;
    let distance = 1;
    while (distance < PROFILE_BINS) {
      const before = index - distance;
      const after = index + distance;
      const found = before >= 0 && Number.isFinite(min[before]) ? before : after < PROFILE_BINS && Number.isFinite(min[after]) ? after : -1;
      if (found >= 0) { min[index] = min[found]; max[index] = max[found]; break; }
      distance += 1;
    }
    if (!Number.isFinite(min[index])) { min[index] = 0; max[index] = 1; }
  }
  // A small vertical blur prevents bin edges from becoming visible in curves.
  for (let pass = 0; pass < 5; pass += 1) {
    const nextMin = [...min], nextMax = [...max];
    for (let index = 1; index < PROFILE_BINS - 1; index += 1) {
      nextMin[index] = (min[index - 1] + min[index] * 2 + min[index + 1]) / 4;
      nextMax[index] = (max[index - 1] + max[index] * 2 + max[index + 1]) / 4;
    }
    min.splice(0, min.length, ...nextMin); max.splice(0, max.length, ...nextMax);
  }
  return { min, max };
}

function sampleProfile(profile: Profile, normalizedY: number) {
  const position = Math.max(0, Math.min(PROFILE_BINS - 1.001, normalizedY * (PROFILE_BINS - 1)));
  const lower = Math.floor(position);
  const upper = Math.min(PROFILE_BINS - 1, lower + 1);
  const linearMix = position - lower;
  const mix = linearMix * linearMix * (3 - 2 * linearMix);
  return {
    min: profile.min[lower] * (1 - mix) + profile.min[upper] * mix,
    max: profile.max[lower] * (1 - mix) + profile.max[upper] * mix,
  };
}

export function synthesizePath(source: OTPath, references: OTPath[], settings: Omit<TransformSettings, "familyName" | "tracking" | "kerning">) {
  if (settings.sourceKind === "model") {
    const path = new opentype.Path();
    path.commands.push(...source.commands.map((command) => ({ ...command })));
    return path;
  }
  const path = new opentype.Path();
  const geometry = roundLinearContours(source, settings.roundness);
  const bounds = geometry.getBoundingBox();
  const width = Math.max(1, bounds.x2 - bounds.x1);
  const height = Math.max(1, bounds.y2 - bounds.y1);
  const centerX = (bounds.x1 + bounds.x2) / 2;
  const centerY = (bounds.y1 + bounds.y2) / 2;
  const xScale = settings.width / 100;
  const styleSlant = settings.slant + (settings.style.italic ? -10 : 0);
  const shear = Math.tan((-styleSlant * Math.PI) / 180);
  const weightGain = (settings.style.weight - 400) / 500 * .075;
  const contrastGain = (settings.contrast - 50) / 50 * .2;
  const phase = (settings.morphSeed % 997) / 997 * Math.PI * 2;
  const organic = .35 + (settings.morphSeed % 7) * .13;
  const sourceProfile = pathProfile(geometry);
  const referenceProfiles = references.filter((path) => path.commands.length > 0).map(pathProfile);
  const sourceAspect = width / height;
  const targetAspect = references.length
    ? references.reduce((sum, path, index) => { const box = path.getBoundingBox(); return sum + Math.max(.15, (box.x2 - box.x1) / Math.max(1, box.y2 - box.y1)) / (index + 1); }, 0) / references.reduce((sum, _path, index) => sum + 1 / (index + 1), 0)
    : sourceAspect;
  const constructionAnchors = [.15, .5, .85].map((position) => {
    const own = sampleProfile(sourceProfile, position);
    let targetMin = own.min, targetMax = own.max, weightTotal = 1;
    referenceProfiles.forEach((profile, index) => {
      const weight = 1 / (index + 2);
      const band = sampleProfile(profile, position);
      targetMin += band.min * weight; targetMax += band.max * weight; weightTotal += weight;
    });
    targetMin /= weightTotal; targetMax /= weightTotal;
    const ownSpan = Math.max(.08, own.max - own.min);
    const targetSpan = Math.max(.08, targetMax - targetMin);
    return {
      scale: Math.max(.82, Math.min(1.18, targetSpan / ownSpan)),
      shift: ((targetMin + targetMax) - (own.min + own.max)) / 2,
    };
  });
  const quadratic = (a: number, b: number, c: number, position: number) => a * (1 - position) * (1 - position) + 2 * b * (1 - position) * position + c * position * position;

  const transform = (x: number, y: number) => {
    const normalizedY = Math.max(0, Math.min(1, (y - bounds.y1) / height));
    const constructionScale = Math.max(.72, Math.min(1.32, targetAspect / sourceAspect));
    const profileScale = quadratic(constructionAnchors[0].scale, constructionAnchors[1].scale, constructionAnchors[2].scale, normalizedY);
    const profileShift = quadratic(constructionAnchors[0].shift, constructionAnchors[1].shift, constructionAnchors[2].shift, normalizedY) * width;
    const stress = Math.cos(normalizedY * Math.PI * 2 + phase * .17);
    const localWidth = xScale * (1 + contrastGain * stress);
    const reconstructedX = centerX + (x - centerX) * profileScale + profileShift * .34;
    const promptWarp = Math.sin((x - bounds.x1) / width * Math.PI * 2 + normalizedY * 1.5 + phase) * organic;
    return {
      x: centerX * xScale + (reconstructedX - centerX) * (localWidth * constructionScale + weightGain) + y * shear + promptWarp,
      y: centerY + (y - centerY) * (1 + weightGain * .12),
    };
  };

  geometry.commands.forEach((raw) => {
    const command = raw as unknown as NumericCommand;
    const next = { ...raw } as typeof raw;
    for (const key of ["x", "x1", "x2"] as PointKey[]) {
      const yKey = key === "x" ? "y" : key === "x1" ? "y1" : "y2";
      const x = Number(command[key]);
      const y = Number(command[yKey]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const point = transform(x, y);
      (next as unknown as NumericCommand)[key] = point.x;
      (next as unknown as NumericCommand)[yKey] = point.y;
    }
    path.commands.push(next);
  });
  return path;
}

function matchingReferencePaths(sources: OTFont[], unicode: number | undefined) {
  if (unicode === undefined) return [];
  const character = String.fromCodePoint(unicode);
  return sources.slice(1).map((font) => font.charToGlyph(character)).filter((glyph) => glyph.index !== 0 && glyph.path.commands.length > 0).map((glyph) => glyph.path);
}

export function buildFont(sources: OTFont[], settings: TransformSettings): OTFont {
  const source = sources[0];
  if (!source) throw new Error("No construction sources loaded");
  const glyphs: OTGlyph[] = [];
  for (let index = 0; index < source.glyphs.length; index += 1) {
    const original = source.glyphs.get(index);
    const glyphName = original.unicode === 0 ? ".null" : (original.name ?? undefined);
    glyphs.push(new opentype.Glyph({
      name: glyphName,
      unicode: original.unicode,
      unicodes: original.unicodes,
      advanceWidth: Math.max(0, Math.round((original.advanceWidth ?? source.unitsPerEm) * (settings.sourceKind === "model" ? 1 : settings.width / 100) + settings.tracking + (settings.sourceKind === "model" ? 0 : (settings.style.weight - 400) * .08))),
      path: synthesizePath(original.path, matchingReferencePaths(sources, original.unicode), settings),
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

export async function exportFont(sources: OTFont[], settings: TransformSettings, type: "otf" | "ttf") {
  const generated = buildFont(sources, settings);
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

export function createTextSvg(sources: OTFont[], settings: TransformSettings, text: string) {
  const generated = buildFont(sources, settings);
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

export function exportTextSvg(sources: OTFont[], settings: TransformSettings, text: string) {
  const { svg, filename } = createTextSvg(sources, settings, text);
  download(svg, filename, "image/svg+xml;charset=utf-8");
}

export function drawPreview(canvas: HTMLCanvasElement, fonts: OTFont[], text: string, settings: Omit<TransformSettings, "familyName">, fontSize: number, showGuides: boolean) {
  const font = fonts[0];
  if (!font) return;
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
    const kern = next ? (settings.kerning[pair] ?? (settings.sourceKind === "model" ? 0 : font.getKerningValue(glyph, next))) : 0;
    return ((glyph.advanceWidth ?? font.unitsPerEm) * (settings.sourceKind === "model" ? 1 : settings.width / 100) + settings.tracking + (settings.sourceKind === "model" ? 0 : (settings.style.weight - 400) * .08) + kern) * scale;
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
    const path = synthesizePath(glyph.path, matchingReferencePaths(fonts, glyph.unicode), settings);
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
