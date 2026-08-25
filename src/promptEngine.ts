export type FontBaseId =
  | "inter"
  | "space-grotesk"
  | "playfair-display"
  | "roboto-mono"
  | "bebas-neue"
  | "pacifico"
  | "rubik"
  | "pt-serif"
  | "russo-one";

export type GeneratedRecipe = {
  baseId: FontBaseId;
  familyName: string;
  classification: string;
  width: number;
  slant: number;
  contrast: number;
  roundness: number;
  tracking: number;
  description: string;
  tags: string[];
};

const has = (value: string, words: string[]) => words.some((word) => value.includes(word));

export function generateRecipe(prompt: string): GeneratedRecipe {
  const value = prompt.toLocaleLowerCase("ru");
  let baseId: FontBaseId = "space-grotesk";
  let classification = "Neo-grotesk";
  let width = 100;
  let slant = 0;
  let contrast = 28;
  let roundness = 35;
  let tracking = 0;
  const tags: string[] = [];

  if (has(value, ["антикв", "serif", "редакцион", "editorial", "книжн", "fashion", "модн"])) {
    baseId = has(value, ["книжн", "book", "спокойн"]) ? "pt-serif" : "playfair-display";
    classification = "Transitional serif";
    contrast = 76;
    roundness = 22;
    tags.push("serif");
  } else if (has(value, ["моно", "mono", "код", "terminal", "технич", "industrial"])) {
    baseId = "roboto-mono";
    classification = "Monospaced";
    width = 94;
    roundness = 20;
    tags.push("mono");
  } else if (has(value, ["рукопис", "script", "каллиграф", "handwritten", "подпись"])) {
    baseId = "pacifico";
    classification = "Display script";
    slant = -5;
    contrast = 54;
    tags.push("script");
  } else if (has(value, ["плакат", "poster", "condensed", "узк", "афиш", "headline"])) {
    baseId = has(value, ["рус", "кирил", "бруталь", "brutal"]) ? "russo-one" : "bebas-neue";
    classification = "Condensed display";
    width = 73;
    contrast = 18;
    tracking = 24;
    tags.push("display");
  } else if (has(value, ["дружелюб", "friendly", "мягк", "soft", "rounded", "кругл"])) {
    baseId = "rubik";
    classification = "Rounded sans";
    roundness = 82;
    tags.push("rounded");
  } else if (has(value, ["нейтраль", "neutral", "ui", "интерфейс", "системн"])) {
    baseId = "inter";
    classification = "UI neo-grotesk";
    contrast = 14;
    roundness = 30;
    tags.push("ui");
  }

  if (has(value, ["широк", "wide", "extended"])) width = 122;
  if (has(value, ["очень узк", "ultra condensed"])) width = 62;
  if (has(value, ["italic", "курсив", "наклон", "динамич"])) slant = -11;
  if (has(value, ["разреж", "airy", "воздух"])) tracking = 70;
  if (has(value, ["плотн", "tight", "компакт"])) tracking = -34;
  if (has(value, ["контраст", "contrast"])) contrast = Math.max(contrast, 70);
  if (has(value, ["геометр", "geometric", "швейцар", "swiss"])) tags.push("geometric");

  const seed = Array.from(value).reduce((sum, char) => sum + char.charCodeAt(0), 0);
  const suffixes = ["Form", "Grotesk", "Studio", "Type", "Matter", "Signal"];
  const prefix = has(value, ["космос", "space"]) ? "Orbit" : has(value, ["мода", "fashion"]) ? "Mode" : has(value, ["город", "urban"]) ? "Metro" : "Prompt";
  const familyName = `${prefix} ${suffixes[seed % suffixes.length]}`;

  return {
    baseId,
    familyName,
    classification,
    width,
    slant,
    contrast,
    roundness,
    tracking,
    description: `${classification}. Пропорции ${width < 90 ? "сжатые" : width > 110 ? "расширенные" : "нормальные"}, ${slant ? "динамический наклон" : "прямое начертание"}, трекинг ${tracking === 0 ? "нейтральный" : tracking > 0 ? "свободный" : "плотный"}.`,
    tags: tags.length ? tags : ["sans", "contemporary"],
  };
}
