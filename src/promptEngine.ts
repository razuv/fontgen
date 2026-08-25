export type FontSeedId =
  | "inter"
  | "space-grotesk"
  | "playfair-display"
  | "roboto-mono"
  | "bebas-neue"
  | "pacifico"
  | "rubik"
  | "pt-serif"
  | "russo-one";

type Reference = {
  name: string;
  seed: FontSeedId;
  category: "sans" | "serif" | "mono" | "display" | "script";
  width: number;
  contrast: number;
  roundness: number;
};

// Metadata index of open families. It is intentionally not exposed as a
// picker: references are ranked from the prompt and only inform synthesis.
const referenceCatalog: Reference[] = [
  { name: "Inter", seed: "inter", category: "sans", width: 100, contrast: 12, roundness: 28 },
  { name: "Roboto", seed: "inter", category: "sans", width: 98, contrast: 18, roundness: 32 },
  { name: "Noto Sans", seed: "inter", category: "sans", width: 101, contrast: 16, roundness: 30 },
  { name: "IBM Plex Sans", seed: "inter", category: "sans", width: 96, contrast: 26, roundness: 18 },
  { name: "Source Sans 3", seed: "inter", category: "sans", width: 97, contrast: 20, roundness: 26 },
  { name: "Work Sans", seed: "inter", category: "sans", width: 102, contrast: 14, roundness: 22 },
  { name: "Manrope", seed: "space-grotesk", category: "sans", width: 106, contrast: 10, roundness: 48 },
  { name: "DM Sans", seed: "space-grotesk", category: "sans", width: 103, contrast: 12, roundness: 44 },
  { name: "Space Grotesk", seed: "space-grotesk", category: "sans", width: 105, contrast: 18, roundness: 30 },
  { name: "Montserrat", seed: "space-grotesk", category: "sans", width: 108, contrast: 8, roundness: 38 },
  { name: "Raleway", seed: "space-grotesk", category: "sans", width: 104, contrast: 34, roundness: 24 },
  { name: "Urbanist", seed: "space-grotesk", category: "sans", width: 107, contrast: 12, roundness: 46 },
  { name: "Outfit", seed: "space-grotesk", category: "sans", width: 108, contrast: 10, roundness: 52 },
  { name: "Rubik", seed: "rubik", category: "sans", width: 103, contrast: 12, roundness: 72 },
  { name: "Nunito", seed: "rubik", category: "sans", width: 105, contrast: 10, roundness: 88 },
  { name: "Quicksand", seed: "rubik", category: "sans", width: 110, contrast: 8, roundness: 92 },
  { name: "Comfortaa", seed: "rubik", category: "display", width: 112, contrast: 6, roundness: 96 },
  { name: "Onest", seed: "rubik", category: "sans", width: 101, contrast: 10, roundness: 66 },
  { name: "Roboto Mono", seed: "roboto-mono", category: "mono", width: 94, contrast: 16, roundness: 18 },
  { name: "JetBrains Mono", seed: "roboto-mono", category: "mono", width: 96, contrast: 10, roundness: 20 },
  { name: "IBM Plex Mono", seed: "roboto-mono", category: "mono", width: 93, contrast: 24, roundness: 12 },
  { name: "Source Code Pro", seed: "roboto-mono", category: "mono", width: 95, contrast: 18, roundness: 16 },
  { name: "Bebas Neue", seed: "bebas-neue", category: "display", width: 68, contrast: 14, roundness: 8 },
  { name: "Oswald", seed: "bebas-neue", category: "display", width: 74, contrast: 18, roundness: 10 },
  { name: "Barlow Condensed", seed: "bebas-neue", category: "display", width: 76, contrast: 20, roundness: 18 },
  { name: "Archivo Narrow", seed: "bebas-neue", category: "display", width: 78, contrast: 14, roundness: 16 },
  { name: "Russo One", seed: "russo-one", category: "display", width: 108, contrast: 8, roundness: 18 },
  { name: "Unbounded", seed: "russo-one", category: "display", width: 122, contrast: 10, roundness: 28 },
  { name: "Playfair Display", seed: "playfair-display", category: "serif", width: 102, contrast: 90, roundness: 16 },
  { name: "Cormorant Garamond", seed: "playfair-display", category: "serif", width: 94, contrast: 86, roundness: 12 },
  { name: "Bodoni Moda", seed: "playfair-display", category: "serif", width: 98, contrast: 96, roundness: 10 },
  { name: "DM Serif Display", seed: "playfair-display", category: "serif", width: 106, contrast: 82, roundness: 22 },
  { name: "PT Serif", seed: "pt-serif", category: "serif", width: 100, contrast: 58, roundness: 20 },
  { name: "Lora", seed: "pt-serif", category: "serif", width: 102, contrast: 64, roundness: 28 },
  { name: "Merriweather", seed: "pt-serif", category: "serif", width: 105, contrast: 54, roundness: 26 },
  { name: "Source Serif 4", seed: "pt-serif", category: "serif", width: 98, contrast: 62, roundness: 18 },
  { name: "Literata", seed: "pt-serif", category: "serif", width: 104, contrast: 66, roundness: 26 },
  { name: "Pacifico", seed: "pacifico", category: "script", width: 112, contrast: 54, roundness: 72 },
  { name: "Caveat", seed: "pacifico", category: "script", width: 104, contrast: 24, roundness: 68 },
  { name: "Lobster", seed: "pacifico", category: "script", width: 106, contrast: 66, roundness: 74 },
  { name: "Marck Script", seed: "pacifico", category: "script", width: 108, contrast: 42, roundness: 70 },
  { name: "Bad Script", seed: "pacifico", category: "script", width: 102, contrast: 30, roundness: 64 },
];

export type GeneratedRecipe = {
  seedId: FontSeedId;
  familyName: string;
  classification: string;
  width: number;
  slant: number;
  contrast: number;
  roundness: number;
  tracking: number;
  morphSeed: number;
  description: string;
  tags: string[];
  referenceNames: string[];
  catalogSize: number;
};

const has = (value: string, words: string[]) => words.some((word) => value.includes(word));

function generatedName(value: string, seed: number) {
  const prefixes = has(value, ["космос", "space", "orbit"]) ? ["Orba", "Astra", "Lunor"]
    : has(value, ["город", "urban", "архитект"]) ? ["Modul", "Forma", "Metro"]
      : has(value, ["мода", "fashion", "журнал"]) ? ["Vela", "Moiré", "Aurea"]
        : has(value, ["тех", "digital", "интерфейс", "ui"]) ? ["Nexa", "Synt", "Kern"]
          : has(value, ["дет", "мяг", "friendly"]) ? ["Milo", "Puffy", "Luma"]
            : ["Vektor", "Forma", "Takt", "Nova"];
  const suffixes = ["Sans", "Text", "Display", "Grotesk", "Mono", "Type"];
  return `${prefixes[seed % prefixes.length]} ${suffixes[Math.floor(seed / 7) % suffixes.length]}`;
}

export function generateRecipe(prompt: string): GeneratedRecipe {
  const value = prompt.toLocaleLowerCase("ru");
  let category: Reference["category"] = "sans";
  let classification = "Synthetic neo-grotesk";
  let width = 100;
  let slant = 0;
  let contrast = 28;
  let roundness = 35;
  let tracking = 0;
  const tags: string[] = [];

  if (has(value, ["антикв", "serif", "редакцион", "editorial", "книжн", "fashion", "модн"])) {
    category = "serif"; classification = "Synthetic serif"; contrast = 76; roundness = 22; tags.push("serif");
  } else if (has(value, ["моно", "mono", "код", "terminal", "технич", "industrial"])) {
    category = "mono"; classification = "Synthetic monospaced"; width = 94; roundness = 20; tags.push("mono");
  } else if (has(value, ["рукопис", "script", "каллиграф", "handwritten", "подпись"])) {
    category = "script"; classification = "Synthetic script"; slant = -6; contrast = 54; roundness = 72; tags.push("script");
  } else if (has(value, ["плакат", "poster", "condensed", "узк", "афиш", "headline", "бруталь"])) {
    category = "display"; classification = "Synthetic display"; width = 73; contrast = 18; tracking = 24; roundness = 12; tags.push("display");
  } else if (has(value, ["дружелюб", "friendly", "мягк", "soft", "rounded", "кругл"])) {
    roundness = 84; tags.push("rounded");
  } else if (has(value, ["нейтраль", "neutral", "ui", "интерфейс", "системн"])) {
    classification = "Synthetic UI sans"; contrast = 14; roundness = 30; tags.push("ui");
  }

  if (has(value, ["широк", "wide", "extended"])) width = 122;
  if (has(value, ["очень узк", "ultra condensed"])) width = 62;
  if (has(value, ["italic", "курсив", "наклон", "динамич"])) slant = -11;
  if (has(value, ["разреж", "airy", "воздух"])) tracking = 70;
  if (has(value, ["плотн", "tight", "компакт"])) tracking = -34;
  if (has(value, ["контраст", "contrast"])) contrast = Math.max(contrast, 78);
  if (has(value, ["геометр", "geometric", "швейцар", "swiss"])) { roundness = Math.max(roundness, 42); tags.push("geometric"); }

  const ranked = referenceCatalog
    .map((item) => ({ item, score: (item.category === category ? 0 : 180) + Math.abs(item.width - width) + Math.abs(item.contrast - contrast) * .65 + Math.abs(item.roundness - roundness) * .45 }))
    .sort((a, b) => a.score - b.score);
  const seed = Array.from(value).reduce((sum, char, index) => sum + char.charCodeAt(0) * (index + 3), 0) || 137;
  const closest = ranked[seed % Math.min(3, ranked.length)].item;

  return {
    seedId: closest.seed,
    familyName: generatedName(value, seed),
    classification,
    width,
    slant,
    contrast,
    roundness,
    tracking,
    morphSeed: seed,
    description: `${classification}. Новые контуры синтезированы по ${ranked.slice(0, 4).length} близким типографическим референсам: ${width < 90 ? "сжатая" : width > 110 ? "расширенная" : "нормальная"} пропорция, ${contrast > 60 ? "высокий" : "умеренный"} контраст и ${roundness > 65 ? "мягкие" : "собранные"} соединения.`,
    tags: tags.length ? tags : ["sans", "contemporary"],
    referenceNames: ranked.slice(0, 4).map(({ item }) => item.name),
    catalogSize: referenceCatalog.length,
  };
}
