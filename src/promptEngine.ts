export type FontSeedId =
  | "inter" | "space-grotesk" | "montserrat" | "pt-sans" | "arsenal"
  | "playfair-display" | "pt-serif" | "roboto-mono" | "bebas-neue"
  | "pacifico" | "rubik" | "russo-one";

type Category = "sans" | "serif" | "mono" | "display" | "script";
type Reference = { name: string; seed: FontSeedId; category: Category; width: number; contrast: number; roundness: number };

const groups: Record<Category, string[]> = {
  sans: ["Inter","Roboto","Open Sans","Noto Sans","IBM Plex Sans","Source Sans 3","Work Sans","Manrope","DM Sans","Space Grotesk","Montserrat","Raleway","Urbanist","Outfit","Rubik","Nunito","Quicksand","Onest","Lato","Poppins","Mulish","Karla","Barlow","Figtree","Plus Jakarta Sans","Public Sans","Instrument Sans","Geologica","Commissioner","Albert Sans","Red Hat Display","Lexend","Jost","Exo 2","Ubuntu","Fira Sans","Cabin","Asap","Hind","Heebo","Assistant","Arimo","Archivo","Chivo","Overpass","Sora","Epilogue","Golos Text","Wix Madefor Text","PT Sans"],
  serif: ["Playfair Display","Cormorant Garamond","Bodoni Moda","DM Serif Display","PT Serif","Lora","Merriweather","Source Serif 4","Literata","Noto Serif","Libre Baskerville","Libre Caslon Text","EB Garamond","Crimson Pro","Spectral","Bitter","Roboto Serif","IBM Plex Serif","Newsreader","Vollkorn","Alegreya","Cardo","Domine","Zilla Slab","Arvo","Rokkitt","Bree Serif","Fraunces","Prata","Yeseva One","Old Standard TT","Noto Serif Display","Cormorant","Arapey","Neuton","Trirong","Brygada 1918","Fira Serif","Lusitana","Inria Serif"],
  mono: ["Roboto Mono","JetBrains Mono","IBM Plex Mono","Source Code Pro","Fira Mono","Fira Code","Space Mono","DM Mono","Inconsolata","Ubuntu Mono","Noto Sans Mono","Geist Mono","Martian Mono","Azeret Mono","Spline Sans Mono","Anonymous Pro","Cousine","PT Mono","Red Hat Mono","Chivo Mono"],
  display: ["Bebas Neue","Oswald","Barlow Condensed","Archivo Narrow","Russo One","Unbounded","Anton","Alumni Sans","Teko","Saira Condensed","Roboto Condensed","DIN Condensed","League Spartan","Archivo Black","Black Ops One","Orbitron","Michroma","Syncopate","Staatliches","Fjalla One","Yanone Kaffeesatz","Big Shoulders Display","Bungee","Bowlby One SC","Monoton","Righteous","Krona One","Titan One","Paytone One","Chango","Ultra","Abril Fatface","Limelight","Poiret One","Forum","Tenor Sans","Cuprum","Arsenal","Ruslan Display","Play"],
  script: ["Pacifico","Caveat","Lobster","Marck Script","Bad Script","Dancing Script","Great Vibes","Sacramento","Satisfy","Kaushan Script","Pattaya","Neucha","Comforter","Yesteryear","Allura","Alex Brush","Parisienne","Yellowtail","Courgette","Cookie","Handlee","Patrick Hand","Shadows Into Light","Permanent Marker","Cedarville Cursive","Nothing You Could Do","Amatic SC","Kelly Slab","Gabriela","Lobster Two"],
};

const seedPools: Record<Category, FontSeedId[]> = {
  sans: ["inter", "space-grotesk", "montserrat", "pt-sans", "rubik", "arsenal"],
  serif: ["playfair-display", "pt-serif", "arsenal"],
  mono: ["roboto-mono", "inter"],
  display: ["bebas-neue", "russo-one", "montserrat", "arsenal"],
  script: ["pacifico", "playfair-display"],
};
const primarySeeds: Record<Category, FontSeedId> = { sans:"inter", serif:"pt-serif", mono:"roboto-mono", display:"russo-one", script:"pacifico" };

const categoryDefaults: Record<Category, [number, number, number]> = {
  sans: [101, 18, 38], serif: [101, 70, 22], mono: [95, 18, 18], display: [82, 22, 18], script: [108, 48, 72],
};

function hash(value: string) {
  let result = 2166136261;
  for (const char of value) { result ^= char.charCodeAt(0); result = Math.imul(result, 16777619); }
  return result >>> 0;
}

const referenceCatalog: Reference[] = (Object.keys(groups) as Category[]).flatMap((category) => {
  const [baseWidth, baseContrast, baseRoundness] = categoryDefaults[category];
  return groups[category].map((name, index) => {
    const value = hash(name);
    return {
      name,
      category,
      seed: seedPools[category][(value + index) % seedPools[category].length],
      width: Math.max(58, Math.min(132, baseWidth + (value % 29) - 14)),
      contrast: Math.max(4, Math.min(98, baseContrast + (Math.floor(value / 31) % 31) - 15)),
      roundness: Math.max(4, Math.min(98, baseRoundness + (Math.floor(value / 997) % 35) - 17)),
    };
  });
});

export type GeneratedRecipe = {
  seedIds: FontSeedId[];
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
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, Math.round(value)));

function generatedName(value: string, seed: number) {
  const prefixes = has(value, ["космос", "space", "orbit"]) ? ["Orba","Astra","Lunor"] : has(value, ["город","urban","архитект"]) ? ["Modul","Forma","Metro"] : has(value, ["мода","fashion","журнал"]) ? ["Vela","Moiré","Aurea"] : has(value, ["тех","digital","интерфейс","ui"]) ? ["Nexa","Synt","Kern"] : has(value, ["дет","мяг","friendly"]) ? ["Milo","Puffy","Luma"] : ["Vektor","Forma","Takt","Nova","Axiom","Lumen"];
  const suffixes = ["Sans","Text","Display","Grotesk","Mono","Type","Serif","Studio"];
  return `${prefixes[seed % prefixes.length]} ${suffixes[Math.floor(seed / 11) % suffixes.length]}`;
}

export function generateRecipe(prompt: string): GeneratedRecipe {
  const value = prompt.trim().toLocaleLowerCase("ru");
  const morphSeed = hash(value || "untitled-fontgen");
  let category: Category = "sans";
  let classification = "Synthetic neo-grotesk";
  let width = 100, slant = 0, contrast = 28, roundness = 35, tracking = 0;
  const tags: string[] = [];

  if (has(value,["антикв","serif","редакцион","editorial","книжн","fashion","модн"])) { category="serif";classification="Synthetic serif";contrast=76;roundness=22;tags.push("serif"); }
  else if (has(value,["моно","mono","код","terminal","технич","industrial"])) { category="mono";classification="Synthetic monospaced";width=94;roundness=20;tags.push("mono"); }
  else if (has(value,["рукопис","script","каллиграф","handwritten","подпись"])) { category="script";classification="Synthetic script";slant=-6;contrast=54;roundness=72;tags.push("script"); }
  else if (has(value,["плакат","poster","condensed","узк","афиш","headline","бруталь"])) { category="display";classification="Synthetic display";width=73;contrast=18;tracking=24;roundness=12;tags.push("display"); }
  else if (has(value,["дружелюб","friendly","мягк","soft","rounded","кругл"])) { roundness=84;tags.push("rounded"); }
  else if (has(value,["нейтраль","neutral","ui","интерфейс","системн"])) { classification="Synthetic UI sans";contrast=14;roundness=30;tags.push("ui"); }

  // Every custom prompt contributes a stable design vector even when it does
  // not contain any known keyword.
  width += (morphSeed % 17) - 8;
  contrast += (Math.floor(morphSeed / 17) % 19) - 9;
  roundness += (Math.floor(morphSeed / 323) % 23) - 11;
  slant += (Math.floor(morphSeed / 7429) % 7) - 3;
  tracking += (Math.floor(morphSeed / 52003) % 17) - 8;
  if (has(value,["широк","wide","extended"])) width=122;
  if (has(value,["очень узк","ultra condensed"])) width=62;
  if (has(value,["italic","курсив","наклон","динамич"])) slant=-11;
  if (has(value,["разреж","airy","воздух"])) tracking=70;
  if (has(value,["плотн","tight","компакт"])) tracking=-34;
  if (has(value,["контраст","contrast"])) contrast=Math.max(contrast,78);
  if (has(value,["геометр","geometric","швейцар","swiss"])) { roundness=Math.max(roundness,42);tags.push("geometric"); }
  width=clamp(width,58,135);contrast=clamp(contrast,0,100);roundness=clamp(roundness,0,100);slant=clamp(slant,-18,12);tracking=clamp(tracking,-100,180);

  const ranked = referenceCatalog.map((item) => ({ item, score:(item.category===category?0:210)+Math.abs(item.width-width)+Math.abs(item.contrast-contrast)*.7+Math.abs(item.roundness-roundness)*.5+((hash(item.name+value)%100)/100) })).sort((a,b)=>a.score-b.score);
  const references = ranked.slice(0,12).map(({item})=>item);
  const seedIds: FontSeedId[] = [primarySeeds[category]];
  for (const { item } of ranked) { if (!seedIds.includes(item.seed)) seedIds.push(item.seed); if (seedIds.length===8) break; }

  return {
    seedIds,
    familyName: generatedName(value,morphSeed),
    classification,width,slant,contrast,roundness,tracking,morphSeed,
    description:`${classification}. Контуры реконструированы по профилям 12 близких конструкций из открытого индекса: ${width<90?"сжатая":width>110?"расширенная":"нормальная"} пропорция, ${contrast>60?"высокий":"умеренный"} контраст и ${roundness>65?"мягкие":"собранные"} соединения.`,
    tags:tags.length?tags:["synthetic","contemporary"],
    referenceNames:references.map((item)=>item.name),
    catalogSize:referenceCatalog.length,
  };
}
