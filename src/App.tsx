import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Font } from "opentype.js";
import { drawPreview, exportFont, exportTextSvg, loadFont, type FontStyle, type TransformSettings } from "./fontEngine";
import { generateRecipe, type FontSeedId, type GeneratedRecipe } from "./promptEngine";

const seedFiles: Record<FontSeedId, string> = {
  "space-grotesk": "space-grotesk.ttf",
  inter: "inter.ttf",
  "playfair-display": "playfair-display.ttf",
  "pt-serif": "pt-serif.ttf",
  "roboto-mono": "roboto-mono.ttf",
  "bebas-neue": "bebas-neue.ttf",
  rubik: "rubik.ttf",
  "russo-one": "russo-one.ttf",
  pacifico: "pacifico.ttf",
  montserrat: "montserrat.ttf",
  "pt-sans": "pt-sans.ttf",
  arsenal: "arsenal.ttf",
};

const promptPool = [
  "Современный геометрический гротеск для технологичного бренда, немного широкий, спокойный и уверенный",
  "Узкий брутальный шрифт для плакатов с кириллицей",
  "Контрастная редакционная антиква для модного журнала",
  "Дружелюбный округлый UI-шрифт для детского приложения",
  "Моноширинный технический шрифт для терминала",
  "Динамичный курсивный гротеск для спортивного бренда",
  "Воздушная широкая гарнитура для архитектурного бюро",
  "Мягкий рукописный шрифт для упаковки шоколада",
  "Нейтральный компактный шрифт для финансового интерфейса",
];
const shuffledPrompts = [...promptPool].sort(() => Math.random() - .5);
const initialPrompt = shuffledPrompts[0];
const initialRecipe = generateRecipe(initialPrompt);
const initialStyles: FontStyle[] = [
  { id: "regular", name: "Regular", weight: 400, italic: false },
  { id: "medium", name: "Medium", weight: 500, italic: false },
  { id: "bold", name: "Bold", weight: 700, italic: false },
];

function baseUrl(path: string) {
  return `${import.meta.env.BASE_URL}${path.replace(/^\/+/, "")}`;
}

function RangeControl({ label, value, min, max, unit, onChange }: { label: string; value: number; min: number; max: number; unit?: string; onChange: (value: number) => void }) {
  const percent = ((value - min) / (max - min)) * 100;
  return <label className="range-control"><span><b>{label}</b><output>{value}{unit}</output></span><input aria-label={label} type="range" min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} style={{ background: `linear-gradient(90deg,#dfe0dc ${percent}%,#303030 ${percent}%)` }} /></label>;
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [prompt, setPrompt] = useState(initialPrompt);
  const [recipe, setRecipe] = useState<GeneratedRecipe>(initialRecipe);
  const [familyName, setFamilyName] = useState(initialRecipe.familyName);
  const [fonts, setFonts] = useState<Font[]>([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [previewText, setPreviewText] = useState("Шрифт создаёт характер");
  const [fontSize, setFontSize] = useState(122);
  const [showGuides, setShowGuides] = useState(true);
  const [width, setWidth] = useState(initialRecipe.width);
  const [slant, setSlant] = useState(initialRecipe.slant);
  const [tracking, setTracking] = useState(initialRecipe.tracking);
  const [contrast, setContrast] = useState(initialRecipe.contrast);
  const [roundness, setRoundness] = useState(initialRecipe.roundness);
  const [styles, setStyles] = useState<FontStyle[]>(initialStyles);
  const [activeStyleId, setActiveStyleId] = useState("regular");
  const [pairLeft, setPairLeft] = useState("A");
  const [pairRight, setPairRight] = useState("V");
  const [kerning, setKerning] = useState<Record<string, number>>({ AV: -70, To: -48, Та: -42, ЛА: -35 });
  const [toast, setToast] = useState("");
  const [exporting, setExporting] = useState<string | null>(null);
  const startupPrompts = useMemo(() => shuffledPrompts.slice(0, 4), []);

  const activeStyle = styles.find((style) => style.id === activeStyleId) ?? styles[0];
  const pair = `${pairLeft.slice(0, 1)}${pairRight.slice(0, 1)}`;
  const pairValue = kerning[pair] ?? 0;

  const loadSeeds = useCallback(async (seedIds: FontSeedId[]) => {
    setLoading(true);
    try { setFonts(await Promise.all(seedIds.map((seedId) => loadFont(baseUrl(`fonts/${seedFiles[seedId]}`))))); }
    catch { setToast("Не удалось инициализировать контурный движок"); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { void loadSeeds(recipe.seedIds); }, [recipe.seedIds, loadSeeds]);

  const currentSettings = useCallback((): TransformSettings => ({ familyName: familyName.trim() || recipe.familyName, width, slant, contrast, roundness, tracking, morphSeed: recipe.morphSeed, style: activeStyle, kerning }), [familyName, recipe.familyName, recipe.morphSeed, width, slant, contrast, roundness, tracking, activeStyle, kerning]);

  const render = useCallback(() => {
    if (!canvasRef.current || !fonts.length || !activeStyle) return;
    const { familyName: _name, ...settings } = currentSettings();
    void _name;
    drawPreview(canvasRef.current, fonts, previewText, settings, fontSize, showGuides);
  }, [fonts, previewText, activeStyle, currentSettings, fontSize, showGuides]);

  useEffect(() => {
    render();
    const observer = new ResizeObserver(render);
    if (canvasRef.current) observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, [render]);
  useEffect(() => { if (!toast) return; const timeout = window.setTimeout(() => setToast(""), 2600); return () => window.clearTimeout(timeout); }, [toast]);

  function generate() {
    if (!prompt.trim()) return;
    setGenerating(true);
    window.setTimeout(() => {
      const next = generateRecipe(prompt);
      setRecipe(next); setFamilyName(next.familyName); setWidth(next.width); setSlant(next.slant); setTracking(next.tracking); setContrast(next.contrast); setRoundness(next.roundness); setActiveStyleId("regular"); setGenerating(false); setToast(`Синтезирована гарнитура ${next.familyName}`);
    }, 560);
  }

  function addStyle() {
    const candidates: FontStyle[] = [
      { id: "light", name: "Light", weight: 300, italic: false },
      { id: "italic", name: "Italic", weight: 400, italic: true },
      { id: "bold-italic", name: "Bold Italic", weight: 700, italic: true },
      { id: "black", name: "Black", weight: 900, italic: false },
    ];
    const next = candidates.find((candidate) => !styles.some((style) => style.id === candidate.id));
    if (!next) { setToast("Все доступные начертания добавлены"); return; }
    setStyles((current) => [...current, next]); setActiveStyleId(next.id); setToast(`Добавлено начертание ${next.name}`);
  }

  function removeStyle(id: string) {
    if (styles.length === 1) return;
    setStyles((current) => current.filter((style) => style.id !== id));
    if (activeStyleId === id) setActiveStyleId(styles.find((style) => style.id !== id)?.id ?? "regular");
  }

  async function handleExport(type: "otf" | "ttf" | "svg") {
    if (!fonts.length || !activeStyle) return;
    setExporting(type);
    try {
      if (type === "svg") exportTextSvg(fonts, currentSettings(), previewText);
      else await exportFont(fonts, currentSettings(), type);
      setToast(type === "svg" ? "Тестовая строка экспортирована в SVG" : `${type.toUpperCase()} готов к скачиванию`);
    } catch (error) { console.error(error); setToast(`Не удалось собрать ${type.toUpperCase()}`); }
    finally { setExporting(null); }
  }

  const glyphCount = fonts[0]?.glyphs.length ?? 0;

  return (
    <main className="studio-shell no-topbar">
      <section className="workspace">
        <aside className="panel source-panel">
          <div className="panel-heading"><span>01</span><h2>Промпт</h2><b className="engine-mark">FONTGEN / SYNTH</b></div>
          <div className="prompt-box">
            <span>ОПИШИТЕ ХАРАКТЕР ШРИФТА</span>
            <textarea value={prompt} maxLength={280} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") generate(); }} />
            <div className="startup-prompts" aria-label="Случайные тестовые промпты">{startupPrompts.map((example, index) => <button type="button" key={example} onClick={() => setPrompt(example)} title={example}>{index + 1}</button>)}<span>СЛУЧАЙНЫЕ ПРОМПТЫ</span></div>
            <div><small>{prompt.length}/280 · ⌘↵</small><button onClick={generate} disabled={generating}>{generating ? "СИНТЕЗ…" : "СОЗДАТЬ ↗"}</button></div>
          </div>

          <div className="prompt-examples">
            <div className="section-title"><span>БЫСТРЫЙ ТЕСТ</span><b>{startupPrompts.length}</b></div>
            {startupPrompts.map((example, index) => <button key={example} onClick={() => setPrompt(example)}><i>0{index + 1}</i><span>{example}</span><b>↗</b></button>)}
          </div>

          <section className="reference-search">
            <div className="search-status"><span>● ПОИСК ЗАВЕРШЁН</span><b>{recipe.catalogSize} OPEN FAMILIES</b></div>
            <h3>Найдены близкие конструкции</h3>
            <p>Референсы используются как типографические координаты. Готовая гарнитура не выбирается из каталога: движок заново строит контуры с параметрами промпта.</p>
            <div className="reference-tags">{recipe.referenceNames.map((name, index) => <span key={name}><i>0{index + 1}</i>{name}</span>)}</div>
            <div className="synthesis-map"><i /><i /><i /><i /><b>NEW<br/>CURVES</b></div>
          </section>
          <div className="license-note"><span>i</span><p>Скрытый индекс основан на открытых семействах Google Fonts. Новый результат получает отдельное редактируемое имя.</p></div>
        </aside>

        <section className="stage" aria-label="Предпросмотр синтезированного шрифта">
          <div className="stage-top"><div><span>● SYNTHESIZED OUTLINES</span><b>{recipe.referenceNames.join(" · ")} → NEW</b></div><div className="stage-actions"><button onClick={() => setShowGuides((value) => !value)} className={showGuides ? "active" : ""}>⌗ НАПРАВЛЯЮЩИЕ</button><button onClick={() => setPreviewText("Hamburgefontsiv AVATAR ТаЛА")}>↺ ТЕСТ</button></div></div>
          <div className="canvas-wrap">
            <div className="specimen-meta"><span>SPECIMEN / {activeStyle?.name.toUpperCase()}</span><span>{fontSize} PX · SEED {String(recipe.morphSeed).slice(-4)}</span></div>
            <canvas ref={canvasRef} />
            {loading && <div className="loader"><i /><b>Синтез контуров</b><small>поиск и реконструкция</small></div>}
            <div className="canvas-caption"><b>{familyName || "Untitled Fontgen"}</b><span>{recipe.description}</span></div>
          </div>
          <div className="test-deck"><label><span>ТЕСТОВЫЙ ТЕКСТ / ЭКСПОРТ SVG</span><input value={previewText} onChange={(event) => setPreviewText(event.target.value)} /></label><RangeControl label="Размер" value={fontSize} min={48} max={210} unit="px" onChange={setFontSize} /></div>
        </section>

        <aside className="panel properties-panel">
          <div className="panel-heading properties-heading"><span>02</span><h2>Параметры</h2><button className="reset-all" onClick={() => { setWidth(recipe.width); setSlant(recipe.slant); setTracking(recipe.tracking); setContrast(recipe.contrast); setRoundness(recipe.roundness); }}>↺ СБРОС</button></div>
          <section className="result-card">
            <div><span>СИНТЕЗИРОВАНО</span><b>● EDITABLE</b></div>
            <label className="family-name"><span>НАЗВАНИЕ ГАРНИТУРЫ</span><input value={familyName} maxLength={42} onChange={(event) => setFamilyName(event.target.value)} placeholder="Untitled Fontgen" /></label>
            <p>{recipe.description}</p><div className="tag-row">{recipe.tags.map((tag) => <span key={tag}>{tag}</span>)}</div>
          </section>

          <details className="property-section" open><summary>Конструкция <span>−</span></summary><div className="section-body stack-controls">
            <RangeControl label="Ширина" value={width} min={60} max={135} unit="%" onChange={setWidth} />
            <RangeControl label="Наклон" value={slant} min={-18} max={12} unit="°" onChange={setSlant} />
            <RangeControl label="Контраст кривых" value={contrast} min={0} max={100} unit="%" onChange={setContrast} />
            <RangeControl label="Скругление узлов" value={roundness} min={0} max={100} unit="%" onChange={setRoundness} />
          </div></details>

          <details className="property-section" open><summary>Spacing <span>−</span></summary><div className="section-body stack-controls">
            <RangeControl label="Трекинг" value={tracking} min={-100} max={180} unit="u" onChange={setTracking} />
            <div className="kerning-editor"><div className="control-label"><span>Кернинг пары</span><output>{pairValue} u</output></div><div className="pair-inputs"><input aria-label="Левый символ" maxLength={1} value={pairLeft} onChange={(event) => setPairLeft(event.target.value)} /><i>+</i><input aria-label="Правый символ" maxLength={1} value={pairRight} onChange={(event) => setPairRight(event.target.value)} /><span className="pair-preview">{pair || "AV"}</span></div><RangeControl label="Коррекция" value={pairValue} min={-200} max={200} unit="u" onChange={(value) => setKerning((current) => ({ ...current, [pair]: value }))} /><div className="common-pairs">{["AV", "To", "Ta", "WA", "Та", "ЛА"].map((item) => <button key={item} className={pair === item ? "active" : ""} onClick={() => { const chars = Array.from(item); setPairLeft(chars[0]); setPairRight(chars[1]); }}>{item}</button>)}</div></div>
          </div></details>

          <details className="property-section" open><summary>Начертания <span>−</span></summary><div className="section-body"><div className="styles-list">{styles.map((style) => <div key={style.id} className={activeStyleId === style.id ? "active" : ""}><button onClick={() => setActiveStyleId(style.id)}><span>Aa</span><b>{style.name}</b><small>{style.weight}{style.italic ? " · italic" : ""}</small></button>{styles.length > 1 && <button className="remove-style" onClick={() => removeStyle(style.id)} aria-label={`Удалить ${style.name}`}>×</button>}</div>)}</div><button className="add-style" onClick={addStyle}><span>＋</span><b>Добавить начертание</b><small>Light, Italic, Black</small></button></div></details>
          <div className="font-meta"><span>{recipe.classification}</span><span>{glyphCount} глифов</span><span>новые контуры</span><span>OFL compatible</span></div>
        </aside>
      </section>

      <footer className="exportbar"><div className="export-title"><span>03</span><div><b>Готово к экспорту</b><small>{familyName || "Untitled Fontgen"} / {activeStyle?.name} · {glyphCount} глифов</small></div></div><div className="export-actions">
        <button onClick={() => void handleExport("svg")} disabled={!fonts.length || exporting !== null}><span>◇</span><div><b>{exporting === "svg" ? "СБОРКА…" : "SVG"}</b><small>Только тестовый текст</small></div></button>
        <button onClick={() => void handleExport("ttf")} disabled={!fonts.length || exporting !== null}><span>T</span><div><b>{exporting === "ttf" ? "СБОРКА…" : "TTF"}</b><small>Вся гарнитура</small></div></button>
        <button className="primary" onClick={() => void handleExport("otf")} disabled={!fonts.length || exporting !== null}><span>↗</span><div><b>{exporting === "otf" ? "СБОРКА…" : "OTF"}</b><small>Вся гарнитура</small></div></button>
      </div></footer>
      {toast && <div className="toast"><span>●</span>{toast}</div>}
    </main>
  );
}
