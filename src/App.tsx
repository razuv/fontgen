import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Font } from "opentype.js";
import { drawPreview, exportFont, loadFont, type FontStyle } from "./fontEngine";
import { generateRecipe, type FontBaseId, type GeneratedRecipe } from "./promptEngine";

type BaseFont = { id: FontBaseId; name: string; category: string; file: string; sample: string };

const baseFonts: BaseFont[] = [
  { id: "space-grotesk", name: "Space Grotesk", category: "Geometric sans", file: "space-grotesk.ttf", sample: "Ag" },
  { id: "inter", name: "Inter", category: "UI sans", file: "inter.ttf", sample: "Aa" },
  { id: "playfair-display", name: "Playfair Display", category: "High contrast serif", file: "playfair-display.ttf", sample: "Gg" },
  { id: "pt-serif", name: "PT Serif", category: "Text serif", file: "pt-serif.ttf", sample: "Rr" },
  { id: "roboto-mono", name: "Roboto Mono", category: "Monospaced", file: "roboto-mono.ttf", sample: "01" },
  { id: "bebas-neue", name: "Bebas Neue", category: "Condensed display", file: "bebas-neue.ttf", sample: "AB" },
  { id: "rubik", name: "Rubik", category: "Rounded sans", file: "rubik.ttf", sample: "Oo" },
  { id: "russo-one", name: "Russo One", category: "Cyrillic display", file: "russo-one.ttf", sample: "ЯR" },
  { id: "pacifico", name: "Pacifico", category: "Script", file: "pacifico.ttf", sample: "Ps" },
];

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
const shuffledPrompts = [...promptPool].sort(() => Math.random() - 0.5);
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
  return (
    <label className="range-control">
      <span><b>{label}</b><output>{value}{unit}</output></span>
      <input aria-label={label} type="range" min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} style={{ background: `linear-gradient(90deg,#dfe0dc ${percent}%,#303030 ${percent}%)` }} />
    </label>
  );
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [prompt, setPrompt] = useState(initialPrompt);
  const [recipe, setRecipe] = useState<GeneratedRecipe>(initialRecipe);
  const [selectedBase, setSelectedBase] = useState<FontBaseId>(initialRecipe.baseId);
  const [font, setFont] = useState<Font | null>(null);
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
  const baseFont = baseFonts.find((item) => item.id === selectedBase) ?? baseFonts[0];
  const pair = `${pairLeft.slice(0, 1)}${pairRight.slice(0, 1)}`;
  const pairValue = kerning[pair] ?? 0;

  const loadSelectedFont = useCallback(async (id: FontBaseId) => {
    const selected = baseFonts.find((item) => item.id === id) ?? baseFonts[0];
    setLoading(true);
    try {
      const loaded = await loadFont(baseUrl(`fonts/${selected.file}`));
      setFont(loaded);
    } catch {
      setToast("Не удалось загрузить базовый шрифт");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void loadSelectedFont(selectedBase); }, [selectedBase, loadSelectedFont]);

  const render = useCallback(() => {
    if (!canvasRef.current || !font || !activeStyle) return;
    drawPreview(canvasRef.current, font, previewText || "Fontgen", { width, slant, tracking, style: activeStyle, kerning }, fontSize, showGuides);
  }, [font, previewText, width, slant, tracking, activeStyle, kerning, fontSize, showGuides]);

  useEffect(() => {
    render();
    const observer = new ResizeObserver(render);
    if (canvasRef.current) observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, [render]);

  useEffect(() => {
    if (!toast) return;
    const timeout = window.setTimeout(() => setToast(""), 2500);
    return () => window.clearTimeout(timeout);
  }, [toast]);

  function generate() {
    if (!prompt.trim()) return;
    setGenerating(true);
    window.setTimeout(() => {
      const next = generateRecipe(prompt);
      setRecipe(next);
      setSelectedBase(next.baseId);
      setWidth(next.width);
      setSlant(next.slant);
      setTracking(next.tracking);
      setContrast(next.contrast);
      setRoundness(next.roundness);
      setActiveStyleId("regular");
      setGenerating(false);
      setToast(`Создан ${next.familyName}`);
    }, 520);
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
    setStyles((current) => [...current, next]);
    setActiveStyleId(next.id);
    setToast(`Добавлено начертание ${next.name}`);
  }

  function removeStyle(id: string) {
    if (styles.length === 1) return;
    setStyles((current) => current.filter((style) => style.id !== id));
    if (activeStyleId === id) setActiveStyleId(styles.find((style) => style.id !== id)?.id ?? "regular");
  }

  async function handleExport(type: "otf" | "ttf" | "svg") {
    if (!font || !activeStyle) return;
    setExporting(type);
    try {
      await exportFont(font, { familyName: recipe.familyName, width, slant, tracking, style: activeStyle, kerning }, type);
      setToast(`${type.toUpperCase()} готов к скачиванию`);
    } catch (error) {
      console.error(error);
      setToast(`Не удалось собрать ${type.toUpperCase()}`);
    } finally {
      setExporting(null);
    }
  }

  const glyphCount = font?.glyphs.length ?? 0;
  const metadata = useMemo(() => [recipe.classification, `${glyphCount} глифов`, "OFL 1.1"], [recipe.classification, glyphCount]);

  return (
    <main className="studio-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="Fontgen"><span>Fg</span> FONTGEN</a>
        <div className="status-pill"><i /> LOCAL ENGINE <b>{loading ? "LOAD" : "READY"}</b></div>
        <button className="top-action" onClick={() => void handleExport("otf")}>Экспортировать шрифт ↗</button>
      </header>

      <section className="workspace">
        <aside className="panel source-panel">
          <div className="panel-heading"><span>01</span><h2>Промпт и база</h2></div>
          <label className="prompt-box">
            <span>ОПИШИТЕ ХАРАКТЕР ШРИФТА</span>
            <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") generate(); }} />
            <div className="startup-prompts" aria-label="Случайные тестовые промпты">
              {startupPrompts.map((example, index) => <button type="button" key={example} onClick={() => setPrompt(example)} title={example}>{index + 1}</button>)}
              <span>СЛУЧАЙНЫЕ ПРОМПТЫ</span>
            </div>
            <div><small>{prompt.length}/280 · ⌘↵</small><button onClick={generate} disabled={generating}>{generating ? "СОЗДАЮ…" : "СОЗДАТЬ ↗"}</button></div>
          </label>

          <div className="prompt-examples">
            <div className="section-title"><span>НАПРАВЛЕНИЯ</span><b>{startupPrompts.length}</b></div>
            {startupPrompts.map((example, index) => <button key={example} onClick={() => setPrompt(example)}><i>0{index + 1}</i><span>{example}</span><b>↗</b></button>)}
          </div>

          <div className="font-library">
            <div className="section-title"><span>GOOGLE FONTS · БАЗА</span><b>{baseFonts.length}</b></div>
            <div className="font-list">
              {baseFonts.map((item) => (
                <button key={item.id} className={selectedBase === item.id ? "active" : ""} onClick={() => setSelectedBase(item.id)}>
                  <span className={`font-thumb font-${item.id}`}>{item.sample}</span>
                  <span><strong>{item.name}</strong><small>{item.category}</small></span>
                  <i>{selectedBase === item.id ? "●" : "○"}</i>
                </button>
              ))}
            </div>
          </div>
          <div className="license-note"><span>i</span><p>Базовые гарнитуры: Google Fonts. Производный шрифт получает новое имя и сохраняет лицензию OFL.</p></div>
        </aside>

        <section className="stage" aria-label="Предпросмотр шрифта">
          <div className="stage-top">
            <div><span>● LIVE OUTLINES</span><b>{baseFont.name} → {recipe.familyName}</b></div>
            <div className="stage-actions"><button onClick={() => setShowGuides((value) => !value)} className={showGuides ? "active" : ""}>⌗ НАПРАВЛЯЮЩИЕ</button><button onClick={() => setPreviewText("Hamburgefontsiv AVATAR ТаЛА")}>↺ ТЕСТ</button></div>
          </div>

          <div className="canvas-wrap">
            <div className="specimen-meta"><span>SPECIMEN / {activeStyle?.name.toUpperCase()}</span><span>{fontSize} PX</span></div>
            <canvas ref={canvasRef} />
            {loading && <div className="loader"><i /><b>Загрузка контуров</b><small>{baseFont.name}</small></div>}
            <div className="canvas-caption"><b>{recipe.familyName}</b><span>{recipe.description}</span></div>
          </div>

          <div className="test-deck">
            <label><span>ТЕСТОВЫЙ ТЕКСТ</span><input value={previewText} onChange={(event) => setPreviewText(event.target.value)} /></label>
            <RangeControl label="Размер" value={fontSize} min={48} max={210} unit="px" onChange={setFontSize} />
          </div>
        </section>

        <aside className="panel properties-panel">
          <div className="panel-heading properties-heading"><span>02</span><h2>Параметры</h2><button className="reset-all" onClick={() => { setWidth(recipe.width); setSlant(recipe.slant); setTracking(recipe.tracking); }}>↺ СБРОС</button></div>

          <section className="result-card">
            <div><span>СГЕНЕРИРОВАНО</span><b>● VALID</b></div>
            <h1>{recipe.familyName}</h1>
            <p>{recipe.description}</p>
            <div className="tag-row">{recipe.tags.map((tag) => <span key={tag}>{tag}</span>)}</div>
          </section>

          <details className="property-section" open>
            <summary>Конструкция <span>−</span></summary>
            <div className="section-body stack-controls">
              <RangeControl label="Ширина" value={width} min={60} max={135} unit="%" onChange={setWidth} />
              <RangeControl label="Наклон" value={slant} min={-18} max={12} unit="°" onChange={setSlant} />
              <RangeControl label="Контраст" value={contrast} min={0} max={100} unit="%" onChange={setContrast} />
              <RangeControl label="Скругление" value={roundness} min={0} max={100} unit="%" onChange={setRoundness} />
            </div>
          </details>

          <details className="property-section" open>
            <summary>Spacing <span>−</span></summary>
            <div className="section-body stack-controls">
              <RangeControl label="Трекинг" value={tracking} min={-100} max={180} unit="u" onChange={setTracking} />
              <div className="kerning-editor">
                <div className="control-label"><span>Кернинг пары</span><output>{pairValue} u</output></div>
                <div className="pair-inputs"><input aria-label="Левый символ" maxLength={1} value={pairLeft} onChange={(event) => setPairLeft(event.target.value)} /><i>+</i><input aria-label="Правый символ" maxLength={1} value={pairRight} onChange={(event) => setPairRight(event.target.value)} /><span className="pair-preview">{pair || "AV"}</span></div>
                <RangeControl label="Коррекция" value={pairValue} min={-200} max={200} unit="u" onChange={(value) => setKerning((current) => ({ ...current, [pair]: value }))} />
                <div className="common-pairs">{["AV", "To", "Ta", "WA", "Та", "ЛА"].map((item) => <button key={item} className={pair === item ? "active" : ""} onClick={() => { const chars = Array.from(item); setPairLeft(chars[0]); setPairRight(chars[1]); }}>{item}</button>)}</div>
              </div>
            </div>
          </details>

          <details className="property-section" open>
            <summary>Начертания <span>−</span></summary>
            <div className="section-body">
              <div className="styles-list">
                {styles.map((style) => <div key={style.id} className={activeStyleId === style.id ? "active" : ""}><button onClick={() => setActiveStyleId(style.id)}><span style={{ fontWeight: style.weight, fontStyle: style.italic ? "italic" : "normal" }}>Aa</span><b>{style.name}</b><small>{style.weight}{style.italic ? " · italic" : ""}</small></button>{styles.length > 1 && <button className="remove-style" onClick={() => removeStyle(style.id)} aria-label={`Удалить ${style.name}`}>×</button>}</div>)}
              </div>
              <button className="add-style" onClick={addStyle}><span>＋</span><b>Добавить начертание</b><small>Light, Italic, Black</small></button>
            </div>
          </details>

          <div className="font-meta">{metadata.map((item) => <span key={item}>{item}</span>)}</div>
        </aside>
      </section>

      <footer className="exportbar">
        <div className="export-title"><span>03</span><div><b>Шрифт готов к экспорту</b><small>{recipe.familyName} / {activeStyle?.name} · {glyphCount} глифов</small></div></div>
        <div className="export-actions">
          <button onClick={() => void handleExport("svg")} disabled={!font || exporting !== null}><span>◇</span><div><b>{exporting === "svg" ? "СБОРКА…" : "SVG"}</b><small>Векторные глифы</small></div></button>
          <button onClick={() => void handleExport("ttf")} disabled={!font || exporting !== null}><span>T</span><div><b>{exporting === "ttf" ? "СБОРКА…" : "TTF"}</b><small>TrueType</small></div></button>
          <button className="primary" onClick={() => void handleExport("otf")} disabled={!font || exporting !== null}><span>↗</span><div><b>{exporting === "otf" ? "СБОРКА…" : "OTF"}</b><small>OpenType · Recommended</small></div></button>
        </div>
      </footer>
      {toast && <div className="toast"><span>●</span>{toast}</div>}
    </main>
  );
}
