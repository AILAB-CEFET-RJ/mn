"use client";

import { useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const MODES = [
  { v: "R4_siglip", label: "R4: Multimodal (SigLIP + Desc + Metadados)" },
  { v: "hybrid", label: "R4: Multimodal (CLIP + Desc + Metadados)" },
  { v: "R4_text", label: "R4: Texto + Metadados (BGE-M3)" },
  { v: "R3_siglip", label: "R3: Imagem (SigLIP) + Descrição Pura" },
  { v: "R3_clip", label: "R3: Imagem (CLIP) + Descrição Pura" },
  { v: "R2", label: "R2: Apenas Descrição Física (BGE-M3)" },
  { v: "R1_siglip", label: "R1: Apenas Imagem (SigLIP Multilíngue)" },
  { v: "images", label: "R1: Apenas Imagem (CLIP ViT-B/32)" },
  { v: "bm25_r4", label: "Baseline Léxico BM25 (R4)" },
  { v: "bm25_r2", label: "Baseline Léxico BM25 (R2)" },
];

export default function Home() {
  const [tab, setTab] = useState("text");
  const [text, setText] = useState("");
  const [mode, setMode] = useState("R4_siglip");
  const [visualModel, setVisualModel] = useState("clip");
  const [topK, setTopK] = useState(12);
  const [categoria, setCategoria] = useState("");
  const [estado, setEstado] = useState("");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [over, setOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);
  const [item, setItem] = useState(null);
  const [activeImg, setActiveImg] = useState(null);
  const fileRef = useRef(null);

  function pickFile(f) {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  async function search(e) {
    e?.preventDefault();
    setError(null);
    setLoading(true);
    try {
      let res;
      if (tab === "text") {
        if (!text.trim()) throw new Error("Digite uma consulta.");
        const filtros = {};
        if (categoria) filtros.categoria = categoria;
        if (estado) filtros.estado = estado;
        res = await fetch(`${API}/search/text`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ texto: text, modo: mode, top_k: topK, filtros }),
        });
      } else {
        if (!file) throw new Error("Selecione uma imagem.");
        const fd = new FormData();
        fd.append("file", file);
        fd.append("top_k", topK);
        fd.append("modelo_visual", visualModel);
        if (categoria) fd.append("categoria", categoria);
        if (estado) fd.append("estado", estado);
        res = await fetch(`${API}/search/image`, { method: "POST", body: fd });
      }
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || "Erro na busca.");
      setData(json);
    } catch (err) {
      setData(null);
      setError(err.message === "Failed to fetch" ? "Não foi possível conectar à API em " + API : err.message);
    } finally {
      setLoading(false);
    }
  }

  async function openItem(r) {
    setActiveImg(r.image_id);
    setItem({ ...r, imagens: [{ image_id: r.image_id, url: r.imagem_url }] });
    try {
      const res = await fetch(`${API}/items/${encodeURIComponent(r.item_id)}`);
      if (res.ok) setItem(await res.json());
    } catch {}
  }

  return (
    <main className="container">
      <header className="hero">
        <span className="eyebrow">Museu Nacional · CEFET/RJ</span>
        <h1>Busca multimodal no acervo</h1>
        <p>Pesquise por consultas em linguagem natural ou por fotografia comparando representações R1, R2, R3, R4 e Baselines.</p>
      </header>

      <section className="panel" aria-label="Busca">
        <div className="tabs" role="tablist">
          <button id="tab-text" role="tab" aria-selected={tab === "text"} className="tab" onClick={() => setTab("text")}>
            Por texto
          </button>
          <button id="tab-image" role="tab" aria-selected={tab === "image"} className="tab" onClick={() => setTab("image")}>
            Por imagem
          </button>
        </div>

        <form onSubmit={search}>
          <div className="searchrow">
            {tab === "text" ? (
              <input
                id="query-input"
                className="input"
                placeholder='Ex.: "instrumentos musicais utilizados em cerimônias", "objetos utilizados em cerimônias religiosas"'
                value={text}
                onChange={(e) => setText(e.target.value)}
                autoFocus
              />
            ) : (
              <div
                id="image-drop"
                className={`drop ${over ? "over" : ""}`}
                onClick={() => fileRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setOver(true); }}
                onDragLeave={() => setOver(false)}
                onDrop={(e) => { e.preventDefault(); setOver(false); pickFile(e.dataTransfer.files[0]); }}
              >
                {preview ? <img src={preview} alt="Imagem enviada" /> : null}
                <span>{file ? file.name : "Arraste uma imagem ou clique para escolher (JPEG, PNG, WebP · até 10 MB)"}</span>
                <input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(e) => pickFile(e.target.files[0])} />
              </div>
            )}
            <button id="search-btn" className="btn" disabled={loading}>{loading ? "Buscando…" : "Buscar"}</button>
          </div>

          <div className="options">
            {tab === "text" ? (
              <label className="field">Representação / Método
                <select id="mode-select" className="select" value={mode} onChange={(e) => setMode(e.target.value)}>
                  {MODES.map((m) => <option key={m.v} value={m.v}>{m.label}</option>)}
                </select>
              </label>
            ) : (
              <label className="field">Modelo Visual (R1)
                <select id="vmodel-select" className="select" value={visualModel} onChange={(e) => setVisualModel(e.target.value)}>
                  <option value="clip">CLIP ViT-B/32</option>
                  <option value="siglip">SigLIP Multilíngue</option>
                </select>
              </label>
            )}
            <label className="field">Resultados
              <select id="topk-select" className="select" value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
                {[6, 12, 24, 48].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <label className="field">Categoria
              <input id="filter-categoria" className="small" placeholder="ex.: Maracá, Cocar" value={categoria} onChange={(e) => setCategoria(e.target.value)} />
            </label>
            <label className="field">Estado
              <input id="filter-estado" className="small" placeholder="ex.: MS" value={estado} onChange={(e) => setEstado(e.target.value)} />
            </label>
          </div>
        </form>
      </section>

      <section aria-live="polite">
        {error && <div className="notice error" style={{ marginTop: 28 }}>{error}</div>}

        {loading && (
          <div className="grid" style={{ marginTop: 34 }}>
            {Array.from({ length: 6 }).map((_, i) => <div key={i} className="skeleton" />)}
          </div>
        )}

        {!loading && data && (
          <>
            <div className="status">
              <h2>{data.total} {data.total === 1 ? "objeto encontrado" : "objetos encontrados"}</h2>
              <span>{data.modo}</span>
            </div>
            {data.aviso && <div className="notice">{data.aviso}</div>}
            <div className="grid">
              {data.resultados.map((r, i) => (
                <button key={r.item_id} className="card" onClick={() => openItem(r)}>
                  <div className="thumb">
                    <img src={`${API}${r.imagem_url}?w=480`} alt={r.descricao?.slice(0, 80) || r.item_id} loading="lazy" />
                    <span className="rank">#{i + 1}</span>
                  </div>
                  <div className="body">
                    <h3>
                      {r.categoria || r.descricao?.split(/[./]/)[0] || r.item_id}
                      {r.identificacao_local ? ` (${r.identificacao_local})` : ""}
                    </h3>
                    <p className="desc">{r.descricao_catalogo || r.descricao}</p>
                    <div className="badges">
                      <span className="badge">{r.item_id}</span>
                      {r.material && <span className="badge">{r.material}</span>}
                      {Object.entries(r.scores || {}).map(([k, v]) => (
                        <span key={k} className="badge score">{k} {v}</span>
                      ))}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </>
        )}
      </section>

      <footer>Projeto de TCC (CEFET/RJ) · resultados refletem exclusivamente os registros da base consultada.</footer>

      {item && (
        <div className="overlay" onClick={() => setItem(null)} role="dialog" aria-modal="true">
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ position: "relative" }}>
            <button id="close-modal" className="close" aria-label="Fechar" onClick={() => setItem(null)}>×</button>
            <div>
              <img className="main-img" src={`${API}/images/${activeImg}?w=900`} alt={item.item_id} />
              <div className="thumbs">
                {item.imagens?.map((im) => (
                  <img key={im.image_id} className={im.image_id === activeImg ? "on" : ""}
                    src={`${API}/images/${im.image_id}?w=128`} alt={im.image_id} onClick={() => setActiveImg(im.image_id)} />
                ))}
              </div>
            </div>
            <div className="meta">
              <h2>
                {item.categoria || item.item_id}
                {item.identificacao_local ? ` / ${item.identificacao_local}` : ""}
              </h2>
              {item.descricao_catalogo && (
                <p style={{ color: "var(--text)", marginBottom: 8, fontSize: 14 }}>
                  <b>Descrição física (R2):</b> {item.descricao_catalogo}
                </p>
              )}
              <p style={{ color: "var(--text-muted)", fontSize: 13 }}>{item.descricao}</p>
              {item.notas_gerais && (
                <p style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 8, fontStyle: "italic" }}>
                  <b>Notas etnográficas:</b> {item.notas_gerais}
                </p>
              )}
              <div className="badges" style={{ marginTop: 14 }}>
                {item.tags?.map((t) => <span key={t} className="badge">{t}</span>)}
              </div>
              <div className="src">
                <div><b>Código do acervo:</b> {item.fonte?.codigo_acervo || item.item_id}</div>
                {item.funcao && <div><b>Função:</b> {item.funcao}</div>}
                {item.material && <div><b>Material/Suporte:</b> {item.material}</div>}
                {item.tecnica && <div><b>Técnica:</b> {item.tecnica}</div>}
                {item.dimensoes && <div><b>Dimensões:</b> {item.dimensoes}</div>}
                {item.artesao && <div><b>Autor/Artesão:</b> {item.artesao}</div>}
                {item.povo && <div><b>Povo / Aldeia:</b> {item.povo}{item.aldeia ? ` — ${item.aldeia}` : ""}</div>}
                <div><b>Coleção:</b> {item.titulo || "—"}</div>
                <div><b>Fonte:</b> {item.fonte?.catalogo || "—"}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
