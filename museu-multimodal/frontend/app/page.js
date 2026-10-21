"use client";

import { useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const MODES = [
  { v: "hybrid", label: "Híbrido (texto + imagem)" },
  { v: "descriptions", label: "Descrições" },
  { v: "images", label: "Imagens (CLIP)" },
];

export default function Home() {
  const [tab, setTab] = useState("text");
  const [text, setText] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [topK, setTopK] = useState(12);
  const [estado, setEstado] = useState("");
  const [pais, setPais] = useState("");
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
        if (estado) filtros.estado = estado;
        if (pais) filtros.pais = pais;
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
        if (estado) fd.append("estado", estado);
        if (pais) fd.append("pais", pais);
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
        <span className="eyebrow">Museu Nacional</span>
        <h1>Busca multimodal no acervo</h1>
        <p>Pesquise por texto ou envie uma foto. Os resultados vêm dos registros catalográficos, com suas fontes.</p>
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
                placeholder='Ex.: "maracá", "cocar de penas", "instrumento musical"'
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
            {tab === "text" && (
              <label className="field">Modo
                <select id="mode-select" className="select" value={mode} onChange={(e) => setMode(e.target.value)}>
                  {MODES.map((m) => <option key={m.v} value={m.v}>{m.label}</option>)}
                </select>
              </label>
            )}
            <label className="field">Resultados
              <select id="topk-select" className="select" value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
                {[6, 12, 24, 48].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <label className="field">Estado
              <input id="filter-estado" className="small" placeholder="ex.: MS" value={estado} onChange={(e) => setEstado(e.target.value)} />
            </label>
            <label className="field">País
              <input id="filter-pais" className="small" placeholder="ex.: Brasil" value={pais} onChange={(e) => setPais(e.target.value)} />
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
              <span>{data.modo === "image" ? "imagem → imagem" : `texto · ${data.modo}`}</span>
            </div>
            {data.aviso && <div className="notice">{data.aviso}</div>}
            <div className="grid">
              {data.resultados.map((r, i) => (
                <button key={r.item_id} className="card" style={{ animationDelay: `${i * 40}ms` }} onClick={() => openItem(r)}>
                  <div className="thumb">
                    <img src={`${API}${r.imagem_url}?w=480`} alt={r.descricao?.slice(0, 80) || r.item_id} loading="lazy" />
                    <span className="rank">#{i + 1}</span>
                  </div>
                  <div className="body">
                    <h3>{r.descricao?.split(/[./]/)[0] || r.item_id}</h3>
                    <p className="desc">{r.descricao}</p>
                    <div className="badges">
                      <span className="badge">{r.item_id}</span>
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

      <footer>Projeto de TCC · resultados refletem apenas os registros da base consultada.</footer>

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
              <h2>{item.descricao?.split(/[./]/)[0] || item.item_id}</h2>
              <p style={{ color: "var(--muted)" }}>{item.descricao}</p>
              <div className="badges" style={{ marginTop: 14 }}>
                {item.tags?.map((t) => <span key={t} className="badge">{t}</span>)}
              </div>
              <div className="src">
                <div><b>Código do acervo:</b> {item.fonte?.codigo_acervo || item.item_id}</div>
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
