"""Etapa 5 (Seção 7.5 do PDF): análise de erros CALCULADA a partir dos dados.

Nenhuma conclusão é escrita à mão: cada frase do relatório é gerada a partir de números medidos aqui.
Fatores do PDF e como são tratados:
- semelhança visual entre categorias diferentes → confusões top-1 em imagem→imagem (CLIP e SigLIP);
- descrições curtas/incompletas → cobertura do termo da categoria no texto R2 e desempenho por tamanho de texto;
- terminologia museológica específica → consultas pelo nome indígena (`identificacao_local`);
- diferenças culturais/linguísticas dos modelos → comparação PT × EN por consulta;
- informação visível mas ausente na descrição / presente nos metadados mas invisível → NÃO calculável
  automaticamente; o relatório lista as consultas piores para revisão manual.

Saída: evaluation/error_analysis.md e error_analysis.json. Requer cross_modal_results.json (rode cross_modal.py antes).
"""
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

# Ajusta sys.path para funcionar tanto como script direto quanto como módulo do pacote
_MUSEU_ROOT = str(Path(__file__).resolve().parents[1])
if _MUSEU_ROOT not in sys.path:
    sys.path.insert(0, _MUSEU_ROOT)

from app.services.search_service import SearchService  # noqa: E402
from evaluation.evaluate import REPO, ROOT, metrics  # noqa: E402

HERE = Path(__file__).parent
STOP = {"de", "da", "do", "e", "a", "o"}
CAT_MODES = ["bm25_r2", "R2", "R1_siglip", "R3_siglip", "R4_text"]
LOCAL_MODES = ["R1_clip", "R1_siglip", "R2", "R4_text", "bm25_r2", "bm25_r4"]


def fold(s: str) -> str:
    n = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in n if not unicodedata.combining(c))


def roots(term: str) -> list[str]:
    """Raiz (prefixo) de cada palavra do termo, para tolerar plural e flexão simples."""
    return [w[:max(4, len(w) - 2)] for w in re.findall(r"[a-z0-9]+", fold(term)) if w not in STOP and len(w) > 1]


def covers(text: str, term: str) -> bool:
    f = fold(text)
    return all(r in f for r in roots(term))


def md_table(headers: list[str], rows: Sequence[Sequence[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def mean(xs: Iterable[float | int | bool]) -> float:
    vals = [float(x) for x in xs]
    return round(statistics.mean(vals), 4) if vals else 0.0


def main():
    q = json.loads((HERE / "queries.json").read_text())
    meta = json.loads((ROOT / "data" / "processed" / "metadata.json").read_text())
    items: dict[str, dict] = {}
    for r in meta:
        items.setdefault(r["item_id"], r)
    svc = SearchService()

    def ap(query: str, mode: str, rel: set[str]) -> float:
        res = svc.search_text(query, mode, top_k=10)
        return metrics([r["item_id"] for r in res], rel)["map@10"]

    report: dict = {}
    md = ["# Análise de Erros (Etapa 5 — Seção 7.5)", "",
          "> Todos os números abaixo foram calculados por `evaluation/error_analysis.py`. "
          "Amostra: 35 objetos de uma única coleção; diferenças pequenas não têm teste de significância.", ""]

    # ---- 1. Semelhança visual (imagem→imagem) ----
    ids = list(items)
    diff_pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:] if items[a]["categoria"] != items[b]["categoria"]]
    base_share = mean(bool(set(items[a].get("materiais_lista") or []) & set(items[b].get("materiais_lista") or []))
                      for a, b in diff_pairs)
    md += ["## 1. Semelhança visual entre categorias diferentes (imagem→imagem, leave-one-out)", ""]
    report["confusoes_visuais"] = {}
    for model, label in (("clip", "CLIP"), ("siglip", "SigLIP")):
        wrong, pairs = [], Counter()
        for x in q["consultas_imagem"]:
            rel = set(x["relevantes"])
            with Image.open(REPO / x["caminho_imagem"]) as im:
                top = svc.search_image(im, 1, exclude_item=x["item_id"], visual_model=model)[0]
            if top["item_id"] not in rel:
                real, pred = items[x["item_id"]], items[top["item_id"]]
                shared = sorted(set(real.get("materiais_lista") or []) & set(pred.get("materiais_lista") or []))
                wrong.append({"consulta": x["item_id"], "imagem": x["image_id"], "categoria_real": x["categoria"],
                              "top1": top["item_id"], "categoria_predita": pred["categoria"],
                              "materiais_em_comum": shared})
                pairs[(x["categoria"], pred["categoria"])] += 1
        n = len(q["consultas_imagem"])
        sh = sum(1 for w in wrong if w["materiais_em_comum"])
        report["confusoes_visuais"][label] = {"consultas": n, "erros_top1": len(wrong), "casos": wrong}
        md += [f"### {label}: {len(wrong)} de {n} consultas ({100 * len(wrong) / n:.0f}%) tiveram o top-1 em outra categoria", ""]
        if wrong:
            md += md_table(["Consulta", "Categoria real", "Top-1 (categoria predita)", "Material em comum"],
                           [[w["consulta"], w["categoria_real"], f"{w['top1']} ({w['categoria_predita']})",
                             ", ".join(w["materiais_em_comum"]) or "nenhum"] for w in wrong])
            md += ["", "Pares mais frequentes (real → predita): " +
                   "; ".join(f"{a} → {b} ({c}×)" for (a, b), c in pairs.most_common(5)),
                   f"Em {sh} de {len(wrong)} erros os dois objetos compartilham ao menos um material "
                   f"(taxa-base entre pares de categorias diferentes: {100 * base_share:.0f}%).", ""]

    # ---- 2. Categorias: R2 incompleto? cobertura do termo e AP por método ----
    rows: list[dict[str, Any]] = []
    cov_pairs = []
    for x in q["consultas_categoria"]:
        rel, cat = set(x["relevantes"]), x["texto_pt"]
        cov = mean(covers(items[i]["texto_r2"], cat) for i in rel)
        aps = {m: round(ap(cat, m, rel), 3) for m in CAT_MODES}
        rows.append({"categoria": cat, "objetos": len(rel), "cobertura_termo_R2": cov, **aps})
        cov_pairs.append((cov, aps["R2"], aps["bm25_r2"]))
    low = [r for r in rows if r["cobertura_termo_R2"] < 0.5]
    high = [r for r in rows if r["cobertura_termo_R2"] >= 0.5]
    report["por_categoria"] = rows
    md += ["## 2. Descrições incompletas: o termo da categoria aparece no texto R2?", "",
           "Cobertura = fração dos objetos da categoria cujo `catalogo_descrição` contém a raiz do nome da categoria "
           "(prefixo; tolera plural, mas pode acusar falso positivo). AP@10 por método (consulta = nome da categoria).", ""]
    md += md_table(["Categoria", "Objetos", "Cobertura R2", *CAT_MODES],
                   [[r["categoria"], r["objetos"], r["cobertura_termo_R2"], *[r[m] for m in CAT_MODES]]
                    for r in sorted(rows, key=lambda r: r["cobertura_termo_R2"])])
    md += ["", f"AP@10 médio de R2: {mean(r['R2'] for r in low)} nas {len(low)} categorias com cobertura < 50% "
               f"contra {mean(r['R2'] for r in high)} nas {len(high)} com cobertura ≥ 50%. "
               f"Para BM25-R2: {mean(r['bm25_r2'] for r in low)} contra {mean(r['bm25_r2'] for r in high)}.",
           "Ressalva: o rótulo de categoria também faz parte do texto de R4, então a vantagem de R4 nesta tabela é "
           "em parte construída (ver notas_tcc.md).", ""]

    # ---- 3. PT × EN por consulta ----
    pt_en: dict[str, list[dict[str, Any]]] = {}
    md += ["## 3. Diferenças linguísticas: consulta em português × inglês (ramo visual, AP@10)", ""]
    for mode, label in (("R1_clip", "CLIP"), ("R1_siglip", "SigLIP")):
        d: list[dict[str, Any]] = []
        for x in q["consultas_texto"]:
            rel = set(x["relevantes"])
            pt_val = round(ap(x["texto_pt"], mode, rel), 3)
            en_val = round(ap(x["texto_en"], mode, rel), 3)
            d.append({
                "id": x["id"],
                "consulta": x["texto_pt"],
                "pt": pt_val,
                "en": en_val,
                "delta": round(en_val - pt_val, 3),
            })
        pt_en[label] = d
        better = sum(1 for e in d if e["delta"] > 0.05)
        worse = sum(1 for e in d if e["delta"] < -0.05)
        md += [f"- **{label}**: AP médio PT = {mean(e['pt'] for e in d)}, EN = {mean(e['en'] for e in d)}; "
               f"inglês melhor em {better}, pior em {worse}, equivalente (|Δ| ≤ 0,05) em {len(d) - better - worse} de {len(d)} consultas."]
    md += [""]
    report["pt_vs_en"] = pt_en

    # ---- 4. Terminologia indígena ----
    by_local: dict[str, set[str]] = defaultdict(set)
    for iid, r in items.items():
        if r.get("identificacao_local"):
            by_local[r["identificacao_local"]].add(iid)
    loc = {m: mean(ap(t, m, rel) for t, rel in by_local.items()) for m in LOCAL_MODES}
    report["termos_indigenas"] = {"termos": len(by_local), "ap_medio": loc}
    md += ["## 4. Terminologia museológica/indígena (consulta = denominação local, ex.: \"Mbaraka\")", "",
           f"{len(by_local)} termos distintos; AP@10 médio por método:", ""]
    md += md_table(["Método", "AP@10 médio"], [[m, v] for m, v in loc.items()])
    md += ["", "Ressalva: `R4_text` e `bm25_r4` recebem a denominação local como campo do texto, portanto o acerto é "
               "esperado por construção; a comparação informativa é entre R1 (CLIP/SigLIP), R2 e BM25-R2.", ""]

    # ---- 5. Tamanho da descrição (descrição→imagem) ----
    cm_path = HERE / "cross_modal_results.json"
    md += ["## 5. Descrições curtas: efeito no descrição→imagem", ""]
    if cm_path.exists():
        recs = json.loads(cm_path.read_text())["descricao_para_imagem"]["por_consulta"]
        recs.sort(key=lambda r: r["palavras"])
        t = len(recs) // 3
        groups = {"curtas (1º terço)": recs[:t], "médias": recs[t:2 * t], "longas (último terço)": recs[2 * t:]}
        rows5 = []
        for g, rs in groups.items():
            rows5.append([g, f"{rs[0]['palavras']}–{rs[-1]['palavras']}", len(rs),
                          mean(1 / r["rank_R1_clip"] for r in rs), mean(1 / r["rank_R1_siglip"] for r in rs)])
        md += md_table(["Grupo", "Palavras", "Objetos", "MRR CLIP", "MRR SigLIP"], rows5)
        worst = sorted(recs, key=lambda r: r["rank_R1_siglip"], reverse=True)[:5]
        report["descricao_curta"] = rows5
        report["piores_descricao_para_imagem"] = worst
        md += ["", "### Cinco piores casos (SigLIP) para revisão manual",
               "Aqui entram os fatores que o script não consegue julgar: informação visível na foto mas ausente da "
               "descrição, ou presente na descrição mas invisível na foto.", ""]
        md += md_table(["Objeto", "Categoria", "Rank do objeto correto (de 35)", "Descrição usada como consulta"],
                       [[w["item_id"], w["categoria"], w["rank_R1_siglip"], w["texto"][:140]] for w in worst])
    else:
        md += ["`cross_modal_results.json` ausente; rode `evaluation/cross_modal.py` antes."]

    (HERE / "error_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (HERE / "error_analysis.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
