"""Etapa 1 da Metodologia (Seção 7.1 do PDF): Análise exploratória e preparação da base.

Investiga e gera estatísticas reprodutíveis sobre:
- número de objetos e número de imagens por objeto;
- campos disponíveis e quantidade de valores ausentes;
- distribuição das categorias, funções, materiais, técnicas e aldeias;
- qualidade e extensão das descrições (R2 vs R4);
- qualidade, resolução (MP) e tamanho (MB) das imagens;
- existência de registros duplicados e consistência entre arquivos no disco e catálogo.

Salva `data/processed/eda_report.json` e `data/processed/eda_report.md`.
"""
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

# Ajusta sys.path para funcionar tanto como script direto quanto como módulo do pacote
_MUSEU_ROOT = str(Path(__file__).resolve().parents[1])
if _MUSEU_ROOT not in sys.path:
    sys.path.insert(0, _MUSEU_ROOT)

from ingestion.load_catalog import BASE, META_DIR, PROCESSED, load_catalog  # noqa: E402


def _stats(nums: Sequence[float | int]) -> dict[str, float | int]:
    if not nums:
        return {"min": 0, "max": 0, "media": 0, "mediana": 0}
    return {
        "min": round(min(nums), 2),
        "max": round(max(nums), 2),
        "media": round(statistics.mean(nums), 2),
        "mediana": round(statistics.median(nums), 2),
    }


def analyze() -> dict:
    records = load_catalog()
    by_item: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_item[r["item_id"]].append(r)

    items = {iid: imgs[0] for iid, imgs in by_item.items()}
    imgs_per_item = [len(v) for v in by_item.values()]

    # Consistência disco vs CSVs e valores ausentes no BaseMN_*.csv
    raw_missing: dict[str, dict] = {}
    disk_consistency: dict[str, dict] = {}
    for csv_path in sorted(META_DIR.glob("*.csv")):
        stem = csv_path.stem
        img_dir = BASE / stem
        base_csv = BASE / f"Base{stem}.csv"
        target_csv = base_csv if base_csv.is_file() else csv_path
        with open(target_csv, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f, delimiter=";"))
        total_rows = len(rows)
        files_in_csv = [(r.get("nome_arquivo") or "").strip() for r in rows]
        unique_files_csv = set(files_in_csv)
        duplicates_in_csv = total_rows - len(unique_files_csv)
        files_on_disk = {p.name for p in img_dir.glob("*") if p.is_file()} if img_dir.is_dir() else set()

        disk_consistency[stem] = {
            "linhas_csv": total_rows,
            "arquivos_unicos_csv": len(unique_files_csv),
            "duplicatas_csv": duplicates_in_csv,
            "arquivos_no_disco": len(files_on_disk),
            "no_csv_mas_ausentes_no_disco": sorted(unique_files_csv - files_on_disk),
            "no_disco_mas_ausentes_no_csv": sorted(files_on_disk - unique_files_csv),
        }

        if rows:
            for col in rows[0].keys():
                empty = sum(1 for r in rows if not (r.get(col) or "").strip())
                raw_missing[col] = {
                    "ausentes": empty,
                    "preenchidos": total_rows - empty,
                    "taxa_ausencia_pct": round(100.0 * empty / total_rows, 1),
                }

    # Distribuições no nível de objeto (35 objetos) e de imagem (113 imagens)
    cat_obj = Counter(it.get("categoria") or "Sem categoria" for it in items.values())
    cat_img = Counter(r.get("categoria") or "Sem categoria" for r in records)
    mat_obj = Counter(m for it in items.values() for m in (it.get("materiais_lista") or ["Não informado"]))
    func_obj = Counter(fn for it in items.values() for fn in (it.get("funcoes_lista") or ["Não informado"]))
    tec_obj = Counter(it.get("tecnica") or "Não informada" for it in items.values())
    aldeia_obj = Counter(it.get("aldeia") or "Não informada" for it in items.values())

    # Qualidade das descrições (em palavras e caracteres, por objeto)
    words_r2 = [len((it.get("descricao_catalogo") or "").split()) for it in items.values()]
    chars_r2 = [len(it.get("descricao_catalogo") or "") for it in items.values()]
    words_leg = [len((it.get("descricao") or "").split()) for it in items.values()]
    words_r4 = [len((it.get("texto_r4") or "").split()) for it in items.values()]

    # Qualidade e resolução das imagens
    mps = [r["megapixels"] for r in records if r.get("megapixels") is not None]
    mbs = [r["tamanho_mb"] for r in records if r.get("tamanho_mb") is not None]
    widths = [r["largura_pixels"] for r in records if r.get("largura_pixels") is not None]
    heights = [r["altura_pixels"] for r in records if r.get("altura_pixels") is not None]
    orient = Counter(r.get("orientacao") or "desconhecida" for r in records)

    report = {
        "resumo_geral": {
            "total_imagens_validas": len(records),
            "total_objetos_unicos": len(items),
            "imagens_por_objeto": _stats(imgs_per_item),
            "distribuicao_imagens_por_objeto": dict(sorted(Counter(imgs_per_item).items())),
        },
        "consistencia_e_duplicatas": disk_consistency,
        "distribuicoes_por_objeto": {
            "categorias": dict(cat_obj.most_common()),
            "categorias_por_imagem": dict(cat_img.most_common()),
            "materiais": dict(mat_obj.most_common()),
            "funcoes": dict(func_obj.most_common()),
            "tecnicas": dict(tec_obj.most_common()),
            "aldeias": dict(aldeia_obj.most_common()),
        },
        "qualidade_descricoes": {
            "descricao_catalogo_R2_palavras": _stats(words_r2),
            "descricao_catalogo_R2_caracteres": _stats(chars_r2),
            "legenda_descricao_palavras": _stats(words_leg),
            "representacao_completa_R4_palavras": _stats(words_r4),
            "textos_R2_unicos": len({it.get("texto_r2") for it in items.values()}),
            "textos_R4_unicos": len({it.get("texto_r4") for it in items.values()}),
        },
        "qualidade_imagens": {
            "megapixels": _stats(mps),
            "tamanho_mb": _stats(mbs),
            "largura_px": _stats(widths),
            "altura_px": _stats(heights),
            "orientacao": dict(orient),
        },
        "campos_e_valores_ausentes": raw_missing,
    }
    return report


def to_markdown(rep: dict) -> str:
    g = rep["resumo_geral"]
    imo = g["imagens_por_objeto"]
    qd = rep["qualidade_descricoes"]
    qi = rep["qualidade_imagens"]
    lines = [
        "# Relatório de Análise Exploratória da Base (Etapa 1 — Seção 7.1)",
        "",
        "## 1. Visão Geral e Granularidade",
        f"- **Objetos museológicos únicos (`codigo_acervo`):** {g['total_objetos_unicos']}",
        f"- **Fotografias válidas indexadas:** {g['total_imagens_validas']}",
        f"- **Imagens por objeto:** média = {imo['media']}, mediana = {imo['mediana']}, mín = {imo['min']}, máx = {imo['max']}",
        f"- **Distribuição (nº de fotos → nº de objetos):** `{g['distribuicao_imagens_por_objeto']}`",
        "",
        "## 2. Consistência entre Catálogo e Arquivos de Imagem",
    ]
    for stem, c in rep["consistencia_e_duplicatas"].items():
        lines.extend([
            f"- **Coleção `{stem}`:** {c['linhas_csv']} linhas no CSV, {c['arquivos_no_disco']} arquivos no disco, "
            f"{c['duplicatas_csv']} duplicatas, {len(c['no_csv_mas_ausentes_no_disco'])} arquivos ausentes no disco.",
        ])
    lines.extend([
        "",
        "## 3. Distribuição das Categorias (`catalogo_identificação_museológica`)",
        "| Categoria | Objetos | Fotografias |",
        "|---|---:|---:|",
    ])
    cat_img = rep["distribuicoes_por_objeto"]["categorias_por_imagem"]
    for cat, n_obj in rep["distribuicoes_por_objeto"]["categorias"].items():
        lines.append(f"| {cat} | {n_obj} | {cat_img.get(cat, 0)} |")

    lines.extend([
        "",
        "## 4. Distribuição de Funções, Materiais e Técnicas (por objeto)",
        f"- **Funções (`catalogo_função`):** {rep['distribuicoes_por_objeto']['funcoes']}",
        f"- **Materiais (`catalogo_material_suporte`):** {rep['distribuicoes_por_objeto']['materiais']}",
        f"- **Técnicas (`catalogo_técnica`):** {rep['distribuicoes_por_objeto']['tecnicas']}",
        "",
        "## 5. Qualidade das Descrições Textuais",
        f"- **Descrição física pura (`R2` - `catalogo_descrição`):** média de {qd['descricao_catalogo_R2_palavras']['media']} palavras "
        f"(mín {qd['descricao_catalogo_R2_palavras']['min']}, máx {qd['descricao_catalogo_R2_palavras']['max']}), "
        f"{qd['textos_R2_unicos']} textos únicos entre {g['total_objetos_unicos']} objetos.",
        f"- **Legenda catalográfica (`descricao`):** média de {qd['legenda_descricao_palavras']['media']} palavras.",
        f"- **Representação enriquecida (`R4` - descrição + metadados):** média de {qd['representacao_completa_R4_palavras']['media']} palavras "
        f"({qd['textos_R4_unicos']} textos únicos).",
        "",
        "## 6. Qualidade e Resolução das Imagens",
        f"- **Resolução (Megapixels):** média = {qi['megapixels']['media']} MP (mín {qi['megapixels']['min']} MP, máx {qi['megapixels']['max']} MP)",
        f"- **Dimensões (px):** largura média = {qi['largura_px']['media']} px, altura média = {qi['altura_px']['media']} px",
        f"- **Tamanho do arquivo (MB):** média = {qi['tamanho_mb']['media']} MB (mín {qi['tamanho_mb']['min']} MB, máx {qi['tamanho_mb']['max']} MB)",
        f"- **Orientação:** `{qi['orientacao']}`",
        "",
        "## 7. Campos Catalográficos e Taxa de Valores Ausentes (`BaseMN_003.csv`)",
        "| Campo | Preenchidos | Ausentes | % Ausência |",
        "|---|---:|---:|---:|",
    ])
    for col, st in rep["campos_e_valores_ausentes"].items():
        if col.startswith("catalogo_") or col in ("descricao", "palavras_chave", "cidade", "estado", "pais"):
            lines.append(f"| `{col}` | {st['preenchidos']} | {st['ausentes']} | {st['taxa_ausencia_pct']}% |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    rep = analyze()
    (PROCESSED / "eda_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    (PROCESSED / "eda_report.md").write_text(to_markdown(rep))
    print("Relatório salvo em data/processed/eda_report.json e eda_report.md")
