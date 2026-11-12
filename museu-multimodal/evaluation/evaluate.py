"""Etapa 4 da Metodologia (Seção 7.4 do PDF): avaliação experimental. A análise de erros fica em error_analysis.py.

Métricas calculadas (nível de objeto `item_id`):
- Precision@k, Recall@k, Mean Reciprocal Rank (MRR), Mean Average Precision (MAP@10) e nDCG@k.

Compara:
- Baselines léxicos: BM25 sobre R2 (descrição pura) e BM25 sobre R4 (descrição + metadados).
- Representações R1 (apenas imagem), R2 (apenas descrição física), R3 (imagem + descrição)
  e R4 (imagem + descrição + metadados selecionados).
- Modelos multimodais: CLIP (`openai/clip-vit-base-patch32`) vs SigLIP (`google/siglip-base-patch16-256-multilingual`).
"""
import json
import math
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

# Ajusta sys.path para funcionar tanto como script direto quanto como módulo do pacote
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import COLLECTION, REPO_ROOT  # noqa: E402
from app.services.search_service import SearchService  # noqa: E402

KS = (1, 5, 10)
REPO = REPO_ROOT


def average_precision(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, item_id in enumerate(ranked[:k], start=1):
        if item_id in relevant:
            hits += 1
            sum_prec += hits / i
    return sum_prec / min(len(relevant), k)


def metrics(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in KS:
        top = ranked[:k]
        hits = sum(1 for x in top if x in relevant)
        out[f"recall@{k}"] = hits / len(relevant) if relevant else 0.0
        out[f"precision@{k}"] = hits / k
        dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(top) if x in relevant)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    out["mrr"] = next((1 / (i + 1) for i, x in enumerate(ranked[:max(KS)]) if x in relevant), 0.0)
    out["map@10"] = average_precision(ranked, relevant, k=10)
    return out


def aggregate(per_query: Sequence[Mapping[str, float]], lat: Sequence[float]) -> dict[str, Any]:
    keys = per_query[0].keys()
    agg: dict[str, Any] = {k: round(statistics.mean(q[k] for q in per_query), 4) for k in keys}
    agg["n"] = len(per_query)
    agg["lat_ms_med"] = round(statistics.median(lat) * 1000, 1) if lat else 0.0
    return agg


def main():
    q = json.loads((Path(__file__).with_name("queries.json")).read_text())

    svc = SearchService()
    cfg = svc.client.get_collection(COLLECTION).config
    col_info = cfg.params.vectors if cfg and cfg.params else {}
    has_siglip = isinstance(col_info, dict) and "siglip_image" in col_info
    K = max(KS)

    text_configs = [
        ("Baseline BM25 (R2: só descrição)", "bm25_r2", "texto_pt"),
        ("Baseline BM25 (R4: desc + metadados)", "bm25_r4", "texto_pt"),
        ("R1: Apenas Imagem — CLIP (PT)", "R1_clip", "texto_pt"),
        ("R1: Apenas Imagem — CLIP (EN)", "R1_clip", "texto_en"),
    ]
    if has_siglip:
        text_configs.extend([
            ("R1: Apenas Imagem — SigLIP Multi (PT)", "R1_siglip", "texto_pt"),
            ("R1: Apenas Imagem — SigLIP Multi (EN)", "R1_siglip", "texto_en"),
        ])
    text_configs.extend([
        ("R2: Apenas Descrição Física — BGE-M3 (PT)", "R2", "texto_pt"),
        ("R3: Imagem (CLIP) + Descrição (R2)", "R3_clip", "texto_pt"),
    ])
    if has_siglip:
        text_configs.append(("R3: Imagem (SigLIP) + Descrição (R2)", "R3_siglip", "texto_pt"))
    text_configs.extend([
        ("R4 (textual): Desc + Metadados — BGE-M3", "R4_text", "texto_pt"),
        ("R4 (multimodal): CLIP + Desc + Metadados", "R4_clip", "texto_pt"),
    ])
    if has_siglip:
        text_configs.append(("R4 (multimodal): SigLIP + Desc + Metadados", "R4_siglip", "texto_pt"))

    # Warmup
    for _, mode, field in text_configs:
        svc.search_text(q["consultas_texto"][0][field], mode, K)

    subsets = [
        ("Tarefa 1: Busca por Categoria (n=13)", q["consultas_categoria"]),
        ("Tarefa 2: Consultas em Linguagem Natural (n=10)", q["consultas_linguagem_natural"]),
        ("Consolidado Texto (Todas as 23 consultas)", q["consultas_texto"]),
    ]

    all_results: dict[str, dict] = {}
    cols = ["recall@1", "recall@5", "recall@10", "precision@5", "ndcg@5", "ndcg@10", "mrr", "map@10", "lat_ms_med"]

    for sub_title, qs in subsets:
        sub_res = {}
        for name, mode, field in text_configs:
            pq, lat = [], []
            for x in qs:
                t0 = time.perf_counter()
                res = svc.search_text(x[field], mode, K)
                lat.append(time.perf_counter() - t0)
                pq.append(metrics([r["item_id"] for r in res], set(x["relevantes"])))
            sub_res[name] = aggregate(pq, lat)
        all_results[sub_title] = sub_res

        print(f"\n### {sub_title}")
        print("| Método / Representação | " + " | ".join(cols) + " |")
        print("|---|" + "---:|" * len(cols))
        for name, r in sub_res.items():
            print(f"| {name} | " + " | ".join(str(r[c]) for c in cols) + " |")

    # Tarefa 3: Imagem -> Imagem
    img_res = {}
    img_models = [("R1: Imagem→Imagem — CLIP (leave-one-out)", "clip")]
    if has_siglip:
        img_models.append(("R1: Imagem→Imagem — SigLIP (leave-one-out)", "siglip"))

    for name, vmod in img_models:
        pq, lat = [], []
        for x in q["consultas_imagem"]:
            with Image.open(REPO / x["caminho_imagem"]) as im:
                t0 = time.perf_counter()
                res = svc.search_image(im, K, exclude_item=x["item_id"], visual_model=vmod)
                lat.append(time.perf_counter() - t0)
            pq.append(metrics([r["item_id"] for r in res], set(x["relevantes"])))
        img_res[name] = aggregate(pq, lat)

    all_results["Tarefa 3: Busca Imagem→Imagem (n=30)"] = img_res
    print("\n### Tarefa 3: Busca Imagem→Imagem (n=30)")
    print("| Método / Modelo Visual | " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for name, r in img_res.items():
        print(f"| {name} | " + " | ".join(str(r[c]) for c in cols) + " |")

    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps({"protocolo": q["protocolo"], "resultados": all_results}, ensure_ascii=False, indent=2))

    print(f"\nSalvo {out.name}. Rode cross_modal.py e error_analysis.py em seguida.")


if __name__ == "__main__":
    main()
