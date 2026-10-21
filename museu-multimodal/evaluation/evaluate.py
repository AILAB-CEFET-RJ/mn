"""Avaliação: Recall@K, Precision@K, nDCG@K (binário, nível de objeto), MRR e latência.

Compara: imagem→imagem, texto→imagem (PT e EN), texto→descrição, híbrido (PT).
Uso: python evaluation/evaluate.py   -> imprime tabela e salva evaluation/results.json
"""
import json
import math
import statistics
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.services.search_service import SearchService  # noqa: E402

KS = (1, 5, 10)
REPO = ROOT.parent  # caminhos de imagem são relativos à raiz do projeto


def metrics(ranked: list[str], relevant: set[str]) -> dict:
    out = {}
    for k in KS:
        top = ranked[:k]
        hits = sum(1 for x in top if x in relevant)
        out[f"recall@{k}"] = hits / len(relevant) if relevant else 0.0
        out[f"precision@{k}"] = hits / k
        dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(top) if x in relevant)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    rr = next((1 / (i + 1) for i, x in enumerate(ranked[:max(KS)]) if x in relevant), 0.0)
    out["mrr"] = rr
    return out


def aggregate(per_query: list[dict], lat: list[float]) -> dict:
    keys = per_query[0].keys()
    agg = {k: round(statistics.mean(q[k] for q in per_query), 4) for k in keys}
    agg["n"] = len(per_query)
    agg["lat_ms_med"] = round(statistics.median(lat) * 1000, 1)
    agg["lat_ms_p95"] = round(sorted(lat)[int(0.95 * (len(lat) - 1))] * 1000, 1)
    return agg


def main():
    q = json.loads((Path(__file__).with_name("queries.json")).read_text())
    print("Rótulos:", q["rotulos"])
    svc = SearchService()
    K = max(KS)
    results: dict[str, dict] = {}

    def run_text(name, mode, field):
        qs = [x for x in q["consultas_texto"] if x.get(field)]
        svc.search_text(qs[0][field], mode, K)  # aquecimento (carrega modelos)
        pq, lat = [], []
        for x in qs:
            t0 = time.perf_counter()
            res = svc.search_text(x[field], mode, K)
            lat.append(time.perf_counter() - t0)
            pq.append(metrics([r["item_id"] for r in res], set(x["relevantes"])))
        results[name] = aggregate(pq, lat)

    run_text("texto→imagem (CLIP, PT)", "images", "texto_pt")
    run_text("texto→imagem (CLIP, EN)", "images", "texto_en")
    run_text("texto→descrição (BGE-M3, PT)", "descriptions", "texto_pt")
    run_text("híbrido RRF (PT)", "hybrid", "texto_pt")

    pq, lat = [], []
    for x in q["consultas_imagem"]:
        with Image.open(REPO / x["caminho_imagem"]) as im:
            t0 = time.perf_counter()
            res = svc.search_image(im, K, exclude_item=x["item_id"])
            lat.append(time.perf_counter() - t0)
        pq.append(metrics([r["item_id"] for r in res], set(x["relevantes"])))
    if pq:
        results["imagem→imagem (CLIP, leave-one-out)"] = aggregate(pq, lat)

    cols = ["recall@1", "recall@5", "recall@10", "precision@5", "ndcg@5", "ndcg@10", "mrr", "n", "lat_ms_med"]
    print("\n| método | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for name, r in results.items():
        print(f"| {name} | " + " | ".join(str(r[c]) for c in cols) + " |")
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps({"rotulos": q["rotulos"], "resultados": results}, ensure_ascii=False, indent=1))
    print(f"\nSalvo em {out.name}")


if __name__ == "__main__":
    main()
