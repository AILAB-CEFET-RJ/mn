"""Tarefas cruzadas (Seção 7.3 e objetivo específico 5 do PDF):

A) descrição → imagem: a consulta é a descrição física do objeto (`catalogo_descrição`, R2) e o sistema
   deve recuperar as imagens do objeto. Só são avaliados os ramos visuais (R1: CLIP e SigLIP). R2 e BM25-R2
   não entram: o texto da consulta é idêntico ao texto indexado, então o acerto seria trivial.
B) imagem → texto: a consulta é uma fotografia e o sistema ranqueia as descrições dos objetos usando os
   encoders de texto do próprio CLIP/SigLIP (zero-shot, sem BGE-M3).

Relevância: o objeto da consulta. Objetos com descrição física idêntica contam como relevantes entre si.
Inclui baseline aleatório (embaralhamento, 200 sorteios por consulta, semente fixa).
Saída: evaluation/cross_modal_results.json (agregados + resultado por consulta).
"""
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_MUSEU_ROOT = str(Path(__file__).resolve().parents[1])
if _MUSEU_ROOT not in sys.path:
    sys.path.insert(0, _MUSEU_ROOT)

from app.services.search_service import SearchService  # noqa: E402
from evaluation.evaluate import aggregate, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = Path(__file__).with_name("cross_modal_results.json")


def rank_of(ranked: list[str], rel: set[str]) -> int:
    for i, x in enumerate(ranked, start=1):
        if x in rel:
            return i
    return len(ranked) + 1


def random_baseline(rel_sets: list[set[str]], universe: list[str], trials: int = 200, seed: int = 0) -> dict:
    rng = random.Random(seed)
    per = []
    for rel in rel_sets:
        acc: dict[str, float] = {}
        for _ in range(trials):
            order = universe[:]
            rng.shuffle(order)
            m = metrics(order, rel)
            acc = m if not acc else {k: acc[k] + m[k] for k in m}
        per.append({k: v / trials for k, v in acc.items()})
    return aggregate(per, [])


def table(title: str, rows: dict[str, dict]) -> str:
    cols = ["recall@1", "recall@5", "recall@10", "ndcg@10", "mrr_completo", "map@10", "n"]
    out = [f"\n### {title}", "| Método | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for name, r in rows.items():
        out.append(f"| {name} | " + " | ".join(str(r.get(c, "—")) for c in cols) + " |")
    return "\n".join(out)


def main():
    meta = json.loads((PROC / "metadata.json").read_text())
    items: dict[str, dict] = {}
    for r in meta:
        items.setdefault(r["item_id"], r)
    text_items: dict[str, set[str]] = defaultdict(set)
    for iid, r in items.items():
        text_items[r["texto_r2"]].add(iid)
    universe = list(items)
    svc = SearchService()

    # ---------- A) descrição → imagem ----------
    modes = [("R1_clip", "R1 — CLIP"), ("R1_siglip", "R1 — SigLIP multilíngue")]
    per_q: dict[str, list[dict]] = {m: [] for m, _ in modes}
    records = []
    for iid, r in items.items():
        rel = text_items[r["texto_r2"]]
        rec = {"item_id": iid, "categoria": r.get("categoria"),
               "palavras": len(r["texto_r2"].split()), "texto": r["texto_r2"]}
        for m, _ in modes:
            res = svc.search_text(r["texto_r2"], m, top_k=len(items))
            ranked = [x["item_id"] for x in res]
            per_q[m].append(metrics(ranked, rel))
            rec[f"rank_{m}"] = rank_of(ranked, rel)
        records.append(rec)
    res_a = {}
    for m, name in modes:
        agg = aggregate(per_q[m], [])
        agg["mrr_completo"] = round(statistics.mean(1 / x[f"rank_{m}"] for x in records), 4)
        res_a[name] = agg
    rb = random_baseline([text_items[r["texto_r2"]] for r in items.values()], universe)
    rb["mrr_completo"] = "—"
    res_a["Aleatório (esperado)"] = rb
    print(table(f"Descrição → imagem (n={len(items)} objetos)", res_a))

    # ---------- B) imagem → texto ----------
    ids_order = json.loads((PROC / "image_ids.json").read_text())
    assert len(ids_order) == len(meta)
    texts = [items[i]["texto_r2"] for i in universe]
    enc = {
        "CLIP (texto→ imagem zero-shot)": (np.load(PROC / "image_embeddings.npy"),
                                           np.array([svc.clip_text_vec(t) for t in texts])),
        "SigLIP multilíngue": (np.load(PROC / "siglip_image_embeddings.npy"),
                               np.array([svc.siglip_text_vec(t) for t in texts])),
    }
    res_b, per_query_b = {}, {}
    rel_sets = [text_items[items[r["item_id"]]["texto_r2"]] for r in meta]
    for name, (img, txt) in enc.items():
        scores = img @ txt.T
        pq, ranks = [], []
        for j, r in enumerate(meta):
            ranked = [universe[k] for k in np.argsort(-scores[j])]
            pq.append(metrics(ranked, rel_sets[j]))
            ranks.append(rank_of(ranked, rel_sets[j]))
        agg = aggregate(pq, [])
        agg["mrr_completo"] = round(statistics.mean(1 / x for x in ranks), 4)
        res_b[name] = agg
        per_query_b[name] = ranks
    rb = random_baseline(rel_sets, universe, trials=100)
    rb["mrr_completo"] = "—"
    res_b["Aleatório (esperado)"] = rb
    print(table(f"Imagem → texto (n={len(meta)} imagens, {len(universe)} descrições candidatas)", res_b))

    OUT.write_text(json.dumps({
        "descricao_para_imagem": {"agregado": res_a, "por_consulta": records},
        "imagem_para_texto": {"agregado": res_b, "ranks_por_imagem": per_query_b,
                              "image_ids": ids_order},
        "nota": "Rótulo = objeto da consulta; descrições idênticas contam como relevantes entre si. "
                "Textos truncados pelos limites dos encoders (CLIP 77 tokens, SigLIP 64).",
    }, ensure_ascii=False, indent=2))
    print(f"\nSalvo em {OUT.name}")


if __name__ == "__main__":
    main()
