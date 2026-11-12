"""Embeddings textuais BGE-M3 (1024-d, CLS + L2) para R2 (descrição pura) e R4 (descrição + metadados).

Atende à Etapa 2 da Metodologia (Seção 7.2 do PDF):
- `text_r2_embeddings.npy`: apenas `catalogo_descrição` (R2 e componente textual de R3).
- `text_embeddings.npy`: descrição + metadados catalográficos selecionados (componente textual de R4).
Incremental: cache único por hash SHA-1 em `text_embeddings.cache.npz`.
"""
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Ajusta sys.path para funcionar tanto como script direto quanto como módulo do pacote
_MUSEU_ROOT = str(Path(__file__).resolve().parents[1])
if _MUSEU_ROOT not in sys.path:
    sys.path.insert(0, _MUSEU_ROOT)

from app.config import BGE_NAME as MODEL_NAME  # noqa: E402
from ingestion.load_catalog import PROCESSED, build_text_r2, build_text_r4  # noqa: E402

log = logging.getLogger("text_embed")
BATCH = int(os.getenv("TEXT_BATCH_SIZE", "8"))


def text_key(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((PROCESSED / "metadata.json").read_text())

    texts_r2 = [m.get("texto_r2") or build_text_r2(m) for m in meta]
    texts_r4 = [m.get("texto_r4") or build_text_r4(m) for m in meta]
    unique = sorted(set(texts_r2) | set(texts_r4))

    cache_path = PROCESSED / "text_embeddings.cache.npz"
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        ck = np.load(cache_path, allow_pickle=False)
        cache = dict(zip(ck["keys"].tolist(), ck["vecs"]))
    todo = [t for t in unique if text_key(t) not in cache]
    log.info(
        "%d registros | %d textos R2 únicos | %d textos R4 únicos | %d novos a embutir (%s)",
        len(meta), len(set(texts_r2)), len(set(texts_r4)), len(todo), device,
    )

    if todo:
        tok: Any = AutoTokenizer.from_pretrained(MODEL_NAME)
        model: Any = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
        todo.sort(key=len)
        for i in tqdm(range(0, len(todo), BATCH), desc="BGE-M3"):
            batch = todo[i:i + BATCH]
            with torch.no_grad():
                enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                out = model(**enc).last_hidden_state[:, 0]
                out = torch.nn.functional.normalize(out, dim=-1).cpu().numpy().astype("float32")
            for t, v in zip(batch, out):
                cache[text_key(t)] = v
        tmp = cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(list(cache)), vecs=np.stack(list(cache.values())))
        tmp.replace(cache_path)

    np.save(PROCESSED / "text_r2_embeddings.npy", np.stack([cache[text_key(t)] for t in texts_r2]))
    np.save(PROCESSED / "text_embeddings.npy", np.stack([cache[text_key(t)] for t in texts_r4]))

    man_path = PROCESSED / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man.update({
        "text_model": MODEL_NAME,
        "text_dim": 1024,
        "text_pooling": "cls",
        "textos_r2_unicos": len(set(texts_r2)),
        "textos_r4_unicos": len(set(texts_r4)),
    })
    man_path.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    log.info("Salvos text_r2_embeddings.npy e text_embeddings.npy (%d registros)", len(meta))


if __name__ == "__main__":
    main()
