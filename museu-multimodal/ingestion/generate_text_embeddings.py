"""Embeddings textuais BGE-M3 (1024-d, CLS + L2) alinhados a metadata.json.

Incremental: cache por texto (hash) em text_embeddings.cache.npz; só textos novos são embutidos.
Rode depois de generate_image_embeddings.py (que define o conjunto final).
"""
import hashlib
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from load_catalog import PROCESSED

log = logging.getLogger("text_embed")
MODEL_NAME = "BAAI/bge-m3"
BATCH = int(os.getenv("TEXT_BATCH_SIZE", "8"))


def build_text(m: dict) -> str:
    parts = [m.get("titulo"), m.get("descricao"), ", ".join(m.get("tags") or [])]
    local = ", ".join(x for x in (m.get("cidade"), m.get("estado"), m.get("pais")) if x)
    parts.append(local)
    return ". ".join(p for p in parts if p)


def text_key(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((PROCESSED / "metadata.json").read_text())
    texts = [build_text(m) for m in meta]
    unique = sorted(set(texts))

    cache_path = PROCESSED / "text_embeddings.cache.npz"
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        ck = np.load(cache_path, allow_pickle=False)
        cache = dict(zip(ck["keys"].tolist(), ck["vecs"]))
    todo = [t for t in unique if text_key(t) not in cache]
    log.info("%d registros, %d textos únicos, %d novos, dispositivo %s", len(texts), len(unique), len(todo), device)

    if todo:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
        # ordena por tamanho: menos padding, mais rápido
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

    np.save(PROCESSED / "text_embeddings.npy", np.stack([cache[text_key(t)] for t in texts]))
    man_path = PROCESSED / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man.update({"text_model": MODEL_NAME, "text_dim": 1024, "text_pooling": "cls",
                "textos_unicos": len(unique)})
    man_path.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    log.info("text_embeddings.npy salvo (%d)", len(texts))


if __name__ == "__main__":
    main()
