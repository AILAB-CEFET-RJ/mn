"""Indexa embeddings (R1 CLIP, R1 SigLIP, R2 descrição pura e R4 descrição+metadados) no Qdrant.

Vetores nomeados na coleção `museum_images`:
- `image`: CLIP ViT-B/32 (512-d, cosine)
- `siglip_image`: SigLIP Multilíngue (dim dinâmica, tipicamente 768-d, cosine)
- `text_r2`: BGE-M3 sobre `catalogo_descrição` pura (1024-d, cosine)
- `text`: BGE-M3 sobre descrição + metadados selecionados — R4 (1024-d, cosine)
"""
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models  # noqa: F401

# Ajusta sys.path para funcionar tanto como script direto quanto como módulo do pacote
_MUSEU_ROOT = str(Path(__file__).resolve().parents[1])
if _MUSEU_ROOT not in sys.path:
    sys.path.insert(0, _MUSEU_ROOT)

from app.config import COLLECTION, QDRANT_URL  # noqa: E402,F401
from app.db.qdrant_client import get_qdrant_client  # noqa: E402
from ingestion.load_catalog import PROCESSED  # noqa: E402

CHUNK = int(os.getenv("UPSERT_CHUNK", "256"))
FORCE = os.getenv("FORCE") == "1"
RECREATE = os.getenv("RECREATE") == "1"
log = logging.getLogger("index_qdrant")

KEYWORD_FIELDS = (
    "item_id",
    "categoria",
    "identificacao_local",
    "material",
    "tecnica",
    "funcao",
    "aldeia",
    "povo",
    "estado",
    "pais",
    "fonte_catalogo",
)


def create_collection(client: QdrantClient, expected_vectors: dict[str, int]):
    vectors = {
        name: models.VectorParams(size=dim, distance=models.Distance.COSINE)
        for name, dim in expected_vectors.items()
    }
    client.create_collection(COLLECTION, vectors_config=vectors)
    for field in KEYWORD_FIELDS:
        client.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)
    log.info("Coleção criada (vetores: %s)", ", ".join(f"{k}:{v}" for k, v in expected_vectors.items()))


def existing_ids(client: QdrantClient, ids: list[str]) -> set[str]:
    found: set[str] = set()
    for i in range(0, len(ids), 1000):
        pts = client.retrieve(COLLECTION, ids=ids[i:i + 1000], with_payload=False, with_vectors=False)
        found.update(str(p.id) for p in pts)
    return found


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    vecs = np.load(PROCESSED / "image_embeddings.npy")
    meta = json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))
    assert len(vecs) == len(meta), "image_embeddings e metadata desalinhados"

    siglip_path = PROCESSED / "siglip_image_embeddings.npy"
    svecs = np.load(siglip_path) if siglip_path.exists() else None
    if svecs is not None and len(svecs) != len(meta):
        svecs = None

    text_path = PROCESSED / "text_embeddings.npy"
    tvecs = np.load(text_path) if text_path.exists() else None
    if tvecs is not None:
        assert len(tvecs) == len(meta), "text_embeddings e metadata desalinhados"

    text_r2_path = PROCESSED / "text_r2_embeddings.npy"
    t2vecs = np.load(text_r2_path) if text_r2_path.exists() else None
    if t2vecs is not None and len(t2vecs) != len(meta):
        t2vecs = None

    expected_vectors: dict[str, int] = {"image": int(vecs.shape[1])}
    if svecs is not None:
        expected_vectors["siglip_image"] = int(svecs.shape[1])
    if tvecs is not None:
        expected_vectors["text"] = int(tvecs.shape[1])
    if t2vecs is not None:
        expected_vectors["text_r2"] = int(t2vecs.shape[1])

    client = get_qdrant_client(timeout=120)
    exists = client.collection_exists(COLLECTION)
    if exists and not RECREATE:
        cfg = client.get_collection(COLLECTION).config
        cur_vectors = cfg.params.vectors if cfg and cfg.params else None
        recreate_now = not isinstance(cur_vectors, dict) or any(k not in cur_vectors for k in expected_vectors)
    else:
        recreate_now = exists

    if recreate_now:
        client.delete_collection(COLLECTION)
        exists = False
    if not exists:
        create_collection(client, expected_vectors)

    todo = list(range(len(meta)))
    if exists and not FORCE:
        have = existing_ids(client, [m["point_id"] for m in meta])
        todo = [j for j in todo if meta[j]["point_id"] not in have]
    log.info("%d pontos no lote, %d a enviar", len(meta), len(todo))

    for i in range(0, len(todo), CHUNK):
        points = []
        for j in todo[i:i + CHUNK]:
            m = meta[j]
            payload = {k: val for k, val in m.items() if k != "point_id"}
            vector = {"image": vecs[j].tolist()}
            if svecs is not None:
                vector["siglip_image"] = svecs[j].tolist()
            if tvecs is not None:
                vector["text"] = tvecs[j].tolist()
            if t2vecs is not None:
                vector["text_r2"] = t2vecs[j].tolist()
            points.append(models.PointStruct(id=m["point_id"], vector=vector, payload=payload))
        client.upsert(COLLECTION, points=points, wait=True)
    log.info("Total na coleção: %d", client.count(COLLECTION).count)


if __name__ == "__main__":
    main()
