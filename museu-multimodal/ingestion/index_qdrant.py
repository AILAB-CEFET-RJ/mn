"""Indexa embeddings + payload no Qdrant de forma incremental e idempotente.

Só envia pontos ainda ausentes na coleção (ou todos com FORCE=1, p.ex. se o payload mudou).
A coleção só é recriada se faltar o vetor `text` ou com RECREATE=1.
"""
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).parent))
from load_catalog import PROCESSED  # noqa: E402

COLLECTION = "museum_images"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CHUNK = int(os.getenv("UPSERT_CHUNK", "256"))
FORCE = os.getenv("FORCE") == "1"
RECREATE = os.getenv("RECREATE") == "1"
log = logging.getLogger("index_qdrant")

KEYWORD_FIELDS = ("item_id", "estado", "pais", "fonte_catalogo")


def create_collection(client: QdrantClient, with_text: bool):
    vectors = {"image": models.VectorParams(size=512, distance=models.Distance.COSINE)}
    if with_text:
        vectors["text"] = models.VectorParams(size=1024, distance=models.Distance.COSINE)
    client.create_collection(COLLECTION, vectors_config=vectors)
    for field in KEYWORD_FIELDS:
        client.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)
    log.info("Coleção criada (vetores: %s)", ", ".join(vectors))


def existing_ids(client: QdrantClient, ids: list[str]) -> set[str]:
    found: set[str] = set()
    for i in range(0, len(ids), 1000):
        pts = client.retrieve(COLLECTION, ids=ids[i:i + 1000], with_payload=False, with_vectors=False)
        found.update(str(p.id) for p in pts)
    return found


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    vecs = np.load(PROCESSED / "image_embeddings.npy")
    meta = json.loads((PROCESSED / "metadata.json").read_text())
    assert len(vecs) == len(meta), "embeddings e metadata desalinhados"

    text_path = PROCESSED / "text_embeddings.npy"
    tvecs = np.load(text_path) if text_path.exists() else None
    if tvecs is not None:
        assert len(tvecs) == len(meta), "text_embeddings e metadata desalinhados (rode generate_text_embeddings)"

    client = QdrantClient(url=QDRANT_URL, timeout=120)
    exists = client.collection_exists(COLLECTION)
    if exists and not RECREATE:
        has_text = "text" in client.get_collection(COLLECTION).config.params.vectors
        if tvecs is not None and not has_text:
            RECREATE_NOW = True  # não dá para adicionar vetor nomeado a coleção existente
        else:
            RECREATE_NOW = False
    else:
        RECREATE_NOW = exists
    if RECREATE_NOW:
        client.delete_collection(COLLECTION)
        exists = False
    if not exists:
        create_collection(client, tvecs is not None)

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
            if tvecs is not None:
                vector["text"] = tvecs[j].tolist()
            points.append(models.PointStruct(id=m["point_id"], vector=vector, payload=payload))
        client.upsert(COLLECTION, points=points, wait=True)
    log.info("Total na coleção: %d", client.count(COLLECTION).count)


if __name__ == "__main__":
    main()
