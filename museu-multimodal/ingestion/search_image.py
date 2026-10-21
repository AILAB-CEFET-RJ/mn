"""Teste de aceite imagem→imagem: python search_image.py <imagem> [top_k]
Resultados agrupados por item_id (um objeto pode ter várias fotos)."""
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent))
from generate_image_embeddings import MODEL_NAME, get_device  # noqa: E402
from index_qdrant import COLLECTION, QDRANT_URL  # noqa: E402


def main():
    path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    device = get_device()
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_NAME)
    with Image.open(path) as im, torch.no_grad():
        vec = model.get_image_features(**proc(images=im.convert("RGB"), return_tensors="pt").to(device))
        if not isinstance(vec, torch.Tensor):
            vec = vec.pooler_output
        vec = torch.nn.functional.normalize(vec, dim=-1)[0].cpu().tolist()

    client = QdrantClient(url=QDRANT_URL)
    res = client.query_points_groups(
        COLLECTION, query=vec, using="image", group_by="item_id", limit=top_k, group_size=1,
        with_payload=True,
    )
    for g in res.groups:
        h = g.hits[0]
        p = h.payload
        print(f"{h.score:.4f}  {p['item_id']}  {p['image_id']}  {(p.get('descricao') or '')[:80]}")
    if not res.groups:
        print("Nenhum registro correspondente encontrado na base consultada.")


if __name__ == "__main__":
    main()
