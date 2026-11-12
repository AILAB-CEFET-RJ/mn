"""Teste de aceite imagem→imagem: python ingestion/search_image.py <imagem> [top_k] [clip|siglip]
Resultados agrupados por item_id (um objeto pode ter várias fotos)."""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.search_service import SearchService  # noqa: E402  # pylint: disable=wrong-import-position


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    visual_model = sys.argv[3] if len(sys.argv) > 3 else "clip"

    svc = SearchService()
    with Image.open(path) as im:
        results = svc.search_image(im, top_k=top_k, visual_model=visual_model)

    for r in results:
        score = r["scores"].get(visual_model, 0.0)
        desc = (r.get("descricao") or r.get("descricao_catalogo") or "")[:80]
        print(f"{score:.4f}  {r['item_id']}  {r['image_id']}  {desc}")
    if not results:
        print("Nenhum registro correspondente encontrado na base consultada.")


if __name__ == "__main__":
    main()
