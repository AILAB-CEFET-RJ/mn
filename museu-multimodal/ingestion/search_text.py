"""CLI: python ingestion/search_text.py "<consulta>" [images|descriptions|hybrid] [top_k] [campo=valor ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.search_service import SearchService  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    query = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    filters = dict(a.split("=", 1) for a in sys.argv[4:])
    results = SearchService().search_text(query, mode, top_k, filters)
    for r in results:
        print(f"{r['scores']}  rrf={r['rrf']}  {r['item_id']}  {r['image_id']}  {(r.get('descricao') or '')[:80]}")
    if not results:
        print("Nenhum registro correspondente encontrado na base consultada.")


if __name__ == "__main__":
    main()
