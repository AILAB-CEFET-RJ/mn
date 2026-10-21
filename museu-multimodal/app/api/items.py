"""GET /items/{item_id} e GET /images/{image_id}[?w=] (miniaturas em cache)."""
import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from PIL import Image
from qdrant_client import models

from app.api.search import serialize

router = APIRouter(tags=["itens"])
ROOT = Path(__file__).resolve().parents[3]  # raiz do projeto (caminho_imagem é relativo a ela)
CACHE = Path(__file__).resolve().parents[2] / "data" / "thumbs"
COLLECTION = "museum_images"


def _by(request: Request, field: str, value: str, limit: int = 100) -> list[dict]:
    pts, _ = request.app.state.svc.client.scroll(
        COLLECTION, scroll_filter=models.Filter(must=[
            models.FieldCondition(key=field, match=models.MatchValue(value=value))]),
        limit=limit, with_payload=True, with_vectors=False)
    return [p.payload for p in pts]


@router.get("/items/{item_id}")
def get_item(item_id: str, request: Request):
    pays = _by(request, "item_id", item_id)
    if not pays:
        raise HTTPException(404, "Item não encontrado na base consultada.")
    pays.sort(key=lambda p: p["image_id"])
    base = serialize(pays[0])
    base.pop("image_id"), base.pop("imagem_url"), base.pop("scores"), base.pop("rrf")
    base["imagens"] = [{"image_id": p["image_id"], "url": f"/images/{p['image_id']}"} for p in pays]
    return base


@router.get("/images/{image_id}")
def get_image(image_id: str, request: Request, w: int = Query(0, ge=0, le=2000)):
    """Serve a imagem pelo image_id (nunca por caminho informado pelo cliente)."""
    pays = _by(request, "image_id", image_id, limit=1)
    if not pays:
        raise HTTPException(404, "Imagem não encontrada.")
    path = ROOT / pays[0]["caminho_imagem"]
    if not path.is_file():
        raise HTTPException(404, "Arquivo de imagem ausente no disco.")
    if not w:
        return FileResponse(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    thumb = CACHE / f"{hashlib.sha1(image_id.encode()).hexdigest()}_{w}.jpg"
    if not thumb.exists():
        with Image.open(path) as im:
            im.draft("RGB", (w * 2, w * 2))
            im = im.convert("RGB")
            im.thumbnail((w, w))
            im.save(thumb, "JPEG", quality=85)
    return FileResponse(thumb, headers={"Cache-Control": "public, max-age=86400"})
