"""Endpoints de busca: POST /search/text e POST /search/image."""
import io
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

router = APIRouter(prefix="/search", tags=["busca"])

MAX_UPLOAD = 10 * 1024 * 1024
ALLOWED = {"image/jpeg", "image/png", "image/webp"}
FILTER_FIELDS = {
    "categoria",
    "identificacao_local",
    "material",
    "tecnica",
    "funcao",
    "aldeia",
    "povo",
    "estado",
    "pais",
    "item_id",
    "fonte_catalogo",
}

SearchMode = Literal[
    "hybrid",
    "descriptions",
    "images",
    "R1_clip",
    "R1_siglip",
    "R2",
    "R3",
    "R3_clip",
    "R3_siglip",
    "R4",
    "R4_text",
    "R4_clip",
    "R4_siglip",
    "bm25",
    "bm25_r2",
    "bm25_r4",
]


class TextQuery(BaseModel):
    texto: str = Field(min_length=1, max_length=500)
    modo: SearchMode = "hybrid"
    top_k: int = Field(default=10, ge=1, le=50)
    filtros: dict[str, str] = {}


def _check_filters(f: dict[str, str]):
    bad = set(f) - FILTER_FIELDS
    if bad:
        raise HTTPException(422, f"Filtros inválidos: {sorted(bad)}. Permitidos: {sorted(FILTER_FIELDS)}")


def serialize(r: dict) -> dict:
    """Resultado com a fonte do registro e metadados catalográficos completos."""
    return {
        "item_id": r["item_id"],
        "image_id": r["image_id"],
        "categoria": r.get("categoria"),
        "identificacao_local": r.get("identificacao_local"),
        "titulo": r.get("titulo"),
        "descricao_catalogo": r.get("descricao_catalogo"),
        "descricao": r.get("descricao"),
        "material": r.get("material"),
        "tecnica": r.get("tecnica"),
        "funcao": r.get("funcao"),
        "dimensoes": r.get("dimensoes"),
        "artesao": r.get("artesao"),
        "povo": r.get("povo"),
        "aldeia": r.get("aldeia"),
        "notas_gerais": r.get("notas_gerais"),
        "tags": r.get("tags", []),
        "local": r.get("aldeia") or ", ".join(x for x in (r.get("cidade"), r.get("estado"), r.get("pais")) if x) or None,
        "imagem_url": f"/images/{r['image_id']}",
        "scores": r.get("scores"),
        "rrf": r.get("rrf"),
        "fonte": {"catalogo": r.get("fonte_catalogo"), "codigo_acervo": r["item_id"]},
    }


@router.post("/text")
def search_text(body: TextQuery, request: Request):
    _check_filters(body.filtros)
    res = request.app.state.svc.search_text(body.texto, body.modo, body.top_k, body.filtros or None)
    return {
        "consulta": body.texto,
        "modo": body.modo,
        "total": len(res),
        "resultados": [serialize(r) for r in res],
        "aviso": None if res else "Nenhum registro correspondente encontrado na base consultada.",
    }


@router.post("/image")
async def search_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Form(10),
    modelo_visual: Literal["clip", "siglip"] = Form("clip"),
    categoria: str | None = Form(None),
    estado: str | None = Form(None),
    pais: str | None = Form(None),
):
    if file.content_type not in ALLOWED:
        raise HTTPException(415, f"Formato não suportado: {file.content_type}. Use JPEG, PNG ou WebP.")
    if not 1 <= top_k <= 50:
        raise HTTPException(422, "top_k deve estar entre 1 e 50.")
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Arquivo maior que 10 MB.")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Arquivo de imagem inválido ou corrompido.")
    filtros = {k: v for k, v in (("categoria", categoria), ("estado", estado), ("pais", pais)) if v}
    res = request.app.state.svc.search_image(im, top_k, filtros or None, visual_model=modelo_visual)
    return {
        "modo": f"image ({modelo_visual})",
        "total": len(res),
        "resultados": [serialize(r) for r in res],
        "aviso": None if res else "Nenhum registro correspondente encontrado na base consultada.",
    }
