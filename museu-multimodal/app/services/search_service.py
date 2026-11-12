"""Serviço de busca multimodal: suporta R1, R2, R3, R4, CLIP vs SigLIP e Baseline BM25.

Atende às Seções 6.1–6.4 e 7.2–7.4 da proposta do TCC (CEFET/RJ):
- Baseline léxico: `bm25_r2` (descrição pura) e `bm25_r4` / `bm25` (descrição + metadados).
- R1 (apenas imagem): `images` / `R1_clip` (CLIP) e `R1_siglip` (SigLIP Multilíngue).
- R2 (apenas descrição textual pura): `R2` (BGE-M3 sobre `catalogo_descrição`).
- R3 (imagem + descrição pura): `R3` / `R3_clip` e `R3_siglip` (RRF entre R1 e R2).
- R4 (imagem + descrição + metadados): `hybrid` / `R4` / `R4_clip` e `R4_siglip` (RRF entre R1 e R4 textual),
  além de `descriptions` / `R4_text` (componente textual semântico de R4).
"""
import json
import os
import sys
from pathlib import Path

import torch
from qdrant_client import models
from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.config import (  # noqa: E402
    BGE_NAME,
    CLIP_NAME,
    COLLECTION,
    META_PATH,
    QDRANT_URL,
    SIGLIP_NAME,
)
from app.db.qdrant_client import get_qdrant_client  # noqa: E402
from app.services.bm25_service import BM25Index, _tokenize_pt  # noqa: E402,F401
from app.services.fusion_service import rrf  # noqa: E402


VALID_MODES = {
    "bm25_r2",
    "bm25_r4",
    "bm25",
    "images",
    "R1_clip",
    "R1_siglip",
    "R2",
    "descriptions",
    "R4_text",
    "R3",
    "R3_clip",
    "R3_siglip",
    "hybrid",
    "R4",
    "R4_clip",
    "R4_siglip",
}


class SearchService:
    def __init__(self, device: str | None = None):
        self.device = device or os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.client = get_qdrant_client()
        self._clip = self._clip_proc = None
        self._siglip = self._siglip_proc = None
        self._bge = self._bge_tok = None
        self._bm25_r2: BM25Index | None = None
        self._bm25_r4: BM25Index | None = None

    def _load_clip(self):
        if self._clip is None:
            self._clip = CLIPModel.from_pretrained(CLIP_NAME).to(self.device).eval()
            self._clip_proc = CLIPProcessor.from_pretrained(CLIP_NAME)

    def _load_siglip(self):
        if self._siglip is None:
            self._siglip = AutoModel.from_pretrained(SIGLIP_NAME).to(self.device).eval()
            self._siglip_proc = AutoProcessor.from_pretrained(SIGLIP_NAME)

    def _load_bge(self):
        if self._bge is None:
            self._bge_tok = AutoTokenizer.from_pretrained(BGE_NAME)
            self._bge = AutoModel.from_pretrained(BGE_NAME).to(self.device).eval()

    def _load_bm25(self):
        if self._bm25_r4 is None:
            meta = json.loads(META_PATH.read_text())
            by_item: dict[str, dict] = {}
            for r in meta:
                by_item.setdefault(r["item_id"], r)
            self._bm25_r2 = BM25Index(by_item, "texto_r2")
            self._bm25_r4 = BM25Index(by_item, "texto_r4")

    @torch.no_grad()
    def clip_text_vec(self, text: str) -> list[float]:
        self._load_clip()
        enc = self._clip_proc(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        v = self._clip.get_text_features(**enc)
        if not isinstance(v, torch.Tensor):
            v = v.pooler_output
        return torch.nn.functional.normalize(v, dim=-1)[0].cpu().tolist()

    @torch.no_grad()
    def siglip_text_vec(self, text: str) -> list[float]:
        self._load_siglip()
        enc = self._siglip_proc(
            text=[text], return_tensors="pt", padding="max_length", truncation=True
        ).to(self.device)
        v = self._siglip.get_text_features(**enc)
        if not isinstance(v, torch.Tensor):
            v = v.pooler_output
        return torch.nn.functional.normalize(v, dim=-1)[0].cpu().tolist()

    @torch.no_grad()
    def bge_vec(self, text: str) -> list[float]:
        self._load_bge()
        enc = self._bge_tok([text], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        v = self._bge(**enc).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(v, dim=-1)[0].cpu().tolist()

    @staticmethod
    def _filter(filters: dict | None, exclude_item: str | None = None):
        if not filters and not exclude_item:
            return None
        must = [models.FieldCondition(key=k, match=models.MatchValue(value=v))
                for k, v in (filters or {}).items()]
        must_not = ([models.FieldCondition(key="item_id", match=models.MatchValue(value=exclude_item))]
                    if exclude_item else None)
        return models.Filter(must=must or None, must_not=must_not)

    @torch.no_grad()
    def clip_image_vec(self, image) -> list[float]:
        self._load_clip()
        enc = self._clip_proc(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        v = self._clip.get_image_features(**enc)
        if not isinstance(v, torch.Tensor):
            v = v.pooler_output
        return torch.nn.functional.normalize(v, dim=-1)[0].cpu().tolist()

    @torch.no_grad()
    def siglip_image_vec(self, image) -> list[float]:
        self._load_siglip()
        enc = self._siglip_proc(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        v = self._siglip.get_image_features(**enc)
        if not isinstance(v, torch.Tensor):
            v = v.pooler_output
        return torch.nn.functional.normalize(v, dim=-1)[0].cpu().tolist()

    def search_image(self, image, top_k: int = 5, filters: dict | None = None,
                     exclude_item: str | None = None, visual_model: str = "clip") -> list[dict]:
        """Imagem→imagem agrupado por `item_id` (CLIP ou SigLIP)."""
        flt = self._filter(filters, exclude_item)
        if visual_model == "siglip":
            hits = self._grouped(self.siglip_image_vec(image), "siglip_image", top_k, flt)
            return [{**p, "scores": {"siglip": round(s, 4)}, "rrf": None} for p, s in hits]
        hits = self._grouped(self.clip_image_vec(image), "image", top_k, flt)
        return [{**p, "scores": {"clip": round(s, 4)}, "rrf": None} for p, s in hits]

    def _grouped(self, vec, using: str, top_k: int, flt):
        """Melhor foto por item_id (compatível com Qdrant Server e Qdrant Local)."""
        try:
            res = self.client.query_points_groups(
                COLLECTION, query=vec, using=using, group_by="item_id", limit=top_k,
                group_size=1, with_payload=True, query_filter=flt,
            )
            return [(g.hits[0].payload, g.hits[0].score) for g in res.groups]
        except Exception:
            res = self.client.query_points(
                COLLECTION, query=vec, using=using, limit=max(top_k * 8, 200),
                with_payload=True, query_filter=flt,
            )
            seen = set()
            grouped = []
            for pt in res.points:
                iid = pt.payload["item_id"]
                if iid not in seen:
                    seen.add(iid)
                    grouped.append((pt.payload, pt.score))
                    if len(grouped) >= top_k:
                        break
            return grouped

    def search_text(self, text: str, mode: str = "hybrid", top_k: int = 5,
                    filters: dict | None = None) -> list[dict]:
        if mode not in VALID_MODES:
            raise ValueError(f"modo inválido: {mode}. Válidos: {sorted(VALID_MODES)}")

        if mode in ("bm25_r2", "bm25_r4", "bm25"):
            self._load_bm25()
            idx = self._bm25_r2 if mode == "bm25_r2" else self._bm25_r4
            hits = idx.search(text, top_k=top_k, filters=filters)
            return [{**p, "rrf": None, "scores": {"bm25": round(s, 4)}} for p, s in hits]

        flt = self._filter(filters)
        pool = max(top_k * 4, 20)
        by_item: dict[str, dict] = {}
        rankings, scores = [], {}

        def run(using, vec, label):
            hits = self._grouped(vec, using, pool, flt)
            for payload, score in hits:
                by_item.setdefault(payload["item_id"], payload)
                scores.setdefault(payload["item_id"], {})[label] = round(score, 4)
            rankings.append([p["item_id"] for p, _ in hits])

        # Ramo visual (R1)
        if mode in ("images", "R1_clip", "R3", "R3_clip", "hybrid", "R4", "R4_clip"):
            run("image", self.clip_text_vec(text), "clip")
        elif mode in ("R1_siglip", "R3_siglip", "R4_siglip"):
            run("siglip_image", self.siglip_text_vec(text), "siglip")

        # Ramo textual: R2 (descrição pura) vs R4 (descrição + metadados)
        if mode in ("R2", "R3", "R3_clip", "R3_siglip"):
            run("text_r2", self.bge_vec(text), "bge_r2")
        elif mode in ("descriptions", "R4_text", "hybrid", "R4", "R4_clip", "R4_siglip"):
            run("text", self.bge_vec(text), "bge_r4")

        is_fused = len(rankings) > 1
        fused = rrf(rankings)[:top_k] if is_fused else [(i, 0.0) for i in rankings[0][:top_k]]
        return [{**by_item[i], "rrf": round(s, 5) if is_fused else None,
                 "scores": scores[i]} for i, s in fused]
