"""Busca multimodal: texto→imagem (CLIP), texto→descrição (BGE-M3), híbrida (RRF)."""
import os
import sys
from pathlib import Path

import torch
from qdrant_client import QdrantClient, models
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.services.fusion_service import rrf  # noqa: E402

COLLECTION = "museum_images"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CLIP_NAME = "openai/clip-vit-base-patch32"
BGE_NAME = "BAAI/bge-m3"


class SearchService:
    def __init__(self, device: str | None = None):
        self.device = device or os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.client = QdrantClient(url=QDRANT_URL)
        self._clip = self._clip_proc = self._bge = self._bge_tok = None

    def _load_clip(self):
        if self._clip is None:
            self._clip = CLIPModel.from_pretrained(CLIP_NAME).to(self.device).eval()
            self._clip_proc = CLIPProcessor.from_pretrained(CLIP_NAME)

    def _load_bge(self):
        if self._bge is None:
            self._bge_tok = AutoTokenizer.from_pretrained(BGE_NAME)
            self._bge = AutoModel.from_pretrained(BGE_NAME).to(self.device).eval()

    @torch.no_grad()
    def clip_text_vec(self, text: str) -> list[float]:
        self._load_clip()
        enc = self._clip_proc(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        v = self._clip.get_text_features(**enc)
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

    def search_image(self, image, top_k: int = 5, filters: dict | None = None,
                     exclude_item: str | None = None) -> list[dict]:
        """Imagem→imagem (agrupado por item). `exclude_item` serve para avaliação leave-one-out."""
        flt = self._filter(filters, exclude_item)
        hits = self._grouped(self.clip_image_vec(image), "image", top_k, flt)
        return [{**p, "scores": {"clip": round(s, 4)}, "rrf": None} for p, s in hits]

    def _grouped(self, vec, using: str, top_k: int, flt):
        """Melhor foto por item_id."""
        res = self.client.query_points_groups(
            COLLECTION, query=vec, using=using, group_by="item_id", limit=top_k,
            group_size=1, with_payload=True, query_filter=flt,
        )
        return [(g.hits[0].payload, g.hits[0].score) for g in res.groups]

    def search_text(self, text: str, mode: str = "hybrid", top_k: int = 5,
                    filters: dict | None = None) -> list[dict]:
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

        if mode in ("images", "hybrid"):
            run("image", self.clip_text_vec(text), "clip")
        if mode in ("descriptions", "hybrid"):
            run("text", self.bge_vec(text), "bge")
        if mode not in ("images", "descriptions", "hybrid"):
            raise ValueError(f"modo inválido: {mode}")

        fused = rrf(rankings)[:top_k] if mode == "hybrid" else [(i, 0.0) for i in rankings[0][:top_k]]
        return [{**by_item[i], "rrf": round(s, 5) if mode == "hybrid" else None,
                 "scores": scores[i]} for i, s in fused]
