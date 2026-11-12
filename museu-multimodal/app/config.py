"""Configurações centralizadas do sistema de busca multimodal (caminhos, modelos e banco vetorial)."""
import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent

DATA_DIR = APP_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
THUMBS_DIR = DATA_DIR / "thumbs"
QDRANT_LOCAL_DIR = DATA_DIR / "qdrant_local"
META_PATH = PROCESSED_DIR / "metadata.json"

COLLECTION = os.getenv("QDRANT_COLLECTION", "museum_images")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

CLIP_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
SIGLIP_NAME = os.getenv("SIGLIP_MODEL", "google/siglip-base-patch16-256-multilingual")
BGE_NAME = os.getenv("BGE_MODEL", "BAAI/bge-m3")

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]

