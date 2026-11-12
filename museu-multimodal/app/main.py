"""API do museu multimodal. Execute de museu-multimodal/: uvicorn app.main:app --port 8000"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.api import items, search  # noqa: E402
from app.config import COLLECTION, CORS_ORIGINS  # noqa: E402
from app.db.qdrant_client import close_qdrant_client  # noqa: E402
from app.services.search_service import SearchService  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.svc = SearchService()  # modelos carregam sob demanda na 1ª consulta
    try:
        yield
    finally:
        close_qdrant_client()


app = FastAPI(title="Museu Nacional — Busca Multimodal", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"], allow_headers=["*"],
)
app.include_router(search.router)
app.include_router(items.router)


@app.get("/health")
def health():
    svc: SearchService = app.state.svc
    try:
        n = svc.client.count(COLLECTION).count
        return {"status": "ok", "qdrant": "ok", "pontos": n, "dispositivo": svc.device}
    except Exception as e:  # Qdrant fora do ar
        return {"status": "degradado", "qdrant": f"erro: {type(e).__name__}", "dispositivo": svc.device}
