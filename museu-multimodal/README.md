# Museu multimodal — Fase 1 (protótipo visual, imagem→imagem)

Execute a partir de `museu-multimodal/`:

```bash
docker compose up -d                       # Qdrant em :6333
python -m venv .venv && source .venv/bin/activate
# PyTorch ROCm (RX 6600; pode exigir HSA_OVERRIDE_GFX_VERSION=10.3.0):
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install -r requirements.txt

LIMIT=0 python ingestion/generate_image_embeddings.py   # LIMIT=N para amostra; DEVICE=cpu para forçar CPU
python ingestion/index_qdrant.py
python ingestion/search_image.py "../Base Museu Nacional/MN_003/MN 003-33.jpg" 5
```

Aceite: a própria imagem aparece com score ≈ 1.0 e fotos do mesmo `item_id` ficam agrupadas.
Catálogo: `Base Museu Nacional/Metadados_Imagens/MN_003.csv` (`codigo_acervo` = `item_id`).
Reexecuções são idempotentes (UUID por `image_id`) e o embedding retoma do checkpoint.

## Fase 5 — API e interface

```bash
# API (de museu-multimodal/)
uvicorn app.main:app --port 8000        # docs em http://localhost:8000/docs
# Frontend
cd frontend && npm install && npm run dev   # http://localhost:3000
```
Endpoints: `POST /search/text`, `POST /search/image`, `GET /items/{item_id}`, `GET /images/{image_id}?w=`, `GET /health`.
