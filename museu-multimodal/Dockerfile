FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEVICE=cpu \
    PORT=8000

WORKDIR /workspace/museu-multimodal

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY app ./app
COPY ingestion ./ingestion
COPY data ./data

EXPOSE 8000

CMD ["sh", "-c", "python ingestion/index_qdrant.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

