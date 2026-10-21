#!/usr/bin/env bash
# Pipeline completo e idempotente: roda só o que for novo. Execute de museu-multimodal/.
set -euo pipefail
PY="${PY:-python}"
cd "$(dirname "$0")/ingestion"
$PY generate_image_embeddings.py
$PY generate_text_embeddings.py
$PY index_qdrant.py
