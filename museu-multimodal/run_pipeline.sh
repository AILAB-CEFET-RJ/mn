#!/usr/bin/env bash
# Pipeline completo e idempotente (Etapas 1 a 5 da metodologia do TCC). Execute de museu-multimodal/.
set -euo pipefail
PY="${PY:-python}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR/ingestion"
$PY eda_report.py >/dev/null
$PY generate_image_embeddings.py
$PY generate_text_embeddings.py
FORCE=1 $PY index_qdrant.py
cd "$ROOT_DIR/evaluation"
$PY build_queries.py
$PY evaluate.py
$PY cross_modal.py
$PY error_analysis.py
