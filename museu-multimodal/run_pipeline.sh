#!/usr/bin/env bash
# Uso (a partir de museu-multimodal/):
#   ./run_pipeline.sh        -> roda tudo (ingestão + experimentos do TCC)
#   ./run_pipeline.sh app    -> roda apenas o necessário para subir a aplicação (embeddings + Qdrant)
#   ./run_pipeline.sh tcc    -> roda apenas os relatórios e avaliações do TCC
set -euo pipefail
PY="${PY:-python}"
MODE="${1:-all}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ "$MODE" == "all" || "$MODE" == "app" || "$MODE" == "ingest" ]]; then
  cd "$ROOT_DIR/ingestion"
  $PY generate_image_embeddings.py
  $PY generate_text_embeddings.py
  FORCE=1 $PY index_qdrant.py
fi

if [[ "$MODE" == "all" || "$MODE" == "tcc" || "$MODE" == "eval" ]]; then
  cd "$ROOT_DIR/ingestion"
  $PY eda_report.py >/dev/null
  cd "$ROOT_DIR/evaluation"
  $PY build_queries.py
  $PY evaluate.py
  $PY cross_modal.py
  $PY error_analysis.py
fi
