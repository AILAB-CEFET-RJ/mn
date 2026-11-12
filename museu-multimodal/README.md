# Museu Multimodal — Recuperação de Informações para o Acervo Digital do Museu Nacional (TCC CEFET/RJ)

Sistema de busca multimodal (texto e imagem) sobre o acervo etnográfico do Museu Nacional, comparando baselines léxicos (**BM25**), representações progressivas (**R1, R2, R3 e R4**) e modelos visuais-linguísticos (**CLIP ViT-B/32** vs **SigLIP Multilíngue**) combinados com embeddings textuais densos (**BGE-M3**) via **Reciprocal Rank Fusion (RRF)**.

---

## 1. Ambiente Local (Python / `.venv`)

Crie o `.venv` na raiz (`/home/lucas/Projetos/mn`) com **Python 3.12**:

```bash
cd /home/lucas/Projetos/mn
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install torch -r museu-multimodal/requirements.txt
```

---

## 2. Pipeline: Preparação da Aplicação vs. Experimentos do TCC

Todos os comandos abaixo são executados dentro de `museu-multimodal/` (e são **incrementais/idempotentes**):

### 2.1. Apenas o necessário para subir a Aplicação (Ingestão + Indexação)
Roda somente a extração dos embeddings e a indexação no Qdrant, sem executar as baterias de avaliação do TCC:

```bash
cd museu-multimodal
./run_pipeline.sh app
```

Ou rodando cada etapa individualmente:
1. `python ingestion/generate_image_embeddings.py` — Gera embeddings visuais CLIP (512-d) e SigLIP (768-d) com checkpoint atômico (`LIMIT=N` para amostra; `DEVICE=cpu` para forçar CPU; `SKIP_SIGLIP=1` para pular SigLIP).
2. `python ingestion/generate_text_embeddings.py` — Gera embeddings textuais BGE-M3 (1024-d) para `R2` (descrição física pura) e `R4` (descrição + metadados) com cache por SHA-1.
3. `FORCE=1 python ingestion/index_qdrant.py` — Indexa os vetores no Qdrant (usa servidor `:6333` se ativo, ou persistência local automática em `data/qdrant_local`).

---

### 2.2. Experimentos, Métricas e Relatórios do TCC
Executa a análise exploratória (EDA), reconstrói as consultas de teste e calcula todas as tabelas de avaliação e análise de erros (requer que a base já esteja indexada pelo passo 2.1):

```bash
cd museu-multimodal
./run_pipeline.sh tcc
```

Ou rodando cada experimento individualmente:
1. `python ingestion/eda_report.py` — Etapa 1 (Seção 7.1): Análise exploratória da base (`data/processed/eda_report.md` e `.json`).
2. `python evaluation/build_queries.py` — Etapa 3 (Seção 7.3): Gera o conjunto de consultas de avaliação (`evaluation/queries.json`).
3. `python evaluation/evaluate.py` — Etapa 4 (Seção 7.4): Calcula Recall@K, Precision@K, nDCG@K, MRR, MAP@10 e latência (`evaluation/results.json`).
4. `python evaluation/cross_modal.py` — Tarefas cruzadas descrição→imagem e imagem→texto (`evaluation/cross_modal_results.json`).
5. `python evaluation/error_analysis.py` — Etapa 5 (Seção 7.5): Relatório quantitativo de análise de erros (`evaluation/error_analysis.md` e `.json`).
6. `python -m unittest discover -s tests -v` — Suíte de testes automatizados (unitários e de integração).

> **Dica:** Para rodar tudo de uma vez (2.1 + 2.2 em sequência), basta executar `./run_pipeline.sh` sem argumentos.

---

## 3. Como Rodar a Aplicação (Backend + Banco + Frontend)

Você pode rodar o backend e o banco de dados de **duas formas** (sempre a partir de `museu-multimodal/`):

### Opção A — Desenvolvimento local (Backend no `.venv`)
Ideal para desenvolvimento rápido com recarregamento automático (`--reload`):

```bash
# 1. (Opcional) Subir apenas o banco Qdrant via Docker:
#    Se não subir o container, a API usa automaticamente o banco local em data/qdrant_local.
docker compose up -d qdrant

# 2. Iniciar a API FastAPI pelo .venv:
uvicorn app.main:app --reload --port 8000
```

### Opção B — Tudo em Docker (Backend + Qdrant em containers)
Ideal para testar o ambiente containerizado ou subir em servidor/VPS:

```bash
docker compose up -d --build
```
O que esse comando faz:
- Sobe o container `museu-qdrant` na porta `6333`.
- Constrói e sobe o container `museu-api` na porta `8000` (usando o `Dockerfile`).
- Conecta automaticamente a API ao Qdrant (`QDRANT_URL=http://qdrant:6333`), garante que os vetores de `data/processed/` estejam indexados, monta `../Base Museu Nacional` em modo somente leitura e persiste o cache dos modelos Hugging Face no volume `hf_cache`.

Para acompanhar os logs ou parar os containers:
```bash
docker compose logs -f api
docker compose down
```

### Subir o Frontend (Next.js)
Em outro terminal:

```bash
cd museu-multimodal/frontend
npm install   # apenas na primeira vez
npm run dev
```

- **Interface Web:** http://localhost:3000
- **Documentação interativa da API (Swagger):** http://localhost:8000/docs
- **Endpoints principais:** `POST /search/text`, `POST /search/image`, `GET /items/{item_id}`, `GET /images/{image_id}?w=`, `GET /health`.

---

## 4. Buscas Rápidas via CLI e Testes Automatizados

Dentro de `museu-multimodal/` (com `.venv` ativo):

```bash
# Busca por texto (modos: R4_siglip, R4_clip, R4_text, R3_siglip, R3_clip, R2, R1_siglip, R1_clip, bm25_r4, bm25_r2)
python ingestion/search_text.py "instrumentos musicais utilizados em cerimônias" R4_siglip 5

# Busca imagem→imagem (modelos visuais: clip ou siglip)
python ingestion/search_image.py "../Base Museu Nacional/MN_003/MN 003-33.jpg" 5 siglip

# Suíte de testes unitários e de integração
python -m unittest discover -s tests -v
```
