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

---

## 5. Arquitetura do Projeto

### 5.1. Estrutura de Diretórios

```text
museu-multimodal/
├── app/                              # Backend FastAPI (busca online)
│   ├── main.py                       # Inicialização da API, CORS, lifespan e /health
│   ├── config.py                     # Configurações centralizadas (modelos, caminhos, Qdrant)
│   ├── api/
│   │   ├── search.py                 # POST /search/text e POST /search/image
│   │   └── items.py                  # GET /items/{item_id} e GET /images/{image_id}?w= (cache de thumbs)
│   ├── db/
│   │   └── qdrant_client.py          # Singleton QdrantClient (Docker/Cloud ou fallback local em disco)
│   └── services/
│       ├── search_service.py         # Orquestração de buscas (R1, R2, R3, R4, CLIP, SigLIP e BM25)
│       ├── bm25_service.py           # Baseline léxico BM25 Okapi sobre R2 e R4
│       └── fusion_service.py         # Fusão de rankings via Reciprocal Rank Fusion (RRF, k=60)
├── ingestion/                        # Pipeline offline de ingestão e CLI
│   ├── load_catalog.py               # Leitura e cruzamento de Base*.csv + Metadados_Imagens/*.csv
│   ├── eda_report.py                 # Etapa 1: Análise exploratória da base (eda_report.md / .json)
│   ├── generate_image_embeddings.py  # Etapa 2: Embeddings visuais CLIP (512-d) e SigLIP (768-d)
│   ├── generate_text_embeddings.py   # Etapa 2: Embeddings textuais BGE-M3 (1024-d) para R2 e R4
│   ├── index_qdrant.py               # Etapa 3: Indexação idempotente dos 4 vetores nomeados no Qdrant
│   ├── search_image.py               # CLI de busca imagem→imagem
│   └── search_text.py                # CLI de busca textual/híbrida
├── evaluation/                       # Avaliação experimental do TCC
│   ├── build_queries.py              # Etapa 3: Geração das consultas de avaliação (queries.json)
│   ├── evaluate.py                   # Etapa 4: Métricas (Recall@K, P@K, nDCG@K, MRR, MAP@10, latência)
│   ├── cross_modal.py                # Tarefas cruzadas descrição→imagem e imagem→texto
│   └── error_analysis.py             # Etapa 5: Análise quantitativa de erros (error_analysis.md / .json)
├── data/
│   ├── processed/                    # Embeddings (.npy), checkpoints (.npz), metadata.json e manifest.json
│   ├── qdrant_local/                 # Persistência local automática do Qdrant (quando sem container)
│   └── thumbs/                       # Cache em disco de miniaturas redimensionadas
├── frontend/                         # Interface Web (Next.js / React)
│   └── app/page.js                   # Busca por texto/imagem, seletor R1–R4/BM25, filtros e modal do item
├── tests/
│   └── test_core.py                  # Testes unitários e de integração (RRF, catálogo, métricas, BM25)
├── Dockerfile                        # Imagem Docker da API FastAPI (PyTorch CPU)
├── docker-compose.yml                # Serviços qdrant + api
├── requirements.txt                  # Dependências Python
├── pyproject.toml                    # Configuração do pacote e pytest
└── run_pipeline.sh                   # Script de automação (app, tcc ou completo)
```

---

### 5.2. Modelos e Representações (`R1` a `R4` + Baselines)

| Representação / Modo | Modos na API | Codificadores | Fontes de Informação |
|---|---|---|---|
| **Baseline Léxico (R2)** | `bm25_r2` | BM25 Okapi (PT) | Apenas descrição física pura (`catalogo_descrição`) |
| **Baseline Léxico (R4)** | `bm25_r4`, `bm25` | BM25 Okapi (PT) | Descrição física + metadados catalográficos selecionados |
| **R1 — Apenas Imagem** | `R1_clip` (`images`), `R1_siglip` | CLIP ViT-B/32 (`512-d`) ou SigLIP Multilíngue (`768-d`) | Fotografia do objeto (busca texto→imagem ou imagem→imagem) |
| **R2 — Apenas Descrição Física** | `R2` | BGE-M3 (`1024-d`, CLS + L2) | Somente `catalogo_descrição` |
| **R3 — Imagem + Descrição Pura** | `R3_clip` (`R3`), `R3_siglip` | Fusão RRF (`R1` + `R2`) | Fotografia + `catalogo_descrição` |
| **R4 (Textual) — Desc + Metadados** | `R4_text` (`descriptions`) | BGE-M3 (`1024-d`, CLS + L2) | Descrição física + categoria, denominação indígena, material, técnica, função, povo, aldeia e notas |
| **R4 (Multimodal) — Completo** | `R4_clip` (`hybrid`, `R4`), `R4_siglip` | Fusão RRF (`R1` + `R4_text`) | Fotografia + descrição física + metadados catalográficos |

---

### 5.3. Modelagem no Banco Vetorial (Qdrant)

- **Coleção:** `museum_images` (1 ponto por fotografia, identificado por `UUIDv5` determinístico derivado de `image_id`).
- **Granularidade de retorno:** como um mesmo objeto museológico (`item_id` = `codigo_acervo`) possui múltiplas fotografias sob diferentes ângulos, as buscas agrupam os resultados por `item_id` mantendo a melhor foto de cada objeto.
- **Quatro vetores nomeados por ponto (distância Cosseno):**
  - `image`: `512-d` (`openai/clip-vit-base-patch32`)
  - `siglip_image`: `768-d` (`google/siglip-base-patch16-256-multilingual`)
  - `text_r2`: `1024-d` (`BAAI/bge-m3` sobre `texto_r2`)
  - `text`: `1024-d` (`BAAI/bge-m3` sobre `texto_r4`)
- **Fusão Multimodal:** os rankings do ramo visual (`R1`) e do ramo textual (`R2` ou `R4_text`) são combinados via **Reciprocal Rank Fusion (RRF, $k=60$)**, evitando somar diretamente scores de modelos com escalas de similaridade distintas.

