# Relatório de Análise Exploratória da Base (Etapa 1 — Seção 7.1)

## 1. Visão Geral e Granularidade
- **Objetos museológicos únicos (`codigo_acervo`):** 35
- **Fotografias válidas indexadas:** 113
- **Imagens por objeto:** média = 3.23, mediana = 2, mín = 1, máx = 8
- **Distribuição (nº de fotos → nº de objetos):** `{1: 2, 2: 16, 3: 7, 4: 3, 5: 2, 7: 4, 8: 1}`

## 2. Consistência entre Catálogo e Arquivos de Imagem
- **Coleção `MN_003`:** 113 linhas no CSV, 113 arquivos no disco, 0 duplicatas, 0 arquivos ausentes no disco.

## 3. Distribuição das Categorias (`catalogo_identificação_museológica`)
| Categoria | Objetos | Fotografias |
|---|---:|---:|
| Maracá | 6 | 15 |
| Cocar | 5 | 27 |
| Borduna | 4 | 11 |
| Apito | 4 | 20 |
| Cordão | 4 | 8 |
| Saia | 3 | 6 |
| Arco | 2 | 4 |
| Arco de boca | 2 | 3 |
| Peneira | 1 | 5 |
| Abanador | 1 | 3 |
| Flecha | 1 | 2 |
| Crânio | 1 | 7 |
| Taquara | 1 | 2 |

## 4. Distribuição de Funções, Materiais e Técnicas (por objeto)
- **Funções (`catalogo_função`):** {'Religioso': 13, 'Ritualistico': 8, 'Político': 8, 'Arma': 7, 'Ritualístico': 7, 'Retomada': 6, 'Ritual': 4, 'Comunicação': 4, 'Proteção': 4, 'Utilitário': 2, 'Conforto doméstico': 1, 'Caça': 1, 'Curativo': 1, 'Terapeutico': 1, 'Vestuário': 1}
- **Materiais (`catalogo_material_suporte`):** {'Madeira': 14, 'Lã': 8, 'Cabaça': 6, 'Algodão': 4, 'Sementes': 4, 'Fibras vegetais': 3, 'Tecelagem': 3, 'Plumárias': 3, 'Taquara': 1, 'Madeira de cedro': 1, 'Osteológico': 1, 'Lona': 1}
- **Técnicas (`catalogo_técnica`):** {'Talhado': 14, 'Não informada': 11, 'Costura': 7, 'Trançado': 3}

## 5. Qualidade das Descrições Textuais
- **Descrição física pura (`R2` - `catalogo_descrição`):** média de 27.6 palavras (mín 6, máx 84), 32 textos únicos entre 35 objetos.
- **Legenda catalográfica (`descricao`):** média de 46.63 palavras.
- **Representação enriquecida (`R4` - descrição + metadados):** média de 155.17 palavras (35 textos únicos).

## 6. Qualidade e Resolução das Imagens
- **Resolução (Megapixels):** média = 26.56 MP (mín 18.55 MP, máx 34.89 MP)
- **Dimensões (px):** largura média = 5087.27 px, altura média = 5421.93 px
- **Tamanho do arquivo (MB):** média = 8.02 MB (mín 2.26 MB, máx 15.46 MB)
- **Orientação:** `{'vertical': 61, 'horizontal': 52}`

## 7. Campos Catalográficos e Taxa de Valores Ausentes (`BaseMN_003.csv`)
| Campo | Preenchidos | Ausentes | % Ausência |
|---|---:|---:|---:|
| `catalogo_número_de_tombo` | 113 | 0 | 0.0% |
| `catalogo_identificação_museológica` | 113 | 0 | 0.0% |
| `catalogo_nome_da_coleção` | 113 | 0 | 0.0% |
| `catalogo_descrição` | 113 | 0 | 0.0% |
| `descricao` | 113 | 0 | 0.0% |
| `palavras_chave` | 113 | 0 | 0.0% |
| `cidade` | 0 | 113 | 100.0% |
| `estado` | 0 | 113 | 100.0% |
| `pais` | 0 | 113 | 100.0% |
| `catalogo_identificação_local` | 113 | 0 | 0.0% |
| `catalogo_material_suporte` | 113 | 0 | 0.0% |
| `catalogo_técnica` | 83 | 30 | 26.5% |
| `catalogo_função` | 113 | 0 | 0.0% |
| `catalogo_quantidade_de_componentes` | 113 | 0 | 0.0% |
| `catalogo_grafismos` | 0 | 113 | 100.0% |
| `catalogo_dimensões` | 113 | 0 | 0.0% |
| `catalogo_estado_de_conservação` | 2 | 111 | 98.2% |
| `catalogo_inscrições_e_marcas` | 0 | 113 | 100.0% |
| `catalogo_localização` | 0 | 113 | 100.0% |
| `catalogo_autor_artesão` | 113 | 0 | 0.0% |
| `catalogo_povo` | 113 | 0 | 0.0% |
| `catalogo_auto_denominação` | 113 | 0 | 0.0% |
| `catalogo_aldeia` | 113 | 0 | 0.0% |
| `catalogo_data_de_coleta` | 113 | 0 | 0.0% |
| `catalogo_data_de_fabricação` | 112 | 1 | 0.9% |
| `catalogo_cedente` | 113 | 0 | 0.0% |
| `catalogo_coletor` | 113 | 0 | 0.0% |
| `catalogo_forma_de_aquisição` | 113 | 0 | 0.0% |
| `catalogo_data_de_entrada_no_mn` | 113 | 0 | 0.0% |
| `catalogo_data_de_tombamento` | 113 | 0 | 0.0% |
| `catalogo_status` | 113 | 0 | 0.0% |
| `catalogo_situação` | 113 | 0 | 0.0% |
| `catalogo_histórico_administrativo` | 113 | 0 | 0.0% |
| `catalogo_histórico_de_exposições` | 0 | 113 | 100.0% |
| `catalogo_documentação` | 0 | 113 | 100.0% |
| `catalogo_bibliografia` | 113 | 0 | 0.0% |
| `catalogo_notas_gerais` | 84 | 29 | 25.7% |
