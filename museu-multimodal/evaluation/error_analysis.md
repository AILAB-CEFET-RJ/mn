# Análise de Erros (Etapa 5 — Seção 7.5)

> Todos os números abaixo foram calculados por `evaluation/error_analysis.py`. Amostra: 35 objetos de uma única coleção; diferenças pequenas não têm teste de significância.

## 1. Semelhança visual entre categorias diferentes (imagem→imagem, leave-one-out)

### CLIP: 5 de 30 consultas (17%) tiveram o top-1 em outra categoria

| Consulta | Categoria real | Top-1 (categoria predita) | Material em comum |
|---|---|---|---|
| MNSEE.003.0029 | Apito | MNSEE.003.0030 (Cordão) | nenhum |
| MNSEE.003.0021 | Cocar | MNSEE.003.0029 (Apito) | nenhum |
| MNSEE.003.0031 | Cordão | MNSEE.003.0029 (Apito) | nenhum |
| MNSEE.003.0032 | Cordão | MNSEE.003.0025 (Cocar) | Lã |
| MNSEE.003.0019 | Saia | MNSEE.003.0032 (Cordão) | nenhum |

Pares mais frequentes (real → predita): Apito → Cordão (1×); Cocar → Apito (1×); Cordão → Apito (1×); Cordão → Cocar (1×); Saia → Cordão (1×)
Em 1 de 5 erros os dois objetos compartilham ao menos um material (taxa-base entre pares de categorias diferentes: 18%).

### SigLIP: 6 de 30 consultas (20%) tiveram o top-1 em outra categoria

| Consulta | Categoria real | Top-1 (categoria predita) | Material em comum |
|---|---|---|---|
| MNSEE.003.0029 | Apito | MNSEE.003.0030 (Cordão) | nenhum |
| MNSEE.003.0010 | Arco de boca | MNSEE.003.0008 (Arco) | Madeira |
| MNSEE.003.0006 | Borduna | MNSEE.003.0007 (Flecha) | Madeira |
| MNSEE.003.0021 | Cocar | MNSEE.003.0001 (Peneira) | Fibras vegetais |
| MNSEE.003.0032 | Cordão | MNSEE.003.0024 (Cocar) | Lã |
| MNSEE.003.0019 | Saia | MNSEE.003.0032 (Cordão) | nenhum |

Pares mais frequentes (real → predita): Apito → Cordão (1×); Arco de boca → Arco (1×); Borduna → Flecha (1×); Cocar → Peneira (1×); Cordão → Cocar (1×)
Em 4 de 6 erros os dois objetos compartilham ao menos um material (taxa-base entre pares de categorias diferentes: 18%).

## 2. Descrições incompletas: o termo da categoria aparece no texto R2?

Cobertura = fração dos objetos da categoria cujo `catalogo_descrição` contém a raiz do nome da categoria (prefixo; tolera plural, mas pode acusar falso positivo). AP@10 por método (consulta = nome da categoria).

| Categoria | Objetos | Cobertura R2 | bm25_r2 | R2 | R1_siglip | R3_siglip | R4_text |
|---|---|---|---|---|---|---|---|
| Abanador | 1 | 0 | 0.0 | 0.143 | 0.25 | 0.5 | 1.0 |
| Apito | 4 | 0 | 0.0 | 0.025 | 0.85 | 0.267 | 1.0 |
| Borduna | 4 | 0 | 0.0 | 0.0 | 0.321 | 0.062 | 1.0 |
| Cocar | 5 | 0 | 0.0 | 0.0 | 0.714 | 0.073 | 1.0 |
| Maracá | 6 | 0 | 0.0 | 0.5 | 0.976 | 0.852 | 1.0 |
| Saia | 3 | 0.3333 | 0.333 | 0.4 | 1.0 | 0.81 | 1.0 |
| Arco de boca | 2 | 0.5 | 1.0 | 0.05 | 0.7 | 0.333 | 1.0 |
| Arco | 2 | 1 | 0.5 | 0.125 | 0.583 | 1.0 | 0.833 |
| Cordão | 4 | 1 | 1.0 | 1.0 | 0.65 | 1.0 | 1.0 |
| Crânio | 1 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Flecha | 1 | 1 | 0.5 | 1.0 | 0.5 | 0.5 | 1.0 |
| Peneira | 1 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Taquara | 1 | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

AP@10 médio de R2: 0.178 nas 6 categorias com cobertura < 50% contra 0.7393 nas 7 com cobertura ≥ 50%. Para BM25-R2: 0.0555 contra 0.8571.
Ressalva: o rótulo de categoria também faz parte do texto de R4, então a vantagem de R4 nesta tabela é em parte construída (ver notas_tcc.md).

## 3. Diferenças linguísticas: consulta em português × inglês (ramo visual, AP@10)

- **CLIP**: AP médio PT = 0.2738, EN = 0.6162; inglês melhor em 19, pior em 3, equivalente (|Δ| ≤ 0,05) em 1 de 23 consultas.
- **SigLIP**: AP médio PT = 0.7733, EN = 0.8462; inglês melhor em 7, pior em 5, equivalente (|Δ| ≤ 0,05) em 11 de 23 consultas.

## 4. Terminologia museológica/indígena (consulta = denominação local, ex.: "Mbaraka")

19 termos distintos; AP@10 médio por método:

| Método | AP@10 médio |
|---|---|
| R1_clip | 0.1546 |
| R1_siglip | 0.2184 |
| R2 | 0.1693 |
| R4_text | 0.9737 |
| bm25_r2 | 0.0526 |
| bm25_r4 | 0.9692 |

Ressalva: `R4_text` e `bm25_r4` recebem a denominação local como campo do texto, portanto o acerto é esperado por construção; a comparação informativa é entre R1 (CLIP/SigLIP), R2 e BM25-R2.

## 5. Descrições curtas: efeito no descrição→imagem

| Grupo | Palavras | Objetos | MRR CLIP | MRR SigLIP |
|---|---|---|---|---|
| curtas (1º terço) | 6–15 | 11 | 0.1503 | 0.7857 |
| médias | 17–31 | 11 | 0.2746 | 0.7727 |
| longas (último terço) | 31–84 | 13 | 0.1732 | 0.4339 |

### Cinco piores casos (SigLIP) para revisão manual
Aqui entram os fatores que o script não consegue julgar: informação visível na foto mas ausente da descrição, ou presente na descrição mas invisível na foto.

| Objeto | Categoria | Rank do objeto correto (de 35) | Descrição usada como consulta |
|---|---|---|---|
| MNSEE.003.0034 | Arco de boca | 11 | Feito de cedro e fibra de caraguatá, guembe pi. Decorado com ferro queimado. Instrumento de corda única, composto por dois arcos que se cruz |
| MNSEE.003.0028 | Apito | 11 | Feito a partir da raiz da goiaba do campo. Ele se diferencia por estar sob um "banco" (apyká). Junto dele está amarrado outro Minby. É um in |
| MNSEE.003.0035 | Saia | 7 | Algodão com detalhes laranja e amarelo. Utilizado em rituais políticos e religiosos. |
| MNSEE.003.0011 | Maracá | 7 | Possui um cordão colorido com fios de lã aglomerados. É atravessado por um pedaço de madeira (cedro). Se usa nos rituais religiosos para se  |
| MNSEE.003.0032 | Cordão | 6 | Cordão feito com sementes e aglomerados de lã em vermelho, branco e verde. |
