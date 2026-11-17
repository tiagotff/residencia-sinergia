# Design: Arquitetura e Implementação do Subsistema RAG

## Context

O SinergIA 2.0 utiliza agentes especializados baseados em CrewAI no Agente A2 (Bem-Estar). Para responder consultas com precisão jurídica e operacional (CLT, NRs, POPs, PCMSO, CIPA, EPIs), o subsistema RAG deve operar de modo encapsulado dentro de `src/sinergia/rag/`, interagindo com o banco relacional PostgreSQL 16 com extensão `pgvector`.

## Goals / Non-Goals

**Goals:**
- Encapsular 100% da lógica de recuperação, ingestão e testes automatizados em `src/sinergia/rag/`.
- Adotar metodologia TDD com suíte de testes unitários (`src/sinergia/rag/tests/unit/`) e testes E2E (`src/sinergia/rag/tests/e2e/`).
- Prover representação vetorial compatível com `vector(768)` usando `tardellirs/colibri-embed-ptbr`.
- Implementar sanitização textual em 4 estágios e chunking híbrido com injeção de breadcrumb hierárquico nos metadados.
- Realizar busca híbrida (Dense HNSW + Lexical BM25/tsvector) com fusão Reciprocal Rank Fusion ($k=60$) retornando top-4 trechos sob latência $p95 \le 8s$.
- Prover interfaces Pydantic estritas e tools CrewAI prontas para os especialistas de F1 a F6.
- Garantir governança documental por invalidação atômica de versões antigas (`vigente = false`).

**Non-Goals:**
- Não gerenciar fluxo de conversação ou orquestração geral de agentes (responsabilidade de A0 e A2).
- Não alterar módulos fora de `src/sinergia/rag/` nesta mudança.
- Não executar chamadas a APIs pagas de embedding proprietário (ex: OpenAI embeddings).

## Decisions

### 1. Modelo de Embedding Local (Colibri PT-BR)
- **Decisão**: Utilizar `tardellirs/colibri-embed-ptbr` via `sentence-transformers`.
- **Racional**: Pré-treinado especificamente em linguagem formal e jurídica em português brasileiro (CLT, NRs); produz vetores densos de 768 dimensões compatíveis com `vector(768)` no pgvector; roda localmente garantindo isolamento LGPD e sem custos de API.
- **Alternativas**: OpenAI `text-embedding-3-small` (descartado por custo de API, latência externa e não conformidade estrita com LGPD local).

### 2. Tabela Física Centralizada com Pré-Filtragem Estrita
- **Decisão**: Centralizar trechos em `rag.trecho` com relacionamentos a `rag.documento` e `rag.base_conhecimento`, aplicando pre-filtering na consulta SQL (`d.id_base = :id_base AND d.vigente = true`).
- **Racional**: Evita complexidade de manutenção de 5 tabelas físicas separadas ou fragmentação de índices, aproveitando índices HNSW com filtros e índice GIN para busca textual.
- **Alternativas**: Criação de tabelas separadas por base (descartado por overhead de manutenção de schemas e migrações repetidas).

### 3. Pipeline de Limpeza em 4 Estágios
- **Decisão**: Implementar em `limpeza.py`:
  1. Remoção de cabeçalhos/rodapés repetitivos de páginas PDF via regex.
  2. Des-hifenização de fim de linha (`ter-\nmo` -> `termo`).
  3. Normalização de caracteres de controle e múltiplos espaços (`\x0c`, bytes nulos).
  4. Preservação mandatória de marcadores (`Art.`, `Item`, `§`, `Inciso`, `Alínea`).
- **Racional**: Garante integridade morfológica dos termos técnicos e preserva os pontos de quebra hierárquica.

### 4. Chunking Híbrido com Breadcrumbs Estruturais
- **Decisão**: Em `chunking.py`, aplicar chunking hierárquico para textos normativos e chunking estrutural com PyMuPDF (`fitz`) para manuais/POPs, injetando cabeçalho estrutural em cada chunk (`Documento > Capítulo > Seção/Artigo > Página`).
- **Racional**: Mantém o contexto íntegro tanto para a atenção do modelo de embeddings quanto para a compreensão do LLM sem descolar artigos de suas exceções.

### 5. Busca Híbrida e Reciprocal Rank Fusion (RRF)
- **Decisão**: Implementar em `retrievers.py` busca densa (cosseno no pgvector) e busca léxica via PostgreSQL Full-Text Search (`to_tsvector('portuguese', ...)`), combinadas por RRF:
  $$\text{RRF\_Score}(d) = \frac{1}{60 + \text{rank}_{\text{denso}}(d)} + \frac{1}{60 + \text{rank}_{\text{lexico}}(d)}$$
  Retornando os top-4 trechos.
- **Racional**: Equilibra similaridade semântica para conceitos subjetivos com busca exata alfanumérica essencial para normas ("Art. 473", número de Certificado de Aprovação de EPI "CA 12345").

### 6. Contratos Fortemente Tipados em Pydantic e Ferramentas CrewAI
- **Decisão**: Criar em `interfaces.py` as classes `MetadadosTrecho`, `TrechoRecuperado` e `ContextoRAG` (com método `formatar_para_prompt()`). Em `tools.py`, disponibilizar ferramentas prontas para os agentes CrewAI (F1 a F6).
- **Racional**: Elimina o tráfego de dados não estruturados ou soltos, garantindo rastreabilidade e citações obrigatórias no prompt.

### 7. Ciclo de Ingestão Idempotente
- **Decisão**: Em `ingestao.py`, disponibilizar interface CLI que executa transação no banco: caso já exista documento com mesmo título e base, marca versões anteriores com `vigente = false`, inserindo a nova versão com `vigente = true` e persistindo chunks e embeddings.
- **Racional**: Governança normativa que previne contradições legais históricas mantendo auditabilidade.

## Structure of Files in `src/sinergia/rag/`

```text
src/sinergia/rag/
├── __init__.py       # Exportações públicas do pacote
├── data/             # Documentos PDF e dados do subsistema
│   ├── mocks/        # PDFs sintéticos de teste/desenvolvimento (layouts heterogêneos)
│   └── README.md     # Documentação de sinalização dos mocks e bases
├── tests/            # Suíte de testes automatizados do RAG (TDD)
│   ├── __init__.py
│   ├── unit/         # Testes unitários (interfaces, limpeza, chunking, embeddings, retrievers)
│   │   └── __init__.py
│   └── e2e/          # Testes ponta a ponta (ingestão dos mocks até busca híbrida RRF)
│       └── __init__.py
├── embeddings.py     # Wrapper do Colibri PT-BR (768d, SentenceTransformer)
├── limpeza.py        # 4 estágios de sanitização e des-hifenização
├── chunking.py       # Chunking hierárquico e estrutural (PyMuPDF) com metadados
├── interfaces.py     # Schemas Pydantic (MetadadosTrecho, TrechoRecuperado, ContextoRAG)
├── retrievers.py     # Busca Híbrida (Dense HNSW + Lexical GIN com RRF k=60)
├── ingestao.py       # Script CLI de ingestão e versionamento idempotente
└── tools.py          # Ferramentas CrewAI para o Agente A2
```

## Risks / Trade-offs

- **[Carga de modelo local de 768 dimensões em CPU]** → Uso de cache do modelo em singleton no `embeddings.py` para evitar recargas frequentes e overhead em cada inferência.
- **[Conflito de dependências com PyMuPDF e Sentence Transformers]** → Dependências documentadas de forma compatível com o ecossistema Python 3.12+ do projeto.
- **[Disponibilidade da extensão pgvector e schema rag no banco]** → Tratar conexões com SQLAlchemy/psycopg de forma resiliente, assumindo o schema `rag` existente conforme especificado pelo time de banco.

