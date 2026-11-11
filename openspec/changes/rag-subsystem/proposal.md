# Proposal: Implementação do Subsistema RAG

## Why

O Agente A2 (Bem-Estar) da solução SinergIA necessita de fundamentação documental precisa para responder às demandas dos colaboradores (pausas, direitos/benefícios, POPs/treinamentos, saúde ocupacional e EPIs), citando fontes e versões documentais oficiais e eliminando alucinações. Para cumprir as metas de qualidade auditadas pelo LLM-as-a-judge e garantir governança normativa e conformidade com a LGPD, é necessária uma infraestrutura de RAG estritamente tipada, com busca híbrida (Dense + BM25 Lexical com RRF), embeddings locais em língua portuguesa e pipeline de ingestão idempotente isolada no módulo `src/sinergia/rag`.

## What Changes

- Criação do wrapper de embeddings utilizando o modelo pré-treinado nativo em português `tardellirs/colibri-embed-ptbr` gerando vetores de 768 dimensões (`embeddings.py`).
- Implementação de pipeline de limpeza e sanitização textual em 4 estágios (`limpeza.py`) para remoção de cabeçalhos/rodapés repetitivos, des-hifenização de quebras de linha e normalização preservando marcadores sintáticos.
- Implementação de chunking híbrido com metadados (`chunking.py`), unindo segmentação hierárquica para normas e leis e Document-Based Chunking estrutural via PyMuPDF para manuais/POPs.
- Criação dos modelos estritos e contratos de dados tipados Pydantic (`interfaces.py`) para intercâmbio com os agentes CrewAI (`MetadadosTrecho`, `TrechoRecuperado`, `ContextoRAG`).
- Implementação da busca híbrida (`retrievers.py`) com dense retrieval no pgvector (HNSW / cosseno) e lexical retrieval no PostgreSQL (`tsvector`), combinados via Reciprocal Rank Fusion (RRF com constante 60) retornando top-4 trechos com pré-filtragem estrita por base e documentos vigentes (`vigente = true`).
- Implementação da CLI e fluxo de ingestão idempotente de documentos (`ingestao.py`), invalidando versões anteriores (`vigente = false`) e persistindo novos trechos e vetores em transação.
- Implementação de ferramentas CrewAI (`tools.py`) para consumo direto pelos especialistas do Agente A2.

## Capabilities

### New Capabilities
- `rag-subsystem`: Recuperação aumentada por geração para os domínios normativos e corporativos do SinergIA, cobrindo processamento, chunking com metadados, vetorização colibri-embed-ptbr, busca híbrida Dense+Lexical RRF sobre PostgreSQL/pgvector e interfaces CrewAI.

### Modified Capabilities
*(Nenhuma capacidade existente modificada — primeira especificação do projeto)*

## Impact

- **Código Afetado**: Estruturação completa do pacote `src/sinergia/rag/` (`__init__.py`, `embeddings.py`, `limpeza.py`, `chunking.py`, `interfaces.py`, `retrievers.py`, `ingestao.py`, `tools.py`).
- **Dependências**: Adição de `sentence-transformers`, `pymupdf` e bibliotecas de suporte a vetorização/pgvector ao ambiente.
- **Banco de Dados**: Interação com o schema `rag` no PostgreSQL 16 com extensão `pgvector` (`rag.base_conhecimento`, `rag.documento`, `rag.trecho`).

