# Sinergia

Solução multiagente de IA para promoção do bem-estar do colaborador, desenvolvida no Challenge 2 (Bem-estar) da Residência em IA.

O **Agente A2 — Bem-Estar** é o núcleo da solução. Os agentes A0 (Orquestrador), A1 (Recepção e Processamento) e A3 (Apoio) fornecem identidade, dados e alertas para ele.

A especificação completa está em [`docs/`](docs/).

## Stack

Python 3.12 · FastAPI · LangGraph · CrewAI · LangChain · PostgreSQL 16 + pgvector · PyTorch + ONNX Runtime

## Estrutura

| Pasta | Conteúdo | Responsável |
| --- | --- | --- |
| `docs/` | Especificação, diagramas e decisões de arquitetura | Tiago |
| `db/` | Schema SQL (3FN) e dados de teste | Tiago |
| `src/sinergia/api/` | API REST (FastAPI) | Leo |
| `src/sinergia/orquestrador/` | A0 — Orquestrador (LangGraph) | Leo |
| `src/sinergia/agentes/a1_recepcao/` | A1 — Recepção e Processamento | Ítalo e Natália |
| `src/sinergia/agentes/a2_bem_estar/` | A2 — Bem-Estar (CrewAI) | Luana e Tiago |
| `src/sinergia/agentes/a3_apoio/` | A3 — Apoio (CrewAI) | A definir |
| `src/sinergia/rag/` | Ingestão, embeddings e retrievers | João |
| `src/sinergia/modelos/rede_neural_beneficios/` | Rede neural de recomendação de benefícios | Ítalo e Natália |
| `src/sinergia/comum/` | Configuração, conexão com o banco e logs | Todos |
| `data/rag/` | Documentos das bases RAG | João |
| `notebooks/` | Exploração e treino de modelos | Todos |
| `tests/` | Testes automatizados | Todos |

## Como rodar localmente

1. Copie `.env.example` para `.env` e preencha as chaves.
2. Suba o banco: `docker compose up -d`
3. Crie as tabelas: `docker compose exec -T db psql -U sinergia -d sinergia < db/schema/001_schema_inicial.sql`
4. Instale as dependências: `pip install -e .`

## Como contribuir

Veja [`CONTRIBUTING.md`](CONTRIBUTING.md).
