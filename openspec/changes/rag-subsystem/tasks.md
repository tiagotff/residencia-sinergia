# Tasks: Implementação do Subsistema RAG (TDD)

## 1. Dependências, Schemas Tipados e Suíte Base

- [ ] 1.1 Atualizar `pyproject.toml` com dependências necessárias (`sentence-transformers`, `pymupdf`, `pgvector`) e verificar consistência com `pip check` ou `ruff check`.
- [ ] 1.2 Implementar testes unitários em `src/sinergia/rag/tests/unit/test_interfaces.py` e em seguida implementar `src/sinergia/rag/interfaces.py` com `MetadadosTrecho`, `TrechoRecuperado` e `ContextoRAG` (incluindo `formatar_para_prompt`), garantindo que os testes passem.

## 2. Processamento Textual (TDD)

- [ ] 2.1 Implementar testes unitários em `src/sinergia/rag/tests/unit/test_embeddings.py` e implementar `src/sinergia/rag/embeddings.py` com carregamento singleton do modelo `tardellirs/colibri-embed-ptbr` e geração de vetores de 768 dimensões.
- [ ] 2.2 Implementar testes unitários em `src/sinergia/rag/tests/unit/test_limpeza.py` cobrindo os 4 estágios de sanitização e em seguida implementar `src/sinergia/rag/limpeza.py`.
- [ ] 2.3 Implementar testes unitários em `src/sinergia/rag/tests/unit/test_chunking.py` cobrindo o corte normativo e Document-Based via PyMuPDF e em seguida implementar `src/sinergia/rag/chunking.py`.

## 3. Recuperação Híbrida e Ferramentas CrewAI (TDD)

- [ ] 3.1 Implementar testes unitários em `src/sinergia/rag/tests/unit/test_retrievers.py` cobrindo o cálculo matemático do RRF ($k=60$), ranking top-4 e filtros de vigência/base, e em seguida implementar `src/sinergia/rag/retrievers.py`.
- [ ] 3.2 Implementar testes em `src/sinergia/rag/tests/unit/test_tools.py` e implementar `src/sinergia/rag/tools.py` expondo as ferramentas tipadas para os especialistas do CrewAI.

## 4. Ingestão Idempotente, Testes E2E e Exportações

- [ ] 4.1 Implementar `src/sinergia/rag/ingestao.py` com CLI de ingestão atômica (`vigente = false` e bulk insert).
- [ ] 4.2 Implementar testes E2E em `src/sinergia/rag/tests/e2e/test_rag_pipeline.py` validando o ciclo completo (ingestão dos PDFs de `src/sinergia/rag/data/mocks/` até busca e formatação de prompt).
- [ ] 4.3 Atualizar `src/sinergia/rag/__init__.py` exportando os módulos e interfaces centrais e executar a suíte completa de testes (`pytest src/sinergia/rag/tests/`).

## Workflow follow-up

- Executar validação dos componentes RAG com testes automatizados.
- Arquivar a mudança com `openspec archive rag-subsystem` após a implementação e revisão.

