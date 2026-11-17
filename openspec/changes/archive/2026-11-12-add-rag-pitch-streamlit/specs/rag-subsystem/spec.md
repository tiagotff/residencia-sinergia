# Spec Delta

## ADDED Requirements

### Requirement: Recuperação Híbrida Instrumentada com Telemetria Detalhada

O subsistema RAG SHALL fornecer uma interface de busca instrumentada que, além do `ContextoRAG` padrão, retorne os artefatos intermediários da execução para fins de auditoria e telemetria: o vetor de embedding de 768 dimensões da consulta (com norma $L_2$ e estatísticas), os scores e posições individuais de cada trecho nos rankings denso e léxico antes da fusão RRF, e os tempos de execução em milissegundos das etapas de geração de embedding e de busca/fusão RRF.

#### Scenario: Execução de busca com coleta de telemetria

- **WHEN** uma consulta é executada através do método instrumentado de busca para uma base de conhecimento (`id_base` de 1 a 5)
- **THEN** o subsistema retorna o `ContextoRAG` final acompanhado dos tempos separados (`embed_ms` e `db_rrf_ms`), dos dados do vetor de 768 dimensões da query e do detalhamento de `dense_score`, `dense_rank`, `lex_score`, `lex_rank` e `rrf_score` para cada trecho retornado.
