# Tasks

## 1. Dependências e Estrutura do Pacote `pitch`

- [x] 1.1 Adicionar `streamlit` e `httpx` às dependências em `pyproject.toml` e verificar que a instalação (`pip install -e .` ou importação no ambiente virtual) conclui sem erros.
- [x] 1.2 Criar a estrutura de diretórios `src/sinergia/rag/pitch/` com `__init__.py` e salvar a estrutura HTML de referência em `src/sinergia/rag/pitch/reference_mockup.html` (contendo o título `SinergIA - RAG Demo` e os fluxos de Visão Global e Raio-X), verificando a presença dos arquivos no pacote.

## 2. Busca Híbrida Instrumentada e Telemetria no Subsistema RAG

- [x] 2.1 Definir os contratos Pydantic de telemetria (`DetalheScoreTrecho`, `EstatisticasEmbedding`, `ResultadoBuscaInstrumentada`) em `src/sinergia/rag/interfaces.py` e verificar com testes unitários em `src/sinergia/rag/tests/unit/` que a serialização e validação dos campos funcionam.
- [x] 2.2 Implementar `calcular_rrf_detalhado` e `BuscadorHibrido.buscar_com_telemetria(query, id_base, limite=4)` em `src/sinergia/rag/retrievers.py` capturando `embed_ms`, `db_rrf_ms`, estatísticas do vetor de 768 dimensões e scores/posições individuais (`dense_score`, `dense_rank`, `lex_score`, `lex_rank`, `rrf_score`), verificando com `pytest src/sinergia/rag/tests/` que tanto os testes existentes quanto os novos testes unitários de telemetria passam.

## 3. Roteamento Autônomo de Base e Serviço Ollama Local

- [x] 3.1 Implementar `src/sinergia/rag/pitch/ollama_service.py` com descoberta automática de modelo local no Ollama (`/api/tags`), roteamento autônomo para decidir qual das 5 ferramentas/bases (`F1` a `F5`) consultar (suportando Tool Calling nativo e fallback JSON estruturado), geração da resposta fundamentada e extração das métricas nativas (`prompt_eval_count`, `eval_count`, `load_duration`, tokens/s), verificando o comportamento com testes unitários mocando as respostas HTTP do Ollama.
- [x] 3.2 Implementar `src/sinergia/rag/pitch/telemetry.py` com os modelos de sessão e funções de agregação da **Visão Global** (tempo médio total, latência média RAG, latência média LLM, velocidade média em tokens/s, total de tokens gerados e contagem/percentual de acionamento por base `F1..F5`), verificando os cálculos com testes unitários em `src/sinergia/rag/tests/unit/`.

## 4. Interface Streamlit (`SinergIA - RAG Demo`) e Tema Monocromático Light

> **DISCLAIMER**: Antes de iniciar a criação dos componentes de UI desta seção (`4.1` e `4.2`), executar/ler obrigatoriamente as skills `/frontend-design` e `/ui-ux-pro-max` para orientar o design visual monocromático em light theme.

- [x] 4.1 Implementar `src/sinergia/rag/pitch/theme.py` com a estilização CSS de **light theme monocromático** (paleta em branco `#FFFFFF`, tons de cinza `#FAFAFA`/`#F4F4F5`/`#71717A` e destaque preto/grafite `#18181B`, ocultando a sidebar e controles laterais), verificando que as regras CSS e helpers de componentes (badges de base consultada e cards) renderizam sem seletores manuais de base ou hiperparâmetros.
- [x] 4.2 Implementar `src/sinergia/rag/pitch/app.py` com o título **`SinergIA - RAG Demo`**, seletor (`st.selectbox`) de modelo Ollama no cabeçalho, layout em duas colunas (Chat à esquerda com indicativo visual da base consultada em cada resposta e ação de clique para inspeção; Painel à direita alternando entre a **Visão Global da Sessão** por padrão e o **Raio-X Nerd** com as 4 abas `Chunks & RRF`, `Embedding (768d)`, `Metadados` e `Tool & Prompt` ao selecionar uma resposta) e fallback automático de carga dos mocks em memória caso o PostgreSQL esteja indisponível, verificando a execução sem erros via teste de fumaça/importação e `pytest`.
