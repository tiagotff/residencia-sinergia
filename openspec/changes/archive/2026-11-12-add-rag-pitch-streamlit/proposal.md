# Proposal

## Why

O subsistema RAG do SinergIA já realiza ingestão, vetorização em 768 dimensões (`tardellirs/colibri-embed-ptbr`) e busca híbrida com Reciprocal Rank Fusion (RRF), mas possui apenas uma CLI de terminal para testes manuais das ferramentas isoladas. Para apresentações e validação técnica (Pitch), é necessário um site em Streamlit (`src/sinergia/rag/pitch/`) intitulado **SinergIA - RAG Demo**, que una um chat fluido conectado a uma LLM local via Ollama — capaz de decidir autonomamente qual base consultar — a um painel de telemetria e diagnóstico ("dados nerds") com métricas globais da sessão e inspeção profunda ao clicar em cada resposta.

## What Changes

- Criação do pacote `src/sinergia/rag/pitch/` contendo a aplicação Streamlit **SinergIA - RAG Demo** com **tema monocromático em light theme** (tons neutros/cinza/preto sobre fundo claro).
- Interface limpa focada na conversa e na telemetria:
  - Seletor discreto (`st.selectbox`) para escolher qual modelo local do Ollama utilizar (auto-detectado via `/api/tags`), sem filtro manual de base/domínio e sem ajustes de hiperparâmetros (`top-k`, `k` RRF, temperatura).
  - Sem rótulos de status de infraestrutura (`PostgreSQL + pgvector` / `Ollama Local`) no cabeçalho.
- Integração com LLM local via **Ollama**, onde a própria LLM analisa a pergunta do usuário, decide autonomamente qual das 5 bases/ferramentas de conhecimento (`F1` a `F5`) consultar, e gera a resposta fundamentada com citações.
- Indicativo visual (badge) em cada resposta do chat mostrando explicitamente qual base de conhecimento (`F1` a `F5`) a LLM consultou.
- Painel lateral dinâmico com dois estados:
  1. **Visão Global (estado padrão quando nenhuma resposta está clicada/selecionada)**: exibe médias de tempo das respostas (total, RAG e LLM), velocidade média em tokens/s, total de tokens gerados na sessão, distribuição de uso por base de conhecimento (`F1..F5`) e atalhos para o histórico de perguntas.
  2. **Modo Raio-X Nerd (ativado ao clicar em uma resposta da LLM)**: exibe os dados técnicos daquela resposta específica organizados em métricas de tempo (`Total`, `Embedding`, `DB + RRF`, `LLM`) e 4 visões de inspeção (`Chunks & RRF` com scores denso/léxico individuais, `Embedding 768d` com norma $L_2$/estatísticas/amostra vetorial, `Metadados & JSONB` e `Tool & Prompt` com métricas nativas do Ollama), além de ação para retornar à Visão Global.
- Extensão não-destrutiva em `src/sinergia/rag/retrievers.py` (e contratos auxiliares) para expor a telemetria intermediária da busca (vetor 768d gerado para a query, rankings denso e léxico antes da fusão RRF e tempos de execução de embedding e busca).
- Atualização de `pyproject.toml` para incluir `streamlit` e suporte à comunicação com o servidor local do Ollama.

## Capabilities

### New Capabilities
- `rag-pitch-demo`: Aplicação web em Streamlit (`src/sinergia/rag/pitch/`) com tema monocromático light, chat integrado ao Ollama local com roteamento autônomo de ferramentas RAG (`F1..F5`), indicativo visual da base consultada, painel de métricas globais da sessão e inspeção detalhada ("Raio-X Nerd") ao selecionar uma resposta.

### Modified Capabilities
- `rag-subsystem`: Adição de requisito de telemetria e rastreabilidade de busca híbrida (exposição opcional do vetor da consulta, scores denso e léxico pré-RRF e tempos de execução por etapa) para viabilizar a inspeção técnica sem alterar o contrato padrão dos agentes.

## Impact

- **Código Novo**: `src/sinergia/rag/pitch/` (`__init__.py`, `app.py`, `ollama_client.py`, `telemetria.py`, e referência de estrutura em `reference_mockup.html` / testes em `src/sinergia/rag/tests/`).
- **Código Modificado**: `src/sinergia/rag/retrievers.py` e `src/sinergia/rag/interfaces.py` (adição de método/modelo de busca com telemetria preservando retrocompatibilidade total com `buscar()`).
- **Dependências (`pyproject.toml`)**: Inclusão de `streamlit` e `httpx` (ou `requests`) para chamadas à API REST local do Ollama (`http://localhost:11434`).
