# Design

## Context

O subsistema RAG em `src/sinergia/rag/` já implementa limpeza em 4 estágios, chunking híbrido, geração de embeddings de 768 dimensões (`tardellirs/colibri-embed-ptbr`) e busca híbrida (cosseno + `tsvector`) com Reciprocal Rank Fusion (`k=60`, `top_k=4`). Atualmente, `BuscadorHibrido.buscar()` descarta os artefatos intermediários (vetor da query, scores denso e léxico pré-RRF e tempos de cada estágio) ao retornar apenas o `ContextoRAG`.

Durante a fase de exploração, foi validado um protótipo de interface (mantido como referência estrutural e funcional) com foco em um layout de duas colunas sem seletores manuais de configuração, operando em **light theme monocromático**, intitulado **SinergIA - RAG Demo**, com LLM local via Ollama decidindo autonomamente qual base consultar. Veja `proposal.md` para a motivação e `specs/` para os contratos comportamentais.

## Goals / Non-Goals

**Goals:**
- Construir a aplicação Streamlit em `src/sinergia/rag/pitch/` fiel à estrutura funcional validada no protótipo HTML, adaptada aos componentes do Streamlit com **tema monocromático em light theme**.
- Garantir que a LLM local (Ollama) faça o roteamento autônomo entre as 5 ferramentas/bases (`F1` a `F5`) sem exigir seleção manual do usuário.
- Instrumentar a busca híbrida para capturar todas as métricas "nerds" reais (tempos por etapa, vetor 768d, scores denso/léxico/RRF, metadados JSONB e telemetria de tokens do Ollama).
- Suportar na coluna direita a alternância fluida entre a **Visão Global da Sessão** (médias e distribuição de uso por base) e o **Raio-X Nerd** individual ao clicar em uma resposta.

**Non-Goals:**
- Expor sliders de temperatura/`top-k`/`k` RRF ou filtros manuais de base na UI (apenas o seletor de modelo Ollama é permitido).
- Alterar o contrato de retorno existente de `BuscadorHibrido.buscar()` ou das funções em `src/sinergia/rag/tools.py` usadas pelos agentes CrewAI.
- Substituir o orquestrador multiagente principal (`src/sinergia/orquestrador/`); o escopo restringe-se ao módulo de demonstração do RAG em `src/sinergia/rag/pitch/`.

## Decisions

### 1. Arquitetura do Módulo `src/sinergia/rag/pitch/` e Mapeamento do Protótipo HTML para Streamlit

A estrutura será dividida em módulos coesos dentro de `src/sinergia/rag/pitch/`:
```text
src/sinergia/rag/pitch/
├── __init__.py
├── app.py                 # Entrypoint Streamlit ("SinergIA - RAG Demo")
├── ollama_service.py      # Cliente Ollama local (roteamento autônomo F1..F5 + geração + métricas)
├── telemetry.py           # Modelos de telemetria por turno e agregadores da Visão Global
├── theme.py               # CSS e configuração de Light Theme Monocromático
└── reference_mockup.html  # Estrutura HTML de referência preservada da exploração
```

Mapeamento dos elementos do HTML para componentes nativos do Streamlit:
- **Layout Principal**: `st.set_page_config(page_title="SinergIA - RAG Demo", layout="wide", initial_sidebar_state="collapsed")` + duas colunas `col_chat, col_panel = st.columns([7, 5], gap="large")`.
- **Cabeçalho e Perguntas Rápidas**: Título `SinergIA - RAG Demo` acompanhado de um `st.selectbox` discreto para seleção do modelo local do Ollama (populado via `GET /api/tags`) e botões horizontais (`st.button`) para disparar perguntas de demonstração (`NR-17`, `CA 12345`, `CLT Art. 473`).
- **Diretriz de Implementação UI**: Antes de criar os componentes visuais (`theme.py` e `app.py`), consultar e aplicar as skills `/frontend-design` e `/ui-ux-pro-max` para garantir acabamento visual refinado no light theme monocromático.
- **Mensagens do Chat + Clique para Raio-X**: Cada resposta da assistente é renderizada em um container estilizado contendo o badge monocromático da base consultada (`Base F1 · Pausas e Ergonomia (NR-17)`), o tempo de resposta, o texto fundamentado, as pílulas de citação e um botão de ação (`"Inspecionar no Raio-X"` / `"Inspecionando"`) que seleciona aquela mensagem no `st.session_state`.
- **Painel Direito**: Renderiza dinamicamente a **Visão Global** (`selected_msg_id is None`) ou o **Raio-X da Resposta** (`selected_msg_id == msg.id`) com `st.tabs(["1. Chunks & RRF", "2. Embedding (768d)", "3. Metadados", "4. Tool & Prompt"])`.

### 2. Estilização Monocromática em Light Theme

Para garantir o visual monocromático claro mesmo que o navegador do usuário esteja em dark mode:
- O módulo `theme.py` injetará variáveis e regras CSS forçando paleta monocromática clara:
  - Fundo principal: `#FFFFFF` (Branco puro)
  - Superfícies secundárias / painéis / cards: `#FAFAFA` e `#F4F4F5` (Zinc-50 / Zinc-100)
  - Bordas sutis: `#E4E4E7` (Zinc-200) e borda ativa `#18181B` (Zinc-900)
  - Tipografia principal: `#09090B` (Zinc-950) e texto secundário `#71717A` (Zinc-500)
  - Acento / botões primários / barras de progresso: `#18181B` (Preto/Grafite monocromático com texto `#FFFFFF`)
- A sidebar nativa (`[data-testid="stSidebar"]`), o botão de colapso da sidebar e a barra de decoração colorida superior do Streamlit serão ocultados via CSS.

### 3. Roteamento Autônomo de Ferramentas e Geração com Ollama Local

Como o usuário não seleciona a base manualmente, a LLM local no Ollama atua em duas etapas coordenadas por `ollama_service.py` (conectando em `OLLAMA_BASE_URL`, padrão `http://localhost:11434`):

```text
[Pergunta do Usuario]
         |
         v
+-------------------------------------------------------------------+
| Passo 1: Roteamento Autonomo (Ollama /api/chat)                   |
| - Avalia as 5 tools (F1 Pausas, F2 CLT, F3 POPs, F4 PCMSO, F5 EPI)|
| - Suporta Tool Calling nativo OU structured JSON output           |
| -> Decide: id_base (1..5), tool_name e query de busca             |
+-------------------------------------------------------------------+
         |
         v
+-------------------------------------------------------------------+
| Passo 2: Busca Hibrida Instrumentada (BuscadorHibrido)            |
| - Gera embedding 768d (mede embed_ms, norma L2, min/max/mean)     |
| - Executa busca densa + lexica + RRF (mede db_rrf_ms)             |
| - Captura scores e ranks individuais (denso, lexico, RRF)         |
+-------------------------------------------------------------------+
         |
         v
+-------------------------------------------------------------------+
| Passo 3: Resposta Fundamentada (Ollama /api/chat)                 |
| - Recebe ContextoRAG.formatar_para_prompt() + pergunta            |
| - Gera resposta com citacoes e coleta metricas nativas do Ollama  |
|   (prompt_eval_count, eval_count, load_duration, tok/s)           |
+-------------------------------------------------------------------+
```

**Alternativas consideradas:**
- *Usar apenas Tool Calling nativo do Ollama*: Modelos menores ou mais antigos no Ollama às vezes não suportam o campo `tools` da API. Por isso, usaremos Tool Calling nativo quando suportado pelo modelo ativo e um fallback transparente com `format: "json"` (esquema `{"id_base": int, "query": str}`) caso o modelo não retorne `tool_calls`.
- *Seleção automática do modelo local*: Para respeitar a regra de não ter seletor de modelo na UI, o serviço lê `OLLAMA_MODEL` do ambiente; se não estiver definido ou não estiver instalado localmente, consulta `GET /api/tags` no Ollama e utiliza automaticamente o primeiro modelo disponível.

### 4. Busca Híbrida Instrumentada (`buscar_com_telemetria`)

Em `src/sinergia/rag/interfaces.py` e `src/sinergia/rag/retrievers.py`:
- Adicionaremos modelos Pydantic `DetalheScoreTrecho`, `EstatisticasEmbedding` e `ResultadoBuscaInstrumentada`.
- O método `calcular_rrf` (ou função auxiliar `calcular_rrf_detalhado`) preservará o score e o rank de cada trecho na lista densa (`dense_score`, `dense_rank`) e na lista léxica (`lex_score`, `lex_rank`) antes de unificá-los pelo score RRF.
- O método `BuscadorHibrido.buscar_com_telemetria(query, id_base, limite=4)` medirá separadamente o tempo de `gerar_embedding(query)` (`embed_ms`) e o tempo da consulta ao banco/memória + cálculo RRF (`db_rrf_ms`).
- `BuscadorHibrido.buscar()` continuará retornando `ContextoRAG` exatamente como hoje, sem impacto nos testes ou ferramentas existentes.

## Risks / Trade-offs

- **[Risco] Banco PostgreSQL desligado no momento da apresentação** → **Mitigação**: Na inicialização do app Streamlit (`@st.cache_resource`), se a conexão com o PostgreSQL falhar, o `BuscadorHibrido` carregará automaticamente os documentos de `src/sinergia/rag/data/mocks/` em memória (`adicionar_trechos_memoria`), garantindo que a demonstração funcione sem interrupções.
- **[Risco] Servidor Ollama local indisponível ou sem modelos baixados** → **Mitigação**: Exibir mensagem clara dentro do próprio chat orientando o comando `ollama serve` / `ollama pull`, além de permitir fallback determinístico de demonstração caso desejado em testes automatizados.
- **[Trade-off] Duas chamadas ao Ollama por pergunta (Roteamento + Geração)** → **Mitigação**: A chamada de roteamento gera apenas ~15 tokens (decisão da tool `F1..F5`) com `temperature=0.0`, adicionando baixa latência (~100-250ms) em troca de precisão no isolamento de domínio exigido pela arquitetura do Agente A2.
