# Design

## Context

See `proposal.md` for motivation. Atualmente:
- `src/sinergia/agentes/a1_recepcao/` já estabelece o padrão arquitetural dos agentes sob o protocolo `sinergia/1` (`config.py`, `schemas.py`, `prompts.py`, `repositorio.py`, `servico.py`).
- `src/sinergia/rag/tools.py` já define as 5 ferramentas `@tool` do CrewAI (`ferramenta_busca_pausas` a `ferramenta_busca_epis`) integradas ao `BuscadorHibrido`.
- `src/sinergia/rag/pitch/` hospeda a interface Streamlit ("SinergIA - RAG Demo") com o painel de telemetria "Raio-X Nerd" e visualização de BBox no PDF, mas está localizada dentro do pacote `rag/` e chama `GroqPitchService` diretamente sem passar pelo Agente 2 (`a2_bem_estar`) nem pelo CrewAI.

## Goals / Non-Goals

**Goals:**
- Estruturar `src/sinergia/agentes/a2_bem_estar/` com fachada `executar(tarefa) -> resultado` no protocolo `sinergia/1`.
- Implementar de ponta a ponta a ação conversacional **`a2.responder`** usando **CrewAI** configurado com a chave **`GROQ_API_KEY`** (suportando seleção de modelo Groq, ex: `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `llama-3.1-8b-instant`).
- Cobrir na ação `a2.responder` tanto consultas normativas (`F1`..`F5`) com citação estruturada (`base`, `titulo`, `versao`) quanto o roteiro de EPI avariado/ausente (`F5`, NR-6) com abertura de solicitação em `apoio.solicitacao_epi` e emissão do evento `solicitacao_epi.aberta`.
- Mover `src/sinergia/rag/pitch/` para `src/sinergia/chat_demo/` e fazer o `app.py` executar o turno conversacional através do Agente 2 (`a2_bem_estar`), preservando 100% da telemetria do "Raio-X Nerd" (Chunks & RRF, Embedding 768d, Metadados, Tool & Prompt e BBox PDF).
- Deixar os contratos Pydantic e o backlog de `tasks.md` prontos para as demais ações do A2 (`a2.gerar_lembretes`, `a2.recomendar_beneficios`, `a2.orientar_epi`).

**Non-Goals:**
- Implementar a lógica completa de agendamento de `a2.gerar_lembretes` ou o treinamento/inferência da Rede Neural de Benefícios (`a2.recomendar_beneficios`) nesta primeira entrega focada no conversacional (eles ficam registrados nas tasks de backlog para a fase seguinte).
- Implementar o Orquestrador `A0` (LangGraph) ou o `A3` neste escopo.

## Decisions

### 1. Arquitetura da Crew do A2 (`src/sinergia/agentes/a2_bem_estar/crew.py`) com CrewAI + Groq
- **Decisão**: Definir no módulo `crew.py` os agentes do CrewAI conforme a Seção 5.1 da documentação v2.2:
  - **Gerente / Roteador da Crew**: Analisa a pergunta do colaborador, o `historico_resumo` e a `intencao` recebida na `TAREFA` (ou decide autonomamente entre as bases `F1` a `F5` e guardrail conversacional quando `intencao` não vier restrita), determinando também se há relato de EPI avariado/perdido/ausente (`F5`).
  - **5 Especialistas (`Agent` do CrewAI)**:
    1. `especialista_pausas` (`F1` — `ferramenta_busca_pausas`, NR-17)
    2. `especialista_direitos_beneficios` (`F2` — `ferramenta_busca_direitos`, CLT e Benefícios)
    3. `especialista_procedimentos` (`F3` — `ferramenta_busca_procedimentos`, POPs e Treinamentos)
    4. `especialista_saude_ocupacional` (`F4` — `ferramenta_busca_saude_ocupacional`, PCMSO/NR-7)
    5. `especialista_epi` (`F5/F6` — `ferramenta_busca_epis`, NR-6/CA + tool de abertura de `solicitacao_epi`)
  - **Redator Final (`Agent` do CrewAI)**: Consolida os insumos recuperados pelos especialistas no tom acolhedor, objetivo e personalizado ao perfil (`cargo`, `setor`, `beneficios`), garantindo os 5 critérios avaliados pelo Judge (`Fundamentação`, `Aderência ao perfil`, `Segurança`, `Tom`, `Clareza`).
- **Configuração do LLM na CrewAI**: Utiliza `crewai.LLM` (ou `ChatGroq` via LiteLLM/LangChain do CrewAI) apontando para `groq/<modelo>` e lendo `GROQ_API_KEY` do ambiente. Para garantir resiliência em testes automatizados offline (CI sem chave ou quando um LLM fake/determinístico é injetado) e evitar falhas por rate-limit (HTTP 429), `ExecutorCrewA2` permite injeção de dependência e fallback controlado mantendo a mesma estrutura de agentes/tools do CrewAI.
- **Alternativas consideradas**:
  - *Crew hierárquica pura (`Process.hierarchical`) com 7 agentes ativos em toda pergunta*: Descartada como caminho único porque em perguntas de domínio único gera 4+ chamadas LLM em série, aumentando latência no Streamlit e consumo de rate-limit da Groq. Em vez disso, o Gerente seleciona os especialistas necessários (1 ou mais domínios) e monta a execução `Crew(..., process=Process.sequential)` com os Especialistas selecionados + Redator Final.

### 2. Instrumentação de Telemetria entre `a2_bem_estar` e `src/sinergia/chat_demo/`
- **Decisão**: Para que o painel **"Raio-X Nerd"** do `chat_demo` continue exibindo os tempos separados (`roteamento_ms`, `embed_ms`, `db_rrf_ms`, `llm_ms`), os chunks com scores denso/léxico/RRF, o vetor 768d e o preview de BBox no PDF, as tools RAG acionadas pela Crew do A2 registrarão a última `ResultadoBuscaInstrumentada` em um coletor de contexto da execução (`TelemetriaExecucaoA2`), que também pode ser retornado ou consultado pelo `chat_demo` junto ao `RESULTADO` oficial do protocolo `sinergia/1`.
- **Racional**: O envelope público `RESULTADO` do protocolo `sinergia/1` permanece 100% fiel ao Anexo C.3 e D.3 (`payload: {base, trechos_usados, id_solicitacao_epi}`, `saidas`, `eventos`, `decisao`), enquanto o `chat_demo` obtém os artefatos ricos do Raio-X sem poluir o contrato padrão entre `A0` e `A2`.

### 3. Migração de `src/sinergia/rag/pitch/` para `src/sinergia/chat_demo/`
- **Decisão**: Mover todos os arquivos de `src/sinergia/rag/pitch/` para `src/sinergia/chat_demo/`.
- **Ajustes de Caminho**:
  - Em `src/sinergia/rag/pitch/app.py`, `DIRETORIO_MOCKS = Path(__file__).resolve().parents[4] / "data" / "rag" / "mocks"` subia 4 níveis (`pitch -> rag -> sinergia -> src -> raiz`). Em `src/sinergia/chat_demo/app.py`, como subiu um nível na árvore, passará a usar `Path(__file__).resolve().parents[3] / "data" / "rag" / "mocks"`.
  - Todos os imports `from sinergia.rag.pitch...` passam para `from sinergia.chat_demo...`.

### 4. Repositório do A2 (`RepositorioA2Protocol`)
- **Decisão**: Seguir o mesmo padrão Dual (`RepositorioA2Memoria` e `RepositorioA2Postgres`) de `a1_recepcao/repositorio.py`, provendo métodos para consultar perfil do colaborador e registrar `apoio.solicitacao_epi` (com status `ABERTA`) quando `a2.responder` identificar um relato de EPI danificado, perdido, não recebido ou de tamanho inadequado.

## Risks / Trade-offs

- **[Risk] Latência e Rate Limit (HTTP 429) da Groq ao rodar múltiplos agentes CrewAI em sequência** → **Mitigação**: Quando a consulta atingir apenas uma base e as tools RAG já trouxerem o contexto estruturado, a task da CrewAI pode executar o Especialista/Redator de forma enxuta com retry de backoff para HTTP 429, mantendo a compatibilidade com streaming/status no `chat_demo`.
- **[Risk] Quebra de referências a `DIRETORIO_MOCKS` após mover `pitch` para `chat_demo`** → **Mitigação**: Ajustar `parents[4]` para `parents[3]` em `src/sinergia/chat_demo/app.py` e validar com testes automatizados de resolução de PDF e BBox.
