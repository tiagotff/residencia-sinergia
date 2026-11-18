# Proposal

## Why

O protótipo conversacional de Bem-Estar encontra-se atualmente acoplado dentro de `src/sinergia/rag/pitch/` realizando chamadas diretas via `httpx` para a LLM, sem utilizar o framework **CrewAI**, sem implementar o contrato oficial do protocolo `sinergia/1` e fora do diretório arquitetural definido na *Documentação - SinergIA v2.2.docx* (`src/sinergia/agentes/a2_bem_estar/`). Precisamos estruturar o Agente 2 (`a2_bem_estar`) com CrewAI + Groq (focado prioritariamente na ação conversacional `a2.responder`) e mover a interface de demonstração de `src/sinergia/rag/pitch/` para `src/sinergia/chat_demo/` integrando-a diretamente ao Agente 2.

## What Changes

- **Novo pacote oficial do Agente 2 (`src/sinergia/agentes/a2_bem_estar/`)**:
  - Implementa a fachada oficial `executar(tarefa) -> resultado` compatível com o envelope JSON `sinergia/1` (Anexos A, C.3 e D.3 da documentação v2.2), com controle de idempotência por `id_mensagem`.
  - Implementa a ação conversacional **`a2.responder`** utilizando **CrewAI** integrado à **API da Groq (`GROQ_API_KEY`)**, estruturando o Gerente/Roteador da Crew, os 5 Especialistas de domínio (`F1` Pausas/NR-17, `F2` Direitos/CLT, `F3` Procedimentos/POPs, `F4` Saúde Ocupacional/PCMSO, `F5` EPIs/CA) com as `@tool` de `src/sinergia/rag/tools.py` e o Redator Final alinhado aos 5 critérios do Judge.
  - Inclui suporte ao roteiro normativo de relato de problema com EPI (`F5`, NR-6): orientação imediata de interrupção da atividade de risco, abertura de solicitação em `apoio.solicitacao_epi` (via repositório Postgres ou em memória) e emissão do evento `solicitacao_epi.aberta`.
  - Mantém mapeadas nos contratos e no backlog de tasks as demais ações do A2 (`a2.gerar_lembretes`, `a2.recomendar_beneficios`, `a2.orientar_epi` e a Rede Neural de Benefícios) para implementação incremental após o fluxo conversacional.
- **BREAKING — Migração de `src/sinergia/rag/pitch/` para `src/sinergia/chat_demo/`**:
  - Move a aplicação Streamlit e seus módulos auxiliares (`app.py`, `ollama_service.py`, `pdf_preview.py`, `telemetry.py`, `theme.py`, `reference_mockup.html`) de `src/sinergia/rag/pitch/` para `src/sinergia/chat_demo/`, deixando `src/sinergia/rag/` exclusivo para o pipeline RAG.
  - Integra `src/sinergia/chat_demo/app.py` ao Agente 2 (`sinergia.agentes.a2_bem_estar`), substituindo a geração direta de LLM pela execução do Agente 2 via CrewAI + Groq enquanto preserva a telemetria do painel "Raio-X Nerd" e a visualização de BBox nos PDFs.

## Capabilities

### New Capabilities
- `a2-bem-estar`: Agente 2 de Bem-Estar (`src/sinergia/agentes/a2_bem_estar/`) implementado com CrewAI e Groq (`GROQ_API_KEY`) sob o protocolo `sinergia/1`, cobrindo a ação conversacional `a2.responder` (especialistas F1..F5, redator final, citações normativas e abertura de solicitação de EPI em F5) e definindo os contratos das ações `a2.gerar_lembretes`, `a2.recomendar_beneficios` e `a2.orientar_epi`.

### Modified Capabilities
- `rag-pitch-demo`: Atualiza a localização do módulo de `src/sinergia/rag/pitch/` para `src/sinergia/chat_demo/` e altera o motor de roteamento e geração conversacional para acionar o Agente 2 (`a2_bem_estar`) com CrewAI + Groq (`GROQ_API_KEY`), mantendo o painel de telemetria "Raio-X Nerd" e a inspeção de trechos em PDF.

## Impact

- **Código Novo**: `src/sinergia/agentes/a2_bem_estar/` (`__init__.py`, `config.py`, `schemas.py`, `prompts.py`, `repositorio.py`, `crew.py`, `servico.py`).
- **Código Movido / Refatorado**: `src/sinergia/rag/pitch/` → `src/sinergia/chat_demo/` e respectivos testes em `src/sinergia/rag/tests/`.
- **Dependências e Configuração**: Uso de `crewai`, `langchain-groq` / `GROQ_API_KEY` (já listados em `pyproject.toml`).
