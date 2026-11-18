# Spec Delta

## Purpose

Fornece o Agente 2 — Bem-Estar (`src/sinergia/agentes/a2_bem_estar/`) como núcleo proativo e conversacional da solução SinergIA sob o protocolo `sinergia/1`, utilizando uma equipe de agentes especialistas em CrewAI integrada à API da Groq (`GROQ_API_KEY`) para responder dúvidas e relatos do colaborador fundamentados nas 5 bases RAG (`F1` a `F5`), registrar solicitações de reposição de EPI (`F5`) e expor os contratos das ações `a2.responder`, `a2.gerar_lembretes`, `a2.recomendar_beneficios` e `a2.orientar_epi`.

## ADDED Requirements

### Requirement: Fachada Oficial do Protocolo sinergia/1 com Idempotência

O Agente 2 SHALL expor em `src/sinergia/agentes/a2_bem_estar/` uma função `executar(tarefa) -> resultado` que valide o envelope `TAREFA` do protocolo `sinergia/1` (`origem="A0"`, `destino="A2"`), execute a ação solicitada (`a2.responder`, `a2.gerar_lembretes`, `a2.recomendar_beneficios` ou `a2.orientar_epi`) e retorne um envelope `RESULTADO` (`origem="A2"`, `destino="A0"`) garantindo idempotência pelo `id_mensagem`.

#### Scenario: Execução idempotente de uma tarefa a2.responder

- **WHEN** o Agente 2 recebe duas vezes uma `TAREFA` com o mesmo `id_mensagem` e ação `a2.responder`
- **THEN** o Agente 2 processa a primeira chamada e devolve na segunda chamada exatamente o mesmo `RESULTADO` sem reexecutar a CrewAI nem duplicar registros no repositório.

#### Scenario: Rejeição estruturada de payload inválido

- **WHEN** o Agente 2 recebe uma `TAREFA` cujo `payload` viola o schema esperado da ação
- **THEN** o Agente 2 retorna um `RESULTADO` com `status="ERRO"` e objeto `erro` contendo `codigo="PAYLOAD_INVALIDO"` e `reprocessavel=False`.

### Requirement: Atendimento Conversacional via CrewAI e Groq (a2.responder)

Na ação `a2.responder`, o Agente 2 SHALL utilizar uma Crew do framework CrewAI conectada à API da Groq (`GROQ_API_KEY`), composta por agentes especialistas por domínio (`F1` a `F5`) equipados com as ferramentas do subsistema RAG e um Redator Final que produza respostas personalizadas ao perfil do colaborador com citação explícita de base, título e versão documental.

#### Scenario: Resposta fundamentada a dúvida trabalhista ou de procedimento

- **WHEN** o Agente 2 executa `a2.responder` com uma pergunta normativa (ex: parcelamento de férias em `F2` ou pausa de digitação em `F1`) e o perfil do colaborador (`cargo`, `setor`, `beneficios`)
- **THEN** a CrewAI aciona o especialista e a ferramenta RAG da base correspondente, sintetiza a resposta pelo Redator Final e retorna `status="OK"`, `payload` com `base` e `trechos_usados`, e `saidas` contendo a mensagem com `fontes` preenchidas (`base`, `titulo`, `versao`).

#### Scenario: Guardrail conversacional para saudações sem busca vetorial

- **WHEN** o Agente 2 executa `a2.responder` com uma saudação ou agradecimento sem intenção de consulta normativa (ex: "Bom dia, tudo bem?")
- **THEN** o Agente 2 responde de forma acolhedora apresentando os serviços de bem-estar (`F1` a `F5`) sem acionar busca vetorial desnecessária (`trechos_usados=0` e `fontes=[]`).

### Requirement: Roteiro Normativo de Relato de EPI Avariado ou Ausente (F5 / NR-6)

Quando a mensagem em `a2.responder` relatar problema com Equipamento de Proteção Individual (EPI avariado, danificado, perdido, não recebido ou de tamanho inadequado), o Agente 2 SHALL orientar imediatamente o colaborador a não executar a atividade de risco sem o EPI adequado nem utilizar o equipamento danificado (conforme NR-6), registrar uma solicitação de reposição com status `ABERTA` no repositório (`apoio.solicitacao_epi`) e emitir o evento `solicitacao_epi.aberta`.

#### Scenario: Colaborador relata luva ou equipamento danificado no chat

- **WHEN** o Agente 2 recebe `a2.responder` com um relato de EPI avariado ou ausente (ex: "Minha luva de proteção rasgou durante o turno")
- **THEN** o Agente 2 abre um registro em `apoio.solicitacao_epi` com status `ABERTA`, orienta na mensagem de saída a interromper a atividade de risco e avisar o gestor com citação da base `epi`, preenche `payload.id_solicitacao_epi` e inclui `{"tipo": "solicitacao_epi.aberta", "dados": {"id_solicitacao": <id>}}` na lista `eventos`.
