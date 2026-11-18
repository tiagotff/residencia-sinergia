# Tasks

## 1. Contratos, Configuração e Repositório do Agente 2 (`a2_bem_estar`)

- [x] 1.1 Criar `src/sinergia/agentes/a2_bem_estar/config.py` com `A2Config` (lendo `GROQ_API_KEY`, `SINERGIA_A2_LLM_MODEL`, `DATABASE_URL` e timeout) e verificar a instanciação com variáveis de ambiente padrão.
- [x] 1.2 Criar `src/sinergia/agentes/a2_bem_estar/schemas.py` com os modelos Pydantic do protocolo `sinergia/1` (`TarefaA2`, `ResultadoA2`, `PayloadResponderTarefa`, `PayloadResponderResultado`, além dos schemas de contrato das ações `a2.gerar_lembretes`, `a2.recomendar_beneficios` e `a2.orientar_epi` conforme Anexos C.3 e D.3) e verificar com testes unitários de validação Pydantic.
- [x] 1.3 Criar `src/sinergia/agentes/a2_bem_estar/repositorio.py` com `RepositorioA2Protocol`, `RepositorioA2Memoria` e `RepositorioA2Postgres` (suportando leitura de perfil do colaborador e abertura de solicitação em `apoio.solicitacao_epi` com status `ABERTA`) e verificar com testes unitários em memória.

## 2. CrewAI com Groq e Ação Conversacional (`a2.responder`)

- [x] 2.1 Criar `src/sinergia/agentes/a2_bem_estar/prompts.py` com os prompts base do Gerente/Roteador, dos 5 Especialistas (`F1` Pausas/NR-17, `F2` Direitos/CLT, `F3` Procedimentos/POPs, `F4` Saúde Ocupacional/PCMSO, `F5` EPIs/NR-6) e do Redator Final (alinhado aos 5 critérios do Judge) e verificar a importação dos templates.
- [x] 2.2 Criar `src/sinergia/agentes/a2_bem_estar/crew.py` implementando a Crew do Agente 2 com `crewai` (`Agent`, `Task`, `Crew`) conectada à Groq (`GROQ_API_KEY`) e às ferramentas RAG de `src/sinergia/rag/tools.py`, incluindo detecção e abertura de solicitação de reposição de EPI (`F5`, NR-6) e captura de telemetria instrumentada (`ResultadoBuscaInstrumentada`), verificando a execução com testes unitários.
- [x] 2.3 Criar `src/sinergia/agentes/a2_bem_estar/servico.py` e `__init__.py` expondo `executar(tarefa) -> resultado` com controle de idempotência por `id_mensagem` e roteamento completo de `a2.responder` (incluindo `saidas` com `fontes[{base, titulo, versao}]`, `payload.{base, trechos_usados, id_solicitacao_epi}` e evento `solicitacao_epi.aberta`), verificando com uma suíte de testes de contrato do Anexo D.3 em `tests/`.
  
## 3. Migração de `rag/pitch` para `src/sinergia/chat_demo` e Integração com o A2

- [x] 3.1 Mover o pacote `src/sinergia/rag/pitch/` para `src/sinergia/chat_demo/`, atualizar os imports internos (`sinergia.chat_demo.*`), corrigir a resolução de `DIRETORIO_MOCKS` (`parents[3] / "data" / "rag" / "mocks"`) e verificar que `src/sinergia/rag/pitch/` não existe mais.
- [x] 3.2 Integrar `src/sinergia/chat_demo/app.py` ao Agente 2 (`sinergia.agentes.a2_bem_estar`), fazendo `executar_turno_rag` despachar a tarefa `a2.responder` via CrewAI + Groq no lugar da geração direta da LLM e populando `TelemetriaTurno` (tempos, chunks & RRF, embedding 768d, metadados, tool/prompt da Crew e citações/BBox PDF), verificando com os testes automatizados da demo atualizados para `sinergia.chat_demo`.

## 4. Evoluções Posteriores do Agente 2 (Backlog pós-conversacional)

- [ ] 4.1 Implementar a ação `a2.orientar_epi` (`F6`) em `src/sinergia/agentes/a2_bem_estar/servico.py` acionando o Especialista de EPI + Redator Final da CrewAI a partir do alerta `A3.4` (`id_alerta`, `id_colaborador`, `id_epi`, `situacao`) e verificar com teste de contrato do Anexo D.3.
- [ ] 4.2 Implementar a ação `a2.gerar_lembretes` (`F1` a `F4`) com verificação de janela de jornada contratual (`jornada_inicio`..`jornada_fim`), suspensão/reagendamento em férias (`apoio.ferias`) e afastamento confirmado (`afastamento.atestado`), gravando em `bem_estar.lembrete` (`status="GERADO"`), e verificar com testes unitários de janela e reagendamento.
- [ ] 4.3 Implementar o módulo `src/sinergia/modelos/rede_neural_beneficios/` (MLP PyTorch/ONNX + fallback cold start por cargo) e conectá-lo à ação mensal `a2.recomendar_beneficios` (`F2`) gravando até 3 recomendações por colaborador em `bem_estar.recomendacao_beneficio` com emissão do evento `recomendacao.gerada`, verificando com teste de contrato do Anexo D.3.
