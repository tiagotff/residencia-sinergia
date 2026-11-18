"""Testes unitários para os prompts e para a CrewAI do Agente 2 (`a2_bem_estar`)."""

from __future__ import annotations

from unittest.mock import MagicMock

from crewai.llms.base_llm import BaseLLM

from sinergia.agentes.a2_bem_estar.config import A2Config
from sinergia.agentes.a2_bem_estar.crew import (
    ExecutorCrewA2,
    detectar_relato_problema_epi,
    verificar_guardrail_conversacional,
)
from sinergia.agentes.a2_bem_estar.prompts import (
    CRITERIOS_JUDGE_A2,
    PROMPT_GERENTE_ROTEADOR_A2,
    PROMPT_REDATOR_FINAL_A2,
    PROMPTS_ESPECIALISTAS_A2,
    construir_task_especialista,
    construir_task_redator_final,
    formatar_perfil_para_prompt,
)
from sinergia.agentes.a2_bem_estar.repositorio import RepositorioA2Memoria
from sinergia.agentes.a2_bem_estar.schemas import PerfilColaboradorResponder
from sinergia.rag.interfaces import MetadadosTrecho, TrechoRecuperado
from sinergia.rag.retrievers import BuscadorHibrido


class _FakeCrewLLM(BaseLLM):
    def __init__(self, resposta: str = "Resposta sintetizada pela CrewAI.") -> None:
        super().__init__(model="openai/gpt-oss-20b")
        self.resposta = resposta
        self.chamadas = 0

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
        self.chamadas += 1
        return self.resposta


def test_importacao_e_construcao_prompts_a2() -> None:
    """Valida todos os templates da Crew A2 (Gerente, 5 Especialistas e Redator Final)."""
    assert "Gerente" in PROMPT_GERENTE_ROTEADOR_A2["role"]
    assert "Redator Final" in PROMPT_REDATOR_FINAL_A2["role"]
    assert set(CRITERIOS_JUDGE_A2.keys()) == {
        "FUNDAMENTACAO",
        "ADERENCIA_PERFIL",
        "SEGURANCA",
        "TOM",
        "CLAREZA",
    }
    assert set(PROMPTS_ESPECIALISTAS_A2.keys()) == {1, 2, 3, 4, 5}
    codigos = [PROMPTS_ESPECIALISTAS_A2[i]["codigo"] for i in range(1, 6)]
    assert codigos == ["F1", "F2", "F3", "F4", "F5"]

    perfil_txt = formatar_perfil_para_prompt(
        PerfilColaboradorResponder(
            cargo="Analista de Operações",
            setor="Logística",
            beneficios=["Vale-Alimentação"],
        )
    )
    assert "Analista de Operações" in perfil_txt
    assert "Logística" in perfil_txt
    assert "Vale-Alimentação" in perfil_txt

    task_f5 = construir_task_especialista(
        5,
        pergunta="Minha luva rasgou",
        contexto_rag="[FONTE 1] Procedimento de reposição de EPI (v1.2)",
        perfil_texto=perfil_txt,
        info_solicitacao_epi="#58 (ABERTA)",
    )
    assert "Minha luva rasgou" in task_f5
    assert "#58 (ABERTA)" in task_f5
    assert "NR-6" in task_f5 or "atividade de risco" in task_f5

    task_redator = construir_task_redator_final(
        pergunta="Posso dividir férias?",
        perfil_texto=perfil_txt,
        usar_rag=True,
        contexto_rag="Política de férias v2026.1",
    )
    assert "Fundamentação" in task_redator
    assert "Aderência ao perfil" in task_redator
    assert "Segurança" in task_redator

    task_chat = construir_task_redator_final(
        pergunta="Bom dia!",
        perfil_texto=perfil_txt,
        usar_rag=False,
    )
    assert "não cite nomes de arquivos PDF" in task_chat


def _criar_buscador_teste() -> BuscadorHibrido:
    mock_emb = MagicMock()
    mock_emb.model_name = "tardellirs/colibri-embed-ptbr"
    mock_emb.gerar_embedding.return_value = [0.02 if i % 2 == 0 else -0.01 for i in range(768)]

    buscador = BuscadorHibrido(gerador_embeddings=mock_emb)
    buscador.adicionar_trechos_memoria(
        [
            TrechoRecuperado(
                id_trecho=10,
                conteudo="17.6.4. Deve haver pausa de 10 minutos a cada 50 minutos trabalhados em digitação.",
                metadados=MetadadosTrecho(
                    id_documento=1,
                    titulo="NR-17 Ergonomia",
                    fonte="Portaria MTP 423",
                    versao="2024.1",
                    secao="Item 17.6.4",
                    pagina=4,
                    id_base=1,
                    vigente=True,
                ),
            ),
            TrechoRecuperado(
                id_trecho=20,
                conteudo="É possível parcelar as férias em até 3 períodos mediante concordância, sendo um de ao menos 14 dias.",
                metadados=MetadadosTrecho(
                    id_documento=2,
                    titulo="Política de férias",
                    fonte="CLT Art. 134",
                    versao="2026.1",
                    secao="Seção Férias",
                    pagina=2,
                    id_base=2,
                    vigente=True,
                ),
            ),
            TrechoRecuperado(
                id_trecho=50,
                conteudo="NR-6: Em caso de avaria ou perda do EPI, interrompa a atividade de risco e solicite reposição imediata.",
                metadados=MetadadosTrecho(
                    id_documento=5,
                    titulo="Procedimento de reposição de EPI",
                    fonte="NR-6",
                    versao="1.2",
                    secao="Item 4.1",
                    pagina=1,
                    id_base=5,
                    vigente=True,
                ),
            ),
        ]
    )
    return buscador


def test_crew_a2_estrutura_especialistas_e_consulta_normativa() -> None:
    """Valida instanciação da CrewAI com os 5 especialistas, busca RAG e telemetria."""
    repo = RepositorioA2Memoria()
    buscador = _criar_buscador_teste()
    fake_llm = _FakeCrewLLM("Você pode dividir as férias em até 3 períodos conforme Política de férias (v2026.1).")
    executor = ExecutorCrewA2(
        config=A2Config(groq_api_key=""),
        repositorio=repo,
        buscador=buscador,
        llm=fake_llm,
    )

    assert len(executor.especialistas) == 5
    assert executor.especialistas[1].tools[0].name == "Busca Base de Pausas e Ergonomia (F1)"
    assert executor.especialistas[5].tools[0].name == "Busca Base de EPIs e Fichas de CA (F5/F6)"

    res = executor.executar_responder(
        pergunta="Posso dividir minhas férias em 3 períodos?",
        id_colaborador=42,
        intencao="F2",
    )

    assert fake_llm.chamadas == 1
    assert res.base_slug == "direitos_beneficios"
    assert res.trechos_usados == 1
    assert res.id_solicitacao_epi is None
    assert len(res.fontes) == 1
    assert res.fontes[0].base == "direitos_beneficios"
    assert res.fontes[0].titulo == "Política de férias"
    assert res.fontes[0].versao == "2026.1"
    assert res.telemetria.estatisticas_embedding.dimensoes == 768
    assert len(res.telemetria.resultados_rag) == 1
    assert res.telemetria.agentes_acionados == [
        "gerente_roteador",
        "especialista_direitos_beneficios",
        "redator_final",
    ]


def test_crew_a2_guardrail_conversacional_sem_busca_rag() -> None:
    """Saudação sem pergunta normativa não deve executar busca vetorial nem citar fontes."""
    assert verificar_guardrail_conversacional("Bom dia, tudo bem?")
    buscador_mock = MagicMock()
    fake_llm = _FakeCrewLLM("Bom dia! Como posso ajudar você com Pausas, CLT, POPs, Saúde Ocupacional ou EPIs?")
    executor = ExecutorCrewA2(
        config=A2Config(groq_api_key=""),
        repositorio=RepositorioA2Memoria(),
        buscador=buscador_mock,
        llm=fake_llm,
    )

    res = executor.executar_responder(pergunta="Bom dia, tudo bem?", id_colaborador=42)
    buscador_mock.buscar_com_telemetria.assert_not_called()
    assert res.trechos_usados == 0
    assert res.fontes == []
    assert not res.telemetria.roteamento.usar_rag
    assert res.telemetria.agentes_acionados == ["gerente_roteador", "redator_final"]


def test_crew_a2_deteccao_e_abertura_solicitacao_epi_nr6() -> None:
    """Relato de EPI avariado deve acionar F5, orientar NR-6 e abrir solicitação ABERTA."""
    eh_epi, motivo = detectar_relato_problema_epi("Minha luva de proteção rasgou durante o turno")
    assert eh_epi is True
    assert motivo == "AVARIA"

    eh_pergunta_comum, _ = detectar_relato_problema_epi("Qual a validade do CA da luva?")
    assert eh_pergunta_comum is False

    repo = RepositorioA2Memoria()
    buscador = _criar_buscador_teste()
    executor = ExecutorCrewA2(
        config=A2Config(groq_api_key=""),
        repositorio=repo,
        buscador=buscador,
    )

    res = executor.executar_responder(
        pergunta="Minha luva de proteção rasgou durante o turno",
        id_colaborador=42,
    )
    assert res.base_slug == "epi"
    assert res.id_solicitacao_epi == 58
    assert res.motivo_solicitacao_epi == "AVARIA"
    assert "58" in res.resposta
    assert "não" in res.resposta.lower()
    assert res.fontes[0].base == "epi"
    sol_db = repo.obter_solicitacao_epi(58)
    assert sol_db is not None
    assert sol_db.status == "ABERTA"
    assert sol_db.id_epi == 1
