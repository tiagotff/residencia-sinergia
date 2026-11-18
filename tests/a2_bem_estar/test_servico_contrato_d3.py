"""Testes de contrato do Anexo D.3 e fachada `executar(tarefa) -> resultado` do Agente 2."""

from __future__ import annotations

from unittest.mock import MagicMock

import jsonschema
import pytest
from crewai.llms.base_llm import BaseLLM

from sinergia.agentes.a2_bem_estar import (
    A2Config,
    RepositorioA2Memoria,
    ResultadoA2,
    ServicoA2,
    executar,
    exportar_json_schemas_a2,
    limpar_cache_idempotencia,
)
from sinergia.rag.interfaces import MetadadosTrecho, TrechoRecuperado
from sinergia.rag.retrievers import BuscadorHibrido


class _FakeContractLLM(BaseLLM):
    def __init__(self, resposta: str) -> None:
        super().__init__(model="openai/gpt-oss-20b")
        self.resposta = resposta
        self.chamadas = 0

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
        self.chamadas += 1
        return self.resposta


@pytest.fixture(autouse=True)
def _limpar_cache() -> None:
    limpar_cache_idempotencia()


def _criar_buscador_contrato() -> BuscadorHibrido:
    mock_emb = MagicMock()
    mock_emb.model_name = "tardellirs/colibri-embed-ptbr"
    mock_emb.gerar_embedding.return_value = [0.03 if i % 2 == 0 else -0.02 for i in range(768)]

    buscador = BuscadorHibrido(gerador_embeddings=mock_emb)
    buscador.adicionar_trechos_memoria(
        [
            TrechoRecuperado(
                id_trecho=201,
                conteudo="Você pode dividir as férias em até 3 períodos, desde que um deles tenha ao menos 14 dias corridos.",
                metadados=MetadadosTrecho(
                    id_documento=2,
                    titulo="Política de férias",
                    fonte="CLT Art. 134",
                    versao="2026.1",
                    secao="Art. 134 §1º",
                    pagina=3,
                    id_base=2,
                    vigente=True,
                ),
            ),
            TrechoRecuperado(
                id_trecho=202,
                conteudo="Para períodos menores de férias, nenhum período pode ser inferior a 5 dias corridos.",
                metadados=MetadadosTrecho(
                    id_documento=2,
                    titulo="Política de férias",
                    fonte="CLT Art. 134",
                    versao="2026.1",
                    secao="Art. 134 §1º",
                    pagina=3,
                    id_base=2,
                    vigente=True,
                ),
            ),
            TrechoRecuperado(
                id_trecho=501,
                conteudo="NR-6: Não execute atividade de risco sem EPI adequado nem utilize luva danificada.",
                metadados=MetadadosTrecho(
                    id_documento=5,
                    titulo="Procedimento de reposição de EPI",
                    fonte="NR-6",
                    versao="1.2",
                    secao="Item 3.2",
                    pagina=1,
                    id_base=5,
                    vigente=True,
                ),
            ),
        ]
    )
    return buscador


def test_contrato_d3_a2_responder_consulta_normativa() -> None:
    """Valida o envelope completo do Anexo D.3 para `a2.responder` em `direitos_beneficios`."""
    schemas = exportar_json_schemas_a2()
    repo = RepositorioA2Memoria()
    buscador = _criar_buscador_contrato()
    fake_llm = _FakeContractLLM(
        "Você pode dividir as férias em até 3 períodos, se a empresa concordar. Um deles precisa ter pelo menos 14 dias."
    )

    tarefa = {
        "protocolo": "sinergia/1",
        "id_mensagem": "t-207-01",
        "tipo": "TAREFA",
        "acao": "a2.responder",
        "origem": "A0",
        "destino": "A2",
        "correlacao": {"id_caso": None, "thread_id": "chat-42-207", "id_tarefa": "t-207-01"},
        "contexto": {
            "id_colaborador": 42,
            "perfil": "colaborador",
            "canal": "chat",
            "data_hora": "2026-11-10T09:00:00-03:00",
        },
        "payload": {
            "pergunta": "Posso dividir minhas férias em 3 períodos?",
            "intencao": "F2",
            "perfil": {
                "cargo": "Analista de Operações",
                "setor": "Logística",
                "beneficios": ["Vale-Alimentação", "Plano de Saúde"],
            },
            "historico_resumo": "Colaborador perguntou sobre férias.",
        },
    }

    resultado = executar(
        tarefa,
        repositorio=repo,
        llm=fake_llm,
        config=A2Config(groq_api_key=""),
        buscador=buscador,
    )

    jsonschema.validate(instance=resultado, schema=schemas["ResultadoA2"])
    ResultadoA2.model_validate(resultado).validar_payload_acao()

    assert resultado["protocolo"] == "sinergia/1"
    assert resultado["tipo"] == "RESULTADO"
    assert resultado["acao"] == "a2.responder"
    assert resultado["origem"] == "A2"
    assert resultado["destino"] == "A0"
    assert resultado["em_resposta_a"] == "t-207-01"
    assert resultado["status"] == "OK"
    assert resultado["estado_proposto"] is None
    assert resultado["payload"]["base"] == "direitos_beneficios"
    assert resultado["payload"]["trechos_usados"] == 2
    assert "id_solicitacao_epi" not in resultado["payload"]
    assert resultado["eventos"] == []
    assert len(resultado["saidas"]) == 1
    assert resultado["saidas"][0]["tipo"] == "mensagem"
    assert resultado["saidas"][0]["para"] == {"perfil": "colaborador", "id_colaborador": 42}
    assert resultado["saidas"][0]["fontes"] == [
        {
            "base": "direitos_beneficios",
            "titulo": "Política de férias",
            "versao": "2026.1",
        }
    ]
    assert resultado["decisao"]["usou_llm"] is True


def test_contrato_d3_a2_responder_epi_avariado_f5() -> None:
    """Valida o roteiro NR-6 do Anexo D.3 com abertura de `solicitacao_epi` e evento."""
    schemas = exportar_json_schemas_a2()
    repo = RepositorioA2Memoria()
    buscador = _criar_buscador_contrato()
    fake_llm = _FakeContractLLM(
        "Não faça a atividade de risco sem a luva adequada e não use a luva danificada. "
        "Abri o pedido de reposição nº 58 e o TST e seu gestor serão avisados."
    )

    tarefa = {
        "protocolo": "sinergia/1",
        "id_mensagem": "t-208-01",
        "tipo": "TAREFA",
        "acao": "a2.responder",
        "origem": "A0",
        "destino": "A2",
        "correlacao": {"id_caso": None, "thread_id": "chat-42-208", "id_tarefa": "t-208-01"},
        "contexto": {
            "id_colaborador": 42,
            "perfil": "colaborador",
            "canal": "chat",
            "data_hora": "2026-11-10T09:30:00-03:00",
        },
        "payload": {
            "pergunta": "Minha luva de proteção rasgou durante o turno",
            "intencao": "F5",
        },
    }

    resultado = executar(
        tarefa,
        repositorio=repo,
        llm=fake_llm,
        config=A2Config(groq_api_key=""),
        buscador=buscador,
    )

    jsonschema.validate(instance=resultado, schema=schemas["ResultadoA2"])
    ResultadoA2.model_validate(resultado).validar_payload_acao()

    assert resultado["status"] == "OK"
    assert resultado["payload"]["base"] == "epi"
    assert resultado["payload"]["id_solicitacao_epi"] == 58
    assert resultado["eventos"] == [
        {"tipo": "solicitacao_epi.aberta", "dados": {"id_solicitacao": 58}}
    ]
    assert resultado["saidas"][0]["fontes"][0]["base"] == "epi"
    assert "58" in resultado["saidas"][0]["texto"]
    assert repo.obter_solicitacao_epi(58) is not None
    assert repo.obter_solicitacao_epi(58).status == "ABERTA"


def test_contrato_idempotencia_guardrail_e_payload_invalido() -> None:
    """Valida idempotência por `id_mensagem`, guardrail de saudação e erro `PAYLOAD_INVALIDO`."""
    repo = RepositorioA2Memoria()
    buscador = _criar_buscador_contrato()
    fake_llm = _FakeContractLLM("Não execute a atividade de risco sem o EPI adequado. Pedido nº 58 aberto.")
    servico = ServicoA2(
        repositorio=repo,
        llm=fake_llm,
        config=A2Config(groq_api_key=""),
        buscador=buscador,
    )

    t_epi = {
        "id_mensagem": "t-idem-01",
        "acao": "a2.responder",
        "correlacao": {"id_tarefa": "t-idem-01"},
        "contexto": {"id_colaborador": 42, "data_hora": "2026-11-10T10:00:00-03:00"},
        "payload": {"pergunta": "Minha luva de proteção rasgou"},
    }
    res_1 = servico.executar(t_epi)
    res_2 = servico.executar(t_epi)
    assert res_1 == res_2
    assert fake_llm.chamadas == 1
    assert len(repo.solicitacoes_epi) == 1

    # Guardrail conversacional
    t_chat = {
        "id_mensagem": "t-chat-01",
        "acao": "a2.responder",
        "correlacao": {"id_tarefa": "t-chat-01"},
        "contexto": {"id_colaborador": 42, "data_hora": "2026-11-10T10:05:00-03:00"},
        "payload": {"pergunta": "Bom dia, tudo bem?"},
    }
    res_chat = servico.executar(t_chat)
    assert res_chat["status"] == "OK"
    assert res_chat["payload"]["trechos_usados"] == 0
    assert res_chat["saidas"][0]["fontes"] == []

    # Payload inválido
    t_inv = {
        "id_mensagem": "t-inv-01",
        "acao": "a2.responder",
        "correlacao": {"id_tarefa": "t-inv-01"},
        "contexto": {"id_colaborador": 42, "data_hora": "2026-11-10T10:10:00-03:00"},
        "payload": {"pergunta": ""},
    }
    res_inv = servico.executar(t_inv)
    assert res_inv["status"] == "ERRO"
    assert res_inv["erro"]["codigo"] == "PAYLOAD_INVALIDO"
    assert res_inv["erro"]["reprocessavel"] is False
