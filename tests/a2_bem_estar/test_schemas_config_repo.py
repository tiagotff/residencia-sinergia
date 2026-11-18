"""Testes unitários de configuração, schemas `sinergia/1` e repositório do Agente 2."""

from __future__ import annotations

import jsonschema
import pytest
from pydantic import ValidationError

from sinergia.agentes.a2_bem_estar.config import A2Config, obter_config_a2
from sinergia.agentes.a2_bem_estar.repositorio import (
    RepositorioA2Memoria,
    RepositorioA2Protocol,
)
from sinergia.agentes.a2_bem_estar.schemas import (
    PayloadGerarLembretesResultado,
    PayloadGerarLembretesTarefa,
    PayloadOrientarEpiTarefa,
    PayloadRecomendarBeneficiosResultado,
    PayloadRecomendarBeneficiosTarefa,
    PayloadResponderResultado,
    PayloadResponderTarefa,
    ResultadoA2,
    TarefaA2,
    exportar_json_schemas_a2,
)


def test_config_a2_padrao_e_variaveis_ambiente(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valida instanciação padrão e leitura de variáveis de ambiente no A2Config."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("SINERGIA_A2_LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SINERGIA_A2_TIMEOUT_S", raising=False)

    cfg_padrao = obter_config_a2()
    assert cfg_padrao.groq_api_key == ""
    assert cfg_padrao.llm_model == "openai/gpt-oss-20b"
    assert "postgresql://" in cfg_padrao.database_url
    assert cfg_padrao.timeout_s == 45.0

    monkeypatch.setenv("GROQ_API_KEY", "gsk_env_123")
    monkeypatch.setenv("SINERGIA_A2_LLM_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@db:5432/sinergia")
    monkeypatch.setenv("SINERGIA_A2_TIMEOUT_S", "30.5")

    cfg_env = A2Config.from_env()
    assert cfg_env.groq_api_key == "gsk_env_123"
    assert cfg_env.llm_model == "qwen/qwen3.8-27b"
    assert cfg_env.database_url == "postgresql://user:pass@db:5432/sinergia"
    assert cfg_env.timeout_s == 30.5


def test_schemas_envelopes_e_payloads_anexo_c3_e_d3() -> None:
    """Valida os contratos Pydantic e JSON Schemas de todas as 4 ações do A2."""
    schemas = exportar_json_schemas_a2()
    assert "TarefaA2" in schemas
    assert "ResultadoA2" in schemas

    # 1. a2.responder
    tarefa_resp = TarefaA2.model_validate(
        {
            "id_mensagem": "t-207-01",
            "acao": "a2.responder",
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
                "historico_resumo": "Colaborador tirou dúvidas anteriores.",
            },
        }
    )
    payload_resp = tarefa_resp.validar_payload_acao()
    assert isinstance(payload_resp, PayloadResponderTarefa)
    assert payload_resp.perfil is not None
    assert payload_resp.perfil.cargo == "Analista de Operações"

    resultado_d3 = ResultadoA2.a_partir_da_tarefa(
        tarefa_resp,
        status="OK",
        payload=PayloadResponderResultado(base="direitos_beneficios", trechos_usados=2),
        saidas=[
            {
                "tipo": "mensagem",
                "para": {"perfil": "colaborador", "id_colaborador": 42},
                "nivel_detalhe": "individual",
                "texto": "Você pode dividir as férias em até 3 períodos, se a empresa concordar.",
                "fontes": [
                    {
                        "base": "direitos_beneficios",
                        "titulo": "Política de férias",
                        "versao": "2026.1",
                    }
                ],
                "agendar_para": None,
            }
        ],
        decisao={
            "motivo": "resposta RAG com 2 trechos vigentes",
            "usou_llm": True,
            "confianca": 0.88,
            "modelo": "openai/gpt-oss-20b",
        },
    )
    assert isinstance(resultado_d3.validar_payload_acao(), PayloadResponderResultado)
    jsonschema.validate(instance=tarefa_resp.model_dump(mode="json"), schema=schemas["TarefaA2"])
    jsonschema.validate(instance=resultado_d3.model_dump(mode="json"), schema=schemas["ResultadoA2"])

    # 2. a2.gerar_lembretes
    t_lemb = TarefaA2.model_validate(
        {
            "id_mensagem": "t-lemb-01",
            "acao": "a2.gerar_lembretes",
            "correlacao": {"id_caso": None, "thread_id": None, "id_tarefa": "t-lemb-01"},
            "contexto": {"perfil": "sistema", "canal": "chat", "data_hora": "2026-11-10T09:00:00-03:00"},
            "payload": {"janela": {"de": "2026-11-10T09:00:00-03:00", "ate": "2026-11-10T09:15:00-03:00"}},
        }
    )
    assert isinstance(t_lemb.validar_payload_acao(), PayloadGerarLembretesTarefa)
    r_lemb = ResultadoA2.a_partir_da_tarefa(
        t_lemb,
        status="OK",
        payload={"janela": {"de": "2026-11-10T09:00:00-03:00", "ate": "2026-11-10T09:15:00-03:00"}, "gerados": 2, "reagendados": 1},
    )
    assert isinstance(r_lemb.validar_payload_acao(), PayloadGerarLembretesResultado)

    # 3. a2.recomendar_beneficios
    t_rec = TarefaA2.model_validate(
        {
            "id_mensagem": "t-rec-01",
            "acao": "a2.recomendar_beneficios",
            "correlacao": {"id_caso": None, "thread_id": None, "id_tarefa": "t-rec-01"},
            "contexto": {"perfil": "sistema", "canal": "chat", "data_hora": "2026-11-01T08:00:00-03:00"},
            "payload": {"competencia": "2026-11"},
        }
    )
    assert isinstance(t_rec.validar_payload_acao(), PayloadRecomendarBeneficiosTarefa)
    r_rec = ResultadoA2.a_partir_da_tarefa(
        t_rec,
        status="OK",
        payload={
            "competencia": "2026-11",
            "modelo": "beneficios-v1",
            "total": 312,
            "exemplo": {
                "id_colaborador": 42,
                "recomendacoes": [{"id_beneficio": 7, "probabilidade": 0.71}],
            },
        },
    )
    assert isinstance(r_rec.validar_payload_acao(), PayloadRecomendarBeneficiosResultado)

    # 4. a2.orientar_epi
    t_epi = TarefaA2.model_validate(
        {
            "id_mensagem": "t-epi-01",
            "acao": "a2.orientar_epi",
            "correlacao": {"id_caso": None, "thread_id": None, "id_tarefa": "t-epi-01"},
            "contexto": {"id_colaborador": 42, "perfil": "colaborador", "canal": "chat", "data_hora": "2026-11-10T09:00:00-03:00"},
            "payload": {
                "id_alerta": 10,
                "id_colaborador": 42,
                "id_epi": 2,
                "situacao": "O protetor auricular chega ao fim da vida útil em 15 dias.",
            },
        }
    )
    assert isinstance(t_epi.validar_payload_acao(), PayloadOrientarEpiTarefa)


def test_schemas_rejeitam_payload_invalido_e_erro_sem_bloco() -> None:
    """Valida que payloads vazios e ResultadoA2 com ERRO sem `erro` falham na validação."""
    t_invalida = TarefaA2.model_validate(
        {
            "id_mensagem": "t-inv-01",
            "acao": "a2.responder",
            "correlacao": {"id_tarefa": "t-inv-01"},
            "contexto": {"id_colaborador": 42, "data_hora": "2026-11-10T09:00:00-03:00"},
            "payload": {"pergunta": "   "},
        }
    )
    with pytest.raises(ValidationError):
        t_invalida.validar_payload_acao()

    with pytest.raises(ValueError):
        ResultadoA2.a_partir_da_tarefa(t_invalida, status="ERRO")


def test_repositorio_a2_memoria_perfil_epis_e_solicitacao_aberta() -> None:
    """Valida leitura de perfil, resolução de EPI por texto e abertura em `apoio.solicitacao_epi`."""
    repo = RepositorioA2Memoria()
    assert isinstance(repo, RepositorioA2Protocol)

    perfil = repo.obter_perfil_colaborador(42)
    assert perfil is not None
    assert perfil.nome == "Carla Souza"
    assert perfil.cargo == "Analista de Operações"
    assert perfil.setor == "Logística"
    assert "Vale-Alimentação" in perfil.beneficios

    epis = repo.listar_epis_colaborador(42)
    assert len(epis) >= 3

    id_luva = repo.resolver_id_epi_por_texto("Minha luva de proteção rasgou", id_colaborador=42)
    assert id_luva == 1

    id_protetor = repo.resolver_id_epi_por_texto("Perdi meu protetor auricular", id_colaborador=42)
    assert id_protetor == 2

    sol = repo.abrir_solicitacao_epi(
        id_colaborador=42,
        id_epi=id_luva,
        motivo="AVARIA",
        descricao="Minha luva de proteção rasgou durante o turno",
    )
    assert sol.id_solicitacao == 58
    assert sol.status == "ABERTA"
    assert sol.id_entrega is None
    assert repo.obter_solicitacao_epi(58) == sol

    with pytest.raises(ValueError):
        repo.abrir_solicitacao_epi(id_colaborador=42, id_epi=999, motivo="AVARIA")

