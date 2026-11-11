"""Testes de contrato da Fase 0 do Agente 1 (Anexos A, B, C.2, D.1 e D.2 da v2.1).

Pode ser executado tanto via `pytest` quanto diretamente via:
`python tests/a1_recepcao/test_contrato_fase0.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite rodar o arquivo diretamente com `python` mesmo sem `pip install -e .`
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sinergia.agentes.a1_recepcao import (  # noqa: E402
    PayloadBaixaFisicaResultado,
    PayloadConfirmarInvalidoResultado,
    PayloadConfirmarTarefa,
    PayloadConfirmarValidoResultado,
    PayloadConversarResultado,
    PayloadProcessarDocumentoResultado,
    ResultadoA1,
    TarefaA1,
)


def test_anexo_d1_tarefa_a1_confirmar() -> None:
    """Valida o JSON oficial de TAREFA `a1.confirmar` do Anexo D.1."""
    exemplo_d1 = {
        "protocolo": "sinergia/1",
        "id_mensagem": "0b9e7d1a-5c2f-4e8b-a1d3-6f7e8c9d0a12",
        "tipo": "TAREFA",
        "acao": "a1.confirmar",
        "origem": "A0",
        "destino": "A1",
        "correlacao": {"id_caso": 105, "thread_id": "chat-42-105", "id_tarefa": "t-105-03"},
        "contexto": {
            "id_colaborador": 42,
            "perfil": "colaborador",
            "canal": "chat",
            "data_hora": "2026-11-09T09:15:00-03:00",
        },
        "registro_inicial": {
            "id_caso": 105,
            "thread_id": "chat-42-105",
            "aberto_em": "2026-11-09T09:05:00-03:00",
            "canal": "chat",
            "id_colaborador": 42,
            "nome": "Carla S.",
            "matricula": "00123",
            "id_area": 2,
            "id_setor": 7,
            "id_cargo": 15,
            "jornada_inicio": "07:00",
            "jornada_fim": "16:00",
            "municipio": "Campinas",
            "uf": "SP",
            "tipo_solicitacao": "atestado_medico",
            "inicio_informado": "2026-11-09",
            "dias_informados": 5,
            "tem_documento": True,
        },
        "payload": {
            "atestado_confirmado": {
                "periodo": {"inicio": "2026-11-09", "fim": "2026-11-11"},
                "data_emissao": "2026-11-09",
            },
            "campos_corrigidos": [],
        },
    }

    tarefa = TarefaA1.model_validate(exemplo_d1)
    assert tarefa.acao == "a1.confirmar"
    assert tarefa.registro_inicial is not None
    assert tarefa.registro_inicial.nome == "Carla S."
    payload_tipado = tarefa.validar_payload_acao()
    assert isinstance(payload_tipado, PayloadConfirmarTarefa)
    assert payload_tipado.campos_corrigidos == []


def test_anexo_d2_exemplos_resultado_a1() -> None:
    """Valida todos os exemplos oficiais de RESULTADO do Anexo D.2."""
    # 1. a1.conversar (envelope completo do Anexo D.2)
    res_conversar_json = {
        "protocolo": "sinergia/1",
        "id_mensagem": "r-105-01",
        "tipo": "RESULTADO",
        "acao": "a1.conversar",
        "origem": "A1",
        "destino": "A0",
        "em_resposta_a": "t-105-01",
        "correlacao": {"id_caso": 105, "thread_id": "chat-42-105", "id_tarefa": "t-105-01"},
        "status": "OK",
        "estado_proposto": "NOVO",
        "payload": {"faltando": ["arquivo"]},
        "saidas": [
            {
                "tipo": "mensagem",
                "para": {"perfil": "colaborador", "id_colaborador": 42},
                "nivel_detalhe": "individual",
                "texto": "Entendi, Carla. Pode me enviar uma foto do atestado?",
                "fontes": [],
                "agendar_para": None,
            }
        ],
        "eventos": [],
        "decisao": {
            "motivo": "falta o arquivo",
            "usou_llm": True,
            "confianca": 0.95,
            "modelo": "groq/llama",
        },
    }
    res_conversar = ResultadoA1.model_validate(res_conversar_json)
    payload_conv = PayloadConversarResultado.model_validate(res_conversar.payload)
    assert payload_conv.faltando == ["arquivo"]

    # 2. a1.processar_documento (Anexo D.2)
    payload_proc_json = {
        "atestado_extraido": {
            "legibilidade": "ok",
            "nome_no_documento": "Carla Souza",
            "periodo": {"inicio": "2026-11-09", "fim": "2026-11-11"},
            "data_emissao": "2026-11-09",
            "profissional": {"conselho": "CRM", "numero": "123456", "uf": "SP"},
            "cid": None,
            "municipio": "Campinas",
        },
        "divergencias": [{"campo": "dias", "no_documento": 3, "informado": 5}],
    }
    payload_proc = PayloadProcessarDocumentoResultado.model_validate(payload_proc_json)
    assert payload_proc.atestado_extraido.profissional.conselho == "CRM"
    assert payload_proc.divergencias[0].campo == "dias"

    # 3. a1.confirmar válido e inválido (Anexo D.2)
    payload_conf_ok = PayloadConfirmarValidoResultado.model_validate(
        {
            "id_atestado": 311,
            "status_atestado": "CONFIRMADO",
            "origem": "IA",
            "prazo_entrega_fisica": "2026-11-16",
        }
    )
    assert payload_conf_ok.status_atestado == "CONFIRMADO"

    payload_conf_inv = PayloadConfirmarInvalidoResultado.model_validate(
        {"motivo": "data de fim anterior à de início"}
    )
    assert "anterior" in payload_conf_inv.motivo

    # 4. a1.registrar_baixa_fisica (Anexo D.2)
    payload_baixa = PayloadBaixaFisicaResultado.model_validate(
        {
            "id_atestado": 311,
            "recebido_fisico_por": "rh.ana",
            "recebido_fisico_em": "2026-11-13T10:20:00-03:00",
        }
    )
    assert payload_baixa.id_atestado == 311


if __name__ == "__main__":
    test_anexo_d1_tarefa_a1_confirmar()
    test_anexo_d2_exemplos_resultado_a1()
    print("OK! Todos os contratos da Fase 0 (Anexos D.1 e D.2 da v2.1) foram validados com sucesso!")
