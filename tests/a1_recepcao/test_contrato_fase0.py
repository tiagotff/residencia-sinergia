"""Testes rigorosos da Fase 0 do Agente 1 (Anexos A, B, C.2, D.1, D.2, E.2, F e G da v2.1).

Pode ser executado tanto via `pytest` quanto diretamente via:
`python tests/a1_recepcao/test_contrato_fase0.py`
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import jsonschema

# Permite rodar o arquivo diretamente com `python` mesmo sem `pip install -e .`
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sinergia.agentes.a1_recepcao import (  # noqa: E402
    AtestadoExtraido,
    PayloadConfirmarInvalidoResultado,
    PayloadConfirmarTarefa,
    PayloadConfirmarValidoResultado,
    PayloadConversarResultado,
    PayloadProcessarDocumentoResultado,
    ResultadoA1,
    TarefaA1,
    exportar_json_schemas_a1,
    obter_config_a1,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_fixtures_anexo_f_pydantic_e_jsonschema() -> None:
    """Valida todos os pares `.tarefa.json` e `.esperado.json` via Pydantic e JSON Schema."""
    schemas_json = exportar_json_schemas_a1()
    arquivos_tarefa = sorted(FIXTURES_DIR.rglob("*.tarefa.json"))
    assert len(arquivos_tarefa) == 5, f"Esperados 5 cenários de fixture, encontrados {len(arquivos_tarefa)}"

    for caminho_tarefa in arquivos_tarefa:
        caminho_esperado = caminho_tarefa.with_name(
            caminho_tarefa.name.replace(".tarefa.json", ".esperado.json")
        )
        assert caminho_esperado.exists(), f"Falta arquivo esperado para {caminho_tarefa}"

        dados_tarefa = json.loads(caminho_tarefa.read_text(encoding="utf-8"))
        dados_esperado = json.loads(caminho_esperado.read_text(encoding="utf-8"))

        # 1. Validação contra o JSON Schema exportado (Anexo A e Anexo F)
        jsonschema.validate(instance=dados_tarefa, schema=schemas_json["TarefaA1"])
        jsonschema.validate(instance=dados_esperado, schema=schemas_json["ResultadoA1"])

        # 2. Validação via Pydantic + payload específico da ação
        tarefa = TarefaA1.model_validate(dados_tarefa)
        payload_in = tarefa.validar_payload_acao()
        assert payload_in is not None

        resultado = ResultadoA1.model_validate(dados_esperado)
        payload_out = resultado.validar_payload_acao()
        assert payload_out is not None
        assert resultado.em_resposta_a == tarefa.id_mensagem
        assert resultado.acao == tarefa.acao


def test_normalizacao_e_regras_anexo_e2_g() -> None:
    """Valida normalização (CRM/UF/CID), cálculo de dias, limiar 0.8 (Anexo E.2) e Política de CID (Anexo G)."""
    extraido = AtestadoExtraido.model_validate(
        {
            "legibilidade": "ok",
            "nome_no_documento": "Carla Souza",
            "periodo": {"inicio": "2026-11-09", "fim": "2026-11-11"},
            "data_emissao": "2026-11-09",
            "profissional": {"conselho": " crm ", "numero": " 123456 ", "uf": " sp "},
            "cid": " m54.5 ",
            "municipio": "Campinas",
            "confianca_campos": {
                "nome_no_documento": 0.95,
                "municipio": 0.65,  # Abaixo de 0.8 -> deve virar None!
                "cid": 0.90,
            },
        }
    )

    # Normalização automática de maiúsculas e espaços
    assert extraido.profissional.conselho == "CRM"
    assert extraido.profissional.numero == "123456"
    assert extraido.profissional.uf == "SP"
    assert extraido.cid == "M54.5"

    # Cálculo inclusivo de dias (09/11 a 11/11 = 3 dias, como no Anexo D.2)
    assert extraido.periodo.dias_calculados == 3

    # Aplicação do limiar de confiança 0.8 (Anexo E.2) com CID habilitado
    filtrado_com_cid = extraido.aplicar_politica_e_limiar(
        extrair_cid=True, limiar_confianca=0.8
    )
    assert filtrado_com_cid.municipio is None  # 0.65 < 0.8
    assert filtrado_com_cid.nome_no_documento == "Carla Souza"  # 0.95 >= 0.8
    assert filtrado_com_cid.cid == "M54.5"

    # Aplicação da Política de CID desabilitada no MVP (Anexo G, opção b)
    filtrado_sem_cid = extraido.aplicar_politica_e_limiar(
        extrair_cid=False, limiar_confianca=0.8
    )
    assert filtrado_sem_cid.cid is None


def test_construtor_resultado_e_validacao_erro() -> None:
    """Valida o construtor `ResultadoA1.a_partir_da_tarefa` e a obrigatoriedade do bloco `erro`."""
    caminho_tarefa = FIXTURES_DIR / "a1.confirmar" / "carla_confirmado_ia.tarefa.json"
    tarefa = TarefaA1.model_validate(json.loads(caminho_tarefa.read_text(encoding="utf-8")))

    payload_in = tarefa.validar_payload_acao()
    assert isinstance(payload_in, PayloadConfirmarTarefa)
    assert payload_in.atestado_confirmado.periodo.inicio == date(2026, 11, 9)
    assert payload_in.atestado_confirmado.periodo.dias_calculados == 3

    res_ok = ResultadoA1.a_partir_da_tarefa(
        tarefa,
        status="OK",
        estado_proposto="VALIDADO",
        payload=PayloadConfirmarValidoResultado(
            id_atestado=311,
            origem="IA",
            prazo_entrega_fisica=date(2026, 11, 16),
        ),
        saidas=[{"texto": "Pronto, atestado registrado."}],
        eventos=[{"tipo": "atestado.registrado", "dados": {"id_atestado": 311}}],
        decisao={"motivo": "regras ok", "usou_llm": False},
    )
    assert res_ok.em_resposta_a == tarefa.id_mensagem
    assert res_ok.payload["prazo_entrega_fisica"] == "2026-11-16"

    # Deve falhar se status == "ERRO" sem bloco `erro`
    falhou_como_esperado = False
    try:
        ResultadoA1.a_partir_da_tarefa(tarefa, status="ERRO")
    except ValueError:
        falhou_como_esperado = True
    assert falhou_como_esperado, "Deveria exigir o bloco 'erro' quando status == 'ERRO'"

    # Configuração centralizada deve carregar padrões válidos
    cfg = obter_config_a1()
    assert cfg.limiar_confianca_campo == 0.8
    assert cfg.prazo_entrega_fisica_dias_uteis == 2


if __name__ == "__main__":
    test_fixtures_anexo_f_pydantic_e_jsonschema()
    test_normalizacao_e_regras_anexo_e2_g()
    test_construtor_resultado_e_validacao_erro()
    print(
        "SUCESSO TOTAL! Todos os 5 pares de fixtures (.tarefa.json e .esperado.json), "
        "JSON Schemas, normalizações e regras dos Anexos A-G da v2.1 passaram nos testes!"
    )
