"""Testes da Frente 2B (`gerador_sintetico.py`, `conftest.py` e `cli_demo.py`).

Valida as Tasks 2B.1 e 2B.2 e o Checkpoint 2:
- Geração dos 12 pares de atestados sintéticos (8 legíveis + 4 modos de
  ilegibilidade, em `.png` + `.pdf`) e gabarito JSON, preservando intacto o
  SHA-256 canônico do caso `105`.
- Bateria completa `OCR (RapidOCR) -> Extrator -> ServicoA1` atingindo a meta
  >= 80% de extração sem correção manual.
- Rejeição automática nos 4 modos distintos de ilegibilidade (borrado,
  subexposto, cortado e manchado/rasurado), tanto em PNG quanto em PDF.
- Variedade de conselhos (`CRM`, `CRO`, `CRP`), declaração de comparecimento,
  sigilo de CID, afastamento longo (30 dias INSS/NR-7), divergências múltiplas
  e fluxo `origem = "MANUAL"`.
- Execução do simulador `cli_demo.py` em todos os modos (`--roteiro carla`,
  `--avaliar-metricas`, `--gerar-atestados`, `--tarefa` e menu interativo).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import jsonschema

from sinergia.agentes.a1_recepcao import (
    A1Config,
    GabaritoAtestadoSintetico,
    MockLLM,
    RepositorioA1Memoria,
    ResultadoA1,
    ServicoA1,
    calcular_marcos_legais,
    exportar_json_schemas_a1,
    garantir_atestados_sinteticos,
)
from sinergia.agentes.a1_recepcao.cli_demo import (
    avaliar_metricas_secao10,
    executar_menu_interativo,
)
from sinergia.agentes.a1_recepcao.cli_demo import (
    main as cli_main,
)
from sinergia.agentes.a1_recepcao.gerador_sintetico import (
    SHA256_CANONICO_CARLA_105_PNG,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _registro_inicial_padrao(
    *,
    id_caso: int,
    nome: str = "Carla S.",
    inicio_informado: str = "2026-11-09",
    dias_informados: int = 3,
    tipo_solicitacao: str = "atestado_medico",
) -> dict[str, Any]:
    return {
        "id_caso": id_caso,
        "thread_id": f"chat-42-{id_caso}",
        "aberto_em": "2026-11-09T09:05:00-03:00",
        "canal": "chat",
        "id_colaborador": 42,
        "nome": nome,
        "matricula": "00123",
        "id_area": 2,
        "id_setor": 7,
        "id_cargo": 15,
        "jornada_inicio": "07:00",
        "jornada_fim": "16:00",
        "municipio": "Campinas",
        "uf": "SP",
        "tipo_solicitacao": tipo_solicitacao,
        "inicio_informado": inicio_informado,
        "dias_informados": dias_informados,
        "tem_documento": True,
    }


def test_geracao_atestados_sinteticos_e_preservacao_sha256_canonico(
    catalogo_atestados_sinteticos: list[GabaritoAtestadoSintetico],
    tmp_path: Path,
) -> None:
    """Garante os 12 pares `.png`/`.pdf`, manifesto e SHA-256 intacto do caso 105."""
    assert len(catalogo_atestados_sinteticos) == 12

    amostra_105 = next(
        a for a in catalogo_atestados_sinteticos if a.id_amostra == "atestado_carla_105"
    )
    assert amostra_105.sha256_png == SHA256_CANONICO_CARLA_105_PNG

    manifesto_storage = Path(".storage/atestados/gabarito_atestados.json")
    assert manifesto_storage.is_file()
    manifesto_json = json.loads(manifesto_storage.read_text(encoding="utf-8"))
    assert manifesto_json["total_amostras"] == 12

    for amostra in catalogo_atestados_sinteticos:
        assert Path(amostra.arquivo_png).is_file()
        assert Path(amostra.arquivo_pdf).is_file()

    # Também valida geração completa do zero em diretório isolado (tmp_path)
    gerados_tmp = garantir_atestados_sinteticos(tmp_path)
    assert len(gerados_tmp) == 12
    assert (tmp_path / "gabarito_atestados.json").is_file()
    for amostra in gerados_tmp:
        assert Path(amostra.arquivo_png).is_file()
        assert Path(amostra.arquivo_pdf).is_file()


def test_bateria_ocr_e_servico_a1_atinge_meta_80_porcento_e_rejeita_ilegiveis(
    catalogo_atestados_sinteticos: list[GabaritoAtestadoSintetico],
    repositorio_memoria_a1: RepositorioA1Memoria,
    mock_llm_a1: MockLLM,
) -> None:
    """Testa a bateria sintética via `executar(a1.processar_documento)` + `confirmar`."""
    cfg = A1Config(groq_api_key="", politica_extrair_cid=True)
    relatorio = avaliar_metricas_secao10(cfg=cfg, modo_llm="mock")

    assert relatorio["legiveis_total"] == 8
    assert relatorio["legiveis_exatos"] == 8
    assert relatorio["taxa_extracao_sem_correcao_pct"] == 100.0
    assert relatorio["meta_extracao_atingida"] is True
    assert relatorio["ilegiveis_total"] == 4
    assert relatorio["ilegiveis_rejeitados"] == 4
    assert relatorio["taxa_baixa_no_prazo_pct"] >= 90.0
    assert relatorio["meta_baixa_atingida"] is True

    # Exercita também PDF sintético + amostra com ruído leve e correção MANUAL
    amostra_ruido = next(
        a
        for a in catalogo_atestados_sinteticos
        if a.id_amostra == "atestado_ruido_leve_manual"
    )
    servico = ServicoA1(
        repositorio=repositorio_memoria_a1, llm=mock_llm_a1, config=cfg
    )
    schemas_json = exportar_json_schemas_a1()

    tarefa_pdf = {
        "protocolo": "sinergia/1",
        "id_mensagem": "f2b-ruido-pdf-01",
        "tipo": "TAREFA",
        "acao": "a1.processar_documento",
        "origem": "A0",
        "destino": "A1",
        "correlacao": {
            "id_caso": 209,
            "thread_id": "chat-42-209",
            "id_tarefa": "t-209-01",
        },
        "contexto": {
            "id_colaborador": 42,
            "perfil": "colaborador",
            "canal": "chat",
            "data_hora": "2026-12-14T09:10:00-03:00",
        },
        "registro_inicial": _registro_inicial_padrao(
            id_caso=209,
            inicio_informado="2026-12-14",
            dias_informados=4,
        ),
        "payload": {
            "arquivo_uri": amostra_ruido.arquivo_pdf,
            "mime": "application/pdf",
            "sha256": amostra_ruido.sha256_pdf,
        },
    }
    res_pdf = servico.executar(tarefa_pdf)
    jsonschema.validate(instance=res_pdf, schema=schemas_json["ResultadoA1"])
    ResultadoA1.model_validate(res_pdf).validar_payload_acao()
    assert res_pdf["status"] == "PRECISA_HUMANO"
    assert res_pdf["estado_proposto"] == "RECEBIDO"
    assert res_pdf["payload"]["divergencias"] == []

    tarefa_conf_manual = {
        **tarefa_pdf,
        "id_mensagem": "f2b-ruido-conf-02",
        "acao": "a1.confirmar",
        "payload": {
            "atestado_confirmado": {
                "periodo": {"inicio": "2026-12-14", "fim": "2026-12-18"},
                "data_emissao": "2026-12-14",
            },
            "campos_corrigidos": amostra_ruido.campos_corrigidos_sugeridos,
        },
    }
    res_conf = servico.executar(tarefa_conf_manual)
    jsonschema.validate(instance=res_conf, schema=schemas_json["ResultadoA1"])
    assert res_conf["status"] == "OK"
    assert res_conf["estado_proposto"] == "VALIDADO"
    assert res_conf["payload"]["origem"] == "MANUAL"


def test_quatro_modos_distintos_de_ilegibilidade_png_e_pdf(
    catalogo_atestados_sinteticos: list[GabaritoAtestadoSintetico],
    repositorio_memoria_a1: RepositorioA1Memoria,
    mock_llm_a1: MockLLM,
) -> None:
    """Valida rejeição segura em borrado, subexposto, cortado e manchado/rasurado."""
    cfg = A1Config(groq_api_key="", politica_extrair_cid=True)
    servico = ServicoA1(
        repositorio=repositorio_memoria_a1, llm=mock_llm_a1, config=cfg
    )
    schemas_json = exportar_json_schemas_a1()

    esperado_usou_llm = {
        "atestado_borrado_ilegivel": False,
        "atestado_escuro_subexposto": False,
        "atestado_cortado_incompleto": True,
        "atestado_manchado_rasurado": True,
    }
    ilegiveis = [
        a
        for a in catalogo_atestados_sinteticos
        if a.esperado.legibilidade == "ilegivel"
    ]
    assert {a.id_amostra for a in ilegiveis} == set(esperado_usou_llm)

    for idx, amostra in enumerate(ilegiveis, start=1):
        for fmt, uri, mime, sha in (
            ("png", amostra.arquivo_png, "image/png", amostra.sha256_png),
            ("pdf", amostra.arquivo_pdf, "application/pdf", amostra.sha256_pdf),
        ):
            tarefa = {
                "protocolo": "sinergia/1",
                "id_mensagem": f"ileg-{amostra.id_amostra}-{fmt}-{idx}",
                "tipo": "TAREFA",
                "acao": "a1.processar_documento",
                "origem": "A0",
                "destino": "A1",
                "correlacao": {
                    "id_caso": 400 + idx,
                    "thread_id": f"chat-42-{400 + idx}",
                    "id_tarefa": f"t-ileg-{idx}-{fmt}",
                },
                "contexto": {
                    "id_colaborador": 42,
                    "perfil": "colaborador",
                    "canal": "chat",
                    "data_hora": "2026-11-09T09:10:00-03:00",
                },
                "registro_inicial": _registro_inicial_padrao(id_caso=400 + idx),
                "payload": {"arquivo_uri": uri, "mime": mime, "sha256": sha},
            }
            res = servico.executar(tarefa)
            jsonschema.validate(instance=res, schema=schemas_json["ResultadoA1"])
            ResultadoA1.model_validate(res).validar_payload_acao()

            assert res["status"] == "PRECISA_HUMANO"
            assert res["estado_proposto"] == "NOVO"
            assert res["payload"]["atestado_extraido"]["legibilidade"] == "ilegivel"
            assert res["payload"]["divergencias"] == []
            assert res["eventos"] == []
            assert "mais nítido" in res["saidas"][0]["texto"]
            assert res["decisao"]["usou_llm"] is esperado_usou_llm[amostra.id_amostra]

    assert len(repositorio_memoria_a1.atestados) == 0


def test_variedade_atestados_legiveis_conselhos_divergencias_e_marcos_30_dias(
    catalogo_atestados_sinteticos: list[GabaritoAtestadoSintetico],
    mock_llm_a1: MockLLM,
) -> None:
    """Testa CRO, CRP, sem CID, declaração de comparecimento, 30 dias e divergências."""
    cfg = A1Config(groq_api_key="", politica_extrair_cid=True)
    schemas_json = exportar_json_schemas_a1()

    # 1. Processa e confirma amostras variadas (CRO, CRP, sem CID, declaração, 30 dias)
    ids_variados = (
        "atestado_odontologico_cro",
        "atestado_psicologico_crp",
        "atestado_sem_cid_crm",
        "declaracao_comparecimento_1dia",
        "atestado_inss_nr7_30dias",
    )
    por_id = {a.id_amostra: a for a in catalogo_atestados_sinteticos}

    for idx, id_amostra in enumerate(ids_variados, start=1):
        amostra = por_id[id_amostra]
        repo = RepositorioA1Memoria()
        servico = ServicoA1(repositorio=repo, llm=mock_llm_a1, config=cfg)

        inicio_esp = amostra.esperado.periodo.inicio
        fim_esp = amostra.esperado.periodo.fim
        assert inicio_esp is not None and fim_esp is not None
        dias_esp = (fim_esp - inicio_esp).days + 1

        tarefa_doc = {
            "protocolo": "sinergia/1",
            "id_mensagem": f"var-doc-{id_amostra}",
            "tipo": "TAREFA",
            "acao": "a1.processar_documento",
            "origem": "A0",
            "destino": "A1",
            "correlacao": {
                "id_caso": 600 + idx,
                "thread_id": f"chat-42-{600 + idx}",
                "id_tarefa": f"t-var-{idx}",
            },
            "contexto": {
                "id_colaborador": 42,
                "perfil": "colaborador",
                "canal": "chat",
                "data_hora": "2026-11-09T09:10:00-03:00",
            },
            "registro_inicial": _registro_inicial_padrao(
                id_caso=600 + idx,
                inicio_informado=inicio_esp.isoformat(),
                dias_informados=dias_esp,
                tipo_solicitacao=(
                    "declaracao_comparecimento"
                    if id_amostra == "declaracao_comparecimento_1dia"
                    else "atestado_medico"
                ),
            ),
            "payload": {
                "arquivo_uri": amostra.arquivo_png,
                "mime": "image/png",
                "sha256": amostra.sha256_png,
            },
        }
        res_doc = servico.executar(tarefa_doc)
        jsonschema.validate(instance=res_doc, schema=schemas_json["ResultadoA1"])
        assert res_doc["status"] == "PRECISA_HUMANO"
        assert res_doc["estado_proposto"] == "RECEBIDO"
        assert res_doc["payload"]["divergencias"] == []
        assert (
            res_doc["payload"]["atestado_extraido"]["profissional"]["conselho"]
            == amostra.esperado.profissional.conselho
        )

        tarefa_conf = {
            **tarefa_doc,
            "id_mensagem": f"var-conf-{id_amostra}",
            "acao": "a1.confirmar",
            "payload": {
                "atestado_confirmado": {
                    "periodo": {
                        "inicio": inicio_esp.isoformat(),
                        "fim": fim_esp.isoformat(),
                    },
                },
                "campos_corrigidos": [],
            },
        }
        res_conf = servico.executar(tarefa_conf)
        jsonschema.validate(instance=res_conf, schema=schemas_json["ResultadoA1"])
        assert res_conf["status"] == "OK"
        assert res_conf["estado_proposto"] == "VALIDADO"
        assert res_conf["eventos"][0]["dados"]["dias"] == dias_esp

    # Confirma marcos legais do afastamento de 30 dias (INSS + NR-07)
    marcos_30 = calcular_marcos_legais(date(2026, 11, 1), date(2026, 11, 30))
    assert marcos_30.exige_encaminhamento_inss is True
    assert marcos_30.data_marco_16_dia == date(2026, 11, 16)
    assert marcos_30.exige_exame_retorno_nr7 is True

    # 2. Testa divergência múltipla simultânea (dias + inicio + nome de terceiro)
    amostra_105 = por_id["atestado_carla_105"]
    repo_div = RepositorioA1Memoria()
    servico_div = ServicoA1(repositorio=repo_div, llm=mock_llm_a1, config=cfg)
    tarefa_div_multipla = {
        "protocolo": "sinergia/1",
        "id_mensagem": "var-div-multipla-01",
        "tipo": "TAREFA",
        "acao": "a1.processar_documento",
        "origem": "A0",
        "destino": "A1",
        "correlacao": {
            "id_caso": 700,
            "thread_id": "chat-42-700",
            "id_tarefa": "t-div-01",
        },
        "contexto": {
            "id_colaborador": 42,
            "perfil": "colaborador",
            "canal": "chat",
            "data_hora": "2026-11-09T09:10:00-03:00",
        },
        "registro_inicial": _registro_inicial_padrao(
            id_caso=700,
            nome="Mariana Oliveira",
            inicio_informado="2026-11-10",
            dias_informados=5,
        ),
        "payload": {
            "arquivo_uri": amostra_105.arquivo_png,
            "mime": "image/png",
            "sha256": amostra_105.sha256_png,
        },
    }
    res_div = servico_div.executar(tarefa_div_multipla)
    jsonschema.validate(instance=res_div, schema=schemas_json["ResultadoA1"])
    campos_div = [d["campo"] for d in res_div["payload"]["divergencias"]]
    assert campos_div == ["dias", "inicio", "nome"]
    assert "diferenças" in res_div["saidas"][0]["texto"]


def test_cli_demo_todos_os_modos_e_menu_interativo(capsys: Any) -> None:
    """Valida `--roteiro carla`, `--avaliar-metricas`, `--gerar-atestados`, `--tarefa` e menu."""
    assert (
        cli_main(["--roteiro", "carla", "--modo-repo", "memoria", "--modo-llm", "mock"])
        == 0
    )
    saida_carla = capsys.readouterr().out
    assert "Roteiro completo da Carla S. validado 100%" in saida_carla

    assert cli_main(["--avaliar-metricas", "--modo-llm", "mock"]) == 0
    saida_metricas = capsys.readouterr().out
    assert "Extraídos sem correção manual" in saida_metricas

    assert cli_main(["--gerar-atestados"]) == 0
    saida_gerar = json.loads(capsys.readouterr().out)
    assert saida_gerar["total_amostras"] == 12

    fixture_conv = FIXTURES_DIR / "a1.conversar" / "falta_arquivo.tarefa.json"
    assert (
        cli_main(
            [
                "--tarefa",
                str(fixture_conv),
                "--modo-repo",
                "memoria",
                "--modo-llm",
                "mock",
            ]
        )
        == 0
    )
    saida_tarefa = capsys.readouterr().out
    assert '"faltando"' in saida_tarefa

    # Menu interativo simulando usuário percorrendo 1 -> 2 -> 3 -> 4 -> 0
    entradas = iter(
        [
            "1",
            "",  # texto padrão
            "2",
            "",  # arquivo padrão (caso 105)
            "3",
            "",  # início padrão
            "",  # fim padrão
            "",  # sem campos corrigidos -> IA
            "4",
            "",  # último id_atestado
            "",  # rh.ana
            "0",
        ]
    )
    cfg_menu = A1Config(groq_api_key="", politica_extrair_cid=False)
    codigo_menu = executar_menu_interativo(
        cfg=cfg_menu,
        modo_repo="memoria",
        modo_llm="mock",
        input_fn=lambda _prompt: next(entradas),
    )
    assert codigo_menu == 0
    saida_menu = capsys.readouterr().out
    assert "atestado.extraido" in saida_menu
    assert "atestado.registrado" in saida_menu
    assert "atestado.baixa_registrada" in saida_menu
