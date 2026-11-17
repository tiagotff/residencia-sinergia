"""Testes da Task 2A.1 — Fachada oficial `executar(tarefa) -> resultado` (`servico.py`).

Valida o roteamento das 4 ações do catálogo do A1, conformidade integral
com as 5 fixtures oficiais do Anexo D.2/F (via JSON Schema e Pydantic),
idempotência por `id_mensagem`, casos de borda e fluxo completo real com
PostgreSQL 16 + PyMuPDF/RapidOCR + ChatGroq.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import psycopg
import pytest
from PIL import Image

from sinergia.agentes.a1_recepcao import (
    A1Config,
    MockLLM,
    RepositorioA1Memoria,
    RepositorioA1Postgres,
    ResultadoA1,
    ServicoA1,
    executar,
    exportar_json_schemas_a1,
    limpar_cache_idempotencia,
    obter_config_a1,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_CFG = obter_config_a1()


def _carregar_fixture(relpath: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / relpath).read_text(encoding="utf-8"))


def _postgres_disponivel(database_url: str) -> bool:
    try:
        with psycopg.connect(database_url, connect_timeout=2) as conn:
            row = conn.execute(
                "SELECT count(*) FROM cadastro.colaborador WHERE id_colaborador = 42"
            ).fetchone()
            return bool(row and row[0] >= 1)
    except psycopg.Error:
        return False


@pytest.fixture(autouse=True)
def _limpar_idempotencia() -> None:
    limpar_cache_idempotencia()


def test_executar_todas_as_5_fixtures_oficiais_anexo_d2_e_f() -> None:
    """Executa os 5 cenários oficiais via `executar()` validando schema e conteúdo."""
    schemas_json = exportar_json_schemas_a1()
    repo = RepositorioA1Memoria()
    mock_llm = MockLLM()
    cfg = A1Config(groq_api_key="", politica_extrair_cid=False)

    # 1. a1.conversar -> falta_arquivo
    t1 = _carregar_fixture("a1.conversar/falta_arquivo.tarefa.json")
    e1 = _carregar_fixture("a1.conversar/falta_arquivo.esperado.json")
    r1 = executar(t1, repositorio=repo, llm=mock_llm, config=cfg)
    jsonschema.validate(instance=r1, schema=schemas_json["ResultadoA1"])
    ResultadoA1.model_validate(r1).validar_payload_acao()
    assert r1["em_resposta_a"] == t1["id_mensagem"]
    assert r1["status"] == e1["status"]
    assert r1["estado_proposto"] == e1["estado_proposto"]
    assert r1["payload"] == e1["payload"]
    assert r1["saidas"] == e1["saidas"]
    assert r1["eventos"] == e1["eventos"]
    assert r1["decisao"]["motivo"] == e1["decisao"]["motivo"]
    assert r1["decisao"]["usou_llm"] is True

    # 2. a1.processar_documento -> carla_divergencia_dias (usa PNG real com sha256!)
    t2 = _carregar_fixture("a1.processar_documento/carla_divergencia_dias.tarefa.json")
    e2 = _carregar_fixture("a1.processar_documento/carla_divergencia_dias.esperado.json")
    r2 = executar(t2, repositorio=repo, llm=mock_llm, config=cfg)
    jsonschema.validate(instance=r2, schema=schemas_json["ResultadoA1"])
    ResultadoA1.model_validate(r2).validar_payload_acao()
    assert r2["em_resposta_a"] == t2["id_mensagem"]
    assert r2["status"] == e2["status"]
    assert r2["estado_proposto"] == e2["estado_proposto"]
    assert r2["payload"] == e2["payload"]
    assert r2["saidas"] == e2["saidas"]
    assert r2["eventos"] == e2["eventos"]
    assert r2["decisao"]["motivo"] == e2["decisao"]["motivo"]
    assert r2["decisao"]["usou_llm"] is True

    # 3. a1.confirmar -> carla_confirmado_ia (sobre o atestado pendente 311)
    t3 = _carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json")
    e3 = _carregar_fixture("a1.confirmar/carla_confirmado_ia.esperado.json")
    r3 = executar(t3, repositorio=repo, llm=mock_llm, config=cfg)
    jsonschema.validate(instance=r3, schema=schemas_json["ResultadoA1"])
    ResultadoA1.model_validate(r3).validar_payload_acao()
    assert r3["em_resposta_a"] == t3["id_mensagem"]
    assert r3["status"] == e3["status"]
    assert r3["estado_proposto"] == e3["estado_proposto"]
    assert r3["payload"] == e3["payload"]
    assert r3["saidas"] == e3["saidas"]
    assert r3["eventos"] == e3["eventos"]
    assert r3["decisao"] == e3["decisao"]

    # 4. a1.registrar_baixa_fisica -> baixa_no_prazo (sobre id_atestado=311)
    t4 = _carregar_fixture("a1.registrar_baixa_fisica/baixa_no_prazo.tarefa.json")
    e4 = _carregar_fixture("a1.registrar_baixa_fisica/baixa_no_prazo.esperado.json")
    r4 = executar(t4, repositorio=repo, llm=mock_llm, config=cfg)
    jsonschema.validate(instance=r4, schema=schemas_json["ResultadoA1"])
    ResultadoA1.model_validate(r4).validar_payload_acao()
    assert r4["em_resposta_a"] == t4["id_mensagem"]
    assert r4["status"] == e4["status"]
    assert r4["estado_proposto"] is None
    assert r4["payload"] == e4["payload"]
    assert r4["saidas"] == e4["saidas"]
    assert r4["eventos"] == e4["eventos"]
    assert r4["decisao"] == e4["decisao"]

    # 5. a1.confirmar -> datas_invalidas (fim < inicio)
    t5 = _carregar_fixture("a1.confirmar/datas_invalidas.tarefa.json")
    e5 = _carregar_fixture("a1.confirmar/datas_invalidas.esperado.json")
    r5 = executar(t5, repositorio=repo, llm=mock_llm, config=cfg)
    jsonschema.validate(instance=r5, schema=schemas_json["ResultadoA1"])
    ResultadoA1.model_validate(r5).validar_payload_acao()
    assert r5["em_resposta_a"] == t5["id_mensagem"]
    assert r5["status"] == e5["status"]
    assert r5["estado_proposto"] == e5["estado_proposto"]
    assert r5["payload"] == e5["payload"]
    assert r5["saidas"] == e5["saidas"]
    assert r5["eventos"] == e5["eventos"]
    assert r5["decisao"] == e5["decisao"]


def test_idempotencia_por_id_mensagem_nao_duplica_nem_rechama_llm() -> None:
    """Reenviar a mesma `id_mensagem` devolve exatamente o mesmo RESULTADO."""
    repo = RepositorioA1Memoria()
    mock_llm = MockLLM()
    cfg = A1Config(groq_api_key="", politica_extrair_cid=False)

    t_proc = _carregar_fixture("a1.processar_documento/carla_divergencia_dias.tarefa.json")
    r_primeiro = executar(t_proc, repositorio=repo, llm=mock_llm, config=cfg)
    r_repetido = executar(t_proc, repositorio=repo, llm=mock_llm, config=cfg)

    assert r_repetido == r_primeiro
    assert mock_llm.chamadas == 1
    assert len(repo.atestados) == 1

    # Também funciona quando se instancia ServicoA1 diretamente para a1.confirmar isolado
    repo_isolado = RepositorioA1Memoria()
    servico = ServicoA1(repositorio=repo_isolado, llm=mock_llm, config=cfg)
    t_conf = _carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json")
    rc1 = servico.executar(t_conf)
    rc2 = servico.executar(t_conf)
    assert rc1 == rc2
    assert len(repo_isolado.atestados) == 1


def test_documento_ilegivel_duplicado_manual_e_payload_invalido(tmp_path) -> None:
    """Cobre documento ilegível, duplicidade de arquivo, origem MANUAL e erro."""
    repo = RepositorioA1Memoria()
    mock_llm = MockLLM()
    cfg = A1Config(groq_api_key="", politica_extrair_cid=True)
    servico = ServicoA1(repositorio=repo, llm=mock_llm, config=cfg)

    # 1. Documento em branco -> legibilidade = "ilegivel", estado NOVO, sem gravar
    img_branca = tmp_path / "branca.png"
    Image.new("RGB", (300, 200), "white").save(img_branca)
    t_ilegivel = copy.deepcopy(
        _carregar_fixture("a1.processar_documento/carla_divergencia_dias.tarefa.json")
    )
    t_ilegivel["id_mensagem"] = "t-ilegivel-01"
    t_ilegivel["payload"] = {"arquivo_uri": str(img_branca), "mime": "image/png"}
    res_ilegivel = servico.executar(t_ilegivel)
    assert res_ilegivel["status"] == "PRECISA_HUMANO"
    assert res_ilegivel["estado_proposto"] == "NOVO"
    assert res_ilegivel["payload"]["atestado_extraido"]["legibilidade"] == "ilegivel"
    assert res_ilegivel["eventos"] == []
    assert res_ilegivel["decisao"]["usou_llm"] is False
    assert len(repo.atestados) == 0

    # 2. Documento válido + duplicado com novo id_mensagem
    t_doc = _carregar_fixture("a1.processar_documento/carla_divergencia_dias.tarefa.json")
    res_doc = servico.executar(t_doc)
    assert res_doc["payload"]["atestado_extraido"]["cid"] == "J06.9"

    t_dup = copy.deepcopy(t_doc)
    t_dup["id_mensagem"] = "t-dup-02"
    res_dup = servico.executar(t_dup)
    assert res_dup["status"] == "PRECISA_HUMANO"
    assert "já foi recebido" in res_dup["saidas"][0]["texto"]
    assert len(repo.atestados) == 1

    # 3. Confirmação com campos_corrigidos -> origem MANUAL e preserva CRM/SP 123456
    t_manual = copy.deepcopy(
        _carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json")
    )
    t_manual["id_mensagem"] = "t-manual-03"
    t_manual["payload"]["atestado_confirmado"]["periodo"]["fim"] = "2026-11-12"
    t_manual["payload"]["campos_corrigidos"] = ["periodo.fim"]
    res_manual = servico.executar(t_manual)
    assert res_manual["status"] == "OK"
    assert res_manual["estado_proposto"] == "VALIDADO"
    assert res_manual["payload"]["origem"] == "MANUAL"
    atestado_db = repo.obter_atestado(res_manual["payload"]["id_atestado"])
    assert atestado_db is not None
    assert repo.profissionais[atestado_db.id_profissional]["numero_registro"] == "123456"

    # 4. Payload inválido retorna status="ERRO" sem quebrar o chamador
    t_erro = copy.deepcopy(t_doc)
    t_erro["id_mensagem"] = "t-erro-04"
    t_erro["payload"] = {"mime": "image/png"}  # falta arquivo_uri obrigatório
    res_erro = servico.executar(t_erro)
    assert res_erro["status"] == "ERRO"
    assert res_erro["erro"]["codigo"] == "PAYLOAD_INVALIDO"
    assert res_erro["erro"]["reprocessavel"] is False


def test_retomada_confirmacao_apos_reinicio_e_repositorio_padrao_cacheado() -> None:
    """Valida retomada via `pendente_db` após limpar memória e cache do repositório padrão."""
    mock_llm = MockLLM()
    cfg_mem = A1Config(groq_api_key="", database_url="", politica_extrair_cid=True)

    # 1. Processa via função de módulo sem passar repositorio= (usa repositório cacheado)
    t_proc = copy.deepcopy(
        _carregar_fixture("a1.processar_documento/carla_divergencia_dias.tarefa.json")
    )
    t_proc["id_mensagem"] = "t-cache-repo-01"
    r_proc = executar(t_proc, llm=mock_llm, config=cfg_mem)
    assert r_proc["payload"]["atestado_extraido"]["cid"] == "J06.9"

    # 2. Confirma apenas com payload parcial de periodo/emissao na mesma instancia cacheada
    t_conf = copy.deepcopy(_carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json"))
    t_conf["id_mensagem"] = "t-cache-repo-02"
    r_conf = executar(t_conf, llm=mock_llm, config=cfg_mem)
    assert r_conf["status"] == "OK"
    assert r_conf["payload"]["id_atestado"] == 311

    # 3. Simula reinício do serviço: mesmo repo com registro PENDENTE, cache limpo e politica CID desligada
    repo_reinicio = RepositorioA1Memoria()
    cfg_com_cid = A1Config(groq_api_key="", politica_extrair_cid=True)
    executar(
        {**t_proc, "id_mensagem": "t-reinicio-01"},
        repositorio=repo_reinicio,
        llm=mock_llm,
        config=cfg_com_cid,
    )
    limpar_cache_idempotencia()

    cfg_sem_cid = A1Config(groq_api_key="", politica_extrair_cid=False)
    t_conf_sem_data_fim = {
        **t_conf,
        "id_mensagem": "t-reinicio-02",
        "payload": {
            "atestado_confirmado": {
                "periodo": {"inicio": "2026-11-09"},
            },
            "campos_corrigidos": [],
        },
    }
    r_retomado = executar(
        t_conf_sem_data_fim,
        repositorio=repo_reinicio,
        llm=mock_llm,
        config=cfg_sem_cid,
    )
    assert r_retomado["status"] == "OK"
    assert r_retomado["estado_proposto"] == "VALIDADO"
    at_final = repo_reinicio.obter_atestado(r_retomado["payload"]["id_atestado"])
    assert at_final is not None
    assert at_final.fim_afastamento.date().isoformat() == "2026-11-11"
    assert at_final.codigo_cid is None
    assert repo_reinicio.profissionais[at_final.id_profissional]["numero_registro"] == "123456"


@pytest.mark.skipif(
    not (_CFG.groq_api_key and _postgres_disponivel(_CFG.database_url)),
    reason="GROQ_API_KEY ou PostgreSQL 16 com seed da Carla indisponível.",
)
def test_fluxo_completo_4_acoes_com_postgres_e_groq_reais() -> None:
    """Roda as 4 ações do caso 105 no PostgreSQL 16 + OCR + ChatGroq reais."""
    cfg = A1Config(
        groq_api_key=_CFG.groq_api_key,
        llm_model=_CFG.llm_model,
        database_url=_CFG.database_url,
        politica_extrair_cid=False,
        limiar_confianca_campo=_CFG.limiar_confianca_campo,
        prazo_entrega_fisica_dias_uteis=_CFG.prazo_entrega_fisica_dias_uteis,
        atestados_storage_dir=_CFG.atestados_storage_dir,
    )
    repo = RepositorioA1Postgres(cfg.database_url)
    servico = ServicoA1(repositorio=repo, config=cfg)

    id_atestado_criado: int | None = None
    id_prof_criado: int | None = None
    try:
        # 1. a1.conversar
        t1 = copy.deepcopy(_carregar_fixture("a1.conversar/falta_arquivo.tarefa.json"))
        t1["id_mensagem"] = "e2e-pg-groq-01"
        r1 = servico.executar(t1)
        assert r1["status"] == "OK"
        assert "arquivo" in r1["payload"]["faltando"]

        # 2. a1.processar_documento (PNG real com marca d'água -> RapidOCR -> Groq)
        t2 = copy.deepcopy(
            _carregar_fixture("a1.processar_documento/carla_divergencia_dias.tarefa.json")
        )
        t2["id_mensagem"] = "e2e-pg-groq-02"
        t2["payload"]["arquivo_uri"] = (
            ".storage/atestados/atestado_carla_105.png"
        )
        r2 = servico.executar(t2)
        assert r2["status"] == "PRECISA_HUMANO"
        assert r2["estado_proposto"] == "RECEBIDO"
        assert r2["payload"]["divergencias"] == [
            {"campo": "dias", "no_documento": 3, "informado": 5}
        ]
        pend = repo.obter_ultimo_atestado_pendente(42)
        assert pend is not None
        id_atestado_criado = pend.id_atestado
        id_prof_criado = pend.id_profissional

        # 3. a1.confirmar
        t3 = copy.deepcopy(_carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json"))
        t3["id_mensagem"] = "e2e-pg-groq-03"
        r3 = servico.executar(t3)
        assert r3["status"] == "OK"
        assert r3["estado_proposto"] == "VALIDADO"
        assert r3["payload"]["id_atestado"] == id_atestado_criado
        assert r3["payload"]["origem"] == "IA"
        assert r3["payload"]["prazo_entrega_fisica"] == "2026-11-16"

        # 4. a1.registrar_baixa_fisica
        t4 = copy.deepcopy(
            _carregar_fixture("a1.registrar_baixa_fisica/baixa_no_prazo.tarefa.json")
        )
        t4["id_mensagem"] = "e2e-pg-groq-04"
        t4["payload"]["id_atestado"] = id_atestado_criado
        r4 = servico.executar(t4)
        assert r4["status"] == "OK"
        assert r4["eventos"] == [
            {
                "tipo": "atestado.baixa_registrada",
                "dados": {"id_atestado": id_atestado_criado},
            }
        ]
        at_final = repo.obter_atestado(id_atestado_criado)
        assert at_final is not None
        assert at_final.recebido_fisico_por == 7
        assert at_final.recebido_fisico_em is not None
    finally:
        with psycopg.connect(cfg.database_url) as conn:
            if id_atestado_criado is not None:
                conn.execute(
                    "DELETE FROM afastamento.atestado WHERE id_atestado = %s",
                    (id_atestado_criado,),
                )
            if id_prof_criado is not None:
                conn.execute(
                    "DELETE FROM afastamento.profissional_saude "
                    "WHERE id_profissional = %s AND NOT EXISTS ("
                    "SELECT 1 FROM afastamento.atestado WHERE id_profissional = %s"
                    ")",
                    (id_prof_criado, id_prof_criado),
                )
            conn.commit()
