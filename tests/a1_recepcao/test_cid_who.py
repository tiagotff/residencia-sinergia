"""Testes offline da integração WHO ICD API (CID-11/CID-10), cache-aside e seed local."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from sinergia.agentes.a1_recepcao import (
    A1Config,
    AgenteA1,
    ClienteCidWHO,
    MockLLM,
    RepositorioA1Memoria,
    cid_sintaxe_valida,
    eh_cid10,
    eh_cid11,
    limpar_cache_idempotencia,
)
from sinergia.agentes.a1_recepcao.seed_cid10 import (
    executar_seed_cid10,
    extrair_cids10_de_zip_bytes,
)
from sinergia.agentes.a1_recepcao.seed_cid11 import (
    executar_seed_cid11,
    gerar_sql_seed_cids,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _carregar_fixture(relpath: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / relpath).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _limpar_caches() -> None:
    limpar_cache_idempotencia()


def test_validacao_sintatica_cid11_e_cid10() -> None:
    # CID-11 válidos
    for cod in ("ME84.2", "CA07.0", "1A00", "6B00", "BA00.Z", "MG26"):
        assert eh_cid11(cod) is True, cod
        assert cid_sintaxe_valida(cod) is True, cod

    # CID-10 válidos (retrocompatibilidade)
    for cod in ("M54.5", "J06.9", "F41.1", "Z00.0", "A00"):
        assert eh_cid10(cod) is True, cod
        assert cid_sintaxe_valida(cod) is True, cod

    # Inválidos
    for cod in ("", "   ", "INVALIDO", "M54.555", "123456789", "O000", "I000"):
        assert cid_sintaxe_valida(cod) is False, cod


def test_cliente_cid_who_oauth2_cid11_pt_cid10_e_retry_401_429() -> None:
    """Valida OAuth2 (cache + refresh em 401), CID-11 pt, CID-10, 404 e retry em 429."""
    contadores = {"token": 0, "codeinfo": 0, "entidade": 0, "cid10": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://icdaccessmanagement.who.int/connect/token":
            contadores["token"] += 1
            return httpx.Response(
                200,
                json={"access_token": f"tok-{contadores['token']}", "expires_in": 3600},
            )

        assert request.headers.get("API-Version") == "v2"
        assert request.headers.get("Authorization", "").startswith("Bearer tok-")

        if url.endswith("/icd/release/11/2026-01/mms/codeinfo/ME84.2"):
            contadores["codeinfo"] += 1
            if contadores["codeinfo"] == 1:
                # Simula token expirado -> deve renovar automaticamente
                return httpx.Response(401)
            assert request.headers.get("Accept-Language") == "pt"
            return httpx.Response(
                200,
                json={
                    "code": "ME84.2",
                    "stemId": "http://id.who.int/icd/release/11/2026-01/mms/123456",
                },
            )

        if url == "https://id.who.int/icd/release/11/2026-01/mms/123456":
            contadores["entidade"] += 1
            return httpx.Response(
                200,
                json={
                    "code": "ME84.2",
                    "classKind": "category",
                    "chapter": "21",
                    "title": {"@language": "pt", "@value": "Dor lombar baixa"},
                    "definition": {
                        "@language": "pt",
                        "@value": "Dor localizada abaixo do rebordo costal e acima das pregas glúteas inferiores.",
                    },
                },
            )

        if url.endswith("/icd/release/10/2019/A00.0"):
            contadores["cid10"] += 1
            if contadores["cid10"] == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(
                200,
                json={
                    "@id": "http://id.who.int/icd/release/10/2019/A00.0",
                    "code": "A00.0",
                    "classKind": "category",
                    "title": {
                        "@language": "en",
                        "@value": "Cholera due to Vibrio cholerae 01, biovar cholerae",
                    },
                },
            )

        if url.endswith("/icd/release/11/2026-01/mms/codeinfo/ZZ99.9"):
            return httpx.Response(404)

        return httpx.Response(404)

    cliente = ClienteCidWHO(
        base_url="https://id.who.int",
        client_id="cli-test",
        client_secret="sec-test",
        release_11="2026-01",
        release_10="2019",
        language="pt",
        transport=httpx.MockTransport(handler),
    )

    # 1. CID-11 em português (com refresh 401 + reescrita http -> https)
    res_11 = cliente.consultar_cid("me84.2")
    assert res_11.status == "ENCONTRADO"
    assert res_11.dados is not None
    assert res_11.dados.codigo_cid == "ME84.2"
    assert res_11.dados.nome_doenca == "Dor lombar baixa"
    assert (
        res_11.dados.descricao
        == "Dor localizada abaixo do rebordo costal e acima das pregas glúteas inferiores."
    )
    assert res_11.dados.capitulo == "21"
    assert res_11.dados.versao_cid == "CID-11"
    assert contadores["token"] == 2  # 1ª emissão + renovação após 401

    # 2. CID-10 com retry em 429 (reaproveita token em cache)
    res_10 = cliente.consultar_cid("A00.0")
    assert res_10.status == "ENCONTRADO"
    assert res_10.dados is not None
    assert res_10.dados.codigo_cid == "A00.0"
    assert res_10.dados.versao_cid == "CID-10"
    assert res_10.dados.capitulo == "A"
    assert contadores["token"] == 2  # token reutilizado do cache

    # 3. 404 -> NAO_ENCONTRADO
    res_404 = cliente.consultar_cid("ZZ99.9")
    assert res_404.status == "NAO_ENCONTRADO"
    assert res_404.dados is None


def test_cliente_cid_who_sem_credenciais_ou_indisponivel() -> None:
    cliente_sem_cred = ClienteCidWHO(
        base_url="https://id.who.int",
        client_id="",
        client_secret="",
    )
    res_sem = cliente_sem_cred.consultar_cid("ME84.2")
    assert res_sem.status == "INDISPONIVEL"

    def handler_erro(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timeout simulado")

    cliente_timeout = ClienteCidWHO(
        base_url="http://localhost:8088",
        requer_oauth=False,
        transport=httpx.MockTransport(handler_erro),
    )
    res_to = cliente_timeout.consultar_cid("ME84.2")
    assert res_to.status == "INDISPONIVEL"


def test_servico_a1_cache_aside_banco_vs_api_externa_e_cid_inexistente(
    tmp_path: Path,
) -> None:
    """Garante fluxo: hit no banco -> 0 chamadas API; miss -> consulta API -> salva no banco."""
    chamadas_api: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/connect/token"):
            return httpx.Response(200, json={"access_token": "tok-ok", "expires_in": 3600})
        chamadas_api.append(url)
        if url.endswith("/codeinfo/ME84.2"):
            return httpx.Response(
                200,
                json={"code": "ME84.2", "stemId": "http://id.who.int/icd/release/11/2026-01/mms/999"},
            )
        if url.endswith("/mms/999"):
            return httpx.Response(
                200,
                json={
                    "code": "ME84.2",
                    "classKind": "category",
                    "chapter": "21",
                    "title": {"@language": "pt", "@value": "Dor lombar baixa"},
                    "definition": {
                        "@language": "pt",
                        "@value": "Dor localizada abaixo do rebordo costal e acima das pregas glúteas inferiores.",
                    },
                },
            )
        if url.endswith("/codeinfo/ZZ99.9"):
            return httpx.Response(404)
        return httpx.Response(404)

    cliente_who = ClienteCidWHO(
        base_url="https://id.who.int",
        client_id="cid-client",
        client_secret="cid-secret",
        transport=httpx.MockTransport(handler),
    )
    repo = RepositorioA1Memoria()
    cfg = A1Config(groq_api_key="", politica_extrair_cid=True)
    agente = AgenteA1(db=repo, llm=MockLLM(), config=cfg, cliente_cid=cliente_who)

    # 1. CID já existente no banco (M54.5) -> 0 chamadas externas
    t_hit = copy.deepcopy(_carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json"))
    t_hit["id_mensagem"] = "who-hit-local-01"
    t_hit["payload"]["atestado_confirmado"]["cid"] = "M54.5"
    r_hit = agente.executar(t_hit)
    assert r_hit["status"] == "OK"
    assert r_hit["estado_proposto"] == "VALIDADO"
    assert len(chamadas_api) == 0
    assert agente.ultimo_diagnostico_cid is not None
    assert agente.ultimo_diagnostico_cid["origem"] == "BANCO_LOCAL"

    # 2. CID ausente no banco (ME84.2) -> consulta API externa e salva cache-aside no banco
    t_miss = copy.deepcopy(t_hit)
    t_miss["id_mensagem"] = "who-miss-api-02"
    t_miss["payload"]["atestado_confirmado"]["cid"] = "ME84.2"
    r_miss = agente.executar(t_miss)
    assert r_miss["status"] == "OK"
    assert r_miss["estado_proposto"] == "VALIDADO"
    assert len(chamadas_api) == 2  # codeinfo + entidade
    assert "ME84.2" in repo.cids
    assert (
        repo.cids["ME84.2"]
        == "Dor localizada abaixo do rebordo costal e acima das pregas glúteas inferiores."
    )
    cid_info = repo.obter_cid("ME84.2")
    assert cid_info is not None
    assert cid_info["nome_doenca"] == "Dor lombar baixa"
    assert (
        cid_info["descricao"]
        == "Dor localizada abaixo do rebordo costal e acima das pregas glúteas inferiores."
    )
    assert agente.ultimo_diagnostico_cid is not None
    assert agente.ultimo_diagnostico_cid["origem"] == "API_WHO_EXTERNA_CACHE_ASIDE"
    assert agente.ultimo_diagnostico_cid["nome_doenca"] == "Dor lombar baixa"
    atestado_salvo = repo.obter_atestado(r_miss["payload"]["id_atestado"])
    assert atestado_salvo is not None
    assert atestado_salvo.codigo_cid == "ME84.2"
    assert atestado_salvo.nome_doenca == "Dor lombar baixa"
    cid_restrito = repo.obter_cid_restrito(r_miss["payload"]["id_atestado"])
    assert cid_restrito is not None
    assert cid_restrito.codigo_cid == "ME84.2"
    assert cid_restrito.nome_doenca == "Dor lombar baixa"

    # 3. Segunda consulta do mesmo ME84.2 -> hit local no banco (0 novas chamadas API!)
    t_cache = copy.deepcopy(t_hit)
    t_cache["id_mensagem"] = "who-cache-hit-03"
    t_cache["payload"]["atestado_confirmado"]["cid"] = "ME84.2"
    r_cache = agente.executar(t_cache)
    assert r_cache["status"] == "OK"
    assert len(chamadas_api) == 2
    assert agente.ultimo_diagnostico_cid["origem"] == "BANCO_LOCAL"

    # 4. CID inexistente na API externa (ZZ99.9) -> rejeitado na confirmação com cid_invalido
    t_inv = copy.deepcopy(t_hit)
    t_inv["id_mensagem"] = "who-invalido-04"
    t_inv["payload"]["atestado_confirmado"]["cid"] = "ZZ99.9"
    r_inv = agente.executar(t_inv)
    assert r_inv["status"] == "OK"
    assert r_inv["estado_proposto"] == "NOVO"
    assert r_inv["eventos"][0]["tipo"] == "atestado.invalido"
    assert r_inv["eventos"][0]["dados"]["motivo"] == "cid_invalido"

    # 5. Em a1.processar_documento com PDF nativo e CID inexistente -> gera divergência sem rejeitar
    import pymupdf

    pdf_path = tmp_path / "atestado_cid_inv.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=350)
    page.insert_text(
        (40, 60),
        (
            "ATESTADO MEDICO\n"
            "Atesto que Carla Souza necessita de 3 dias de afastamento\n"
            "de 09/11/2026 a 11/11/2026.\n"
            "CID-11: ZZ99.9\n"
            "Campinas - SP, 09 de novembro de 2026.\n"
            "Dr. Marcos - CRM/SP 123456"
        ),
        fontsize=14,
    )
    doc.save(pdf_path)
    doc.close()

    t_doc = copy.deepcopy(
        _carregar_fixture("a1.processar_documento/carla_divergencia_dias.tarefa.json")
    )
    t_doc["id_mensagem"] = "who-doc-div-05"
    t_doc["payload"] = {"arquivo_uri": str(pdf_path), "mime": "application/pdf"}
    r_doc = agente.executar(t_doc)
    assert r_doc["status"] == "PRECISA_HUMANO"
    campos_div = [d["campo"] for d in r_doc["payload"]["divergencias"]]
    assert "cid" in campos_div


def test_seed_cid11_container_local_e_exportacao_sql(tmp_path: Path) -> None:
    """Valida extração da árvore ICD-11 do container local e geração do arquivo SQL."""
    arvore: dict[str, dict[str, Any]] = {
        "/icd/release/11/2026-01/mms": {
            "child": ["http://id.who.int/icd/release/11/2026-01/mms/cap21"]
        },
        "/icd/release/11/2026-01/mms/cap21": {
            "classKind": "chapter",
            "code": "21",
            "title": {"@language": "pt", "@value": "Sintomas, sinais ou achados clínicos"},
            "child": ["http://id.who.int/icd/release/11/2026-01/mms/blk1"],
        },
        "/icd/release/11/2026-01/mms/blk1": {
            "classKind": "block",
            "title": {"@language": "pt", "@value": "Sintomas musculoesqueléticos"},
            "child": [
                "http://id.who.int/icd/release/11/2026-01/mms/me84",
                "http://id.who.int/icd/release/11/2026-01/mms/me842",
            ],
        },
        "/icd/release/11/2026-01/mms/me84": {
            "classKind": "category",
            "code": "ME84",
            "title": {"@language": "pt", "@value": "Dor na coluna vertebral"},
            "child": [],
        },
        "/icd/release/11/2026-01/mms/me842": {
            "classKind": "category",
            "code": "ME84.2",
            "title": {"@language": "pt", "@value": "Dor lombar baixa"},
            "child": [],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        # Container local não deve enviar header Authorization
        assert "Authorization" not in request.headers
        assert request.headers.get("Accept-Language") == "pt"
        payload = arvore.get(request.url.path)
        if payload is None:
            return httpx.Response(404)
        return httpx.Response(200, json=payload)

    cliente_seed = ClienteCidWHO(
        base_url="http://localhost:8088",
        release_11="2026-01",
        language="pt",
        requer_oauth=False,
        transport=httpx.MockTransport(handler),
    )

    sql_destino = tmp_path / "seed_cid11_teste.sql"
    cids, inseridos = executar_seed_cid11(
        seed_url="http://localhost:8088",
        dry_run=True,
        gerar_sql=sql_destino,
        cliente=cliente_seed,
    )

    assert inseridos == 0
    assert {c.codigo_cid for c in cids} == {"ME84", "ME84.2"}
    assert all(c.capitulo == "21" for c in cids)
    assert sql_destino.is_file()
    conteudo_sql = sql_destino.read_text(encoding="utf-8")
    assert "INSERT INTO afastamento.cid (codigo_cid, nome_doenca, descricao, capitulo)" in conteudo_sql
    assert "'ME84.2', 'Dor lombar baixa', 'Dor lombar baixa', '21'" in conteudo_sql

    # Verifica que gerar_sql_seed_cids escapa aspas simples com segurança
    sql_escape = tmp_path / "escape.sql"
    gerar_sql_seed_cids(
        [
            cids[0].model_copy(update={"descricao": "Doença de Crohn's"}),
        ],
        sql_escape,
    )
    assert "Crohn''s" in sql_escape.read_text(encoding="utf-8")


def test_seed_ufs_brasil_contem_todas_as_27_ufs() -> None:
    caminho = Path("db/seeds/002_seed_ufs_brasil.sql")
    assert caminho.is_file()
    sql = caminho.read_text(encoding="utf-8")
    ufs_encontradas = set(re.findall(r"\('([A-Z]{2})',", sql))
    assert len(ufs_encontradas) == 27
    assert {"SP", "RJ", "MG", "BA", "RS", "DF", "AM", "PE"}.issubset(ufs_encontradas)


def test_cliente_cid_who_dois_fallbacks_local_e_externo() -> None:
    """Valida cadeia: Container Local (Fallback 1) -> API Externa OAuth2 (Fallback 2)."""
    chamadas_local: list[str] = []
    chamadas_externas: list[str] = []

    def handler_local_ativo(request: httpx.Request) -> httpx.Response:
        chamadas_local.append(str(request.url))
        if request.url.path.endswith("/codeinfo/ME84.2"):
            return httpx.Response(
                200,
                json={"code": "ME84.2", "stemId": "http://id.who.int/icd/release/11/2026-01/mms/me842"},
            )
        if request.url.path.endswith("/mms/me842"):
            return httpx.Response(
                200,
                json={
                    "code": "ME84.2",
                    "classKind": "category",
                    "chapter": "21",
                    "title": {"@language": "pt", "@value": "Dor lombar baixa"},
                },
            )
        # Container local só tem CID-11 -> retorna 404 para CID-10
        return httpx.Response(404)

    def handler_externo(request: httpx.Request) -> httpx.Response:
        chamadas_externas.append(str(request.url))
        if str(request.url).endswith("/connect/token"):
            return httpx.Response(200, json={"access_token": "tok-ext", "expires_in": 3600})
        if request.url.path.endswith("/icd/release/10/2019/J06.9"):
            return httpx.Response(
                200,
                json={
                    "code": "J06.9",
                    "classKind": "category",
                    "title": {"@language": "en", "@value": "Acute upper respiratory infection, unspecified"},
                },
            )
        if request.url.path.endswith("/codeinfo/CA07.0"):
            return httpx.Response(
                200,
                json={"code": "CA07.0", "stemId": "http://id.who.int/icd/release/11/2026-01/mms/ca070"},
            )
        if request.url.path.endswith("/mms/ca070"):
            return httpx.Response(
                200,
                json={
                    "code": "CA07.0",
                    "classKind": "category",
                    "chapter": "12",
                    "title": {"@language": "pt", "@value": "Nasofaringite aguda"},
                },
            )
        return httpx.Response(404)

    # 1. Container local responde CID-11 -> 0 chamadas na API externa
    cli_ok = ClienteCidWHO(
        base_url="https://id.who.int",
        local_base_url="http://localhost:8088",
        client_id="cli",
        client_secret="sec",
        transport=httpx.MockTransport(handler_externo),
        local_transport=httpx.MockTransport(handler_local_ativo),
    )
    res_local = cli_ok.consultar_cid("ME84.2")
    assert res_local.status == "ENCONTRADO"
    assert res_local.origem_consulta == "CONTAINER_LOCAL"
    assert res_local.dados is not None
    assert res_local.dados.nome_doenca == "Dor lombar baixa"
    assert len(chamadas_local) == 2
    assert len(chamadas_externas) == 0

    # 2. CID-10 dá 404 no container local (só CID-11) -> cai no Fallback 2 (API externa)
    res_10_ext = cli_ok.consultar_cid("J06.9")
    assert res_10_ext.status == "ENCONTRADO"
    assert res_10_ext.origem_consulta == "API_EXTERNA"
    assert res_10_ext.dados is not None
    assert res_10_ext.dados.versao_cid == "CID-10"
    assert any("/icd/release/10/2019/J06.9" in u for u in chamadas_externas)

    # 3. Container local desligado (ConnectError) -> cai no Fallback 2 (API externa)
    def handler_local_offline(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("container offline")

    cli_offline_local = ClienteCidWHO(
        base_url="https://id.who.int",
        local_base_url="http://localhost:8088",
        client_id="cli",
        client_secret="sec",
        transport=httpx.MockTransport(handler_externo),
        local_transport=httpx.MockTransport(handler_local_offline),
    )
    res_fallback_ext = cli_offline_local.consultar_cid("CA07.0")
    assert res_fallback_ext.status == "ENCONTRADO"
    assert res_fallback_ext.origem_consulta == "API_EXTERNA"
    assert res_fallback_ext.dados is not None
    assert res_fallback_ext.dados.nome_doenca == "Nasofaringite aguda"


def test_servico_a1_registra_origem_container_local_e_cache_aside() -> None:
    def handler_local(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/codeinfo/ME84.2"):
            return httpx.Response(
                200,
                json={"code": "ME84.2", "stemId": "http://id.who.int/icd/release/11/2026-01/mms/me842"},
            )
        if request.url.path.endswith("/mms/me842"):
            return httpx.Response(
                200,
                json={
                    "code": "ME84.2",
                    "classKind": "category",
                    "chapter": "21",
                    "title": {"@language": "pt", "@value": "Dor lombar baixa"},
                },
            )
        return httpx.Response(404)

    cliente_who = ClienteCidWHO(
        base_url="https://id.who.int",
        local_base_url="http://localhost:8088",
        client_id="",
        client_secret="",
        local_transport=httpx.MockTransport(handler_local),
    )
    repo = RepositorioA1Memoria()
    agente = AgenteA1(
        db=repo,
        llm=MockLLM(),
        config=A1Config(groq_api_key="", politica_extrair_cid=True),
        cliente_cid=cliente_who,
    )
    t_miss = copy.deepcopy(_carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json"))
    t_miss["id_mensagem"] = "who-container-local-01"
    t_miss["payload"]["atestado_confirmado"]["cid"] = "ME84.2"
    res = agente.executar(t_miss)
    assert res["status"] == "OK"
    assert res["estado_proposto"] == "VALIDADO"
    assert agente.ultimo_diagnostico_cid is not None
    assert agente.ultimo_diagnostico_cid["origem"] == "CONTAINER_WHO_LOCAL_CACHE_ASIDE"
    assert "ME84.2" in repo.cids


def test_seed_cid10_datasus_zip_e_arquivo_versionado(tmp_path: Path) -> None:
    """Valida parser do pacote DATASUS CID10CSV.zip, geração SQL e seed versionado."""
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "CID-10-CAPITULOS.CSV",
            (
                "NUMCAP;CATINIC;CATFIM;DESCRICAO;DESCRABREV;\n"
                "1;A00;B99;Capítulo I - Doenças infecciosas;I;\n"
                "13;M00;M99;Capítulo XIII - Sistema osteomuscular;XIII;\n"
            ).encode("latin-1"),
        )
        zf.writestr(
            "CID-10-CATEGORIAS.CSV",
            (
                "CAT;CLASSIF;DESCRICAO;DESCRABREV;REFER;EXCLUIDOS;\n"
                "A00;;Cólera;A00 Colera;;;\n"
                "M54;;Dorsalgia;M54 Dorsalgia;;;\n"
            ).encode("latin-1"),
        )
        zf.writestr(
            "CID-10-SUBCATEGORIAS.CSV",
            (
                "SUBCAT;CLASSIF;RESTRSEXO;CAUSAOBITO;DESCRICAO;DESCRABREV;REFER;EXCLUIDOS;\n"
                "A000;;;;Cólera devida a Vibrio cholerae 01, biótipo cholerae;A00.0;;;\n"
                "M545;;;;Dor lombar baixa;M54.5;;;\n"
            ).encode("latin-1"),
        )

    zip_path = tmp_path / "CID10CSV.zip"
    zip_path.write_bytes(buf.getvalue())

    cids_mem = extrair_cids10_de_zip_bytes(buf.getvalue())
    assert {c.codigo_cid for c in cids_mem} == {"A00", "M54", "A00.0", "M54.5"}

    sql_destino = tmp_path / "seed_cid10_teste.sql"
    cids, inseridos = executar_seed_cid10(
        arquivo_zip=zip_path,
        gerar_sql=sql_destino,
        dry_run=True,
    )
    assert inseridos == 0
    assert len(cids) == 4
    por_codigo = {c.codigo_cid: c for c in cids}
    assert por_codigo["M54.5"].capitulo == "13"
    assert por_codigo["M54.5"].nome_doenca == "Dor lombar baixa"
    assert por_codigo["A00.0"].capitulo == "01"
    conteudo_sql = sql_destino.read_text(encoding="utf-8")
    assert "ON CONFLICT (codigo_cid) DO UPDATE" in conteudo_sql
    assert "'M54.5', 'Dor lombar baixa', 'Dor lombar baixa', '13'" in conteudo_sql

    seed_repo = Path("db/seeds/004_seed_cid10_pt.sql")
    assert seed_repo.is_file()
    sql_repo = seed_repo.read_text(encoding="utf-8")
    assert "'M54.5', 'Dor lombar baixa'" in sql_repo
    assert "'J06.9'" in sql_repo
