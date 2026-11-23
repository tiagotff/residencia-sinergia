"""Testes de conformidade com o `SinergIA_Guia_A1.pdf` (v0.7), segurança, integridade e velocidade."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Self

import jsonschema
import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from sinergia.agentes.a1_recepcao import (
    A1Config,
    AgenteA1,
    AtestadoExtraido,
    MockLLM,
    PeriodoAfastamento,
    ProfissionalExtraido,
    RepositorioA1Memoria,
    RepositorioA1Postgres,
    ResultadoA1,
    criar_app_a1,
    exportar_json_schemas_a1,
    extrair_atestado_via_llm,
    limpar_cache_idempotencia,
    obter_config_a1,
    ocr,
    processar_arquivo_atestado,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _carregar_fixture(relpath: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / relpath).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _limpar_caches() -> None:
    limpar_cache_idempotencia()


def test_config_politica_cid_guia_v07(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLITICA_CID", "nao_extrair")
    monkeypatch.setenv("POLITICA_EXTRAIR_CID", "true")
    cfg = obter_config_a1()
    assert cfg.politica_extrair_cid is False
    assert cfg.politica_cid_modo == "nao_extrair"

    monkeypatch.setenv("POLITICA_CID", "extrair_restrito")
    cfg2 = obter_config_a1()
    assert cfg2.politica_extrair_cid is True
    assert cfg2.politica_cid_modo == "extrair_restrito"

    monkeypatch.setenv("POLITICA_CID", "valor_invalido")
    with pytest.raises(ValueError, match="POLITICA_CID"):
        obter_config_a1()


def test_campos_guia_v07_baixa_confianca_prazos_legais_e_aliases(tmp_path: Path) -> None:
    """Valida tipo_documento, campos com baixa confiança, prazos_legais e aliases v0.7."""
    schemas_json = exportar_json_schemas_a1()
    repo = RepositorioA1Memoria()

    # Simula extração com número de CRM abaixo do limiar 0.8 (Guia A1 §1.8)
    mock_baixa_conf = MockLLM(
        extracao_padrao=AtestadoExtraido(
            tipo_documento="declaracao_comparecimento",
            legibilidade="ok",
            nome_no_documento="Carla Souza",
            periodo=PeriodoAfastamento(
                inicio=date(2026, 11, 1), fim=date(2026, 11, 30)
            ),
            data_emissao=date(2026, 11, 1),
            profissional=ProfissionalExtraido(
                conselho="CRM", numero="123456", uf="SP"
            ),
            cid=None,
            municipio="Campinas",
            confianca_campos={
                "nome_no_documento": 0.95,
                "periodo.inicio": 0.95,
                "periodo.fim": 0.95,
                "data_emissao": 0.92,
                "profissional.conselho": 0.94,
                "profissional.numero": 0.41,
                "profissional.uf": 0.95,
                "municipio": 0.90,
            },
        )
    )
    agente = AgenteA1(
        db=repo,
        llm=mock_baixa_conf,
        config=A1Config(groq_api_key="", politica_extrair_cid=False),
    )

    img = tmp_path / "atestado_30d.png"
    img_obj = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(img_obj).text((20, 40), "ATESTADO 30 DIAS", fill="black")
    img_obj.save(img)

    t_doc = copy.deepcopy(
        _carregar_fixture("a1.processar_documento/carla_divergencia_dias.tarefa.json")
    )
    t_doc["id_mensagem"] = "guia-v07-doc-01"
    t_doc["contexto"]["id_empresa"] = 1
    t_doc["payload"] = {"arquivo_uri": str(img), "mime": "image/png"}

    r_doc = agente.executar(t_doc)
    jsonschema.validate(instance=r_doc, schema=schemas_json["ResultadoA1"])
    ResultadoA1.model_validate(r_doc).validar_payload_acao()

    ext = r_doc["payload"]["atestado_extraido"]
    assert ext["tipo_documento"] == "declaracao_comparecimento"
    assert ext["periodo"]["dias"] == 30
    assert ext["profissional"]["numero"] is None
    assert ext["campos"]["profissional.numero"] == {
        "confianca": 0.41,
        "origem": "IA",
        "vazio_por_baixa_confianca": True,
    }
    assert "número do registro profissional" in r_doc["saidas"][0]["texto"]
    campos_div = [d["campo"] for d in r_doc["payload"]["divergencias"]]
    assert "tipo_documento" in campos_div

    # Confirma 30 dias -> deve emitir prazos_legais ["inss_16_dias", "exame_retorno"]
    t_conf = copy.deepcopy(_carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json"))
    t_conf["id_mensagem"] = "guia-v07-conf-02"
    t_conf["payload"] = {
        "atestado_confirmado": {
            "periodo": {"inicio": "2026-11-01", "fim": "2026-11-30"},
            "data_emissao": "2026-11-01",
            "profissional": {"conselho": "CRM", "numero": "123456", "uf": "SP"},
        },
        "campos_corrigidos": ["profissional.numero"],
    }
    r_conf = agente.executar(t_conf)
    jsonschema.validate(instance=r_conf, schema=schemas_json["ResultadoA1"])
    assert r_conf["status"] == "OK"
    assert r_conf["estado_proposto"] == "VALIDADO"
    assert r_conf["payload"]["prazos_legais"] == ["inss_16_dias", "exame_retorno"]
    assert r_conf["eventos"][0]["dados"]["prazos_legais"] == [
        "inss_16_dias",
        "exame_retorno",
    ]
    id_atestado = r_conf["payload"]["id_atestado"]

    # Baixa física usando os nomes do Guia v0.7 (recebido_por / data_hora)
    t_baixa = copy.deepcopy(
        _carregar_fixture("a1.registrar_baixa_fisica/baixa_no_prazo.tarefa.json")
    )
    t_baixa["id_mensagem"] = "guia-v07-baixa-03"
    t_baixa["payload"] = {
        "id_atestado": id_atestado,
        "recebido_por": "rh.ana",
        "data_hora": "2026-12-01T10:20:00-03:00",
    }
    r_baixa = agente.executar(t_baixa)
    jsonschema.validate(instance=r_baixa, schema=schemas_json["ResultadoA1"])
    assert r_baixa["status"] == "OK"
    assert r_baixa["payload"]["recebido_por"] == "rh.ana"
    assert r_baixa["payload"]["recebido_fisico_por"] == "rh.ana"


def test_endpoint_fastapi_post_executar_e_health() -> None:
    """Valida a interface HTTP isolada POST /executar e GET /health (Guia A1 §2.2 e §3)."""
    repo = RepositorioA1Memoria()
    agente = AgenteA1(
        db=repo,
        llm=MockLLM(),
        config=A1Config(groq_api_key="", politica_extrair_cid=False),
    )
    client = TestClient(criar_app_a1(agente=agente))

    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["agente"] == "A1"

    tarefa_conv = _carregar_fixture("a1.conversar/falta_arquivo.tarefa.json")
    resp_exec = client.post("/executar", json=tarefa_conv)
    assert resp_exec.status_code == 200
    body = resp_exec.json()
    assert body["status"] == "OK"
    assert body["estado_proposto"] == "NOVO"
    assert body["payload"]["faltando"] == ["arquivo"]


def test_seguranca_ocr_path_traversal_unc_tamanho_e_paginas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Garante bloqueios contra path traversal, UNC, file:// remoto e limites Anti-DoS."""
    with pytest.raises(ValueError, match="path traversal"):
        ocr._resolver_arquivo("../../Windows/System32/drivers/etc/hosts")

    with pytest.raises(ValueError, match="UNC"):
        ocr._resolver_arquivo(r"\\attacker-server\share\atestado.png")

    with pytest.raises(ValueError, match="host remoto"):
        ocr._resolver_arquivo("file://evil.example.com/share/atestado.png")

    with pytest.raises(ValueError, match="byte nulo"):
        ocr._resolver_arquivo("atestado.png\x00.pdf")

    # Limite de tamanho de arquivo
    img_grande = tmp_path / "grande.png"
    Image.new("RGB", (200, 200), "white").save(img_grande)
    monkeypatch.setenv("MAX_ARQUIVO_BYTES", "10")
    with pytest.raises(ValueError, match="limite máximo"):
        processar_arquivo_atestado(str(img_grande), mime="image/png")
    monkeypatch.delenv("MAX_ARQUIVO_BYTES", raising=False)

    # Limite de páginas de PDF
    pdf_6p = tmp_path / "seis_paginas.pdf"
    doc = pymupdf.open()
    for _ in range(6):
        doc.new_page(width=200, height=200)
    doc.save(pdf_6p)
    doc.close()
    with pytest.raises(ValueError, match="limite máximo de 5 páginas"):
        processar_arquivo_atestado(str(pdf_6p), mime="application/pdf")


def test_seguranca_rbac_baixa_fisica_e_sanitizacao_erro(tmp_path: Path) -> None:
    """Bloqueia colaborador comum na baixa física e mascara segredos em ERRO_INTERNO_A1."""
    repo = RepositorioA1Memoria()
    agente = AgenteA1(
        db=repo,
        llm=MockLLM(),
        config=A1Config(groq_api_key="", politica_extrair_cid=False),
    )

    # Registra atestado confirmado 311
    t_conf = copy.deepcopy(_carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json"))
    r_conf = agente.executar(t_conf)
    assert r_conf["status"] == "OK"

    # 1. Contexto com perfil 'colaborador' tentando dar baixa física
    t_baixa_colab = copy.deepcopy(
        _carregar_fixture("a1.registrar_baixa_fisica/baixa_no_prazo.tarefa.json")
    )
    t_baixa_colab["id_mensagem"] = "rbac-colab-01"
    t_baixa_colab["contexto"]["perfil"] = "colaborador"
    r_rbac1 = agente.executar(t_baixa_colab)
    assert r_rbac1["status"] == "ERRO"
    assert r_rbac1["erro"]["codigo"] == "NAO_AUTORIZADO"

    # 2. Contexto 'rh' mas informando usuário de login com perfil COLABORADOR
    t_baixa_user = copy.deepcopy(
        _carregar_fixture("a1.registrar_baixa_fisica/baixa_no_prazo.tarefa.json")
    )
    t_baixa_user["id_mensagem"] = "rbac-user-02"
    t_baixa_user["payload"]["recebido_fisico_por"] = "carla.souza"
    r_rbac2 = agente.executar(t_baixa_user)
    assert r_rbac2["status"] == "ERRO"
    assert r_rbac2["erro"]["codigo"] == "NAO_AUTORIZADO"

    # 3. Erro interno não deve vazar URLs/senhas/chaves no payload de erro
    class _RepoExplosivo(RepositorioA1Memoria):
        def conferir_colaborador_ativo(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                "Falha em postgresql://admin:segredo123@db:5432/sinergia chave gsk_abc123SECRET"
            )

    agente_erro = AgenteA1(db=_RepoExplosivo(), llm=MockLLM())
    t_err = copy.deepcopy(_carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json"))
    t_err["id_mensagem"] = "erro-sanitizado-01"
    r_err = agente_erro.executar(t_err)
    assert r_err["status"] == "ERRO"
    assert r_err["erro"]["codigo"] == "ERRO_INTERNO_A1"
    assert "segredo123" not in r_err["erro"]["mensagem"]
    assert "gsk_abc123SECRET" not in r_err["erro"]["mensagem"]


def test_integridade_duplicidade_sha256_sobreposicao_inativo_e_concorrencia(
    tmp_path: Path,
) -> None:
    """Valida duplicidade por SHA-256, sobreposição, colaborador inativo e idempotência concorrente."""
    repo = RepositorioA1Memoria()
    mock_llm = MockLLM()
    agente = AgenteA1(
        db=repo,
        llm=mock_llm,
        config=A1Config(groq_api_key="", politica_extrair_cid=False),
    )

    # Copia o mesmo PNG canônico para outro caminho/nome
    origem_png = Path(".storage/atestados/atestado_carla_105.png")
    copia_png = tmp_path / "copia_carla_outro_nome.png"
    copia_png.write_bytes(origem_png.read_bytes())
    sha_copia = hashlib.sha256(copia_png.read_bytes()).hexdigest()

    t_doc1 = copy.deepcopy(
        _carregar_fixture("a1.processar_documento/carla_divergencia_dias.tarefa.json")
    )
    t_doc1["id_mensagem"] = "int-doc-01"
    r_doc1 = agente.executar(t_doc1)
    assert r_doc1["status"] == "PRECISA_HUMANO"
    assert mock_llm.chamadas == 1

    # Envia o mesmo arquivo com outro arquivo_uri -> deve detectar duplicidade por SHA-256 sem chamar LLM
    t_doc2 = copy.deepcopy(t_doc1)
    t_doc2["id_mensagem"] = "int-doc-02"
    t_doc2["payload"] = {
        "arquivo_uri": str(copia_png),
        "mime": "image/png",
        "sha256": sha_copia,
    }
    r_doc2 = agente.executar(t_doc2)
    assert r_doc2["status"] == "PRECISA_HUMANO"
    assert "já foi recebido" in r_doc2["saidas"][0]["texto"]
    assert mock_llm.chamadas == 1

    # Confirma o primeiro atestado (09/11 a 11/11)
    t_conf = copy.deepcopy(_carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json"))
    t_conf["id_mensagem"] = "int-conf-01"
    r_conf = agente.executar(t_conf)
    assert r_conf["status"] == "OK"

    # Novo atestado diferente no período 10/11 a 12/11 -> detecta sobreposição
    img_sobrep = tmp_path / "atestado_sobreposto.png"
    im = Image.new("RGB", (620, 260), "white")
    draw = ImageDraw.Draw(im)
    draw.text(
        (20, 30),
        "ATESTADO MEDICO\n"
        "Atesto que a Sra. Carla Souza necessita de 3 dias\n"
        "de afastamento de 10/11/2026 a 12/11/2026.\n"
        "Belo Horizonte - MG, 10 de novembro de 2026.\n"
        "Dr. Lucas - CRM/MG 554433",
        fill="black",
    )
    im.save(img_sobrep)
    t_sobrep = copy.deepcopy(t_doc1)
    t_sobrep["id_mensagem"] = "int-sobrep-01"
    t_sobrep["payload"] = {"arquivo_uri": str(img_sobrep), "mime": "image/png"}
    r_sobrep = agente.executar(t_sobrep)
    campos_div = [d["campo"] for d in r_sobrep["payload"]["divergencias"]]
    assert "sobreposicao" in campos_div

    # Colaborador inativo no período -> além da mensagem ao colaborador, envia saída para perfil 'rh'
    t_inativo = copy.deepcopy(t_conf)
    t_inativo["id_mensagem"] = "int-inativo-01"
    t_inativo["payload"] = {
        "atestado_confirmado": {
            "periodo": {"inicio": "2024-01-01", "fim": "2024-01-03"},
            "data_emissao": "2024-01-01",
        },
        "campos_corrigidos": [],
    }
    r_inat = agente.executar(t_inativo)
    assert r_inat["estado_proposto"] == "NOVO"
    assert any(s.get("para", {}).get("perfil") == "rh" for s in r_inat["saidas"])

    # Idempotência concorrente em múltiplas threads com o mesmo id_mensagem
    mock_conc = MockLLM()
    agente_conc = AgenteA1(db=RepositorioA1Memoria(), llm=mock_conc)
    t_conc = copy.deepcopy(_carregar_fixture("a1.conversar/falta_arquivo.tarefa.json"))
    t_conc["id_mensagem"] = "conc-msg-01"
    with ThreadPoolExecutor(max_workers=4) as pool:
        respostas = list(pool.map(lambda _: agente_conc.executar(t_conc), range(4)))
    assert all(r == respostas[0] for r in respostas)
    assert mock_conc.chamadas == 1


def test_velocidade_pdf_texto_nativo_fast_path_e_extrator_municipio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Garante que PDF com camada textual nativa não aciona RapidOCR e extrai municípios variados."""
    pdf = tmp_path / "atestado_digital.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=350)
    page.insert_text(
        (40, 60),
        (
            "ATESTADO MEDICO\n\n"
            "Atesto que Carla Souza necessita de 2 dias de afastamento\n"
            "de 10/11/2026 a 11/11/2026.\n\n"
            "Belo Horizonte - MG, 10 de novembro de 2026.\n"
            "Dr. Marcos Lima - CRM/MG 123456"
        ),
        fontsize=16,
    )
    doc.save(pdf)
    doc.close()

    def _nao_deve_rodar_rapidocr(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("RapidOCR não deveria ser invocado para PDF com texto nativo.")

    monkeypatch.setattr(ocr, "_executar_rapidocr", _nao_deve_rodar_rapidocr)
    res_ocr = processar_arquivo_atestado(str(pdf), mime="application/pdf")
    assert res_ocr.legibilidade == "ok"
    assert res_ocr.confianca_media == 1.0

    mock_llm = MockLLM()
    extraido, _ = extrair_atestado_via_llm(res_ocr, llm=mock_llm)
    assert extraido.municipio == "Belo Horizonte"
    assert extraido.profissional.uf == "MG"


def test_migracao_sql_003_modelo_fisico_guia_v07() -> None:
    """Valida que a migração 003 define as 5 tabelas v0.7, RLS, views e mantém isolamento."""
    migracao = Path("db/schema/003_alinhar_afastamento_guia_v07.sql")
    assert migracao.is_file()
    sql = migracao.read_text(encoding="utf-8")

    assert "ALTER TABLE afastamento.atestado" in sql
    assert "ADD COLUMN IF NOT EXISTS id_caso" in sql
    assert "ADD COLUMN IF NOT EXISTS tipo_documento" in sql
    assert "ADD COLUMN IF NOT EXISTS prazos_legais" in sql
    assert "CREATE TABLE IF NOT EXISTS afastamento.atestado_arquivo" in sql
    assert "CREATE TABLE IF NOT EXISTS afastamento.atestado_campo" in sql
    assert "CREATE TABLE IF NOT EXISTS afastamento.baixa_fisica" in sql
    assert "CREATE TABLE IF NOT EXISTS afastamento.cid_restrito" in sql
    assert "ALTER TABLE afastamento.cid_restrito ENABLE ROW LEVEL SECURITY" in sql
    assert "CREATE OR REPLACE VIEW afastamento.vw_atestado_sem_cid" in sql
    assert "CREATE OR REPLACE VIEW afastamento.vw_baixa_fisica" in sql
    assert not re.search(r"\borquestracao\.", sql, re.IGNORECASE)


def test_persistencia_5_tabelas_v07_stateless_auditoria_e_cid_restrito(
    tmp_path: Path,
) -> None:
    """Valida persistência completa nas 5 tabelas v0.7, retomada stateless e isolamento de CID."""
    repo = RepositorioA1Memoria()
    cfg_restrito = A1Config(groq_api_key="", politica_extrair_cid=True)

    # 1. Processamento com baixa confiança no número do CRM e CID J06.9
    mock_extrator = MockLLM(
        extracao_padrao=AtestadoExtraido(
            tipo_documento="atestado_medico",
            legibilidade="ok",
            nome_no_documento="Carla Souza",
            periodo=PeriodoAfastamento(
                inicio=date(2026, 11, 9), fim=date(2026, 11, 11)
            ),
            data_emissao=date(2026, 11, 9),
            profissional=ProfissionalExtraido(
                conselho="CRM", numero="123456", uf="SP", nome="Dr. Marcos Lima"
            ),
            cid="J06.9",
            municipio="Campinas",
            confianca_campos={
                "nome_no_documento": 0.96,
                "periodo.inicio": 0.95,
                "periodo.fim": 0.95,
                "data_emissao": 0.94,
                "profissional.conselho": 0.96,
                "profissional.numero": 0.41,
                "profissional.uf": 0.96,
                "cid": 0.93,
                "municipio": 0.91,
            },
        )
    )
    agente = AgenteA1(db=repo, llm=mock_extrator, config=cfg_restrito)

    t_doc = copy.deepcopy(
        _carregar_fixture("a1.processar_documento/carla_divergencia_dias.tarefa.json")
    )
    t_doc["id_mensagem"] = "db5t-doc-01"
    r_doc = agente.executar(t_doc)
    assert r_doc["status"] == "PRECISA_HUMANO"

    pendente = repo.obter_ultimo_atestado_pendente(42)
    assert pendente is not None
    id_atestado = pendente.id_atestado
    assert pendente.id_caso == 105
    assert pendente.tipo_documento == "atestado_medico"
    # CID isolado em afastamento.cid_restrito, jamais exposto na tabela principal
    assert pendente.codigo_cid is None
    cid_restrito = repo.obter_cid_restrito(id_atestado)
    assert cid_restrito is not None
    assert cid_restrito.codigo_cid == "J06.9"

    arquivos = repo.listar_arquivos_atestado(id_atestado)
    assert len(arquivos) == 1
    assert arquivos[0].sha256 == t_doc["payload"]["sha256"]
    assert arquivos[0].mime == "image/png"

    campos_ext = {c.nome_campo: c for c in repo.listar_campos_atestado(id_atestado)}
    assert campos_ext["profissional.numero"].vazio_por_baixa_confianca is True
    assert campos_ext["profissional.numero"].valor_extraido is None
    assert campos_ext["periodo.fim"].valor_extraido == "2026-11-11"
    assert campos_ext["cid"].valor_extraido == "J06.9"

    # 2. Simula reinício completo (nova instância sem cache em memória) + correção manual
    agente_reiniciado = AgenteA1(db=repo, llm=mock_extrator, config=cfg_restrito)
    t_conf = copy.deepcopy(
        _carregar_fixture("a1.confirmar/carla_confirmado_ia.tarefa.json")
    )
    t_conf["id_mensagem"] = "db5t-conf-02"
    t_conf["payload"] = {
        "atestado_confirmado": {
            "periodo": {"inicio": "2026-11-09", "fim": "2026-11-24"},
            "profissional": {"conselho": "CRM", "numero": "123456", "uf": "SP"},
        },
        "campos_corrigidos": ["periodo.fim", "profissional.numero"],
    }
    r_conf = agente_reiniciado.executar(t_conf)
    assert r_conf["status"] == "OK"
    assert r_conf["payload"]["origem"] == "MANUAL"
    assert r_conf["payload"]["prazos_legais"] == ["inss_16_dias"]

    at_conf = repo.obter_atestado(id_atestado)
    assert at_conf is not None
    assert at_conf.prazos_legais == ["inss_16_dias"]
    assert at_conf.codigo_cid is None

    campos_conf = {c.nome_campo: c for c in repo.listar_campos_atestado(id_atestado)}
    assert campos_conf["periodo.fim"].valor_extraido == "2026-11-11"
    assert campos_conf["periodo.fim"].valor_confirmado == "2026-11-24"
    assert campos_conf["periodo.fim"].origem == "MANUAL"
    assert campos_conf["profissional.numero"].valor_confirmado == "123456"
    assert campos_conf["profissional.numero"].origem == "MANUAL"
    assert campos_conf["nome_no_documento"].valor_confirmado == "Carla Souza"
    assert campos_conf["nome_no_documento"].origem == "IA"
    assert campos_conf["cid"].valor_confirmado == "J06.9"

    # 3. Baixa física popula também afastamento.baixa_fisica
    t_baixa = copy.deepcopy(
        _carregar_fixture("a1.registrar_baixa_fisica/baixa_no_prazo.tarefa.json")
    )
    t_baixa["id_mensagem"] = "db5t-baixa-03"
    t_baixa["payload"]["id_atestado"] = id_atestado
    r_baixa = agente_reiniciado.executar(t_baixa)
    assert r_baixa["status"] == "OK"
    baixa_db = repo.obter_baixa_fisica(id_atestado)
    assert baixa_db is not None
    assert baixa_db.recebido_por == "rh.ana"
    assert baixa_db.id_usuario_rh == 7

    # 4. Janela de 60 dias enxerga CID isolado em cid_restrito
    assert (
        repo.calcular_dias_afastamento_ultimos_60_dias(
            42, date(2026, 11, 30), cid="J06.9"
        )
        == 16
    )

    # 5. Política nao_extrair não cria registro em cid_restrito
    repo_sem_cid = RepositorioA1Memoria()
    agente_sem_cid = AgenteA1(
        db=repo_sem_cid,
        llm=MockLLM(),
        config=A1Config(groq_api_key="", politica_extrair_cid=False),
    )
    t_sem_cid = copy.deepcopy(t_doc)
    t_sem_cid["id_mensagem"] = "db5t-semcid-04"
    agente_sem_cid.executar(t_sem_cid)
    pend2 = repo_sem_cid.obter_ultimo_atestado_pendente(42)
    assert pend2 is not None
    assert repo_sem_cid.obter_cid_restrito(pend2.id_atestado) is None


def test_repositorio_postgres_queries_5_tabelas_guia_v07(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valida queries SQL do RepositorioA1Postgres nas 5 tabelas v0.7 sem violar matriz C.2."""
    queries: list[str] = []
    agora = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)

    class FakeCursor5T:
        def __init__(self) -> None:
            self._sql = ""

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            self._sql = sql
            queries.append(sql)

        def fetchone(self) -> dict[str, Any] | None:
            s = self._sql.lower()
            if "from afastamento.atestado_arquivo" in s:
                return {
                    "id_atestado": 311,
                    "id_caso": 105,
                    "id_colaborador": 42,
                    "id_profissional": 99,
                    "codigo_cid": None,
                    "id_municipio": 3509502,
                    "tipo_documento": "atestado_medico",
                    "data_emissao": date(2026, 11, 9),
                    "inicio_afastamento": agora,
                    "fim_afastamento": agora,
                    "arquivo_uri": ".storage/atestados/carla.png",
                    "origem_dados": "IA",
                    "status": "PENDENTE",
                    "postado_em": agora,
                    "prazo_entrega_fisica": None,
                    "prazos_legais": [],
                    "recebido_fisico_em": None,
                    "recebido_fisico_por": None,
                }
            if "from cadastro.usuario" in s:
                return {"id_usuario": 7, "perfil": "RH"}
            if "into afastamento.atestado_campo" in s:
                return {
                    "id_campo": 1,
                    "id_atestado": 311,
                    "id_caso": 105,
                    "nome_campo": "periodo.fim",
                    "valor_extraido": "2026-11-11",
                    "valor_confirmado": "2026-11-24",
                    "confianca": 0.95,
                    "origem": "MANUAL",
                    "vazio_por_baixa_confianca": False,
                    "atualizado_em": agora,
                }
            if "into afastamento.atestado" in s or "update afastamento.atestado" in s:
                return {
                    "id_atestado": 311,
                    "id_caso": 105,
                    "id_colaborador": 42,
                    "id_profissional": 99,
                    "codigo_cid": None,
                    "id_municipio": 3509502,
                    "tipo_documento": "atestado_medico",
                    "data_emissao": date(2026, 11, 9),
                    "inicio_afastamento": agora,
                    "fim_afastamento": agora,
                    "arquivo_uri": ".storage/atestados/carla.png",
                    "origem_dados": "IA",
                    "status": "CONFIRMADO",
                    "postado_em": agora,
                    "prazo_entrega_fisica": date(2026, 11, 16),
                    "prazos_legais": ["inss_16_dias"],
                    "recebido_fisico_em": agora,
                    "recebido_fisico_por": 7,
                }
            return None

        def fetchall(self) -> list[dict[str, Any]]:
            return []

    class FakeConn5T:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def cursor(self) -> FakeCursor5T:
            return FakeCursor5T()

        def commit(self) -> None:
            return None

    import sinergia.agentes.a1_recepcao.repositorio as repo_mod

    monkeypatch.setattr(repo_mod.psycopg, "connect", lambda *a, **kw: FakeConn5T())
    repo_pg = RepositorioA1Postgres("postgresql://fake:fake@localhost:5432/sinergia")

    dup = repo_pg.verificar_duplicidade_arquivo(
        ".storage/atestados/outro.png", id_colaborador=42, sha256="a" * 64
    )
    assert dup is not None and dup.id_atestado == 311

    repo_pg.salvar_atestado_pendente(
        id_colaborador=42,
        id_profissional=99,
        id_municipio=3509502,
        data_emissao=date(2026, 11, 9),
        inicio_afastamento=date(2026, 11, 9),
        fim_afastamento=date(2026, 11, 11),
        arquivo_uri=".storage/atestados/carla.png",
        codigo_cid="J06.9",
        arquivo_sha256="b" * 64,
        id_caso=105,
        tipo_documento="atestado_medico",
        arquivo_mime="image/png",
        politica_cid_restrito=True,
    )
    campos = repo_pg.salvar_campos_atestado(
        311,
        [{"nome_campo": "periodo.fim", "valor_confirmado": "2026-11-24", "origem": "MANUAL"}],
        id_caso=105,
    )
    assert len(campos) == 1
    repo_pg.confirmar_atestado_no_banco(
        311,
        inicio_afastamento=date(2026, 11, 9),
        fim_afastamento=date(2026, 11, 24),
        data_emissao=date(2026, 11, 9),
        origem_dados="MANUAL",
        prazo_entrega_fisica=date(2026, 11, 27),
        codigo_cid="J06.9",
        prazos_legais=["inss_16_dias"],
        politica_cid_restrito=True,
    )
    repo_pg.registrar_baixa_fisica_no_banco(
        311, "rh.ana", "2026-11-13T10:20:00-03:00"
    )
    repo_pg.calcular_dias_afastamento_ultimos_60_dias(42, date(2026, 11, 30), cid="J06.9")

    unificado = "\n".join(queries)
    assert "afastamento.atestado_arquivo" in unificado
    assert "afastamento.atestado_campo" in unificado
    assert "afastamento.baixa_fisica" in unificado
    assert "afastamento.cid_restrito" in unificado
    for sql in queries:
        assert not re.search(r"\borquestracao\.", sql, re.IGNORECASE)
        assert not re.search(r"\bbem_estar\.", sql, re.IGNORECASE)
        assert not re.search(r"\bapoio\.", sql, re.IGNORECASE)
        assert not re.search(r"\brag\.", sql, re.IGNORECASE)


def test_fastapi_modo_local_por_variaveis_ambiente(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garante que A1_MODO_REPO=memoria e A1_MODO_LLM=mock sobem API sem Postgres/Groq."""
    monkeypatch.setenv("A1_MODO_REPO", "memoria")
    monkeypatch.setenv("A1_MODO_LLM", "mock")
    monkeypatch.setenv("DATABASE_URL", "postgresql://inativo:inativo@127.0.0.1:59999/inativo")
    monkeypatch.setenv("GROQ_API_KEY", "")

    client = TestClient(criar_app_a1())

    res_health = client.get("/health")
    assert res_health.status_code == 200
    body_health = res_health.json()
    assert body_health["status"] == "ok"
    assert body_health["protocolo"] == "sinergia/1"
    assert body_health["modo_repositorio"] == "memoria"
    assert body_health["modo_llm"] == "mock"

    tarefa = copy.deepcopy(_carregar_fixture("a1.conversar/falta_arquivo.tarefa.json"))
    tarefa["id_mensagem"] = "api-env-local-01"
    res_exec = client.post("/executar", json=tarefa)
    assert res_exec.status_code == 200
    assert res_exec.json()["status"] == "OK"
