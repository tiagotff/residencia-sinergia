"""Testes unitários e de UI (AppTest) para o painel Streamlit A0 <-> A1."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

from sinergia.agentes.a1_recepcao import (
    A1Config,
    AgenteA1,
    ClienteCidWHO,
    MockLLM,
    RepositorioA1Memoria,
    criar_app_a1,
)
from sinergia.agentes.a1_recepcao.painel import (
    ClienteConexaoA0A1,
    aplicar_resultado_ao_caso,
    calcular_campos_corrigidos,
    carregar_fixtures_oficiais,
    criar_caso_padrao,
    gerar_preview_documento_png,
    limpar_uploads_temporarios,
    montar_tarefa_baixa_fisica,
    montar_tarefa_confirmar,
    montar_tarefa_conversar,
    montar_tarefa_processar_documento,
    resumir_diff_edicao,
    salvar_upload_temporario,
    verificar_catalogo_sintetico,
    verificar_variaveis_env,
    verificar_who_icd_api,
)
from sinergia.agentes.a1_recepcao.painel.tema import (
    renderizar_badge_origem_cid,
    renderizar_card_origem_cid,
)

APP_PAINEL_PATH = str(
    Path(__file__).resolve().parents[2]
    / "src"
    / "sinergia"
    / "agentes"
    / "a1_recepcao"
    / "painel"
    / "app.py"
)


def test_diagnostico_env_mascara_segredos_e_valida_catalogo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garante mascaramento de segredos no diagnóstico .env e integridade do catálogo."""
    segredo_groq = "gsk_super_secreto_nao_vazar_12345"
    segredo_jwt = "jwt_ultra_secreto_67890"
    segredo_who = "who_secret_super_sigiloso_999"
    monkeypatch.setenv("GROQ_API_KEY", segredo_groq)
    monkeypatch.setenv("JWT_SECRET", segredo_jwt)
    monkeypatch.setenv("WHO_ICD_CLIENT_ID", "who_client_abc")
    monkeypatch.setenv("WHO_ICD_CLIENT_SECRET", segredo_who)
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://sinergia:senhaSuperSecreta@localhost:5432/sinergia"
    )

    cfg = A1Config.from_env()
    diag_env = verificar_variaveis_env(cfg)
    serializado = diag_env.model_dump_json()

    assert segredo_groq not in serializado
    assert segredo_jwt not in serializado
    assert segredo_who not in serializado
    assert "senhaSuperSecreta" not in serializado
    assert "preenchido" in diag_env.detalhes["a1"]["GROQ_API_KEY"]
    assert "preenchido" in diag_env.detalhes["a1"]["WHO_ICD_CLIENT_SECRET"]
    assert "preenchido" in diag_env.detalhes["a0"]["JWT_SECRET"]

    diag_who = verificar_who_icd_api(cfg, testar_rede=False)
    assert diag_who.status == "ok"
    assert segredo_who not in diag_who.model_dump_json()

    diag_cat = verificar_catalogo_sintetico()
    assert diag_cat.status == "ok"
    assert diag_cat.detalhes["integro"] is True


def test_cliente_conexao_in_process_http_a1_e_inspetor_lgpd() -> None:
    """Valida transporte in_process, HTTP FastAPI A1, idempotência e RLS LGPD Art. 11."""
    cfg = A1Config(groq_api_key="", politica_extrair_cid=True)
    repo = RepositorioA1Memoria()
    cache_idem: dict[str, dict] = {}
    pendentes: dict = {}

    cliente_inproc = ClienteConexaoA0A1(
        modo_transporte="in_process",
        config=cfg,
        modo_repo="memoria",
        modo_llm="mock",
        retornar_modelo=True,
        repositorio_memoria=repo,
        cache_idempotencia=cache_idem,
        extracoes_pendentes=pendentes,
    )

    caso = criar_caso_padrao(105)

    # 1. a1.conversar
    t_conv = montar_tarefa_conversar(caso, "Preciso enviar um atestado médico de 5 dias.")
    tr_conv = cliente_inproc.enviar_tarefa(t_conv)
    assert tr_conv.schema_tarefa_valido is True
    assert tr_conv.schema_resultado_valido is True
    assert tr_conv.retornou_modelo_pydantic is True
    assert tr_conv.resultado["status"] == "OK"
    caso = aplicar_resultado_ao_caso(caso, tr_conv)

    # 2. a1.processar_documento (PNG real do caso 105 com CID J06.9)
    png_path = Path(".storage/atestados/atestado_carla_105.png")
    sha = hashlib.sha256(png_path.read_bytes()).hexdigest()
    t_doc = montar_tarefa_processar_documento(caso, str(png_path), "image/png", sha)
    tr_doc = cliente_inproc.enviar_tarefa(t_doc)
    assert tr_doc.schema_resultado_valido is True
    assert tr_doc.resultado["status"] == "PRECISA_HUMANO"
    assert tr_doc.resultado["estado_proposto"] == "RECEBIDO"
    caso = aplicar_resultado_ao_caso(caso, tr_doc)
    assert caso.aguardando_hitl is True
    assert caso.ultima_revisao_colaborador is not None
    assert "resumo_markdown" in caso.ultima_revisao_colaborador
    assert len(caso.ultima_revisao_colaborador["itens"]) >= 10

    # Replay idempotente da mesma id_mensagem
    tr_doc_replay = cliente_inproc.enviar_tarefa(t_doc, eh_replay=True)
    assert tr_doc_replay.resultado == tr_doc.resultado
    assert len(repo.atestados) == 1

    # 3. a1.confirmar
    assert caso.ultimo_atestado_extraido is not None
    t_conf = montar_tarefa_confirmar(caso, caso.ultimo_atestado_extraido, [])
    tr_conf = cliente_inproc.enviar_tarefa(t_conf)
    assert tr_conf.resultado["status"] == "OK"
    assert tr_conf.resultado["estado_proposto"] == "VALIDADO"
    caso = aplicar_resultado_ao_caso(caso, tr_conf)
    assert caso.aguardando_hitl is False
    assert caso.id_atestado_confirmado is not None

    # 4. a1.registrar_baixa_fisica
    t_baixa = montar_tarefa_baixa_fisica(
        caso,
        id_atestado=caso.id_atestado_confirmado,
        recebido_por="rh.ana",
        data_hora_iso="2026-11-13T10:20:00-03:00",
    )
    tr_baixa = cliente_inproc.enviar_tarefa(t_baixa)
    assert tr_baixa.resultado["status"] == "OK"
    caso = aplicar_resultado_ao_caso(caso, tr_baixa)
    assert caso.baixa_registrada is True

    # Inspetor das 5 tabelas: RH enxerga codigo_cid/nome_doenca e cid_restrito, Gestor/A2 não enxerga
    visao_rh = cliente_inproc.inspecionar_tabelas_afastamento(perfil_acesso="A1_RH")
    assert len(visao_rh["atestado"]) == 1
    assert visao_rh["atestado"][0]["codigo_cid"] == "J06.9"
    assert visao_rh["atestado"][0]["nome_doenca"] == "Infecção aguda das vias aéreas superiores"
    assert len(visao_rh["atestado_arquivo"]) == 1
    assert len(visao_rh["atestado_campo"]) > 0
    assert len(visao_rh["baixa_fisica"]) == 1
    assert len(visao_rh["cid_restrito"]) == 1
    assert visao_rh["cid_restrito"][0]["codigo_cid"] == "J06.9"
    assert visao_rh["cid_restrito"][0]["nome_doenca"] == "Infecção aguda das vias aéreas superiores"

    visao_gestor = cliente_inproc.inspecionar_tabelas_afastamento(perfil_acesso="GESTOR_A2")
    assert visao_gestor["cid_restrito"] == []
    assert "codigo_cid" not in visao_gestor["atestado"][0]
    assert "nome_doenca" not in visao_gestor["atestado"][0]

    # Transporte HTTP FastAPI A1 via TestClient
    repo_http = RepositorioA1Memoria()
    app_fastapi = criar_app_a1(
        agente=AgenteA1(db=repo_http, llm=MockLLM(), config=cfg)
    )
    test_client = TestClient(app_fastapi)
    cliente_http = ClienteConexaoA0A1(
        modo_transporte="http_a1",
        config=cfg,
        url_a1_http="",
        http_client=test_client,
    )
    tr_http = cliente_http.enviar_tarefa(t_conv)
    assert tr_http.schema_resultado_valido is True
    assert tr_http.resultado["status"] == "OK"
    assert tr_http.resultado["payload"]["faltando"] == ["arquivo"]


def test_emulador_a0_multi_caso_uploads_e_campos_corrigidos(tmp_path: Path) -> None:
    """Valida uploads isolados em uploads_painel/, preview PNG/PDF e diff de campos."""
    # Cria arquivo fora de uploads_painel para garantir que limpeza não o apague
    arquivo_canonico = tmp_path / "canonico.png"
    im = Image.new("RGB", (240, 120), "white")
    ImageDraw.Draw(im).text((20, 40), "CANONICO", fill="black")
    im.save(arquivo_canonico)

    caminho_up, mime_up, sha_up = salvar_upload_temporario(
        "../../meu atestado!.png",
        arquivo_canonico.read_bytes(),
        tmp_path,
    )
    assert Path(caminho_up).parent.name == "uploads_painel"
    assert mime_up == "image/png"
    assert len(sha_up) == 64

    preview_png = gerar_preview_documento_png(caminho_up)
    assert preview_png is not None and preview_png.startswith(b"\x89PNG")

    # Preview de PDF via PyMuPDF
    pdf_path = tmp_path / "uploads_painel" / "teste.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((30, 60), "ATESTADO PDF")
    doc.save(pdf_path)
    doc.close()
    preview_pdf = gerar_preview_documento_png(pdf_path)
    assert preview_pdf is not None and preview_pdf.startswith(b"\x89PNG")

    removidos = limpar_uploads_temporarios(tmp_path)
    assert removidos == 2
    assert arquivo_canonico.is_file()

    extraido = {
        "tipo_documento": "atestado_medico",
        "nome_no_documento": "Carla Souza",
        "periodo": {"inicio": "2026-11-09", "fim": "2026-11-11"},
        "data_emissao": "2026-11-09",
        "profissional": {"conselho": "CRM", "numero": "123456", "uf": "SP"},
        "cid": None,
        "municipio": "Campinas",
    }
    assert calcular_campos_corrigidos(extraido, extraido) == []

    editado = {
        **extraido,
        "periodo": {"inicio": "2026-11-09", "fim": "2026-11-12"},
        "profissional": {
            "conselho": "CRM",
            "numero": "654321",
            "uf": "SP",
            "nome": "Dra. Helena Costa",
        },
    }
    assert calcular_campos_corrigidos(extraido, editado) == [
        "periodo.fim",
        "profissional.numero",
        "profissional.nome",
    ]
    diffs_resumo = resumir_diff_edicao(extraido, editado)
    assert [d["campo"] for d in diffs_resumo] == [
        "periodo.fim",
        "profissional.numero",
        "profissional.nome",
    ]
    assert diffs_resumo[1]["antes_ia"] == "123456"
    assert diffs_resumo[1]["depois_manual"] == "654321"
    assert len(carregar_fixtures_oficiais()) == 5

    caso_106 = criar_caso_padrao(106, inicio_informado=date(2026, 11, 16), dias_informados=2)
    assert caso_106.thread_id == "chat-42-106"


def test_apptest_renderizacao_e_fluxo_basico_streamlit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executa o app Streamlit via AppTest cobrindo espelho HITL, edição guiada e modos."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("POLITICA_CID", "nao_extrair")
    monkeypatch.setenv("A1_MODO_REPO", "memoria")

    at = AppTest.from_file(APP_PAINEL_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 4
    assert at.radio(key="sb_modo_repo").value == "memoria"

    # 1. Envia a1.conversar
    at.button(key="btn_conv_105").click().run()
    assert not at.exception
    assert any("Pode me enviar uma foto" in md.value for md in at.markdown)

    # 2. Processa documento canônico da Carla -> exibe Espelho de Validação + alerta divergência
    at.button(key="btn_proc_105").click().run()
    assert not at.exception
    assert len(at.warning) >= 1
    assert any("O que o Agente A1 leu no documento" in md.value for md in at.markdown)
    assert any("Dados lidos no documento" in md.value for md in at.markdown)

    # 3. Confirma em 1 clique no Espelho (origem IA)
    at.button(key="btn_conf_105").click().run()
    assert not at.exception
    assert at.session_state["painel_casos"][105].estado_atual == "VALIDADO"

    # 4. Cria Caso 106, processa e testa edição guiada (origem MANUAL)
    at.button(key="sim_btn_novo_caso").click().run()
    assert not at.exception
    assert at.session_state["painel_id_caso_ativo"] == 106

    at.button(key="btn_proc_106").click().run()
    assert not at.exception
    at.toggle(key="toggle_editar_106").set_value(True).run()
    assert not at.exception
    at.text_input(key="hitl_num_106").set_value("654321").run()
    at.button(key="btn_conf_editado_106").click().run()
    assert not at.exception
    caso_106 = at.session_state["painel_casos"][106]
    assert caso_106.estado_atual == "VALIDADO"
    assert caso_106.trocas[-1].resultado["payload"]["origem"] == "MANUAL"

    # 5. Alterna para Modo Apresentação sem erros
    at.radio(key="painel_modo_visao").set_value("apresentacao").run()
    assert not at.exception
    at.radio(key="painel_modo_visao").set_value("engenharia").run()
    assert not at.exception

    # 6. Testa replay idempotente e diagnóstico pré-voo
    at.button(key="btn_replay_106").click().run()
    assert not at.exception
    assert any("Idempotência 100% íntegra" in s.value for s in at.success)

    at.button(key="diag_btn_run").click().run()
    assert not at.exception

    # 7. Testa consulta de CID na aba Inspetor (indicativo visual Banco Local)
    at.text_input(key="insp_cid_consulta").set_value("M54.5")
    at.button(key="insp_btn_consultar_cid").click().run()
    assert not at.exception
    assert at.session_state["painel_consulta_cid"]["origem"] == "BANCO_LOCAL"
    assert any("Banco Local" in md.value for md in at.markdown)


def test_painel_api_who_cid_origem_banco_vs_api_externa_in_process_e_http() -> None:
    """Valida fallback WHO ICD-11 no painel e indicativo visual Banco Local vs API Externa."""
    chamadas_api: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/connect/token"):
            return httpx.Response(200, json={"access_token": "tok-painel", "expires_in": 3600})
        chamadas_api.append(url)
        if url.endswith("/codeinfo/ME84.2"):
            return httpx.Response(
                200,
                json={"code": "ME84.2", "stemId": "http://id.who.int/icd/release/11/2026-01/mms/777"},
            )
        if url.endswith("/mms/777"):
            return httpx.Response(
                200,
                json={
                    "code": "ME84.2",
                    "classKind": "category",
                    "chapter": "21",
                    "title": {"@language": "pt", "@value": "Dor lombar baixa"},
                    "definition": {
                        "@language": "pt",
                        "@value": "Condição definida como dor na região lombar.",
                    },
                },
            )
        return httpx.Response(404)

    cliente_who = ClienteCidWHO(
        base_url="https://id.who.int",
        client_id="cli-painel",
        client_secret="sec-painel",
        transport=httpx.MockTransport(handler),
    )
    cfg = A1Config(
        groq_api_key="",
        politica_extrair_cid=True,
        who_icd_enabled=True,
        who_icd_client_id="cli-painel",
        who_icd_client_secret="sec-painel",
    )
    repo = RepositorioA1Memoria()
    cliente = ClienteConexaoA0A1(
        modo_transporte="in_process",
        config=cfg,
        modo_repo="memoria",
        modo_llm="mock",
        repositorio_memoria=repo,
        cliente_cid=cliente_who,
    )

    # 1. Código existente no banco -> BANCO_LOCAL (0 chamadas externas)
    diag_local = cliente.consultar_cid("M54.5")
    assert diag_local["origem"] == "BANCO_LOCAL"
    assert len(chamadas_api) == 0
    assert "Banco Local" in renderizar_card_origem_cid(diag_local)
    assert "sinergia-badge--ok" in renderizar_badge_origem_cid(diag_local)

    # 2. Código CID-11 ausente -> API Externa OMS + cache-aside no banco
    diag_api = cliente.consultar_cid("ME84.2")
    assert diag_api["origem"] == "API_WHO_EXTERNA_CACHE_ASIDE"
    assert diag_api["nome_doenca"] == "Dor lombar baixa"
    assert diag_api["descricao"] == "Condição definida como dor na região lombar."
    assert len(chamadas_api) == 2
    assert "ME84.2" in repo.cids
    cid_info = repo.obter_cid("ME84.2")
    assert cid_info is not None
    assert cid_info["nome_doenca"] == "Dor lombar baixa"
    assert cid_info["descricao"] == "Condição definida como dor na região lombar."
    card_api = renderizar_card_origem_cid(diag_api)
    assert "API Externa OMS" in card_api
    assert "Nome da doença:" in card_api
    assert "Descrição:" in card_api
    assert "Salvo em afastamento.cid" in card_api
    assert "sinergia-badge--api" in renderizar_badge_origem_cid(diag_api)

    # 3. Segunda consulta do mesmo código -> vem do Banco Local (cache-aside)
    diag_cache = cliente.consultar_cid("ME84.2")
    assert diag_cache["origem"] == "BANCO_LOCAL"
    assert len(chamadas_api) == 2

    # 4. Fluxo HTTP FastAPI A1 propaga cabeçalhos X-Sinergia-Cid-* sem alterar schema sinergia/1
    repo_http = RepositorioA1Memoria()
    app_fastapi = criar_app_a1(
        agente=AgenteA1(
            db=repo_http,
            llm=MockLLM(),
            config=cfg,
            cliente_cid=cliente_who,
        )
    )
    cliente_http = ClienteConexaoA0A1(
        modo_transporte="http_a1",
        config=cfg,
        url_a1_http="",
        http_client=TestClient(app_fastapi),
    )
    caso = criar_caso_padrao(109)
    t_conf = montar_tarefa_confirmar(
        caso,
        {
            "tipo_documento": "atestado_medico",
            "nome_no_documento": "Carla Souza",
            "periodo": {"inicio": "2026-11-09", "fim": "2026-11-11"},
            "data_emissao": "2026-11-09",
            "profissional": {"conselho": "CRM", "numero": "123456", "uf": "SP"},
            "cid": "ME84.2",
            "municipio": "Campinas",
        },
        [],
    )
    tr_http = cliente_http.enviar_tarefa(t_conf)
    assert tr_http.schema_resultado_valido is True
    assert tr_http.resultado["status"] == "OK"
    assert tr_http.diagnostico_cid is not None
    assert tr_http.diagnostico_cid["origem"] == "API_WHO_EXTERNA_CACHE_ASIDE"
    assert tr_http.diagnostico_cid["nome_doenca"] == "Dor lombar baixa"
    assert tr_http.diagnostico_cid["descricao"] == "Condição definida como dor na região lombar."
    caso_atualizado = aplicar_resultado_ao_caso(caso, tr_http)
    assert caso_atualizado.ultimo_diagnostico_cid is not None
    assert caso_atualizado.ultimo_diagnostico_cid["origem"] == "API_WHO_EXTERNA_CACHE_ASIDE"


def test_apptest_inicia_com_postgres_por_padrao_quando_database_url_configurada(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garante que o seletor de repositório inicia em postgres quando DATABASE_URL está ativa."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.delenv("A1_MODO_REPO", raising=False)
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://sinergia:sinergia@localhost:5432/sinergia"
    )

    at = AppTest.from_file(APP_PAINEL_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert at.radio(key="sb_modo_repo").value == "postgres"


def test_card_campo_revisao_cid_exibe_nome_e_descricao_doenca() -> None:
    """Garante que o card de CID da etapa 3 exiba nome e descrição da doença."""
    from sinergia.agentes.a1_recepcao.painel.tema import (
        renderizar_card_campo_revisao,
        renderizar_detalhes_doenca_cid,
    )

    diag = {
        "codigo_cid": "ME84.2",
        "nome_doenca": "Dor lombar baixa",
        "descricao": "Condição definida como dor na região lombar.",
    }
    detalhes_html = renderizar_detalhes_doenca_cid(diag)
    assert "Nome da doença:" in detalhes_html
    assert "Dor lombar baixa" in detalhes_html
    assert "Descrição:" in detalhes_html
    assert "Condição definida como dor na região lombar." in detalhes_html

    item_cid = {
        "campo": "cid",
        "rotulo": "Código CID-10",
        "valor_exibicao": "ME84.2",
        "confianca": 0.96,
        "status": "lido_ok",
    }
    card_cid = renderizar_card_campo_revisao(item_cid, diag_cid=diag)
    assert "ME84.2" in card_cid
    assert "Dor lombar baixa" in card_cid
    assert "Condição definida como dor na região lombar." in card_cid

    item_outro = {
        "campo": "municipio",
        "rotulo": "Município de emissão",
        "valor_exibicao": "Campinas",
        "confianca": 0.95,
        "status": "lido_ok",
    }
    card_outro = renderizar_card_campo_revisao(item_outro, diag_cid=diag)
    assert "Dor lombar baixa" not in card_outro
    assert renderizar_detalhes_doenca_cid(None) == ""
