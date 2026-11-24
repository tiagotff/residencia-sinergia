"""Testes unitários e de UI (AppTest) para o painel Streamlit A0 <-> A1."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from streamlit.testing.v1 import AppTest

from sinergia.agentes.a1_recepcao import (
    A1Config,
    AgenteA1,
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
    salvar_upload_temporario,
    verificar_catalogo_sintetico,
    verificar_variaveis_env,
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
    monkeypatch.setenv("GROQ_API_KEY", segredo_groq)
    monkeypatch.setenv("JWT_SECRET", segredo_jwt)
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://sinergia:senhaSuperSecreta@localhost:5432/sinergia"
    )

    cfg = A1Config.from_env()
    diag_env = verificar_variaveis_env(cfg)
    serializado = diag_env.model_dump_json()

    assert segredo_groq not in serializado
    assert segredo_jwt not in serializado
    assert "senhaSuperSecreta" not in serializado
    assert "preenchido" in diag_env.detalhes["a1"]["GROQ_API_KEY"]
    assert "preenchido" in diag_env.detalhes["a0"]["JWT_SECRET"]

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

    # Inspetor das 5 tabelas: RH enxerga cid_restrito, Gestor/A2 não enxerga
    visao_rh = cliente_inproc.inspecionar_tabelas_afastamento(perfil_acesso="A1_RH")
    assert len(visao_rh["atestado"]) == 1
    assert len(visao_rh["atestado_arquivo"]) == 1
    assert len(visao_rh["atestado_campo"]) > 0
    assert len(visao_rh["baixa_fisica"]) == 1
    assert len(visao_rh["cid_restrito"]) == 1
    assert visao_rh["cid_restrito"][0]["codigo_cid"] == "J06.9"

    visao_gestor = cliente_inproc.inspecionar_tabelas_afastamento(perfil_acesso="GESTOR_A2")
    assert visao_gestor["cid_restrito"] == []
    assert "codigo_cid" not in visao_gestor["atestado"][0]

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
        "profissional": {"conselho": "CRM", "numero": "654321", "uf": "SP"},
    }
    assert calcular_campos_corrigidos(extraido, editado) == [
        "periodo.fim",
        "profissional.numero",
    ]
    assert len(carregar_fixtures_oficiais()) == 5

    caso_106 = criar_caso_padrao(106, inicio_informado=date(2026, 11, 16), dias_informados=2)
    assert caso_106.thread_id == "chat-42-106"


def test_apptest_renderizacao_e_fluxo_basico_streamlit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executa o app Streamlit via AppTest (offline) cobrindo fluxo completo HITL."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("POLITICA_CID", "nao_extrair")

    at = AppTest.from_file(APP_PAINEL_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 4

    # 1. Envia a1.conversar
    at.button(key="btn_conv_105").click().run()
    assert not at.exception
    assert any("Pode me enviar uma foto" in md.value for md in at.markdown)

    # 2. Processa documento canônico da Carla (gera pausa humana HITL)
    at.button(key="btn_proc_105").click().run()
    assert not at.exception
    assert len(at.warning) >= 1  # divergência de dias detectada

    # 3. Confirma no formulário HITL
    at.button(key="btn_conf_105").click().run()
    assert not at.exception

    # 4. Testa replay idempotente
    at.button(key="btn_replay_105").click().run()
    assert not at.exception
    assert any("Idempotência 100% íntegra" in s.value for s in at.success)

    # 5. Executa diagnóstico pré-voo
    at.button(key="diag_btn_run").click().run()
    assert not at.exception
