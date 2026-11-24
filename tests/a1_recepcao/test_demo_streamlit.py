"""Testes headless (`AppTest`) do site de demonstração em Streamlit do Agente 1."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(
    Path(__file__).resolve().parents[2]
    / "src"
    / "sinergia"
    / "agentes"
    / "a1_recepcao"
    / "demo_streamlit.py"
)


def test_demo_streamlit_carrega_e_navega_entre_secoes() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert not at.exception
    assert len(at.title) >= 1
    assert "Agente 1" in at.title[0].value

    # Garante modo Memória local offline para testes determinísticos
    at.segmented_control(key="cfg_repo_escolha").set_value("Memória local").run()
    assert not at.exception

    # 1. Gera PDF sintético com CID-10 M54.5 (presente no banco em memória)
    at.button(key="btn_gen_cid10").click().run()
    assert not at.exception
    assert any("Dor lombar baixa" in c.value for c in at.caption)
    assert len(at.info) >= 1

    # 2. Alterna para Formulário manual e registra atestado com M54.5
    at.segmented_control(key="nav_secao_principal").set_value("manual").run()
    assert not at.exception
    assert len(at.form_submit_button) >= 1
    at.form_submit_button[0].click().run()
    assert not at.exception
    assert len(at.success) >= 1

    # 3. Alterna para Consulta e catálogo CID e testa código existente no banco
    at.segmented_control(key="nav_secao_principal").set_value("catalogo").run()
    assert not at.exception
    at.text_input(key="input_teste_cid").set_value("M54.5")
    at.button(key="btn_consultar_cid").click().run()
    assert not at.exception
    assert any("Dor lombar baixa" in c.value for c in at.caption)

    # 4. Alterna para Atestados no banco e ativa toggle LGPD
    at.segmented_control(key="nav_secao_principal").set_value("atestados").run()
    assert not at.exception
    at.toggle(key="toggle_mostrar_cid_restrito").set_value(True).run()
    assert not at.exception
    assert len(at.dataframe) >= 1
