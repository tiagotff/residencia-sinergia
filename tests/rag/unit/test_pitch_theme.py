"""Testes unitários para o tema monocromático claro e helpers HTML do módulo pitch."""

import unittest

from sinergia.rag.pitch.ollama_service import DecisaoRoteamento
from sinergia.rag.pitch.theme import (
    PALETA_MONOCROMATICA,
    obter_css_monocromatico_light,
    renderizar_badge_base,
    renderizar_badges_roteamento,
    renderizar_barra_distribuicao,
    renderizar_card_kpi,
    renderizar_citacoes_html,
    renderizar_grafico_embedding_html,
)


class TestPitchTheme(unittest.TestCase):
    def test_css_monocromatico_light_e_ocultacao_sidebar(self):
        css = obter_css_monocromatico_light()
        for hex_color in PALETA_MONOCROMATICA.values():
            self.assertIn(hex_color, css)

        self.assertIn('[data-testid="stSidebar"]', css)
        self.assertIn("display: none !important;", css)
        self.assertIn("color-scheme: light !important;", css)

        # Garantir que não injeta seletores manuais de base ou hiperparâmetros
        self.assertNotIn("top-k", css.lower())
        self.assertNotIn("temperatura", css.lower())

    def test_helpers_renderizacao_html_escapados(self):
        badge = renderizar_badge_base("F1", "Pausas & <script>", ativo=True)
        self.assertIn("Base F1", badge)
        self.assertNotIn("<script>", badge)
        self.assertIn("sinergia-badge-base--ativa", badge)

        kpi = renderizar_card_kpi("Tempo Médio Total", "1.42 s", "End-to-end")
        self.assertIn("Tempo Médio Total", kpi)
        self.assertIn("1.42 s", kpi)

        barra = renderizar_barra_distribuicao("F5", "EPIs (NR-6)", 2, 50.0)
        self.assertIn("width: 50.0%;", barra)
        self.assertIn("2 (50%)", barra)

        cit = renderizar_citacoes_html(["NR-17 (v2024.1) · Item 17.6.4 · Pág. 4"])
        self.assertIn("sinergia-citation-pill", cit)

        emb_bars = renderizar_grafico_embedding_html([0.05, -0.03, 0.08])
        self.assertIn("sinergia-emb-bar-pos", emb_bars)
        self.assertIn("sinergia-emb-bar-neg", emb_bars)

    def test_renderizar_badges_roteamento_sem_rag(self):
        decisao = DecisaoRoteamento(
            id_base=None,
            codigo_base="CHAT",
            nome_base="Chat Direto (Sem RAG)",
            tool_name="nenhuma",
            query_busca="",
            modo_roteamento="guardrail_conversacional",
            usar_rag=False,
        )
        badge = renderizar_badges_roteamento(decisao, ativo=True)
        self.assertIn("Chat Direto · Sem consulta documental", badge)
        self.assertIn("sinergia-badge-base--ativa", badge)


if __name__ == "__main__":
    unittest.main()
