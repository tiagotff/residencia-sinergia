"""Testes unitários e de UI (AppTest) para a aplicação Streamlit SinergIA - RAG Demo."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from sinergia.rag.interfaces import MetadadosTrecho, TrechoRecuperado
from sinergia.rag.pitch.app import carregar_trechos_fallback_mocks
from sinergia.rag.pitch.ollama_service import (
    PRESETS_MODELOS_GROQ,
    PRESETS_MODELOS_LOCAL,
    GroqPitchService,
)
from sinergia.rag.retrievers import BuscadorHibrido

APP_PATH = str(Path(__file__).resolve().parents[2] / "pitch" / "app.py")


def _fake_http_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


class TestPitchApp(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()

    def tearDown(self):
        st.cache_resource.clear()

    def test_carregar_trechos_fallback_mocks_carrega_cinco_bases(self):
        trechos = carregar_trechos_fallback_mocks()
        self.assertGreater(len(trechos), 0)
        bases_carregadas = {t.metadados.id_base for t in trechos}
        self.assertEqual(bases_carregadas, {1, 2, 3, 4, 5})

    @patch("sinergia.rag.pitch.ollama_service.httpx.post")
    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    @patch("sinergia.rag.pitch.app._banco_disponivel_com_dados", return_value=False)
    def test_fluxo_completo_apptest_visao_global_e_raio_x(
        self, mock_pg_check, mock_http_get, mock_http_post
    ):
        mock_http_get.return_value = _fake_http_response(
            {"models": [{"name": "qwen2.5:3b"}, {"name": "qwen3.5:4b"}]}
        )
        mock_http_post.side_effect = [
            # 1) Roteamento autônomo
            _fake_http_response(
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "ferramenta_busca_pausas",
                                    "arguments": {"query": "pausa digitação NR-17"},
                                }
                            }
                        ],
                    }
                }
            ),
            # 2) Geração fundamentada
            _fake_http_response(
                {
                    "message": {
                        "role": "assistant",
                        "content": "Conforme a NR-17 (Item 17.6.4), a pausa é de 10 minutos a cada 50 minutos.",
                    },
                    "prompt_eval_count": 450,
                    "eval_count": 48,
                    "load_duration": 10_000_000,
                    "eval_duration": 800_000_000,
                    "total_duration": 900_000_000,
                }
            ),
        ]

        mock_emb = MagicMock()
        mock_emb.model_name = "tardellirs/colibri-embed-ptbr"
        mock_emb.gerar_embedding.return_value = [
            0.03 if i % 2 == 0 else -0.02 for i in range(768)
        ]

        buscador_mock = BuscadorHibrido(gerador_embeddings=mock_emb)
        buscador_mock.adicionar_trechos_memoria(
            [
                TrechoRecuperado(
                    id_trecho=42,
                    conteudo="17.6.4. Deve haver, no mínimo, uma pausa de 10 minutos para cada 50 minutos trabalhados.",
                    metadados=MetadadosTrecho(
                        id_documento=1,
                        titulo="NR-17 Ergonomia",
                        fonte="Portaria MTP nº 423",
                        versao="2024.1",
                        secao="Item 17.6.4",
                        pagina=4,
                        id_base=1,
                        breadcrumb="NR-17 > 17.6 > 17.6.4",
                        vigente=True,
                        extra={"hierarquia": ["NR-17", "17.6.4"], "page_start": 4},
                    ),
                )
            ]
        )

        with patch(
            "sinergia.rag.pitch.app.obter_buscador_pitch", return_value=buscador_mock
        ):
            # 1. Carga inicial -> Visão Global por padrão
            at = AppTest.from_file(APP_PATH, default_timeout=15).run()
            self.assertFalse(at.exception)

            # Toggle Local/Cloud + seletor de modelo Ollama (instalados + presets) presentes; sem sliders
            self.assertEqual(len(at.toggle), 1)
            self.assertFalse(at.toggle(key="provider_cloud_toggle").value)
            self.assertEqual(len(at.selectbox), 1)
            self.assertEqual(
                at.selectbox[0].options,
                ["qwen2.5:3b", "qwen3.5:4b", *PRESETS_MODELOS_LOCAL],
            )
            self.assertEqual(len(at.slider), 0)

            subheaders = [s.value for s in at.subheader]
            self.assertIn("Métricas agregadas do pipeline", subheaders)

            # 2. Enviar pergunta pelo chat_input
            at.chat_input(key="chat_input_pitch").set_value(
                "Qual a pausa para digitação contínua?"
            ).run()
            self.assertFalse(at.exception)

            markdown_text = " ".join(m.value for m in at.markdown)
            self.assertIn("10 minutos", markdown_text)
            self.assertTrue(any(b.label == "Inspecionar no Raio-X" for b in at.button))

            # 3. Clicar em "Inspecionar no Raio-X" -> alterna para Raio-X Nerd com 4 abas
            inspect_btn = next(
                b for b in at.button if b.label == "Inspecionar no Raio-X"
            )
            inspect_btn.click().run()
            self.assertFalse(at.exception)

            tab_labels = [t.label for t in at.tabs]
            self.assertEqual(
                tab_labels,
                [
                    "1. Chunks & RRF",
                    "2. Embedding (768d)",
                    "3. Metadados",
                    "4. Tool & Prompt",
                ],
            )
            self.assertTrue(
                any(b.label == "Voltar para Visão Global" for b in at.button)
            )

            # 4. Clicar em "Voltar para Visão Global" -> retorna à Visão Global
            voltar_btn = next(
                b for b in at.button if b.label == "Voltar para Visão Global"
            )
            voltar_btn.click().run()
            self.assertFalse(at.exception)
            subheaders_after = [s.value for s in at.subheader]
            self.assertIn("Métricas agregadas do pipeline", subheaders_after)

    @patch("sinergia.rag.pitch.ollama_service.httpx.post")
    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    @patch("sinergia.rag.pitch.app._banco_disponivel_com_dados", return_value=False)
    def test_toggle_cloud_groq_alterna_provedor_e_executa_consulta(
        self, mock_pg_check, mock_http_get, mock_http_post
    ):
        def _dispatch_get(url, **_kwargs):
            if "api.groq.com" in url:
                return _fake_http_response(
                    {"data": [{"id": "qwen/qwen3.8-27b"}, {"id": "openai/gpt-oss-20b"}]}
                )
            return _fake_http_response({"models": [{"name": "qwen2.5:3b"}]})

        mock_http_get.side_effect = _dispatch_get
        mock_http_post.side_effect = [
            _fake_http_response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "ferramenta_busca_pausas",
                                            "arguments": '{"query": "pausa digitação NR-17"}',
                                        }
                                    }
                                ],
                            }
                        }
                    ]
                }
            ),
            _fake_http_response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Resposta via Groq Cloud: pausa de 10 minutos (NR-17).",
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 210,
                        "completion_tokens": 30,
                        "queue_time": 0.02,
                        "completion_time": 0.10,
                        "total_time": 0.12,
                    },
                }
            ),
        ]

        mock_emb = MagicMock()
        mock_emb.model_name = "tardellirs/colibri-embed-ptbr"
        mock_emb.gerar_embedding.return_value = [
            0.03 if i % 2 == 0 else -0.02 for i in range(768)
        ]

        buscador_mock = BuscadorHibrido(gerador_embeddings=mock_emb)
        buscador_mock.adicionar_trechos_memoria(
            [
                TrechoRecuperado(
                    id_trecho=42,
                    conteudo="17.6.4. Deve haver pausa de 10 minutos para cada 50 minutos trabalhados.",
                    metadados=MetadadosTrecho(
                        id_documento=1,
                        titulo="NR-17 Ergonomia",
                        fonte="Portaria MTP nº 423",
                        versao="2024.1",
                        secao="Item 17.6.4",
                        pagina=4,
                        id_base=1,
                        vigente=True,
                    ),
                )
            ]
        )

        with (
            patch(
                "sinergia.rag.pitch.app.obter_buscador_pitch",
                return_value=buscador_mock,
            ),
            patch(
                "sinergia.rag.pitch.app.obter_servico_groq",
                return_value=GroqPitchService(api_key="gsk_test_key"),
            ),
        ):
            at = AppTest.from_file(APP_PATH, default_timeout=15).run()
            self.assertFalse(at.exception)

            at.toggle(key="provider_cloud_toggle").set_value(True).run()
            self.assertFalse(at.exception)
            self.assertEqual(
                at.selectbox[0].options[:2],
                ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
            )
            for p in PRESETS_MODELOS_GROQ:
                self.assertIn(p, at.selectbox[0].options)

            at.selectbox(key="ollama_model_select").select("openai/gpt-oss-20b").run()
            self.assertFalse(at.exception)

            at.chat_input(key="chat_input_pitch").set_value(
                "Qual a pausa na NR-17?"
            ).run()
            self.assertFalse(at.exception)
            markdown_text = " ".join(m.value for m in at.markdown)
            self.assertIn("Resposta via Groq Cloud", markdown_text)
            ultimo_turno = at.session_state["turnos"][-1]
            self.assertEqual(ultimo_turno.metricas_ollama.modelo, "openai/gpt-oss-20b")


if __name__ == "__main__":
    unittest.main()
