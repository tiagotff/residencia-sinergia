"""Testes unitários e de UI (AppTest) para a aplicação Streamlit SinergIA - RAG Demo."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from sinergia.rag.interfaces import MetadadosTrecho, TrechoRecuperado
from sinergia.rag.pitch.app import (
    carregar_trechos_fallback_mocks,
    executar_turno_rag,
    obter_historico_chat_sessao,
)
from sinergia.rag.pitch.ollama_service import (
    PRESETS_MODELOS_GROQ,
    ChamadaFerramentaRAG,
    DecisaoRoteamento,
    GroqPitchService,
    MetricasOllama,
    RespostaGeradaOllama,
)
from sinergia.rag.retrievers import BuscadorHibrido

APP_PATH = str(Path(__file__).resolve().parents[2] / "pitch" / "app.py")


def _fake_http_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _fake_stream_response(lines: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.iter_lines.return_value = iter(lines)
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm


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

    @patch("sinergia.rag.pitch.ollama_service.httpx.stream")
    @patch("sinergia.rag.pitch.ollama_service.httpx.post")
    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    @patch("sinergia.rag.pitch.app._banco_disponivel_com_dados", return_value=False)
    def test_fluxo_completo_apptest_visao_global_e_raio_x(
        self, mock_pg_check, mock_http_get, mock_http_post, mock_http_stream
    ):
        mock_http_get.return_value = _fake_http_response(
            {"data": [{"id": "qwen/qwen3.8-27b"}, {"id": "openai/gpt-oss-20b"}]}
        )
        mock_http_post.return_value = _fake_http_response(
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
        )
        mock_http_stream.return_value = _fake_stream_response(
            [
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "Conforme a NR-17 (Item 17.6.4), a pausa é de 10 minutos a cada 50 minutos.",
                                }
                            }
                        ]
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 450,
                            "completion_tokens": 48,
                            "queue_time": 0.02,
                            "completion_time": 0.80,
                            "total_time": 0.90,
                        },
                    }
                ),
                "data: [DONE]",
            ]
        )

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

        with (
            patch(
                "sinergia.rag.pitch.app.obter_buscador_pitch", return_value=buscador_mock
            ),
            patch(
                "sinergia.rag.pitch.app.obter_servico_groq",
                return_value=GroqPitchService(api_key="gsk_test_key"),
            ),
        ):
            # 1. Carga inicial -> Visão Global por padrão, sem opção de LLM local e presetado em openai/gpt-oss-20b
            at = AppTest.from_file(APP_PATH, default_timeout=15).run()
            self.assertFalse(at.exception)

            self.assertEqual(len(at.toggle), 0)
            self.assertEqual(len(at.selectbox), 1)
            self.assertEqual(at.selectbox(key="ollama_model_select").value, "openai/gpt-oss-20b")
            self.assertEqual(
                at.selectbox[0].options[:2],
                ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
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

    @patch("sinergia.rag.pitch.ollama_service.httpx.stream")
    @patch("sinergia.rag.pitch.ollama_service.httpx.post")
    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    @patch("sinergia.rag.pitch.app._banco_disponivel_com_dados", return_value=False)
    def test_groq_preset_gpt_oss_20b_padrao_e_executa_consulta(
        self, mock_pg_check, mock_http_get, mock_http_post, mock_http_stream
    ):
        mock_http_get.return_value = _fake_http_response(
            {"data": [{"id": "qwen/qwen3.8-27b"}, {"id": "openai/gpt-oss-20b"}]}
        )
        mock_http_post.return_value = _fake_http_response(
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
        )
        mock_http_stream.return_value = _fake_stream_response(
            [
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "Resposta via Groq Cloud: pausa de 10 minutos (NR-17).",
                                }
                            }
                        ]
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 210,
                            "completion_tokens": 30,
                            "queue_time": 0.02,
                            "completion_time": 0.10,
                            "total_time": 0.12,
                        },
                    }
                ),
                "data: [DONE]",
            ]
        )

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

            self.assertEqual(len(at.toggle), 0)
            self.assertEqual(
                at.selectbox[0].options[:2],
                ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
            )
            for p in PRESETS_MODELOS_GROQ:
                self.assertIn(p, at.selectbox[0].options)
            self.assertEqual(at.selectbox(key="ollama_model_select").value, "openai/gpt-oss-20b")

            at.chat_input(key="chat_input_pitch").set_value(
                "Qual a pausa na NR-17?"
            ).run()
            self.assertFalse(at.exception)
            markdown_text = " ".join(m.value for m in at.markdown)
            self.assertIn("Resposta via Groq Cloud", markdown_text)
            ultimo_turno = at.session_state["turnos"][-1]
            self.assertEqual(ultimo_turno.metricas_ollama.modelo, "openai/gpt-oss-20b")

    def test_executar_turno_rag_mescla_multiplas_bases_por_peso(self):
        mock_emb = MagicMock()
        mock_emb.model_name = "tardellirs/colibri-embed-ptbr"
        mock_emb.gerar_embedding.return_value = [0.02] * 768

        buscador_mock = BuscadorHibrido(gerador_embeddings=mock_emb)
        trechos_teste = [
            TrechoRecuperado(
                id_trecho=i,
                conteudo=f"Regra PCMSO saúde ocupacional ASO número {i}.",
                metadados=MetadadosTrecho(
                    id_documento=4,
                    titulo="PCMSO NR-7",
                    fonte="NR-7",
                    versao="2024.1",
                    secao=f"Item 7.{i}",
                    pagina=i,
                    id_base=4,
                    vigente=True,
                ),
            )
            for i in range(1, 6)
        ] + [
            TrechoRecuperado(
                id_trecho=10 + i,
                conteudo=f"Ficha EPI luva CA número {i}.",
                metadados=MetadadosTrecho(
                    id_documento=5,
                    titulo="EPI NR-6",
                    fonte="NR-6",
                    versao="2024.1",
                    secao=f"CA {i}",
                    pagina=i,
                    id_base=5,
                    vigente=True,
                ),
            )
            for i in range(1, 5)
        ]
        buscador_mock.adicionar_trechos_memoria(trechos_teste)

        servico_mock = MagicMock()
        servico_mock.decidir_base_autonomamente.return_value = DecisaoRoteamento(
            id_base=4,
            codigo_base="F4",
            nome_base="Saúde Ocupacional (PCMSO)",
            tool_name="ferramenta_busca_saude_ocupacional",
            query_busca="PCMSO ASO",
            modo_roteamento="tool_calling",
            roteamento_ms=12.0,
            chamadas=[
                ChamadaFerramentaRAG(
                    id_base=4,
                    codigo_base="F4",
                    nome_base="Saúde Ocupacional (PCMSO)",
                    tool_name="ferramenta_busca_saude_ocupacional",
                    query_busca="PCMSO ASO",
                    peso=0.75,
                    limite_trechos=4,
                ),
                ChamadaFerramentaRAG(
                    id_base=5,
                    codigo_base="F5",
                    nome_base="EPIs e Fichas de CA (NR-6)",
                    tool_name="ferramenta_busca_epis",
                    query_busca="EPI luva CA",
                    peso=0.25,
                    limite_trechos=2,
                ),
            ],
        )

        def _fake_gerar(pergunta, contexto, **_kwargs):
            return RespostaGeradaOllama(
                resposta="Resposta mesclada PCMSO + EPI.",
                citacoes=[],
                metricas=MetricasOllama(modelo="teste"),
                llm_ms=10.0,
                prompt_enviado=contexto.formatar_para_prompt(),
            )

        servico_mock.gerar_resposta_fundamentada.side_effect = _fake_gerar

        st.session_state.next_msg_id = 1
        turno = executar_turno_rag(
            pergunta="Dúvida sobre PCMSO ASO e qual EPI usar",
            modelo_selecionado="teste",
            buscador=buscador_mock,
            servico_llm=servico_mock,
        )

        bases_recuperadas = [d.metadados.id_base for d in turno.detalhes_trechos]
        self.assertEqual(bases_recuperadas.count(4), 4)
        self.assertEqual(bases_recuperadas.count(5), 2)
        self.assertIn("PCMSO NR-7", turno.prompt_enviado)
        self.assertIn("EPI NR-6", turno.prompt_enviado)
        self.assertEqual(turno.roteamento.codigos_combinados, "F4+F5")

    def test_executar_turno_rag_guardrail_nao_busca_vetores_nem_retorna_pdfs(self):
        buscador_mock = MagicMock()
        servico_mock = MagicMock()
        servico_mock.decidir_base_autonomamente.return_value = DecisaoRoteamento(
            id_base=None,
            codigo_base="CHAT",
            nome_base="Chat Direto (Sem RAG)",
            tool_name="nenhuma",
            query_busca="",
            modo_roteamento="guardrail_conversacional",
            roteamento_ms=1.0,
            chamadas=[],
            usar_rag=False,
        )
        servico_mock.gerar_resposta_fundamentada.return_value = RespostaGeradaOllama(
            resposta="Bom dia! Como posso ajudar você hoje no trabalho?",
            citacoes=["PDF Indevido"],
            metricas=MetricasOllama(modelo="openai/gpt-oss-20b"),
            llm_ms=20.0,
            prompt_enviado="prompt",
        )

        st.session_state.turnos = []
        st.session_state.next_msg_id = 1
        turno = executar_turno_rag(
            pergunta="Olá",
            modelo_selecionado="openai/gpt-oss-20b",
            buscador=buscador_mock,
            servico_llm=servico_mock,
        )

        buscador_mock.buscar_com_telemetria.assert_not_called()
        self.assertFalse(turno.roteamento.usar_rag)
        self.assertEqual(turno.citacoes, [])
        self.assertEqual(turno.detalhes_trechos, [])
        self.assertEqual(turno.estatisticas_embedding.dimensoes, 0)

    def test_obter_historico_chat_sessao_e_continuidade_na_sessao(self):
        buscador_mock = MagicMock()
        servico_mock = MagicMock()
        servico_mock.decidir_base_autonomamente.return_value = DecisaoRoteamento(
            id_base=None,
            codigo_base="CHAT",
            nome_base="Chat Direto (Sem RAG)",
            tool_name="nenhuma",
            query_busca="",
            modo_roteamento="guardrail_conversacional",
            roteamento_ms=1.0,
            chamadas=[],
            usar_rag=False,
        )
        servico_mock.gerar_resposta_fundamentada.return_value = RespostaGeradaOllama(
            resposta="Bom dia! Qual é a sua dúvida sobre ergonomia?",
            citacoes=[],
            metricas=MetricasOllama(modelo="openai/gpt-oss-20b"),
            llm_ms=15.0,
            prompt_enviado="prompt 1",
        )

        st.session_state.turnos = []
        st.session_state.next_msg_id = 1
        turno_1 = executar_turno_rag(
            pergunta="Bom dia, ergonomia",
            modelo_selecionado="openai/gpt-oss-20b",
            buscador=buscador_mock,
            servico_llm=servico_mock,
        )
        st.session_state.turnos.append(turno_1)

        historico_extraido = obter_historico_chat_sessao()
        self.assertEqual(
            historico_extraido,
            [
                {"role": "user", "content": "Bom dia, ergonomia"},
                {"role": "assistant", "content": "Bom dia! Qual é a sua dúvida sobre ergonomia?"},
            ],
        )

        # Turno 2 deve receber automaticamente o histórico da sessão
        executar_turno_rag(
            pergunta="qual o tempo de pausa?",
            modelo_selecionado="openai/gpt-oss-20b",
            buscador=buscador_mock,
            servico_llm=servico_mock,
        )
        kwargs_roteamento = servico_mock.decidir_base_autonomamente.call_args.kwargs
        kwargs_geracao = servico_mock.gerar_resposta_fundamentada.call_args.kwargs
        self.assertEqual(kwargs_roteamento["historico"], historico_extraido)
        self.assertEqual(kwargs_geracao["historico"], historico_extraido)


if __name__ == "__main__":
    unittest.main()
