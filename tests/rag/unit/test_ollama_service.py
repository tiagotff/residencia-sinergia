"""Testes unitários para o serviço Ollama local do módulo pitch."""

import json
import unittest
from unittest.mock import MagicMock, patch

import httpx

from sinergia.rag.interfaces import ContextoRAG, MetadadosTrecho, TrechoRecuperado
from sinergia.rag.pitch.ollama_service import (
    PRESETS_MODELOS_GROQ,
    PRESETS_MODELOS_LOCAL,
    GroqPitchService,
    OllamaPitchService,
    verificar_guardrail_conversacional,
)


def _mock_response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _mock_stream_response(lines: list[str], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status.return_value = None
    resp.iter_lines.return_value = iter(lines)
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    return cm


class TestOllamaPitchService(unittest.TestCase):
    def setUp(self):
        self.service = OllamaPitchService(base_url="http://localhost:11434")

    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    def test_listar_modelos_disponiveis_filtra_embedding(self, mock_get):
        mock_get.return_value = _mock_response(
            {
                "models": [
                    {"name": "qwen3-embedding:0.6b"},
                    {"name": "qwen2.5:3b"},
                    {"name": "qwen3.5:4b"},
                ]
            }
        )

        modelos = self.service.listar_modelos_disponiveis()
        self.assertEqual(modelos, ["qwen2.5:3b", "qwen3.5:4b"])

    @patch("sinergia.rag.pitch.ollama_service.httpx.post")
    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    def test_roteamento_autonomo_via_tool_calling(self, mock_get, mock_post):
        mock_get.return_value = _mock_response({"models": [{"name": "qwen2.5:3b"}]})
        mock_post.return_value = _mock_response(
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ferramenta_busca_epis",
                                "arguments": {"query": "CA 12345 luva nitrílica"},
                            }
                        }
                    ],
                }
            }
        )

        decisao = self.service.decidir_base_autonomamente("Qual a validade do CA 12345?")
        self.assertEqual(decisao.id_base, 5)
        self.assertEqual(decisao.codigo_base, "F5")
        self.assertEqual(decisao.tool_name, "ferramenta_busca_epis")
        self.assertEqual(decisao.query_busca, "CA 12345 luva nitrílica")
        self.assertEqual(decisao.modo_roteamento, "tool_calling")

    @patch("sinergia.rag.pitch.ollama_service.httpx.post")
    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    def test_roteamento_autonomo_fallback_json_estruturado(self, mock_get, mock_post):
        mock_get.return_value = _mock_response({"models": [{"name": "llama3.2:1b"}]})
        # 1ª chamada (tool calling) não retorna tool_calls; 2ª chamada (JSON) retorna id_base=2
        mock_post.side_effect = [
            _mock_response({"message": {"role": "assistant", "content": "Não sei usar tools"}}),
            _mock_response(
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"id_base": 2, "query": "falta casamento Art 473 CLT"}),
                    }
                }
            ),
        ]

        decisao = self.service.decidir_base_autonomamente("Quantos dias de licença casamento?")
        self.assertEqual(decisao.id_base, 2)
        self.assertEqual(decisao.codigo_base, "F2")
        self.assertEqual(decisao.tool_name, "ferramenta_busca_direitos")
        self.assertEqual(decisao.query_busca, "falta casamento Art 473 CLT")
        self.assertEqual(decisao.modo_roteamento, "json_fallback")

    @patch("sinergia.rag.pitch.ollama_service.httpx.stream")
    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    def test_gerar_resposta_fundamentada_extrai_metricas_nativas(self, mock_get, mock_stream):
        mock_get.return_value = _mock_response({"models": [{"name": "qwen2.5:3b"}]})
        mock_stream.return_value = _mock_stream_response(
            [
                json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "content": "De acordo com a NR-17 (Item 17.6.4, Pág. 4), a pausa é de 10 minutos.",
                        },
                        "done": True,
                        "prompt_eval_count": 512,
                        "eval_count": 64,
                        "load_duration": 15_000_000,  # 15 ms
                        "eval_duration": 1_000_000_000,  # 1000 ms -> 64.0 tok/s
                        "total_duration": 1_200_000_000,  # 1200 ms
                    }
                )
            ]
        )

        contexto = ContextoRAG(
            query="pausa digitação",
            id_base=1,
            trechos=[
                TrechoRecuperado(
                    id_trecho=42,
                    conteudo="Deve haver pausa de 10 minutos para cada 50 minutos trabalhados.",
                    score=0.0328,
                    metadados=MetadadosTrecho(
                        titulo="NR-17 Ergonomia",
                        fonte="Portaria MTP nº 423",
                        versao="2024.1",
                        secao="Item 17.6.4",
                        pagina=4,
                        id_base=1,
                        vigente=True,
                    ),
                )
            ],
            total_encontrado=1,
        )

        resultado = self.service.gerar_resposta_fundamentada(
            pergunta="Qual a pausa para digitação contínua?",
            contexto=contexto,
            modelo="qwen2.5:3b",
        )

        self.assertIn("10 minutos", resultado.resposta)
        self.assertEqual(resultado.citacoes, ["NR-17 Ergonomia (v2024.1) · Item 17.6.4 · Pág. 4"])
        self.assertEqual(resultado.metricas.modelo, "qwen2.5:3b")
        self.assertEqual(resultado.metricas.prompt_eval_count, 512)
        self.assertEqual(resultado.metricas.eval_count, 64)
        self.assertAlmostEqual(resultado.metricas.load_duration_ms, 15.0)
        self.assertAlmostEqual(resultado.metricas.tokens_por_segundo, 64.0)
        self.assertIn("NR-17 Ergonomia", resultado.prompt_enviado)

    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    def test_presets_local_e_merge_com_instalados(self, mock_get):
        self.assertEqual(len(PRESETS_MODELOS_LOCAL), 5)
        joined = " ".join(PRESETS_MODELOS_LOCAL).lower()
        for familia in ("llama", "gemma", "qwen", "mimo", "deepseek-r1"):
            self.assertIn(familia, joined)

        mock_get.return_value = _mock_response(
            {"models": [{"name": "qwen2.5:3b"}, {"name": "qwen3:8b"}]}
        )
        combinados = self.service.listar_modelos_com_presets()
        self.assertEqual(combinados[0], "qwen2.5:3b")
        self.assertEqual(combinados.count("qwen3:8b"), 1)
        self.assertIn("mimo-v2.6:9b-thinking", combinados)
        self.assertTrue(self.service.modelo_esta_instalado("qwen2.5:3b"))
        self.assertFalse(self.service.modelo_esta_instalado("mimo-v2.6:9b-thinking"))

    @patch("sinergia.rag.pitch.ollama_service.httpx.post")
    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    def test_roteamento_multi_tool_com_pesos_e_alocacao_proporcional(self, mock_get, mock_post):
        mock_get.return_value = _mock_response({"models": [{"name": "qwen2.5:3b"}]})
        mock_post.return_value = _mock_response(
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ferramenta_busca_epis",
                                "arguments": {"query": "luva nitrílica CA", "peso": 0.25},
                            }
                        },
                        {
                            "function": {
                                "name": "ferramenta_busca_saude_ocupacional",
                                "arguments": {"query": "ASO exame periódico PCMSO", "peso": 0.75},
                            }
                        },
                    ],
                }
            }
        )

        decisao = self.service.decidir_base_autonomamente(
            "Quero saber detalhes do ASO periódico no PCMSO e também o CA da luva."
        )
        self.assertTrue(decisao.multi_base)
        self.assertEqual(decisao.codigos_combinados, "F4+F5")
        self.assertEqual(len(decisao.chamadas), 2)
        # Ordenado por peso decrescente: F4 (75%) -> 4 trechos, F5 (25%) -> 2 trechos
        self.assertEqual(decisao.chamadas[0].id_base, 4)
        self.assertAlmostEqual(decisao.chamadas[0].peso, 0.75)
        self.assertEqual(decisao.chamadas[0].limite_trechos, 4)
        self.assertEqual(decisao.chamadas[1].id_base, 5)
        self.assertAlmostEqual(decisao.chamadas[1].peso, 0.25)
        self.assertEqual(decisao.chamadas[1].limite_trechos, 2)

    @patch("sinergia.rag.pitch.ollama_service.httpx.post")
    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    def test_roteamento_via_ferramenta_busca_multiplas_bases(self, mock_get, mock_post):
        mock_get.return_value = _mock_response({"models": [{"name": "qwen2.5:3b"}]})
        mock_post.return_value = _mock_response(
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "ferramenta_busca_multiplas_bases",
                                "arguments": {
                                    "consultas": [
                                        {"id_base": 4, "query": "exame periódico PCMSO", "peso": 0.8},
                                        {"id_base": 5, "query": "luva EPI CA", "peso": 0.2},
                                    ]
                                },
                            }
                        }
                    ],
                }
            }
        )

        decisao = self.service.decidir_base_autonomamente(
            "Quero saber principalmente sobre exame periódico do PCMSO e qual luva usar."
        )
        self.assertTrue(decisao.multi_base)
        self.assertEqual(decisao.modo_roteamento, "tool_calling")
        self.assertEqual(decisao.codigos_combinados, "F4+F5")
        self.assertEqual(decisao.chamadas[0].tool_name, "ferramenta_busca_saude_ocupacional")
        self.assertEqual(decisao.chamadas[0].limite_trechos, 4)
        self.assertEqual(decisao.chamadas[1].tool_name, "ferramenta_busca_epis")
        self.assertEqual(decisao.chamadas[1].limite_trechos, 2)

    @patch("sinergia.rag.pitch.ollama_service.httpx.post")
    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    def test_roteamento_multi_base_json_fallback_e_heuristica(self, mock_get, mock_post):
        mock_get.return_value = _mock_response({"models": [{"name": "llama3.2:1b"}]})
        mock_post.side_effect = [
            _mock_response({"message": {"role": "assistant", "content": ""}}),
            _mock_response(
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "consultas": [
                                    {"id_base": 4, "query": "exame periódico PCMSO", "peso": 0.8},
                                    {"id_base": 5, "query": "CA luva EPI", "peso": 0.2},
                                ]
                            }
                        ),
                    }
                }
            ),
        ]

        decisao_json = self.service.decidir_base_autonomamente("PCMSO e EPI")
        self.assertEqual(decisao_json.modo_roteamento, "json_fallback")
        self.assertEqual(len(decisao_json.chamadas), 2)
        self.assertEqual(decisao_json.chamadas[0].id_base, 4)
        self.assertEqual(decisao_json.chamadas[0].limite_trechos, 4)
        self.assertEqual(decisao_json.chamadas[1].id_base, 5)
        self.assertEqual(decisao_json.chamadas[1].limite_trechos, 2)

        # Heurística com múltiplos termos de saúde (3 matches) e 1 de EPI (1 match)
        decisao_heur = self.service._decidir_heuristica(
            "Como funciona o exame periódico ASO de saúde no PCMSO e qual epi usar?", 0.0
        )
        self.assertEqual(decisao_heur.modo_roteamento, "heuristic_fallback")
        self.assertEqual(len(decisao_heur.chamadas), 2)
        self.assertEqual(decisao_heur.chamadas[0].id_base, 4)
        self.assertGreater(decisao_heur.chamadas[0].limite_trechos, decisao_heur.chamadas[1].limite_trechos)


class TestGroqPitchService(unittest.TestCase):
    def setUp(self):
        self.groq = GroqPitchService(api_key="gsk_test_fake_key")

    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    def test_listar_modelos_groq_filtra_audio_e_guard(self, mock_get):
        mock_get.return_value = _mock_response(
            {
                "data": [
                    {"id": "whisper-large-v3"},
                    {"id": "canopylabs/orpheus-v1-english"},
                    {"id": "meta-llama/llama-prompt-guard-2-86m"},
                    {"id": "openai/gpt-oss-safeguard-20b"},
                    {"id": "openai/gpt-oss-20b"},
                    {"id": "qwen/qwen3.8-27b"},
                    {"id": "openai/gpt-oss-120b"},
                ]
            }
        )
        modelos = self.groq.listar_modelos_disponiveis()
        self.assertEqual(
            modelos,
            [
                "openai/gpt-oss-20b",
                "qwen/qwen3.8-27b",
                "openai/gpt-oss-120b",
                "openai/gpt-oss-safeguard-20b",
            ],
        )
        com_presets = self.groq.listar_modelos_com_presets()
        for preset in PRESETS_MODELOS_GROQ:
            self.assertIn(preset, com_presets)

    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    def test_selecionar_modelo_ativo_nunca_sobrescreve_escolha_explicita_com_modelo_padrao(
        self, mock_get
    ):
        mock_get.return_value = _mock_response(
            {"data": [{"id": "qwen/qwen3.8-27b"}, {"id": "openai/gpt-oss-120b"}]}
        )
        groq_com_padrao = GroqPitchService(
            api_key="gsk_test_fake_key", modelo_padrao="openai/gpt-oss-120b"
        )
        self.assertEqual(
            groq_com_padrao.selecionar_modelo_ativo("openai/gpt-oss-20b"),
            "openai/gpt-oss-20b",
        )
        self.assertEqual(
            groq_com_padrao.selecionar_modelo_ativo("llama-3.1-8b-instant"),
            "llama-3.1-8b-instant",
        )

    @patch("sinergia.rag.pitch.ollama_service.httpx.stream")
    @patch("sinergia.rag.pitch.ollama_service.httpx.post")
    @patch("sinergia.rag.pitch.ollama_service.httpx.get")
    def test_groq_roteamento_e_geracao_com_metricas(self, mock_get, mock_post, mock_stream):
        mock_get.return_value = _mock_response(
            {"data": [{"id": "qwen/qwen3.8-27b"}, {"id": "openai/gpt-oss-20b"}]}
        )
        mock_post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "ferramenta_busca_direitos",
                                        "arguments": '{"query": "falta casamento CLT Art 473"}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        )
        mock_stream.return_value = _mock_stream_response(
            [
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "<think>Analisando Art. 473...</think>Até 3 dias consecutivos conforme Art. 473, II da CLT.",
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
                            "prompt_tokens": 320,
                            "completion_tokens": 45,
                            "queue_time": 0.05,
                            "completion_time": 0.15,
                            "total_time": 0.20,
                        },
                    }
                ),
                "data: [DONE]",
            ]
        )

        decisao = self.groq.decidir_base_autonomamente(
            "Quantos dias de falta por casamento?", modelo="qwen/qwen3.8-27b"
        )
        self.assertEqual(decisao.id_base, 2)
        self.assertEqual(decisao.modo_roteamento, "tool_calling")

        contexto = ContextoRAG(
            query=decisao.query_busca,
            id_base=2,
            trechos=[
                TrechoRecuperado(
                    id_trecho=95,
                    conteudo="Art. 473, II - até 3 (três) dias consecutivos, em virtude de casamento.",
                    score=0.032,
                    metadados=MetadadosTrecho(
                        titulo="CLT",
                        fonte="Decreto-Lei nº 5.452",
                        versao="2024.1",
                        secao="Art. 473",
                        pagina=19,
                        id_base=2,
                        vigente=True,
                    ),
                )
            ],
            total_encontrado=1,
        )
        resp = self.groq.gerar_resposta_fundamentada(
            "Quantos dias de falta por casamento?", contexto, modelo="qwen/qwen3.8-27b"
        )
        self.assertNotIn("<think>", resp.resposta)
        self.assertIn("3 dias consecutivos", resp.resposta)
        self.assertEqual(resp.metricas.prompt_eval_count, 320)
        self.assertEqual(resp.metricas.eval_count, 45)
        self.assertAlmostEqual(resp.metricas.load_duration_ms, 50.0)
        self.assertAlmostEqual(resp.metricas.tokens_por_segundo, 300.0)

    @patch("sinergia.rag.pitch.ollama_service.httpx.post")
    def test_guardrail_conversacional_bloqueia_saudacoes_e_termos_soltos(self, mock_post):
        for msg in ("Olá", "Bom dia", "Oi, tudo bem?", "Obrigado!", "Bom dia, ergonomia", "Olá EPI"):
            self.assertTrue(verificar_guardrail_conversacional(msg), msg)
            decisao = self.groq.decidir_base_autonomamente(msg, modelo="openai/gpt-oss-20b")
            self.assertFalse(decisao.usar_rag)
            self.assertIsNone(decisao.id_base)
            self.assertEqual(decisao.chamadas, [])
            self.assertEqual(decisao.modo_roteamento, "guardrail_conversacional")
            self.assertEqual(decisao.codigos_combinados, "CHAT")
        mock_post.assert_not_called()

    @patch("sinergia.rag.pitch.ollama_service.httpx.post")
    def test_guardrail_permite_saudacao_com_pergunta_de_dominio(self, mock_post):
        self.assertFalse(verificar_guardrail_conversacional("Bom dia, qual a pausa da NR-17?"))
        mock_post.return_value = _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "ferramenta_busca_pausas",
                                        "arguments": '{"query": "pausa NR-17"}',
                                    }
                                }
                            ],
                        }
                    }
                ]
            }
        )
        decisao = self.groq.decidir_base_autonomamente(
            "Bom dia, qual a pausa da NR-17?", modelo="openai/gpt-oss-20b"
        )
        self.assertTrue(decisao.usar_rag)
        self.assertEqual(decisao.id_base, 1)
        self.assertEqual(decisao.codigo_base, "F1")
        mock_post.assert_called_once()

    def test_prompts_geracao_focados_no_colaborador(self):
        contexto = ContextoRAG(
            query="pausa NR-17",
            id_base=1,
            trechos=[
                TrechoRecuperado(
                    id_trecho=1,
                    conteudo="Pausa de 10 minutos para cada 50 minutos trabalhados.",
                    metadados=MetadadosTrecho(
                        titulo="NR-17",
                        fonte="NR-17",
                        versao="2024",
                        secao="17.6.4",
                        pagina=4,
                        id_base=1,
                    ),
                )
            ],
            total_encontrado=1,
        )
        sys_rag, user_rag, _ = self.groq._montar_prompts_geracao(
            "Qual a pausa?", contexto, usar_rag=True
        )
        self.assertIn("colaborador (trabalhador da empresa)", sys_rag.lower())
        self.assertIn("Fontes recuperadas no RAG", user_rag)

        sys_chat, user_chat, _ = self.groq._montar_prompts_geracao(
            "Bom dia", ContextoRAG(), usar_rag=False
        )
        self.assertIn("colaborador (trabalhador da empresa)", sys_chat.lower())
        self.assertIn("NÃO cite nomes de arquivos PDF", sys_chat)
        self.assertNotIn("Fontes recuperadas no RAG", user_chat)

    @patch("sinergia.rag.pitch.ollama_service.httpx.stream")
    def test_historico_sessao_enviado_para_llm_e_roteamento_continuacao(self, mock_stream):
        historico = [
            {"role": "user", "content": "Bom dia, ergonomia"},
            {"role": "assistant", "content": "Bom dia! Qual é a sua dúvida sobre ergonomia?"},
        ]

        # Continuação sem palavra-chave explícita no turno 2 deve usar histórico na heurística
        decisao_heur = self.groq._decidir_heuristica("qual o limite?", 0.0, historico=historico)
        self.assertEqual(decisao_heur.id_base, 1)
        self.assertIn("ergonomia", decisao_heur.query_busca.lower())

        mock_stream.return_value = _mock_stream_response(
            [
                "data: "
                + json.dumps({"choices": [{"delta": {"content": "Por nada! Estou à disposição."}}]}),
                "data: "
                + json.dumps(
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 120, "completion_tokens": 15, "completion_time": 0.1},
                    }
                ),
                "data: [DONE]",
            ]
        )

        resp = self.groq.gerar_resposta_fundamentada(
            pergunta="Obrigado",
            contexto=ContextoRAG(),
            modelo="openai/gpt-oss-20b",
            usar_rag=False,
            historico=historico,
        )
        self.assertEqual(resp.citacoes, [])
        self.assertIn("Bom dia, ergonomia", resp.prompt_enviado)
        payload_enviado = mock_stream.call_args.kwargs["json"]
        roles_enviadas = [m["role"] for m in payload_enviado["messages"]]
        self.assertEqual(roles_enviadas, ["system", "user", "assistant", "user"])
        self.assertEqual(payload_enviado["messages"][1]["content"], "Bom dia, ergonomia")

    @patch("sinergia.rag.pitch.ollama_service.time.sleep")
    @patch("sinergia.rag.pitch.ollama_service.httpx.stream")
    def test_groq_gerar_resposta_trata_erro_429_stream_fechado_sem_crash(
        self, mock_stream, mock_sleep
    ):
        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.is_closed = True
        resp_429.read.side_effect = httpx.StreamClosed()
        erro_429 = httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp_429)
        resp_429.raise_for_status.side_effect = erro_429

        cm = MagicMock()
        cm.__enter__.return_value = resp_429
        cm.__exit__.return_value = False
        mock_stream.return_value = cm

        resultado = self.groq.gerar_resposta_fundamentada(
            pergunta="Bom dia",
            contexto=ContextoRAG(),
            modelo="openai/gpt-oss-20b",
            usar_rag=False,
        )

        self.assertEqual(mock_stream.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertIn("429", resultado.resposta)
        self.assertIn("Limite temporário", resultado.resposta)
        self.assertNotIn("Resumo direto do RAG", resultado.resposta)
        self.assertEqual(resultado.citacoes, [])


if __name__ == "__main__":
    unittest.main()
