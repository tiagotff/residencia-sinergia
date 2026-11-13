"""Testes unitários para os agregadores de telemetria da Visão Global do pitch."""

import unittest

from sinergia.rag.interfaces import (
    DetalheScoreTrecho,
    EstatisticasEmbedding,
    MetadadosTrecho,
)
from sinergia.rag.pitch.ollama_service import DecisaoRoteamento, MetricasOllama
from sinergia.rag.pitch.telemetry import (
    TelemetriaTurno,
    TemposTurno,
    calcular_visao_global,
)


def _criar_turno_teste(
    id_turno: int,
    id_base: int,
    codigo: str,
    total_ms: float,
    embed_ms: float,
    db_rrf_ms: float,
    llm_ms: float,
    eval_count: int,
    tok_s: float,
) -> TelemetriaTurno:
    return TelemetriaTurno(
        id_turno=id_turno,
        pergunta=f"Pergunta {id_turno}",
        resposta=f"Resposta {id_turno}",
        citacoes=["Doc (v1.0) · Seção 1 · Pág. 1"],
        roteamento=DecisaoRoteamento(
            id_base=id_base,
            codigo_base=codigo,
            nome_base=f"Base {codigo}",
            tool_name=f"tool_{codigo.lower()}",
            query_busca="query",
            modo_roteamento="tool_calling",
            roteamento_ms=10.0,
        ),
        tempos=TemposTurno(
            total_ms=total_ms,
            roteamento_ms=10.0,
            embed_ms=embed_ms,
            db_rrf_ms=db_rrf_ms,
            llm_ms=llm_ms,
        ),
        estatisticas_embedding=EstatisticasEmbedding(dimensoes=768, norma_l2=1.0),
        detalhes_trechos=[
            DetalheScoreTrecho(
                id_trecho=id_turno,
                conteudo="Trecho",
                metadados=MetadadosTrecho(
                    titulo="Doc", fonte="Fonte", versao="1.0", id_base=id_base, vigente=True
                ),
                rrf_score=0.03,
                dense_score=0.8,
                dense_rank=1,
                lex_score=0.5,
                lex_rank=1,
            )
        ],
        metricas_ollama=MetricasOllama(
            modelo="qwen2.5:3b",
            prompt_eval_count=400,
            eval_count=eval_count,
            tokens_por_segundo=tok_s,
        ),
        prompt_enviado="prompt",
    )


class TestPitchTelemetry(unittest.TestCase):
    def test_visao_global_sessao_vazia(self):
        resumo = calcular_visao_global([])
        self.assertEqual(resumo.total_consultas, 0)
        self.assertEqual(resumo.tempo_medio_total_s, 0.0)
        self.assertEqual(resumo.total_tokens_gerados, 0)
        self.assertEqual(len(resumo.distribuicao_bases), 5)
        self.assertTrue(all(d.contagem == 0 and d.percentual == 0.0 for d in resumo.distribuicao_bases))

    def test_visao_global_agregacao_multiplos_turnos(self):
        t1 = _criar_turno_teste(1, 1, "F1", total_ms=1400.0, embed_ms=20.0, db_rrf_ms=10.0, llm_ms=1300.0, eval_count=80, tok_s=60.0)
        t2 = _criar_turno_teste(2, 5, "F5", total_ms=1000.0, embed_ms=16.0, db_rrf_ms=8.0, llm_ms=900.0, eval_count=60, tok_s=70.0)
        t3 = _criar_turno_teste(3, 1, "F1", total_ms=1200.0, embed_ms=18.0, db_rrf_ms=9.0, llm_ms=1100.0, eval_count=70, tok_s=65.0)

        resumo = calcular_visao_global([t1, t2, t3])
        self.assertEqual(resumo.total_consultas, 3)
        self.assertAlmostEqual(resumo.tempo_medio_total_ms, 1200.0)
        self.assertAlmostEqual(resumo.tempo_medio_total_s, 1.2)
        # RAG médio: (30 + 24 + 27) / 3 = 27.0 ms
        self.assertAlmostEqual(resumo.latencia_media_rag_ms, 27.0)
        # LLM médio: (1300 + 900 + 1100) / 3 = 1100.0 ms -> 1.1 s
        self.assertAlmostEqual(resumo.latencia_media_llm_ms, 1100.0)
        self.assertAlmostEqual(resumo.latencia_media_llm_s, 1.1)
        self.assertAlmostEqual(resumo.velocidade_media_tok_s, 65.0)
        self.assertEqual(resumo.total_tokens_gerados, 210)
        self.assertEqual(resumo.total_prompt_tokens, 1200)

        por_codigo = {d.codigo: d for d in resumo.distribuicao_bases}
        self.assertEqual(por_codigo["F1"].contagem, 2)
        self.assertAlmostEqual(por_codigo["F1"].percentual, 66.7)
        self.assertEqual(por_codigo["F5"].contagem, 1)
        self.assertAlmostEqual(por_codigo["F5"].percentual, 33.3)
        self.assertEqual(por_codigo["F2"].contagem, 0)


if __name__ == "__main__":
    unittest.main()
