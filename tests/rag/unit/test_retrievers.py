import unittest
from unittest.mock import MagicMock, patch
from sinergia.rag.interfaces import (
    ContextoRAG,
    MetadadosTrecho,
    ResultadoBuscaInstrumentada,
    TrechoRecuperado,
)
from sinergia.rag.retrievers import BuscadorHibrido, calcular_rrf, calcular_rrf_detalhado


class TestRetrievers(unittest.TestCase):
    def setUp(self):
        self.meta_a = MetadadosTrecho(
            id_documento=1,
            titulo="Doc A",
            fonte="Fonte A",
            versao="1.0",
            secao="Art. 1",
            id_base=1,
            vigente=True,
        )
        self.meta_b = MetadadosTrecho(
            id_documento=1,
            titulo="Doc B",
            fonte="Fonte B",
            versao="1.0",
            secao="Art. 2",
            id_base=1,
            vigente=True,
        )
        self.meta_c = MetadadosTrecho(
            id_documento=2,
            titulo="Doc C",
            fonte="Fonte C",
            versao="1.0",
            secao="Art. 3",
            id_base=1,
            vigente=True,
        )
        self.meta_d = MetadadosTrecho(
            id_documento=2,
            titulo="Doc D",
            fonte="Fonte D",
            versao="1.0",
            secao="Art. 4",
            id_base=1,
            vigente=True,
        )
        self.meta_e = MetadadosTrecho(
            id_documento=3,
            titulo="Doc E",
            fonte="Fonte E",
            versao="1.0",
            secao="Art. 5",
            id_base=1,
            vigente=True,
        )
        self.meta_f = MetadadosTrecho(
            id_documento=3,
            titulo="Doc F",
            fonte="Fonte F",
            versao="1.0",
            secao="Art. 6",
            id_base=1,
            vigente=True,
        )

        self.t_a = TrechoRecuperado(id_trecho=1, conteudo="Conteudo A", metadados=self.meta_a)
        self.t_b = TrechoRecuperado(id_trecho=2, conteudo="Conteudo B", metadados=self.meta_b)
        self.t_c = TrechoRecuperado(id_trecho=3, conteudo="Conteudo C", metadados=self.meta_c)
        self.t_d = TrechoRecuperado(id_trecho=4, conteudo="Conteudo D", metadados=self.meta_d)
        self.t_e = TrechoRecuperado(id_trecho=5, conteudo="Conteudo E", metadados=self.meta_e)
        self.t_f = TrechoRecuperado(id_trecho=6, conteudo="Conteudo F", metadados=self.meta_f)

    def test_calcular_rrf_matematica(self):
        # Denso: [A (rank 1), B (rank 2), D (rank 3)]
        # Lexico: [C (rank 1), A (rank 2), D (rank 3)]
        densos = [self.t_a, self.t_b, self.t_d]
        lexicos = [self.t_c, self.t_a, self.t_d]

        k = 60
        esperado_score_a = (1.0 / (k + 1)) + (1.0 / (k + 2))
        esperado_score_d = (1.0 / (k + 3)) + (1.0 / (k + 3))
        esperado_score_c = 1.0 / (k + 1)
        esperado_score_b = 1.0 / (k + 2)

        fundidos = calcular_rrf(densos, lexicos, k=60, top_k=4)

        self.assertEqual(len(fundidos), 4)
        # Primeiro colocado deve ser A
        self.assertEqual(fundidos[0].id_trecho, 1)
        self.assertAlmostEqual(fundidos[0].score, esperado_score_a, places=6)

        # Segundo colocado deve ser D
        self.assertEqual(fundidos[1].id_trecho, 4)
        self.assertAlmostEqual(fundidos[1].score, esperado_score_d, places=6)

        # Terceiro deve ser C
        self.assertEqual(fundidos[2].id_trecho, 3)
        self.assertAlmostEqual(fundidos[2].score, esperado_score_c, places=6)

        # Quarto deve ser B
        self.assertEqual(fundidos[3].id_trecho, 2)
        self.assertAlmostEqual(fundidos[3].score, esperado_score_b, places=6)

    def test_calcular_rrf_top_4_limite(self):
        densos = [self.t_a, self.t_b, self.t_c, self.t_d, self.t_e, self.t_f]
        lexicos = [self.t_f, self.t_e, self.t_d, self.t_c, self.t_b, self.t_a]

        fundidos = calcular_rrf(densos, lexicos, k=60, top_k=4)
        self.assertEqual(len(fundidos), 4)

    @patch.object(BuscadorHibrido, "_buscar_banco", return_value=None)
    def test_buscador_hibrido_com_mock(self, mock_buscar_banco):
        mock_embedding = MagicMock()
        mock_embedding.gerar_embedding.return_value = [0.1] * 768

        buscador = BuscadorHibrido(gerador_embeddings=mock_embedding)

        # Simular base de dados em memória do buscador
        # Item válido da base 1 e vigente
        item_valido = TrechoRecuperado(
            id_trecho=10,
            conteudo="Trecho de ergonomia e pausas",
            metadados=MetadadosTrecho(
                titulo="NR-17",
                fonte="Portaria",
                versao="2024",
                id_base=1,
                vigente=True,
            ),
        )
        # Item obsoleto (vigente=False) da base 1
        item_obsoleto = TrechoRecuperado(
            id_trecho=11,
            conteudo="Trecho antigo de ergonomia",
            metadados=MetadadosTrecho(
                titulo="NR-17 Antiga",
                fonte="Portaria",
                versao="1990",
                id_base=1,
                vigente=False,
            ),
        )
        # Item de outra base (id_base=2)
        item_outra_base = TrechoRecuperado(
            id_trecho=12,
            conteudo="Trecho da CLT",
            metadados=MetadadosTrecho(
                titulo="CLT",
                fonte="Decreto",
                versao="1943",
                id_base=2,
                vigente=True,
            ),
        )

        buscador.adicionar_trechos_memoria([item_valido, item_obsoleto, item_outra_base])

        contexto = buscador.buscar(query="ergonomia e pausas", id_base=1)

        self.assertIsInstance(contexto, ContextoRAG)
        self.assertEqual(len(contexto.trechos), 1)
        self.assertEqual(contexto.trechos[0].id_trecho, 10)
        self.assertEqual(contexto.trechos[0].metadados.id_base, 1)
        self.assertTrue(contexto.trechos[0].metadados.vigente)
        self.assertGreater(contexto.trechos[0].score, 0.0)

    @patch.object(BuscadorHibrido, "_buscar_banco", return_value=None)
    def test_buscador_limite_4_registros_e_score(self, mock_buscar_banco):
        mock_embedding = MagicMock()
        mock_embedding.gerar_embedding.return_value = [0.1] * 768

        buscador = BuscadorHibrido(gerador_embeddings=mock_embedding)
        itens = [
            TrechoRecuperado(
                id_trecho=i,
                conteudo=f"Conteudo de teste {i}",
                metadados=MetadadosTrecho(
                    titulo=f"Doc {i}",
                    fonte="Fonte",
                    versao="1.0",
                    id_base=1,
                    vigente=True,
                ),
            )
            for i in range(1, 10)
        ]
        buscador.adicionar_trechos_memoria(itens)

        contexto = buscador.buscar(query="teste", id_base=1, limite=4)
        self.assertEqual(len(contexto.trechos), 4)
        for t in contexto.trechos:
            self.assertIsNotNone(t.score)
            self.assertGreater(t.score, 0.0)

    def test_calcular_rrf_detalhado_preserva_ranks_e_scores(self):
        t1 = TrechoRecuperado(id_trecho=1, conteudo="A", score=0.91, metadados=self.meta_a)
        t2 = TrechoRecuperado(id_trecho=2, conteudo="B", score=0.75, metadados=self.meta_b)
        t3_lex = TrechoRecuperado(id_trecho=3, conteudo="C", score=0.88, metadados=self.meta_c)
        t1_lex = TrechoRecuperado(id_trecho=1, conteudo="A", score=0.62, metadados=self.meta_a)

        trechos, detalhes = calcular_rrf_detalhado(
            densos=[t1, t2],
            lexicos=[t3_lex, t1_lex],
            k=60,
            top_k=4,
        )

        self.assertEqual(len(trechos), 3)
        self.assertEqual(len(detalhes), 3)
        primeiro = detalhes[0]
        self.assertEqual(primeiro.id_trecho, 1)
        self.assertAlmostEqual(primeiro.dense_score, 0.91)
        self.assertEqual(primeiro.dense_rank, 1)
        self.assertAlmostEqual(primeiro.lex_score, 0.62)
        self.assertEqual(primeiro.lex_rank, 2)
        self.assertAlmostEqual(primeiro.rrf_score, (1.0 / 61) + (1.0 / 62), places=6)

        somente_lex = [d for d in detalhes if d.id_trecho == 3][0]
        self.assertIsNone(somente_lex.dense_rank)
        self.assertEqual(somente_lex.lex_rank, 1)
        self.assertAlmostEqual(somente_lex.lex_score, 0.88)

    @patch.object(BuscadorHibrido, "_buscar_banco", return_value=None)
    def test_buscar_com_telemetria(self, mock_buscar_banco):
        mock_embedding = MagicMock()
        mock_embedding.model_name = "tardellirs/colibri-embed-ptbr"
        vetor_teste = [0.02 if i % 2 == 0 else -0.01 for i in range(768)]
        mock_embedding.gerar_embedding.return_value = vetor_teste

        buscador = BuscadorHibrido(gerador_embeddings=mock_embedding)
        item = TrechoRecuperado(
            id_trecho=42,
            conteudo="Pausa de 10 minutos para digitação contínua na NR-17.",
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
        buscador.adicionar_trechos_memoria([item])

        resultado = buscador.buscar_com_telemetria(
            query="pausa digitação NR-17", id_base=1, limite=4
        )

        self.assertIsInstance(resultado, ResultadoBuscaInstrumentada)
        self.assertGreaterEqual(resultado.embed_ms, 0.0)
        self.assertGreaterEqual(resultado.db_rrf_ms, 0.0)
        self.assertEqual(resultado.estatisticas_embedding.dimensoes, 768)
        self.assertEqual(len(resultado.estatisticas_embedding.amostra), 12)
        self.assertEqual(len(resultado.estatisticas_embedding.vetor), 768)
        self.assertGreater(resultado.estatisticas_embedding.norma_l2, 0.0)
        self.assertEqual(len(resultado.detalhes_trechos), 1)
        det = resultado.detalhes_trechos[0]
        self.assertEqual(det.id_trecho, 42)
        self.assertEqual(det.dense_rank, 1)
        self.assertEqual(det.lex_rank, 1)
        self.assertGreater(det.rrf_score, 0.0)



if __name__ == "__main__":
    unittest.main()

