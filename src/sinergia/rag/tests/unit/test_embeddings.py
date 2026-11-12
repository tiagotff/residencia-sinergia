import unittest
from unittest.mock import MagicMock, patch
from sinergia.rag.embeddings import GeradorEmbeddings, get_gerador_embeddings


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        GeradorEmbeddings.reset_instance()

    def tearDown(self):
        GeradorEmbeddings.reset_instance()

    def test_singleton_pattern(self):
        inst1 = get_gerador_embeddings()
        inst2 = get_gerador_embeddings()
        self.assertIs(inst1, inst2)

    def test_dimensao_768(self):
        gerador = get_gerador_embeddings()
        vetor = gerador.gerar_embedding("Norma Regulamentadora 17 sobre ergonomia")
        self.assertIsInstance(vetor, list)
        self.assertEqual(len(vetor), 768)
        self.assertTrue(all(isinstance(x, float) for x in vetor))

    def test_gerar_embeddings_lote(self):
        gerador = get_gerador_embeddings()
        textos = [
            "Pausas para descanso na NR-17",
            "Artigo 473 da CLT sobre faltas justificadas",
            "Certificado de Aprovação de EPI",
        ]
        vetores = gerador.gerar_embeddings(textos)
        self.assertEqual(len(vetores), 3)
        for v in vetores:
            self.assertEqual(len(v), 768)

    @patch("sinergia.rag.embeddings.SentenceTransformer")
    def test_chamada_sentence_transformer(self, mock_st_cls):
        mock_model = MagicMock()
        fake_vector = [0.1] * 768
        mock_model.encode.return_value = fake_vector
        mock_st_cls.return_value = mock_model

        gerador = GeradorEmbeddings(model_name="tardellirs/colibri-embed-ptbr")
        gerador._carregar_modelo()
        vetor = gerador.gerar_embedding("Texto de teste")

        self.assertEqual(len(vetor), 768)
        self.assertEqual(vetor[0], 0.1)


if __name__ == "__main__":
    unittest.main()
