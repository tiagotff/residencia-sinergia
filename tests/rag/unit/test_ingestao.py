import os
import unittest
from unittest.mock import MagicMock, patch
from sinergia.rag.ingestao import ingerir_documento, ingerir_diretorio_mocks, mapear_mock_para_metadados

CAMINHO_MOCKS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "rag", "mocks"
)


class TestIngestao(unittest.TestCase):
    def test_mapear_mock_para_metadados(self):
        meta = mapear_mock_para_metadados("base1_nr17_ergonomia.pdf")
        self.assertEqual(meta["id_base"], 1)
        self.assertIn("NR-17", meta["titulo"])

        meta_clt = mapear_mock_para_metadados("base2_clt_direitos_beneficios.pdf")
        self.assertEqual(meta_clt["id_base"], 2)

    def test_ingerir_documento_com_mock_db(self):
        pdf_path = os.path.join(CAMINHO_MOCKS, "base1_nr17_ergonomia.pdf")
        if not os.path.exists(pdf_path):
            self.skipTest(f"Mock não encontrado em {pdf_path}")

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = [42]  # id_documento retornado

        mock_psycopg = MagicMock()
        mock_psycopg.connect.return_value.__enter__.return_value = mock_conn

        with patch.dict("sys.modules", {"psycopg": mock_psycopg}):
            res = ingerir_documento(
                caminho_pdf=pdf_path,
                titulo="NR-17 Ergonomia",
                id_base=1,
                versao="2024.2",
                fonte="Portaria MTP 423",
                database_url="postgresql://sinergia:sinergia@localhost:5432/sinergia",
            )

            self.assertEqual(res["status"], "sucesso")
            self.assertEqual(res["id_documento"], 42)
            self.assertGreater(res["chunks_ingeridos"], 0)

            # Verificar que a invalidação da versão anterior foi chamada
            calls_sql = [call[0][0] for call in mock_cursor.execute.call_args_list]
            self.assertTrue(any("UPDATE rag.documento" in s and "SET vigente = false" in s for s in calls_sql))
            # Verificar inserção do documento novo
            self.assertTrue(any("INSERT INTO rag.documento" in s for s in calls_sql))
            # Verificar inserção dos trechos com metadados JSONB
            self.assertTrue(any("INSERT INTO rag.trecho" in s and "metadados" in s for s in calls_sql))
            # Verificar commit da transação
            mock_conn.commit.assert_called_once()

    def test_ingerir_documento_arquivo_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            ingerir_documento(
                caminho_pdf="arquivo_fantasma.pdf",
                titulo="Inexistente",
                id_base=1,
                versao="1.0",
            )


if __name__ == "__main__":
    unittest.main()
