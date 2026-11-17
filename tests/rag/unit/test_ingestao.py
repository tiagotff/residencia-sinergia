import os
import unittest
from unittest.mock import MagicMock, patch

from sinergia.rag.ingestao import (
    ingerir_diretorio_mocks,
    ingerir_documento,
    mapear_mock_para_metadados,
)

CAMINHO_MOCKS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "rag", "mocks"
)


class TestIngestao(unittest.TestCase):
    def test_mapear_mock_para_metadados(self):
        meta_erg = mapear_mock_para_metadados("rag_ergonomia.pdf")
        self.assertEqual(meta_erg["id_base"], 1)
        self.assertIn("Ergonomia", meta_erg["titulo"])

        meta_trab = mapear_mock_para_metadados("rag_trabalho.pdf")
        self.assertEqual(meta_trab["id_base"], 2)
        self.assertIn("Benefícios", meta_trab["titulo"])

        meta_pcmso = mapear_mock_para_metadados("rag_pcmso_saude_ocupacional.pdf")
        self.assertEqual(meta_pcmso["id_base"], 4)

        meta_epi = mapear_mock_para_metadados("rag_epi.pdf")
        self.assertEqual(meta_epi["id_base"], 5)

        # Compatibilidade com nomes legados
        self.assertEqual(mapear_mock_para_metadados("base1_nr17_ergonomia.pdf")["id_base"], 1)
        self.assertEqual(mapear_mock_para_metadados("base3_pop_procedimentos.pdf")["id_base"], 3)

    def test_ingerir_documento_com_mock_db(self):
        pdf_path = os.path.join(CAMINHO_MOCKS, "rag_ergonomia.pdf")
        self.assertTrue(os.path.exists(pdf_path), f"Mock não encontrado em {pdf_path}")

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = [42]

        mock_psycopg = MagicMock()
        mock_psycopg.connect.return_value.__enter__.return_value = mock_conn

        with patch.dict("sys.modules", {"psycopg": mock_psycopg}):
            res = ingerir_documento(
                caminho_pdf=pdf_path,
                id_base=1,
                database_url="postgresql://sinergia:sinergia@localhost:5432/sinergia",
                max_chars=700,
                overlap=120,
            )

            self.assertEqual(res["status"], "sucesso")
            self.assertEqual(res["id_documento"], 42)
            self.assertEqual(res["titulo"], "Base de Conhecimento RH: Pausas e Ergonomia")
            self.assertEqual(res["versao"], "2.0")
            self.assertEqual(res["chunks_ingeridos"], 8)

            calls_sql = [call[0][0] for call in mock_cursor.execute.call_args_list]
            self.assertTrue(any("UPDATE rag.documento" in s and "SET vigente = false" in s for s in calls_sql))
            self.assertTrue(any("INSERT INTO rag.documento" in s for s in calls_sql))
            self.assertTrue(any("INSERT INTO rag.trecho" in s and "metadados" in s for s in calls_sql))
            mock_conn.commit.assert_called_once()

    def test_ingerir_diretorio_mocks_dry_run(self):
        resultados = ingerir_diretorio_mocks(
            CAMINHO_MOCKS,
            dry_run=True,
            max_chars=700,
            overlap=120,
        )
        self.assertEqual(len(resultados), 4)
        for r in resultados:
            self.assertEqual(r["status"], "sucesso")
            self.assertEqual(r["modo"], "dry-run")
            self.assertGreater(r["chunks_ingeridos"], 0)
            self.assertEqual(r["max_chars"], 700)
            self.assertEqual(r["overlap"], 120)

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
