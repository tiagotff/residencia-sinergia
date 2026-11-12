import os
import unittest
from unittest.mock import MagicMock, patch
from sinergia.rag.chunking import (
    ChunkDoc,
    chunking_normativo,
    chunking_estrutural,
    chunking_documento_pdf,
    extrair_texto_pdf,
)

CAMINHO_MOCKS = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "mocks"
)


class TestChunking(unittest.TestCase):
    def test_chunking_normativo_preserva_artigos_e_excecoes(self):
        texto = (
            "CAPÍTULO IV - DO TEMPO DE TRABALHO\n\n"
            "Art. 473 - O empregado poderá deixar de comparecer ao serviço sem prejuízo do salário:\n"
            "I - até 2 (dois) dias consecutivos, em caso de falecimento do cônjuge;\n"
            "II - até 3 (três) dias consecutivos, em virtude de casamento;\n"
            "§ 1º O prazo referido no inciso I será contado a partir do óbito.\n"
            "§ 2º Em caso de emergência, admite-se prorrogação.\n\n"
            "Art. 474 - A suspensão do empregado por mais de 30 dias importa na rescisão do contrato."
        )
        chunks = chunking_normativo(
            texto=texto,
            titulo_documento="CLT",
            pagina=15,
        )

        self.assertGreaterEqual(len(chunks), 2)
        # Primeiro chunk deve conter o Art. 473 completo com incisos e parágrafos
        chunk1 = chunks[0]
        self.assertIn("Art. 473", chunk1.secao)
        self.assertIn("falecimento do cônjuge", chunk1.conteudo)
        self.assertIn("§ 1º", chunk1.conteudo)
        self.assertIn("§ 2º", chunk1.conteudo)
        self.assertIn("CLT", chunk1.breadcrumb)
        self.assertIn("Art. 473", chunk1.breadcrumb)
        self.assertEqual(chunk1.pagina, 15)

        # Segundo chunk deve ser Art. 474
        chunk2 = chunks[1]
        self.assertIn("Art. 474", chunk2.secao)
        self.assertIn("suspensão do empregado", chunk2.conteudo)

    def test_chunking_normativo_itens(self):
        texto = (
            "Item 17.5.3 Em todos os locais e situações de trabalho deve haver iluminação adequada.\n"
            "Item 17.5.3.1 A iluminação deve ser projetada para evitar reflexos incômodos."
        )
        chunks = chunking_normativo(
            texto=texto,
            titulo_documento="NR-17",
            pagina=5,
        )
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("17.5.3", chunks[0].secao)

    def test_chunking_estrutural(self):
        texto = (
            "PROCEDIMENTO OPERACIONAL PADRÃO - POP 03\n\n"
            "1. OBJETIVO\n"
            "Descrever o procedimento de integração do colaborador.\n\n"
            "2. RESPONSABILIDADES\n"
            "O setor de RH é responsável pelo agendamento dos exames e treinamentos."
        )
        chunks = chunking_estrutural(
            texto=texto,
            titulo_documento="POP 03",
            pagina=1,
        )
        self.assertGreaterEqual(len(chunks), 1)
        for c in chunks:
            self.assertIn("POP 03", c.breadcrumb)
            self.assertEqual(c.pagina, 1)

    def test_extrair_texto_pdf_e_chunking_pdf(self):
        pdf_path = os.path.join(CAMINHO_MOCKS, "base1_nr17_ergonomia.pdf")
        if not os.path.exists(pdf_path):
            self.skipTest(f"Mock PDF não encontrado em {pdf_path}")

        paginas = extrair_texto_pdf(pdf_path)
        self.assertGreaterEqual(len(paginas), 1)

        chunks = chunking_documento_pdf(
            caminho_pdf=pdf_path,
            titulo_documento="NR-17 Ergonomia",
        )
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(c, ChunkDoc) for c in chunks))
        self.assertTrue(all(c.breadcrumb is not None for c in chunks))
        self.assertTrue(all(c.ordem >= 1 for c in chunks))

    def test_extrair_texto_pdf_com_pymupdf(self):
        pdf_path = os.path.join(CAMINHO_MOCKS, "base1_nr17_ergonomia.pdf")
        if not os.path.exists(pdf_path):
            self.skipTest(f"Mock PDF não encontrado em {pdf_path}")

        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Página 1 do documento mockado"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Página 2 do documento mockado"

        mock_doc = MagicMock()
        mock_doc.__enter__.return_value = [mock_page1, mock_page2]
        mock_doc.__exit__.return_value = False

        mock_pymupdf = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch("sinergia.rag.chunking.pymupdf", mock_pymupdf):
            paginas = extrair_texto_pdf(pdf_path)

        mock_pymupdf.open.assert_called_once_with(pdf_path)
        self.assertEqual(len(paginas), 2)
        self.assertEqual(paginas[0], (1, "Página 1 do documento mockado"))
        self.assertEqual(paginas[1], (2, "Página 2 do documento mockado"))

    def test_extrair_texto_pdf_fallback_sem_pymupdf(self):
        pdf_path = os.path.join(CAMINHO_MOCKS, "base1_nr17_ergonomia.pdf")
        if not os.path.exists(pdf_path):
            self.skipTest(f"Mock PDF não encontrado em {pdf_path}")

        with patch("sinergia.rag.chunking.pymupdf", None):
            paginas = extrair_texto_pdf(pdf_path)
            self.assertGreaterEqual(len(paginas), 1)

    def test_extrair_texto_pdf_arquivo_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            extrair_texto_pdf("arquivo_que_nao_existe.pdf")


if __name__ == "__main__":
    unittest.main()
