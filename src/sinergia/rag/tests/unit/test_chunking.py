import os
import unittest
from unittest.mock import MagicMock, patch
from sinergia.rag.chunking import (
    ChunkDoc,
    chunking_estrutural,
    chunking_documento_pdf,
)

CAMINHO_MOCKS = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "mocks"
)


class TestChunking(unittest.TestCase):
    def test_chunking_estrutural_texto_compatibilidade(self):
        """Testa o chunking estrutural em modo compatibilidade para blocos de texto/markdown."""
        texto = (
            "# PROCEDIMENTO OPERACIONAL PADRÃO - POP 03\n\n"
            "## 1. OBJETIVO\n"
            "Descrever o procedimento de integração do colaborador.\n\n"
            "## 2. RESPONSABILIDADES\n"
            "O setor de RH é responsável pelo agendamento dos exames e treinamentos."
        )
        chunks = chunking_estrutural(
            documento=texto,
            titulo_documento="POP 03",
            pagina=1,
        )
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks:
            self.assertIn("POP 03", c.breadcrumb)
            self.assertEqual(c.pagina, 1)
            self.assertIn("hierarquia", c.metadados)
            self.assertIn("secoes_map", c.metadados)

    def test_chunking_documento_pdf_pop_com_pymupdf(self):
        """Testa o chunking estrutural unificado em PDF de manual/POP com tabelas e etapas."""
        pdf_path = os.path.join(CAMINHO_MOCKS, "base3_pop_procedimentos_treinamentos.pdf")
        if not os.path.exists(pdf_path):
            self.skipTest(f"Mock PDF não encontrado em {pdf_path}")

        chunks = chunking_documento_pdf(
            caminho_pdf=pdf_path,
            titulo_documento="POP-RH-042",
        )

        self.assertGreaterEqual(len(chunks), 5)
        # Validar ordem sequencial contínua
        self.assertEqual([c.ordem for c in chunks], list(range(1, len(chunks) + 1)))

        primeiro = chunks[0]
        self.assertIn("POP-RH-042", primeiro.breadcrumb)
        self.assertIn("pág.", primeiro.breadcrumb)
        # O conteúdo agora é texto limpo (não injeta mais [breadcrumb] no início do conteúdo)
        self.assertNotIn("[POP-RH-042", primeiro.conteudo)
        self.assertIn("breadcrumb", primeiro.metadados)

        # Validar metadados dinâmicos
        self.assertIn("hierarquia", primeiro.metadados)
        self.assertIn("secoes_map", primeiro.metadados)
        self.assertIn("coordenadas", primeiro.metadados)
        self.assertIn("bboxes", primeiro.metadados)
        self.assertTrue(isinstance(primeiro.metadados["hierarquia"], list))
        self.assertGreaterEqual(len(primeiro.metadados["hierarquia"]), 1)

        # Validar coordenadas das sentenças/blocos
        coordenadas = primeiro.metadados["coordenadas"]
        self.assertGreaterEqual(len(coordenadas), 1)
        coord1 = coordenadas[0]
        self.assertIn("pagina", coord1)
        self.assertIn("x0", coord1)
        self.assertIn("y0", coord1)
        self.assertIn("x1", coord1)
        self.assertIn("y1", coord1)
        self.assertGreater(coord1["x1"], coord1["x0"])
        self.assertGreater(coord1["y1"], coord1["y0"])

        # Validar chunks com seções H2/H3
        tem_secao_numerada = any(
            "1. HISTÓRICO" in c.secao or "2. OBJETIVO" in c.secao
            for c in chunks
        )
        self.assertTrue(tem_secao_numerada)

    def test_chunking_documento_pdf_normativo_clt(self):
        """Testa o chunking unificado em documento normativo (CLT), sem ruído de regex."""
        pdf_path = os.path.join(CAMINHO_MOCKS, "base2_clt_direitos_beneficios.pdf")
        if not os.path.exists(pdf_path):
            self.skipTest(f"Mock PDF não encontrado em {pdf_path}")

        chunks = chunking_documento_pdf(
            caminho_pdf=pdf_path,
            titulo_documento="CLT",
        )

        self.assertGreaterEqual(len(chunks), 5)
        self.assertEqual([c.ordem for c in chunks], list(range(1, len(chunks) + 1)))

        # Verifica se as seções de capítulos e artigos foram capturadas na hierarquia do PyMuPDF
        textos_completos = " ".join(c.conteudo for c in chunks)
        self.assertIn("473", textos_completos)
        for c in chunks:
            self.assertIn("CLT", c.breadcrumb)
            self.assertIn("coordenadas", c.metadados)
            self.assertIn("hierarquia", c.metadados)

    def test_chunking_documento_pdf_normativo_nr17(self):
        """Testa o chunking unificado em documento da NR-17."""
        pdf_path = os.path.join(CAMINHO_MOCKS, "base1_nr17_ergonomia.pdf")
        if not os.path.exists(pdf_path):
            self.skipTest(f"Mock PDF não encontrado em {pdf_path}")

        chunks = chunking_documento_pdf(
            caminho_pdf=pdf_path,
            titulo_documento="NR-17 Ergonomia",
        )

        self.assertGreaterEqual(len(chunks), 5)
        self.assertTrue(all(isinstance(c, ChunkDoc) for c in chunks))
        self.assertTrue(all(c.breadcrumb is not None for c in chunks))
        self.assertTrue(all(c.ordem >= 1 for c in chunks))

    def test_chunking_documento_pdf_arquivo_inexistente(self):
        """Testa que erro FileNotFoundError é levantado se o PDF não existir."""
        with self.assertRaises(FileNotFoundError):
            chunking_documento_pdf("arquivo_inexistente.pdf", "Doc")


if __name__ == "__main__":
    unittest.main()
