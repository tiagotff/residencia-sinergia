import os
import unittest

from sinergia.rag.chunking import (
    MAX_CHARS_PADRAO,
    OVERLAP_PADRAO,
    ChunkDoc,
    chunking_documento_pdf,
    chunking_estrutural,
    dividir_texto_com_overlap,
    extrair_metadados_cabecalho_pdf,
)
from sinergia.rag.interfaces import MetadadosCabecalhoKB

CAMINHO_MOCKS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "rag", "mocks"
)


class TestChunking(unittest.TestCase):
    def test_dividir_texto_com_overlap_respeita_limites(self):
        """Valida limite estrito de caracteres e sobreposição real entre janelas consecutivas."""
        sentencas = [
            f"Regra operacional número {i:02d} para segurança e prevenção de riscos ocupacionais no posto."
            for i in range(1, 12)
        ]
        texto = " ".join(sentencas)

        partes = dividir_texto_com_overlap(texto, max_chars=220, overlap=60)
        self.assertGreaterEqual(len(partes), 4)
        for parte in partes:
            self.assertLessEqual(len(parte), 220)

        for idx in range(len(partes) - 1):
            palavras_fim = partes[idx].split()[-4:]
            self.assertTrue(
                any(p in partes[idx + 1] for p in palavras_fim),
                f"Esperava overlap entre parte {idx + 1} e {idx + 2}",
            )

    def test_chunking_estrutural_texto_compatibilidade(self):
        """Testa o chunking estrutural em modo compatibilidade com max_chars e overlap."""
        texto = (
            "# PROCEDIMENTO OPERACIONAL PADRÃO - POP 03\n\n"
            "## 1. OBJETIVO\n"
            "Descrever o procedimento de integração do colaborador.\n\n"
            "## 2. RESPONSABILIDADES\n"
            + " ".join(
                [
                    "O setor de RH é responsável pelo agendamento dos exames ocupacionais e treinamentos iniciais."
                    for _ in range(5)
                ]
            )
        )
        chunks = chunking_estrutural(
            documento=texto,
            titulo_documento="POP 03",
            pagina=1,
            max_chars=240,
            overlap=50,
        )
        self.assertGreaterEqual(len(chunks), 3)
        for c in chunks:
            self.assertIn("POP 03", c.breadcrumb)
            self.assertEqual(c.pagina, 1)
            self.assertIn("hierarquia", c.metadados)
            self.assertIn("secoes_map", c.metadados)
            self.assertLessEqual(len(c.conteudo), 240)

    def test_extrair_metadados_cabecalho_todos_pdfs_kb(self):
        """Valida extração da tabela de governança da página 1 nos 4 PDFs de data/rag/mocks."""
        esperados = {
            "rag_ergonomia.pdf": ("Base de Conhecimento RH: Pausas e Ergonomia", "KB-RH-ERG-001", "2.0"),
            "rag_trabalho.pdf": ("Base de Conhecimento RH: Benefícios e Direitos Trabalhistas", "KB-RH-BEN-002", "1.0"),
            "rag_pcmso_saude_ocupacional.pdf": ("Base de Conhecimento Saúde Ocupacional", None, "1.0"),
            "rag_epi.pdf": ("Base de Conhecimento EPI", None, "1.2"),
        }
        for nome_pdf, (titulo_esp, codigo_esp, versao_esp) in esperados.items():
            pdf_path = os.path.join(CAMINHO_MOCKS, nome_pdf)
            self.assertTrue(os.path.exists(pdf_path), f"PDF ausente: {pdf_path}")

            cab = extrair_metadados_cabecalho_pdf(pdf_path)
            self.assertIsInstance(cab, MetadadosCabecalhoKB)
            self.assertEqual(cab.titulo, titulo_esp)
            self.assertEqual(cab.codigo, codigo_esp)
            self.assertEqual(cab.versao, versao_esp)
            self.assertTrue(cab.convencoes)

    def test_chunking_layout_kb_rag_ergonomia(self):
        """Valida parsing do layout KB em rag_ergonomia.pdf, incluindo Q&A que atravessa página."""
        pdf_path = os.path.join(CAMINHO_MOCKS, "rag_ergonomia.pdf")
        chunks = chunking_documento_pdf(
            caminho_pdf=pdf_path,
            titulo_documento="Base de Conhecimento RH: Pausas e Ergonomia",
            max_chars=MAX_CHARS_PADRAO,
            overlap=OVERLAP_PADRAO,
        )

        self.assertEqual(len(chunks), 8)
        self.assertEqual([c.ordem for c in chunks], list(range(1, 9)))

        for c in chunks:
            self.assertLessEqual(len(c.conteudo), MAX_CHARS_PADRAO)
            self.assertNotIn("<mark", c.conteudo)
            self.assertNotIn("\ufb01", c.conteudo)
            self.assertNotIn("Aprovação", c.conteudo)
            self.assertIn("id_pergunta", c.metadados)
            self.assertIn("tambem_perguntam", c.metadados)
            self.assertIn("fonte_item", c.metadados)
            self.assertTrue(c.metadados["coordenadas"])

        # RH-ERG-Q003 começa na página 1 e sua Fonte fica no topo da página 2
        q003 = next(c for c in chunks if c.metadados["id_pergunta"] == "RH-ERG-Q003")
        self.assertEqual(q003.metadados["page_start"], 1)
        self.assertEqual(q003.metadados["page_end"], 2)
        self.assertIn("Fonte: Política Interna RH/TST (INTERNO)", q003.conteudo)
        self.assertEqual(q003.metadados["natureza_fonte"], ["INTERNO"])

    def test_chunking_layout_kb_rag_trabalho_com_subchunk_e_overlap(self):
        """Valida sub-chunking com overlap, limpeza de ligaturas e continuação entre páginas em rag_trabalho.pdf."""
        pdf_path = os.path.join(CAMINHO_MOCKS, "rag_trabalho.pdf")
        chunks = chunking_documento_pdf(
            caminho_pdf=pdf_path,
            max_chars=700,
            overlap=120,
        )

        self.assertGreaterEqual(len(chunks), 12)
        for c in chunks:
            self.assertLessEqual(len(c.conteudo), 700)
            self.assertNotIn("<mark", c.conteudo)
            self.assertNotIn("\ufb01", c.conteudo)

        # Verifica limpeza de ligatura em 13º Salário (Gratificação) e Faltas (justificada)
        q002 = next(c for c in chunks if c.metadados["id_pergunta"] == "RH-BEN-Q002")
        self.assertIn("13º Salário (Gratificação Natalina)", q002.conteudo)

        # RH-BEN-Q003 e RH-BEN-Q007 excedem 700 chars e devem ter sido divididos mantendo âncora Q&A
        partes_q003 = [c for c in chunks if c.metadados["id_pergunta"] == "RH-BEN-Q003"]
        self.assertGreaterEqual(len(partes_q003), 2)
        for p in partes_q003:
            self.assertIn("RH-BEN-Q003 | Quais são os prazos e regras das Licenças Remuneradas?", p.conteudo)
            self.assertIn("Fonte: CLT (OFICIAL); Programa Empresa Cidadã (INTERNO)", p.conteudo)
            self.assertLessEqual(len(p.conteudo), 700)

        # Valida sobreposição textual entre parte 1 e parte 2 de RH-BEN-Q003
        self.assertEqual(partes_q003[0].metadados["total_partes"], len(partes_q003))
        self.assertIn("Luto (Nojo)", partes_q003[0].conteudo)
        self.assertIn("Luto (Nojo)", partes_q003[1].conteudo)

        # RH-BEN-Q007 atravessa pág. 2 -> pág. 3
        partes_q007 = [c for c in chunks if c.metadados["id_pergunta"] == "RH-BEN-Q007"]
        self.assertGreaterEqual(len(partes_q007), 2)
        self.assertEqual(partes_q007[0].metadados["page_start"], 2)
        self.assertEqual(partes_q007[0].metadados["page_end"], 3)

    def test_chunking_layout_kb_pcmso_e_epi_respeitam_max_chars_personalizado(self):
        """Valida parsing de rag_pcmso_saude_ocupacional.pdf e rag_epi.pdf incluindo max_chars=350."""
        for nome_pdf, prefixo_id in [
            ("rag_pcmso_saude_ocupacional.pdf", "SO-Q"),
            ("rag_epi.pdf", "EPI-Q"),
        ]:
            pdf_path = os.path.join(CAMINHO_MOCKS, nome_pdf)
            chunks_padrao = chunking_documento_pdf(pdf_path, max_chars=700, overlap=120)
            self.assertGreaterEqual(len(chunks_padrao), 55)
            for c in chunks_padrao:
                self.assertLessEqual(len(c.conteudo), 700)
                self.assertTrue(c.metadados["id_pergunta"].startswith(prefixo_id))
                self.assertNotIn("<mark", c.conteudo)
                self.assertNotIn("\ufb01", c.conteudo)

            chunks_curtos = chunking_documento_pdf(pdf_path, max_chars=350, overlap=80)
            self.assertGreater(len(chunks_curtos), len(chunks_padrao))
            for c in chunks_curtos:
                self.assertLessEqual(len(c.conteudo), 350)

        # Verifica reconstrução da lista numerada de EPI-Q021
        epi_chunks = chunking_documento_pdf(
            os.path.join(CAMINHO_MOCKS, "rag_epi.pdf"),
            max_chars=700,
            overlap=120,
        )
        q021_partes = [c for c in epi_chunks if c.metadados["id_pergunta"] == "EPI-Q021"]
        self.assertGreaterEqual(len(q021_partes), 2)
        self.assertIn("1. Relato:", q021_partes[0].conteudo)
        self.assertIn("5. Uso correto:", q021_partes[-1].conteudo)

    def test_chunking_documento_pdf_arquivo_inexistente(self):
        """Testa que erro FileNotFoundError é levantado se o PDF não existir."""
        with self.assertRaises(FileNotFoundError):
            chunking_documento_pdf("arquivo_inexistente.pdf", "Doc")


if __name__ == "__main__":
    unittest.main()
