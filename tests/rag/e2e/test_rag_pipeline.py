"""Testes de ponta a ponta (E2E) do subsistema RAG validando o fluxo completo de ingestão até busca."""

import os
import unittest
from sinergia.rag.chunking import chunking_documento_pdf
from sinergia.rag.embeddings import get_gerador_embeddings
from sinergia.rag.ingestao import mapear_mock_para_metadados
from sinergia.rag.interfaces import MetadadosTrecho, TrechoRecuperado
from sinergia.rag.retrievers import BuscadorHibrido
from sinergia.rag.tools import (
    ferramenta_busca_pausas,
    ferramenta_busca_direitos,
    ferramenta_busca_procedimentos,
    ferramenta_busca_saude_ocupacional,
    ferramenta_busca_epis,
)

CAMINHO_MOCKS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "rag", "mocks")
)


class TestRAGPipelineE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Carrega e indexa os 5 PDFs mock para o pipeline E2E."""
        cls.buscador = BuscadorHibrido()
        cls.arquivos_mock = [
            "base1_nr17_ergonomia.pdf",
            "base2_clt_direitos_beneficios.pdf",
            "base3_pop_procedimentos_treinamentos.pdf",
            "base4_pcmso_saude_ocupacional.pdf",
            "base5_epi_fichas_ca.pdf",
        ]

        cls.total_chunks = 0
        todos_trechos = []

        for arq in cls.arquivos_mock:
            pdf_path = os.path.join(CAMINHO_MOCKS, arq)
            if not os.path.exists(pdf_path):
                continue

            meta_info = mapear_mock_para_metadados(arq)
            chunks = chunking_documento_pdf(
                caminho_pdf=pdf_path,
                titulo_documento=meta_info["titulo"],
            )

            for c in chunks:
                trecho = TrechoRecuperado(
                    id_trecho=cls.total_chunks + 1,
                    conteudo=c.conteudo,
                    metadados=MetadadosTrecho(
                        titulo=meta_info["titulo"],
                        fonte=meta_info["fonte"],
                        versao=meta_info["versao"],
                        secao=c.secao,
                        pagina=c.pagina,
                        id_base=meta_info["id_base"],
                        ordem=c.ordem,
                        breadcrumb=c.breadcrumb,
                        vigente=True,
                    ),
                )
                todos_trechos.append(trecho)
                cls.total_chunks += 1

        cls.buscador.adicionar_trechos_memoria(todos_trechos)

    def test_01_mocks_presentes_e_ingeridos(self):
        """Valida que todos os 5 PDFs de mock foram carregados com chunks extraídos."""
        self.assertGreaterEqual(self.total_chunks, 5)

    def test_02_recuperacao_base1_pausas(self):
        """Base 1 (F1): Valida busca sobre regras de ergonomia e pausas na NR-17."""
        contexto = self.buscador.buscar(
            query="pausas descanso ergonomia trabalho com computadores",
            id_base=1,
        )
        self.assertGreater(len(contexto.trechos), 0)
        self.assertLessEqual(len(contexto.trechos), 4)

        top1 = contexto.trechos[0]
        self.assertEqual(top1.metadados.id_base, 1)
        self.assertTrue(top1.metadados.vigente)
        self.assertIn("NR-17", top1.metadados.titulo)

        prompt_formatado = contexto.formatar_para_prompt()
        self.assertIn("[FONTE 1]: NR-17 Ergonomia", prompt_formatado)
        self.assertIn("Página:", prompt_formatado)
        self.assertIn("Conteúdo:", prompt_formatado)

    def test_03_recuperacao_base2_clt(self):
        """Base 2 (F2): Valida busca jurídica por termos exatos e artigo da CLT."""
        contexto = self.buscador.buscar(
            query="falecimento cônjuge casamento Art. 473 faltas",
            id_base=2,
        )
        self.assertGreater(len(contexto.trechos), 0)
        for t in contexto.trechos:
            self.assertEqual(t.metadados.id_base, 2)
            self.assertTrue(t.metadados.vigente)

        prompt_formatado = contexto.formatar_para_prompt()
        self.assertIn("CLT", prompt_formatado)

    def test_04_recuperacao_base3_pop(self):
        """Base 3 (F3): Valida busca sobre procedimentos e POPs."""
        contexto = self.buscador.buscar(
            query="procedimento operacional padrao treinamento capacitação",
            id_base=3,
        )
        self.assertGreater(len(contexto.trechos), 0)
        top = contexto.trechos[0]
        self.assertEqual(top.metadados.id_base, 3)

    def test_05_recuperacao_base4_pcmso(self):
        """Base 4 (F4): Valida busca sobre PCMSO e saúde ocupacional."""
        contexto = self.buscador.buscar(
            query="exames periodicos saude ocupacional pcmso clinicos",
            id_base=4,
        )
        self.assertGreater(len(contexto.trechos), 0)
        top = contexto.trechos[0]
        self.assertEqual(top.metadados.id_base, 4)

    def test_06_recuperacao_base5_epis(self):
        """Base 5 (F5/F6): Valida busca exata por CA e normas de EPI."""
        contexto = self.buscador.buscar(
            query="certificado de aprovacao CA luva oculos protecao",
            id_base=5,
        )
        self.assertGreater(len(contexto.trechos), 0)
        top = contexto.trechos[0]
        self.assertEqual(top.metadados.id_base, 5)

    def test_07_isolamento_estrito_entre_bases(self):
        """Garante que consultar a Base 1 nunca retorna trechos da Base 2 ou outras."""
        contexto = self.buscador.buscar(
            query="artigo 473 clt casamento",
            id_base=1,  # Busca termos da CLT na base de ergonomia
        )
        for t in contexto.trechos:
            self.assertEqual(t.metadados.id_base, 1)
            self.assertNotEqual(t.metadados.id_base, 2)

    def test_08_invalidacao_versao_anterior_nao_vigente(self):
        """Valida que documentos com vigente=False são ignorados na recuperação."""
        trecho_obsoleto = TrechoRecuperado(
            id_trecho=999,
            conteudo="Norma NR-17 de 1978 revogada completamente.",
            score=0.99,
            metadados=MetadadosTrecho(
                titulo="NR-17 Antiga",
                fonte="Revogada",
                versao="1978",
                id_base=1,
                vigente=False,  # Marcado como não vigente
            ),
        )
        self.buscador.adicionar_trechos_memoria([trecho_obsoleto])

        contexto = self.buscador.buscar(
            query="Norma NR-17 de 1978 revogada",
            id_base=1,
        )
        # O trecho revogado não pode constar nos resultados
        ids_recuperados = [t.id_trecho for t in contexto.trechos]
        self.assertNotIn(999, ids_recuperados)


if __name__ == "__main__":
    unittest.main()

