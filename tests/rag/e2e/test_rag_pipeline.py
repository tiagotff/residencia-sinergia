"""Testes de ponta a ponta (E2E) do subsistema RAG validando o fluxo completo de ingestão até busca."""

import os
import unittest

from sinergia.rag.chunking import chunking_documento_pdf
from sinergia.rag.ingestao import mapear_mock_para_metadados
from sinergia.rag.interfaces import MetadadosTrecho, TrechoRecuperado
from sinergia.rag.retrievers import BuscadorHibrido

CAMINHO_MOCKS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "rag", "mocks")
)


class TestRAGPipelineE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Carrega e indexa os PDFs de Base de Conhecimento para o pipeline E2E."""
        cls.buscador = BuscadorHibrido()
        cls.arquivos_mock = [
            "rag_ergonomia.pdf",
            "rag_trabalho.pdf",
            "rag_pcmso_saude_ocupacional.pdf",
            "rag_epi.pdf",
        ]

        cls.total_chunks = 0
        todos_trechos = []
        bases_carregadas = set()

        for arq in cls.arquivos_mock:
            pdf_path = os.path.join(CAMINHO_MOCKS, arq)
            if not os.path.exists(pdf_path):
                continue

            meta_info = mapear_mock_para_metadados(arq)
            chunks = chunking_documento_pdf(
                caminho_pdf=pdf_path,
                titulo_documento=meta_info["titulo"],
                max_chars=700,
                overlap=120,
            )

            bases_carregadas.add(meta_info["id_base"])
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
                        extra=c.metadados,
                    ),
                )
                todos_trechos.append(trecho)
                cls.total_chunks += 1

        # Caso Base 3 (POP) não possua PDF em data/rag/mocks, adiciona trecho sintético para validar isolamento
        if 3 not in bases_carregadas:
            todos_trechos.append(
                TrechoRecuperado(
                    id_trecho=cls.total_chunks + 1,
                    conteudo="POP-001 | Qual é o procedimento operacional padrão para treinamento e capacitação inicial?",
                    metadados=MetadadosTrecho(
                        titulo="POP - Procedimentos e Treinamentos Corporativos",
                        fonte="Diretoria de Operações e Recursos Humanos",
                        versao="1.0",
                        secao="POP-001 | Treinamento inicial",
                        pagina=1,
                        id_base=3,
                        ordem=1,
                        breadcrumb="POP > Treinamento",
                        vigente=True,
                    ),
                )
            )
            cls.total_chunks += 1

        cls.buscador.adicionar_trechos_memoria(todos_trechos)

    def test_01_mocks_presentes_e_ingeridos(self):
        """Valida que os PDFs de mock foram carregados com chunks estruturados dentro do limite."""
        self.assertGreaterEqual(self.total_chunks, 100)

    def test_02_recuperacao_base1_pausas(self):
        """Base 1 (F1): Valida busca sobre regras de ergonomia, pausas e regra 20-20-20."""
        contexto = self.buscador.buscar(
            query="pausas descanso ergonomia regra 20-20-20 teleatendimento",
            id_base=1,
        )
        self.assertGreater(len(contexto.trechos), 0)
        self.assertLessEqual(len(contexto.trechos), 4)

        top1 = contexto.trechos[0]
        self.assertEqual(top1.metadados.id_base, 1)
        self.assertTrue(top1.metadados.vigente)
        self.assertIn("Ergonomia", top1.metadados.titulo)

        prompt_formatado = contexto.formatar_para_prompt()
        self.assertIn("[FONTE 1]: Base de Conhecimento RH: Pausas e Ergonomia", prompt_formatado)
        self.assertIn("ID Pergunta:", prompt_formatado)
        self.assertIn("Fonte do Trecho:", prompt_formatado)
        self.assertIn("Página:", prompt_formatado)
        self.assertIn("Conteúdo:", prompt_formatado)

    def test_03_recuperacao_base2_clt(self):
        """Base 2 (F2): Valida busca por licenças remuneradas, férias e vale-transporte."""
        contexto = self.buscador.buscar(
            query="quantos dias de licença maternidade paternidade casamento falecimento",
            id_base=2,
        )
        self.assertGreater(len(contexto.trechos), 0)
        for t in contexto.trechos:
            self.assertEqual(t.metadados.id_base, 2)
            self.assertTrue(t.metadados.vigente)

        prompt_formatado = contexto.formatar_para_prompt()
        self.assertIn("Benefícios e Direitos Trabalhistas", prompt_formatado)

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
        """Base 4 (F4): Valida busca sobre PCMSO, exames periódicos e saúde ocupacional."""
        contexto = self.buscador.buscar(
            query="exame periódico audiometria PCMSO ASO",
            id_base=4,
        )
        self.assertGreater(len(contexto.trechos), 0)
        top = contexto.trechos[0]
        self.assertEqual(top.metadados.id_base, 4)

    def test_06_recuperacao_base5_epis(self):
        """Base 5 (F5/F6): Valida busca por Certificado de Aprovação (CA) e conservação de EPIs."""
        contexto = self.buscador.buscar(
            query="certificado de aprovacao CA luva nitrilica validade",
            id_base=5,
        )
        self.assertGreater(len(contexto.trechos), 0)
        top = contexto.trechos[0]
        self.assertEqual(top.metadados.id_base, 5)

    def test_07_isolamento_estrito_entre_bases(self):
        """Garante que consultar a Base 1 nunca retorna trechos da Base 2 ou outras."""
        contexto = self.buscador.buscar(
            query="décimo terceiro salário férias abono pecuniário",
            id_base=1,
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
                vigente=False,
            ),
        )
        self.buscador.adicionar_trechos_memoria([trecho_obsoleto])

        contexto = self.buscador.buscar(
            query="Norma NR-17 de 1978 revogada",
            id_base=1,
        )
        ids_recuperados = [t.id_trecho for t in contexto.trechos]
        self.assertNotIn(999, ids_recuperados)


if __name__ == "__main__":
    unittest.main()
