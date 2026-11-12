import unittest
from unittest.mock import MagicMock, patch
from sinergia.rag.interfaces import MetadadosTrecho, TrechoRecuperado, ContextoRAG
from sinergia.rag.tools import (
    ferramenta_busca_pausas,
    ferramenta_busca_direitos,
    ferramenta_busca_procedimentos,
    ferramenta_busca_saude_ocupacional,
    ferramenta_busca_epis,
    consultar_base_conhecimento,
)


class TestToolsRAG(unittest.TestCase):
    def setUp(self):
        self.meta = MetadadosTrecho(
            titulo="Documento Teste",
            fonte="Manual Interno",
            versao="1.0",
            secao="Seção 1",
            pagina=3,
            id_base=1,
            vigente=True,
        )
        self.trecho = TrechoRecuperado(
            id_trecho=1,
            conteudo="Instruções sobre descanso e pausas na jornada de trabalho.",
            metadados=self.meta,
        )
        self.contexto_mock = ContextoRAG(
            query="pausas",
            id_base=1,
            trechos=[self.trecho],
            total_encontrado=1,
        )

    @patch("sinergia.rag.tools._get_buscador")
    def test_ferramenta_busca_pausas(self, mock_get_buscador):
        mock_buscador = MagicMock()
        mock_buscador.buscar.return_value = self.contexto_mock
        mock_get_buscador.return_value = mock_buscador

        resposta = ferramenta_busca_pausas("como funcionam as pausas?")
        mock_buscador.buscar.assert_called_once_with(query="como funcionam as pausas?", id_base=1)
        self.assertIn("Documento Teste", resposta)
        self.assertIn("Página: 3", resposta)

    @patch("sinergia.rag.tools._get_buscador")
    def test_ferramenta_busca_direitos(self, mock_get_buscador):
        mock_buscador = MagicMock()
        mock_buscador.buscar.return_value = self.contexto_mock
        mock_get_buscador.return_value = mock_buscador

        resposta = ferramenta_busca_direitos("faltas justificadas")
        mock_buscador.buscar.assert_called_once_with(query="faltas justificadas", id_base=2)
        self.assertIn("Documento Teste", resposta)

    @patch("sinergia.rag.tools._get_buscador")
    def test_ferramenta_busca_procedimentos(self, mock_get_buscador):
        mock_buscador = MagicMock()
        mock_buscador.buscar.return_value = self.contexto_mock
        mock_get_buscador.return_value = mock_buscador

        resposta = ferramenta_busca_procedimentos("POP de onboarding")
        mock_buscador.buscar.assert_called_once_with(query="POP de onboarding", id_base=3)
        self.assertIn("Documento Teste", resposta)

    @patch("sinergia.rag.tools._get_buscador")
    def test_ferramenta_busca_saude_ocupacional(self, mock_get_buscador):
        mock_buscador = MagicMock()
        mock_buscador.buscar.return_value = self.contexto_mock
        mock_get_buscador.return_value = mock_buscador

        resposta = ferramenta_busca_saude_ocupacional("exame periódico")
        mock_buscador.buscar.assert_called_once_with(query="exame periódico", id_base=4)
        self.assertIn("Documento Teste", resposta)

    @patch("sinergia.rag.tools._get_buscador")
    def test_ferramenta_busca_epis(self, mock_get_buscador):
        mock_buscador = MagicMock()
        mock_buscador.buscar.return_value = self.contexto_mock
        mock_get_buscador.return_value = mock_buscador

        resposta = ferramenta_busca_epis("óculos de proteção CA 9876")
        mock_buscador.buscar.assert_called_once_with(query="óculos de proteção CA 9876", id_base=5)
        self.assertIn("Documento Teste", resposta)

    @patch("sinergia.rag.tools._get_buscador")
    def test_consultar_base_conhecimento_generico(self, mock_get_buscador):
        mock_buscador = MagicMock()
        mock_buscador.buscar.return_value = self.contexto_mock
        mock_get_buscador.return_value = mock_buscador

        resposta = consultar_base_conhecimento(query="consulta geral", id_base=1)
        mock_buscador.buscar.assert_called_once_with(query="consulta geral", id_base=1)
        self.assertIn("Instruções sobre descanso", resposta)


if __name__ == "__main__":
    unittest.main()

