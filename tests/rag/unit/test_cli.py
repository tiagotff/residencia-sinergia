import io
import unittest
from unittest.mock import MagicMock, patch

from sinergia.rag.cli import _executar_ferramenta, exibir_cabecalho, modo_interativo


class TestCLI(unittest.TestCase):
    def test_exibir_cabecalho(self):
        f = io.StringIO()
        with patch("sys.stdout", f):
            exibir_cabecalho()
        saida = f.getvalue()
        self.assertIn("SINERGIA - RAG Tools CLI", saida)
        self.assertIn("[1] F1 - Pausas e Ergonomia (NR-17)", saida)
        self.assertIn("[5] F5 - EPIs e Fichas de CA (NR-6)", saida)
        self.assertIn("[0] Sair", saida)

    def test_executar_ferramenta_funcao_simples(self):
        func = MagicMock(spec=["__call__"], return_value="resposta ok")
        res = _executar_ferramenta(func, "query de teste")
        func.assert_called_once_with(query="query de teste")
        self.assertEqual(res, "resposta ok")

    def test_executar_ferramenta_objeto_com_run(self):
        tool_obj = MagicMock()
        tool_obj.run.return_value = "resposta crewai tool"
        res = _executar_ferramenta(tool_obj, "query de teste")
        tool_obj.run.assert_called_once_with(query="query de teste")
        self.assertEqual(res, "resposta crewai tool")

    @patch("builtins.input", side_effect=["0"])
    def test_modo_interativo_sair_imediato(self, mock_input):
        f = io.StringIO()
        with patch("sys.stdout", f):
            modo_interativo()
        saida = f.getvalue()
        self.assertIn("Encerrando CLI de testes do RAG. Até logo!", saida)

    @patch("builtins.input", side_effect=["99", "1", "como funcionam as pausas?", "0"])
    @patch("sinergia.rag.cli.FERRAMENTAS")
    def test_modo_interativo_fluxo_busca(self, mock_ferramentas, mock_input):
        mock_funcao = MagicMock(spec=["run"])
        mock_funcao.run.return_value = "[Trecho 1: Pausas NR-17]"
        mock_ferramentas.__getitem__.return_value = {
            "codigo": "F1",
            "id_base": 1,
            "nome": "Pausas e Ergonomia (NR-17)",
            "funcao": mock_funcao,
            "descricao": "Pausas e descansos",
        }
        mock_ferramentas.__contains__.side_effect = lambda x: x == "1"
        mock_ferramentas.items.return_value = [
            (
                "1",
                {
                    "codigo": "F1",
                    "nome": "Pausas e Ergonomia (NR-17)",
                },
            )
        ]

        f = io.StringIO()
        with patch("sys.stdout", f):
            modo_interativo()
        saida = f.getvalue()

        self.assertIn("Opção inválida", saida)
        self.assertIn("Selecionada: [F1] Pausas e Ergonomia (NR-17)", saida)
        self.assertIn("[Buscando na Base 1...]", saida)
        self.assertIn("[Trecho 1: Pausas NR-17]", saida)

    @patch("builtins.input", side_effect=["1", "", "0"])
    def test_modo_interativo_query_vazia(self, mock_input):
        f = io.StringIO()
        with patch("sys.stdout", f):
            modo_interativo()
        saida = f.getvalue()
        self.assertIn("Retornando à seleção de opções", saida)

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_modo_interativo_keyboard_interrupt(self, mock_input):
        f = io.StringIO()
        with patch("sys.stdout", f):
            modo_interativo()
        saida = f.getvalue()
        self.assertIn("Operação cancelada pelo usuário. Até logo!", saida)
