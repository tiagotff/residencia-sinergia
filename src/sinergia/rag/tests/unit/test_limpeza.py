import unittest
from sinergia.rag.limpeza import (
    remover_cabecalhos_rodapes,
    deshifenizar,
    normalizar_espacos_e_controles,
    preservar_marcadores_juridicos,
    sanitizar_texto,
)


class TestLimpezaTextual(unittest.TestCase):
    def test_estagio_1_remover_cabecalhos_rodapes(self):
        texto = (
            "MINISTÉRIO DO TRABALHO E EMPREGO\n"
            "Norma Regulamentadora NR-17\n"
            "Conteúdo importante da norma.\n"
            "Página 1 de 10\n"
            "Mais conteúdo relevante.\n"
            "-- 2 --\n"
            "Pág. 3/10\n"
        )
        limpo = remover_cabecalhos_rodapes(texto)
        self.assertNotIn("Página 1 de 10", limpo)
        self.assertNotIn("Pág. 3/10", limpo)
        self.assertNotIn("-- 2 --", limpo)
        self.assertIn("Conteúdo importante da norma.", limpo)
        self.assertIn("Mais conteúdo relevante.", limpo)

    def test_estagio_2_deshifenizar(self):
        texto = "Este documento é confor-\nme as regras e de cumprimento obri-\ngatório."
        limpo = deshifenizar(texto)
        self.assertIn("conforme", limpo)
        self.assertIn("obrigatório", limpo)
        self.assertNotIn("confor-", limpo)
        self.assertNotIn("obri-", limpo)

    def test_estagio_3_normalizar_espacos_e_controles(self):
        texto = "Texto com\x00caracteres nulos e\x0cform feed.\n\n   Múltiplos   espaços   aqui.   "
        limpo = normalizar_espacos_e_controles(texto)
        self.assertNotIn("\x00", limpo)
        self.assertNotIn("\x0c", limpo)
        self.assertNotIn("   ", limpo)
        self.assertIn("Texto com caracteres nulos e form feed.", limpo)

    def test_estagio_4_preservar_marcadores_juridicos(self):
        texto = "algum texto preliminar Art. 473 O empregado poderá faltar § 1º Sem prejuízo Item 17.5 Ergonomia"
        resultado = preservar_marcadores_juridicos(texto)
        self.assertIn("Art. 473", resultado)
        self.assertIn("§ 1º", resultado)
        self.assertIn("Item 17.5", resultado)

    def test_pipeline_completo_sanitizar_texto(self):
        texto_bruto = (
            "MINISTÉRIO DO TRABALHO\n"
            "Página 1 de 5\n\n"
            "Art. 168 - O exame médico é confor-\n"
            "me legislação obri-\n"
            "gatória.\x0c\n"
            "-- 1 --\n"
            "§ 1º O exame clínico deve ser realizado."
        )
        sanitizado = sanitizar_texto(texto_bruto)
        self.assertNotIn("Página 1 de 5", sanitizado)
        self.assertNotIn("-- 1 --", sanitizado)
        self.assertNotIn("\x0c", sanitizado)
        self.assertIn("conforme", sanitizado)
        self.assertIn("obrigatória", sanitizado)
        self.assertIn("Art. 168", sanitizado)
        self.assertIn("§ 1º", sanitizado)


if __name__ == "__main__":
    unittest.main()

