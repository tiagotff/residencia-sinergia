import unittest
from sinergia.rag.interfaces import (
    ContextoRAG,
    DetalheScoreTrecho,
    EstatisticasEmbedding,
    MetadadosTrecho,
    ResultadoBuscaInstrumentada,
    TrechoRecuperado,
)


class TestInterfacesRAG(unittest.TestCase):
    def test_metadados_trecho_criacao(self):
        meta = MetadadosTrecho(
            titulo="NR-17 Ergonomia",
            fonte="Portaria MTP nº 423",
            versao="2024.1",
            secao="Item 17.5",
            pagina=2,
            id_base=1,
            ordem=1,
            breadcrumb="NR-17 > Item 17.5 > pág. 2",
            vigente=True,
        )
        self.assertEqual(meta.titulo, "NR-17 Ergonomia")
        self.assertEqual(meta.versao, "2024.1")
        self.assertEqual(meta.secao, "Item 17.5")
        self.assertEqual(meta.pagina, 2)
        self.assertTrue(meta.vigente)

    def test_trecho_recuperado_criacao(self):
        meta = MetadadosTrecho(
            titulo="CLT",
            fonte="Decreto-Lei nº 5.452",
            versao="1943",
            secao="Art. 473",
            pagina=15,
        )
        trecho = TrechoRecuperado(
            id_trecho=101,
            conteudo="O empregado poderá deixar de comparecer ao serviço sem prejuízo do salário...",
            score=0.032,
            metadados=meta,
        )
        self.assertEqual(trecho.id_trecho, 101)
        self.assertIn("deixar de comparecer", trecho.conteudo)
        self.assertEqual(trecho.score, 0.032)
        self.assertEqual(trecho.metadados.secao, "Art. 473")

    def test_contexto_rag_formatar_para_prompt(self):
        meta1 = MetadadosTrecho(
            titulo="NR-17 Ergonomia",
            fonte="Portaria MTP nº 423",
            versao="2024.1",
            secao="Item 17.5.3",
            pagina=4,
            breadcrumb="Normas Regulamentadoras > NR-17 Ergonomia > Item 17.5.3",
        )
        t1 = TrechoRecuperado(
            id_trecho=1,
            conteudo="Devem ser incluídas pausas para descanso na jornada...",
            score=0.033,
            metadados=meta1,
        )
        meta2 = MetadadosTrecho(
            titulo="CLT",
            fonte="Decreto-Lei nº 5.452",
            versao="1943",
            secao="Art. 71",
            pagina=20,
        )
        t2 = TrechoRecuperado(
            id_trecho=2,
            conteudo="Em qualquer trabalho contínuo cuja duração exceda de 6 horas...",
            score=0.031,
            metadados=meta2,
        )
        contexto = ContextoRAG(
            query="quais são as regras de pausas?",
            id_base=1,
            trechos=[t1, t2],
            total_encontrado=2,
        )

        prompt_str = contexto.formatar_para_prompt()
        self.assertIn("NR-17 Ergonomia", prompt_str)
        self.assertIn("- Trilha: Normas Regulamentadoras > NR-17 Ergonomia > Item 17.5.3", prompt_str)
        self.assertIn("2024.1", prompt_str)
        self.assertIn("Item 17.5.3", prompt_str)
        self.assertIn("Página: 4", prompt_str)
        self.assertIn("Decreto-Lei nº 5.452", prompt_str)
        self.assertIn("CLT", prompt_str)
        self.assertIn("Art. 71", prompt_str)
        self.assertIn("Página: 20", prompt_str)
        self.assertIn("Devem ser incluídas pausas", prompt_str)

    def test_contexto_rag_vazio(self):
        contexto = ContextoRAG(query="teste", id_base=1, trechos=[])
        prompt_str = contexto.formatar_para_prompt()
        self.assertIn("Nenhum trecho ou documento encontrado", prompt_str)

    def test_pydantic_serializacao(self):
        meta = MetadadosTrecho(
            titulo="PCMSO",
            fonte="Doc 1",
            versao="1.0",
            extra={"hierarquia": ["H1", "H2"], "coordenadas": [{"pagina": 1, "x0": 10.0}]},
        )
        trecho = TrechoRecuperado(
            conteudo="Exames periódicos",
            metadados=meta,
        )
        contexto = ContextoRAG(trechos=[trecho])
        data = contexto.model_dump()
        self.assertEqual(len(data["trechos"]), 1)
        self.assertEqual(data["trechos"][0]["metadados"]["titulo"], "PCMSO")
        self.assertEqual(data["trechos"][0]["metadados"]["extra"]["hierarquia"], ["H1", "H2"])

    def test_modelos_telemetria_serializacao(self):
        meta = MetadadosTrecho(
            id_documento=1,
            titulo="NR-17 Ergonomia",
            fonte="Portaria MTP nº 423",
            versao="2024.1",
            secao="Item 17.6.4",
            pagina=4,
            id_base=1,
            vigente=True,
        )
        trecho = TrechoRecuperado(
            id_trecho=42,
            conteudo="Pausa de 10 minutos para cada 50 minutos trabalhados.",
            score=0.0328,
            metadados=meta,
        )
        detalhe = DetalheScoreTrecho(
            id_trecho=42,
            conteudo=trecho.conteudo,
            metadados=meta,
            rrf_score=0.0328,
            dense_score=0.8412,
            dense_rank=1,
            lex_score=0.4120,
            lex_rank=1,
        )
        stats = EstatisticasEmbedding(
            modelo="tardellirs/colibri-embed-ptbr",
            dimensoes=768,
            norma_l2=1.0,
            min_val=-0.11,
            max_val=0.13,
            mean_val=0.001,
            amostra=[0.01, -0.02],
            vetor=[0.01, -0.02],
        )
        contexto = ContextoRAG(
            query="pausa digitação",
            id_base=1,
            trechos=[trecho],
            total_encontrado=1,
        )
        resultado = ResultadoBuscaInstrumentada(
            contexto=contexto,
            embed_ms=15.2,
            db_rrf_ms=6.8,
            estatisticas_embedding=stats,
            detalhes_trechos=[detalhe],
        )

        dumped = resultado.model_dump()
        self.assertEqual(dumped["embed_ms"], 15.2)
        self.assertEqual(dumped["db_rrf_ms"], 6.8)
        self.assertEqual(dumped["estatisticas_embedding"]["dimensoes"], 768)
        self.assertEqual(dumped["detalhes_trechos"][0]["dense_rank"], 1)
        self.assertEqual(dumped["detalhes_trechos"][0]["lex_rank"], 1)
        self.assertAlmostEqual(dumped["detalhes_trechos"][0]["rrf_score"], 0.0328)
        self.assertEqual(dumped["contexto"]["total_encontrado"], 1)


if __name__ == "__main__":
    unittest.main()

