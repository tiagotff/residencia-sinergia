"""Testes da Task 1A.3 — Extrator e Conversador LangChain + Groq (`extrator_llm.py`).

Cobre:
- Curto-circuito offline para OCR ilegível/vazio sem gastar chamada de LLM;
- `MockLLM` determinístico e customizado (Política de CID, limiar `0.8`,
  rebaixamento para `ilegivel` quando campos essenciais ficam vazios);
- `analisar_conversa_atestado` sobre a fixture `falta_arquivo.tarefa.json` e
  cenário de alteração de contato (Anexo E.2);
- Cache de `obter_cliente_llm` (Anexo C.2) e validação de `GROQ_API_KEY`;
- Integração real ponta a ponta (`.storage/atestados/atestado_carla_105.png` ->
  `1A.1 ocr.py` -> `1A.3 extrator_llm.py` -> `1B.1 validador.py`) contra a Groq.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from sinergia.agentes.a1_recepcao import (
    A1Config,
    AnaliseConversaA1,
    AtestadoExtraido,
    MockLLM,
    PeriodoAfastamento,
    ProfissionalExtraido,
    ResultadoOCR,
    TarefaA1,
    analisar_conversa_atestado,
    detectar_divergencias,
    extrair_atestado_via_llm,
    gerar_mensagem_confirmacao_extracao,
    obter_cliente_llm,
    obter_config_a1,
    processar_arquivo_atestado,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
ATESTADO_CARLA_PNG = Path(".storage/atestados/atestado_carla_105.png")
_CFG = obter_config_a1()


def _carregar_fixture(relpath: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / relpath).read_text(encoding="utf-8"))


def test_curto_circuito_ocr_ilegivel_ou_vazio_nao_chama_llm() -> None:
    """Entradas vazias ou marcadas como ilegíveis no OCR não acionam o LLM."""
    mock = MockLLM()

    # 1. ResultadoOCR explicitamente ilegível
    ocr_ilegivel = ResultadoOCR(
        arquivo_uri=".storage/atestados/borrado.png",
        mime="image/png",
        texto_extraido="algum ruido",
        confianca_media=0.80,
        legibilidade="ilegivel",
    )
    ext1, dec1 = extrair_atestado_via_llm(ocr_ilegivel, llm=mock)
    assert ext1.legibilidade == "ilegivel"
    assert dec1.usou_llm is False
    assert dec1.modelo is None

    # 2. Texto vazio / espaços
    ext2, dec2 = extrair_atestado_via_llm("   \n ", llm=mock)
    assert ext2.legibilidade == "ilegivel"
    assert dec2.usou_llm is False

    # 3. Confiança média do OCR abaixo do mínimo (0.30)
    ext3, dec3 = extrair_atestado_via_llm(
        "ATESTADO MEDICO", llm=mock, confianca_media_ocr=0.10
    )
    assert ext3.legibilidade == "ilegivel"
    assert dec3.usou_llm is False
    assert dec3.confianca == 0.10
    assert mock.chamadas == 0


def test_mock_llm_extracao_politica_cid_e_limiar_confianca() -> None:
    """Valida extração offline com `MockLLM`, alternância de CID e limiar 0.8."""
    texto_carla = (
        "CLINICA MEDICA CAMPINAS\n"
        "Atesto que a Sra. Carla Souza necessita de 3 dias de afastamento "
        "a partir de 09/11/2026.\n"
        "CID: J06.9\n"
        "Campinas - SP, 09/11/2026\n"
        "Dr. Marcos Lima - CRM/SP 123456"
    )
    mock = MockLLM()

    ext_com_cid, dec_com = extrair_atestado_via_llm(
        texto_carla, llm=mock, extrair_cid=True, limiar_confianca=0.8
    )
    assert ext_com_cid.legibilidade == "ok"
    assert ext_com_cid.nome_no_documento == "Carla Souza"
    assert ext_com_cid.periodo.inicio == date(2026, 11, 9)
    assert ext_com_cid.periodo.fim == date(2026, 11, 11)
    assert ext_com_cid.profissional.conselho == "CRM"
    assert ext_com_cid.profissional.numero == "123456"
    assert ext_com_cid.profissional.uf == "SP"
    assert ext_com_cid.cid == "J06.9"
    assert dec_com.usou_llm is True
    assert dec_com.modelo == "mock/sinergia-a1"

    ext_sem_cid, _ = extrair_atestado_via_llm(
        texto_carla, llm=mock, extrair_cid=False, limiar_confianca=0.8
    )
    assert ext_sem_cid.cid is None
    assert mock.chamadas == 2

    # Limiar 0.8 zera campos duvidosos mantendo os confiáveis
    mock_parcial = MockLLM(
        extracao_padrao=AtestadoExtraido(
            legibilidade="ok",
            nome_no_documento="Carla Souza",
            periodo=PeriodoAfastamento(inicio=date(2026, 11, 9), fim=date(2026, 11, 11)),
            data_emissao=date(2026, 11, 9),
            profissional=ProfissionalExtraido(conselho="CRM", numero="123456", uf="SP"),
            cid="J06.9",
            municipio="Campinas",
            confianca_campos={
                "nome_no_documento": 0.95,
                "periodo.inicio": 0.94,
                "periodo.fim": 0.94,
                "data_emissao": 0.90,
                "profissional.conselho": 0.96,
                "profissional.numero": 0.95,
                "profissional.uf": 0.96,
                "cid": 0.72,
                "municipio": 0.50,
            },
        )
    )
    ext_limiar, _ = extrair_atestado_via_llm(
        texto_carla, llm=mock_parcial, extrair_cid=True, limiar_confianca=0.8
    )
    assert ext_limiar.legibilidade == "ok"
    assert ext_limiar.cid is None
    assert ext_limiar.municipio is None
    assert ext_limiar.profissional.numero == "123456"

    # Se campos essenciais (inicio, fim, numero) caírem abaixo do limiar, vira ilegível
    mock_baixo = MockLLM(
        extracao_padrao=AtestadoExtraido(
            legibilidade="ok",
            nome_no_documento="Carla Souza",
            periodo=PeriodoAfastamento(inicio=date(2026, 11, 9), fim=date(2026, 11, 11)),
            profissional=ProfissionalExtraido(conselho="CRM", numero="123456", uf="SP"),
            confianca_campos={
                "nome_no_documento": 0.85,
                "periodo.inicio": 0.40,
                "periodo.fim": 0.40,
                "profissional.numero": 0.50,
            },
        )
    )
    ext_rebaixado, dec_reb = extrair_atestado_via_llm(
        texto_carla, llm=mock_baixo, limiar_confianca=0.8
    )
    assert ext_rebaixado.legibilidade == "ilegivel"
    assert ext_rebaixado.periodo.inicio is None
    assert ext_rebaixado.profissional.numero is None
    assert "limiar" in dec_reb.motivo


def test_fallback_with_structured_output_sem_argumento_method() -> None:
    """Aceita runnables mock externos cuja assinatura não recebe `method=`."""

    class _RunnableSimples:
        def invoke(self, _mensagens: Any) -> AtestadoExtraido:
            return AtestadoExtraido(
                legibilidade="ok",
                nome_no_documento="Carla Souza",
                periodo=PeriodoAfastamento(
                    inicio=date(2026, 11, 9), fim=date(2026, 11, 11)
                ),
                profissional=ProfissionalExtraido(
                    conselho="CRM", numero="123456", uf="SP"
                ),
            )

    class _MockSemMethod:
        model_name = "custom/sem-method"

        def with_structured_output(self, _schema: type[Any]) -> _RunnableSimples:
            return _RunnableSimples()

    ext, dec = extrair_atestado_via_llm("ATESTADO", llm=_MockSemMethod())
    assert ext.legibilidade == "ok"
    assert dec.modelo == "custom/sem-method"
    assert dec.confianca == 0.92


def test_mock_llm_conversar_fixture_e_alteracao_contato() -> None:
    """Valida `analisar_conversa_atestado` com `MockLLM` e garantia de `faltando`."""
    tarefa_raw = _carregar_fixture("a1.conversar/falta_arquivo.tarefa.json")
    esperado = _carregar_fixture("a1.conversar/falta_arquivo.esperado.json")
    tarefa = TarefaA1.model_validate(tarefa_raw)
    payload = tarefa.validar_payload_acao()

    mock = MockLLM()
    analise, dec = analisar_conversa_atestado(
        payload.texto,
        historico_resumo=payload.historico_resumo,
        registro_inicial=tarefa.registro_inicial,
        llm=mock,
    )
    assert analise.faltando == esperado["payload"]["faltando"]
    assert analise.texto_resposta == esperado["saidas"][0]["texto"]
    assert analise.solicitou_alteracao_contato is False
    assert dec.usou_llm is True
    assert dec.modelo == "mock/sinergia-a1"

    # Caso de alteração de contato
    analise_contato, _ = analisar_conversa_atestado(
        "Meu novo telefone é (19) 99999-1234",
        registro_inicial=tarefa.registro_inicial,
        llm=mock,
    )
    assert analise_contato.solicitou_alteracao_contato is True
    assert "arquivo" in analise_contato.faltando
    assert "rh" in analise_contato.texto_resposta.lower()

    # Garante 'arquivo' em faltando se tem_documento=False mesmo com retorno vazio
    mock_sem_faltando = MockLLM(
        conversa_padrao=AnaliseConversaA1(
            faltando=[],
            texto_resposta="Olá!",
            motivo="teste",
            confianca=0.9,
        )
    )
    analise_forcada, _ = analisar_conversa_atestado(
        "Oi", registro_inicial=tarefa.registro_inicial, llm=mock_sem_faltando
    )
    assert analise_forcada.faltando == ["arquivo"]


def test_obter_cliente_llm_valida_chave_e_reutiliza_instancia_cacheada() -> None:
    """Valida cache único por configuração (Anexo C.2) e erro sem `GROQ_API_KEY`."""
    cfg_sem_chave = A1Config(groq_api_key="")
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        obter_cliente_llm(config=cfg_sem_chave)

    cfg_fake = A1Config(groq_api_key="gsk_teste_cache_123", llm_model="openai/gpt-oss-120b")
    c1 = obter_cliente_llm(config=cfg_fake, temperature=0)
    c2 = obter_cliente_llm(config=cfg_fake, temperature=0.0)
    assert c1 is c2


@pytest.mark.skipif(
    not (_CFG.groq_api_key and ATESTADO_CARLA_PNG.is_file()),
    reason="GROQ_API_KEY ou .storage/atestados/atestado_carla_105.png indisponível.",
)
def test_pipeline_real_1a1_1a3_1b1_com_atestado_png_e_groq() -> None:
    """Valida `.storage/atestados/atestado_carla_105.png` -> OCR -> Groq -> Validador."""
    tarefa_doc = TarefaA1.model_validate(
        _carregar_fixture("a1.processar_documento/carla_divergencia_dias.tarefa.json")
    )
    esperado_doc = _carregar_fixture(
        "a1.processar_documento/carla_divergencia_dias.esperado.json"
    )

    # 1A.1: OCR real sobre a imagem oficial com marca d'água
    res_ocr = processar_arquivo_atestado(str(ATESTADO_CARLA_PNG), mime="image/png")
    assert res_ocr.legibilidade == "ok"

    # 1A.3: Extrator real via ChatGroq (sem CID para bater com a fixture)
    extraido, decisao = extrair_atestado_via_llm(res_ocr, extrair_cid=False)
    assert decisao.usou_llm is True
    assert decisao.modelo == _CFG.llm_model
    assert extraido.legibilidade == "ok"
    assert extraido.nome_no_documento == "Carla Souza"
    assert extraido.periodo.inicio == date(2026, 11, 9)
    assert extraido.periodo.fim == date(2026, 11, 11)
    assert extraido.profissional.conselho == "CRM"
    assert extraido.profissional.numero == "123456"
    assert extraido.profissional.uf == "SP"
    assert extraido.cid is None

    # 1B.1: Validador reproduz exatamente a divergência e a saída da fixture
    divergencias = detectar_divergencias(extraido, tarefa_doc.registro_inicial)
    assert [d.model_dump(mode="json") for d in divergencias] == esperado_doc["payload"][
        "divergencias"
    ]
    assert (
        gerar_mensagem_confirmacao_extracao(extraido, divergencias)
        == esperado_doc["saidas"][0]["texto"]
    )

    # 1A.3 Conversar real sobre fixture da Carla
    tarefa_conv = TarefaA1.model_validate(
        _carregar_fixture("a1.conversar/falta_arquivo.tarefa.json")
    )
    payload_conv = tarefa_conv.validar_payload_acao()
    analise, dec_conv = analisar_conversa_atestado(
        payload_conv.texto,
        historico_resumo=payload_conv.historico_resumo,
        registro_inicial=tarefa_conv.registro_inicial,
    )
    assert dec_conv.usou_llm is True
    assert "arquivo" in analise.faltando
    assert "carla" in analise.texto_resposta.lower()
