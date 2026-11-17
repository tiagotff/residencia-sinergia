"""Extrator estruturado e conversador LangChain + Groq do Agente 1.

Implementa a Task 1A.3 alinhada à ``Documentação - SinergIA v2.2.docx``
(Seção 5.3 e Anexos A, C.2, D.2, E.2, F e G):

- ``obter_cliente_llm``: instancia e cacheia ``ChatGroq`` uma única vez por
  configuração (Anexo C.2).
- ``MockLLM``: cliente determinístico injetável para testes de contrato offline
  (Anexo F).
- ``extrair_atestado_via_llm``: converte ``ResultadoOCR`` ou texto bruto em
  ``(AtestadoExtraido, DecisaoAgente)``, aplicando o limiar ``0.8`` e a
  Política de CID.
- ``analisar_conversa_atestado``: interpreta mensagens em ``a1.conversar`` e
  retorna ``(AnaliseConversaA1, DecisaoAgente)``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

from langchain_groq import ChatGroq
from pydantic import BaseModel

from sinergia.agentes.a1_recepcao.config import A1Config, obter_config_a1
from sinergia.agentes.a1_recepcao.prompts import (
    AnaliseConversaA1,
    construir_mensagens_conversar,
    construir_mensagens_extracao,
)
from sinergia.agentes.a1_recepcao.schemas import (
    AtestadoExtraido,
    DecisaoAgente,
    PeriodoAfastamento,
    ProfissionalExtraido,
    RegistroInicialCaso,
    ResultadoOCR,
)

_LIMIAR_OCR_ILEGIVEL = 0.30
_CONFIANCA_PADRAO_EXTRATOR = 0.92

_RE_OCR_TAG = re.compile(r"<texto_ocr>\s*(.*?)\s*</texto_ocr>", re.DOTALL | re.IGNORECASE)
_RE_MSG_TAG = re.compile(
    r"<mensagem_colaborador>\s*(.*?)\s*</mensagem_colaborador>",
    re.DOTALL | re.IGNORECASE,
)
_RE_DATA_BR = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")
_RE_INTERVALO_BR = re.compile(
    r"(\d{2}/\d{2}/\d{4})\s*(?:a|at[eé]|-)\s*(\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)
_RE_DIAS = re.compile(r"\b(\d{1,3})\s*(?:\([^)]+\)\s*)?dias?\b", re.IGNORECASE)
_RE_CONSELHO = re.compile(
    r"\b(CRM|CRO|CRP)\s*[/\-\s]?\s*([A-Z]{2})?\s*[/\-\s:]?\s*(\d{3,10})\s*(?:[/\-\s]?\s*([A-Z]{2}))?",
    re.IGNORECASE,
)
_RE_CID = re.compile(r"\bCID(?:-10)?\s*[:\-]?\s*([A-Z]\d{2}(?:\.\d{1,2})?)\b", re.IGNORECASE)
_RE_CONTATO = re.compile(
    r"(telefone|celular|whatsapp|e-?mail|@|\(\d{2}\)\s*\d{4,5}[-\s]?\d{4})",
    re.IGNORECASE,
)


def _extrair_texto_mensagens(mensagens: Any) -> tuple[str, str]:
    """Separa o conteúdo de sistema e de usuário a partir de mensagens LangChain."""
    sistema: list[str] = []
    humano: list[str] = []
    if isinstance(mensagens, str):
        return "", mensagens

    if isinstance(mensagens, Sequence):
        for item in mensagens:
            if isinstance(item, tuple) and len(item) >= 2:
                role, content = str(item[0]).lower(), str(item[1])
            elif hasattr(item, "type") and hasattr(item, "content"):
                role, content = str(item.type).lower(), str(item.content)
            else:
                role, content = "human", str(item)
            if role == "system":
                sistema.append(content)
            else:
                humano.append(content)
    return "\n".join(sistema), "\n".join(humano)


def _extrair_atestado_deterministico(mensagens: Any) -> AtestadoExtraido:
    """Heurística offline determinística usada por `MockLLM` quando não há callback."""
    sistema, humano = _extrair_texto_mensagens(mensagens)
    match_ocr = _RE_OCR_TAG.search(humano)
    texto_ocr = match_ocr.group(1).strip() if match_ocr else humano.strip()

    if (
        not texto_ocr
        or "[TEXTO OCR VAZIO" in texto_ocr.upper()
        or "ILEGIVEL" in texto_ocr.upper()
    ):
        return AtestadoExtraido(legibilidade="ilegivel")

    extrair_cid = "POLITICA_EXTRAIR_CID=false" not in sistema

    datas: list[date] = []
    for dia_s, mes_s, ano_s in _RE_DATA_BR.findall(texto_ocr):
        try:
            datas.append(date(int(ano_s), int(mes_s), int(dia_s)))
        except ValueError:
            continue

    inicio: date | None = datas[0] if datas else None
    fim: date | None = None
    match_intervalo = _RE_INTERVALO_BR.search(texto_ocr)
    if match_intervalo:
        d1 = _RE_DATA_BR.findall(match_intervalo.group(1))[0]
        d2 = _RE_DATA_BR.findall(match_intervalo.group(2))[0]
        inicio = date(int(d1[2]), int(d1[1]), int(d1[0]))
        fim = date(int(d2[2]), int(d2[1]), int(d2[0]))
    elif inicio is not None:
        match_dias = _RE_DIAS.search(texto_ocr)
        if match_dias:
            qtd_dias = int(match_dias.group(1))
            fim = inicio + timedelta(days=max(qtd_dias, 1) - 1)
        elif len(datas) >= 2 and datas[1] > datas[0]:
            fim = datas[1]
        else:
            fim = inicio

    data_emissao = datas[-1] if datas else inicio
    if match_intervalo and len(datas) == 2:
        data_emissao = inicio

    conselho = numero = uf = None
    match_prof = _RE_CONSELHO.search(texto_ocr)
    if match_prof:
        conselho = match_prof.group(1).upper()
        uf_antes, numero, uf_depois = match_prof.group(2), match_prof.group(3), match_prof.group(4)
        uf_bruta = uf_antes or uf_depois
        uf = uf_bruta.upper() if uf_bruta else None

    cid = None
    if extrair_cid:
        match_cid = _RE_CID.search(texto_ocr)
        if match_cid:
            cid = match_cid.group(1).upper()

    nome_paciente = None
    match_nome = re.search(
        r"(?:Sra?\.|paciente)\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)+?)(?=\s+necessita|\s*,|\s*\n|$)",
        texto_ocr,
        re.IGNORECASE,
    )
    if match_nome:
        nome_paciente = match_nome.group(1).strip()

    municipio = None
    if "campinas" in texto_ocr.lower():
        municipio = "Campinas"
    elif "são paulo" in texto_ocr.lower() or "sao paulo" in texto_ocr.lower():
        municipio = "São Paulo"

    return AtestadoExtraido(
        legibilidade="ok",
        nome_no_documento=nome_paciente,
        periodo=PeriodoAfastamento(inicio=inicio, fim=fim),
        data_emissao=data_emissao,
        profissional=ProfissionalExtraido(
            conselho=conselho,
            numero=numero,
            uf=uf,
        ),
        cid=cid,
        municipio=municipio,
        confianca_campos={
            "nome_no_documento": 0.96 if nome_paciente else 0.0,
            "periodo.inicio": 0.96 if inicio else 0.0,
            "periodo.fim": 0.95 if fim else 0.0,
            "data_emissao": 0.95 if data_emissao else 0.0,
            "profissional.conselho": 0.97 if conselho else 0.0,
            "profissional.numero": 0.97 if numero else 0.0,
            "profissional.uf": 0.97 if uf else 0.0,
            "cid": 0.95 if cid else 1.0,
            "municipio": 0.94 if municipio else 0.0,
        },
    )


def _analisar_conversa_deterministica(mensagens: Any) -> AnaliseConversaA1:
    """Heurística offline determinística usada por `MockLLM` para `a1.conversar`."""
    _, humano = _extrair_texto_mensagens(mensagens)
    match_msg = _RE_MSG_TAG.search(humano)
    msg_colaborador = match_msg.group(1).strip() if match_msg else humano.strip()

    match_nome = re.search(r"Primeiro nome do colaborador:\s*([^\n]+)", humano)
    primeiro_nome = match_nome.group(1).strip() if match_nome else None
    tem_doc = "Documento já anexado (`tem_documento`): sim" in humano
    alterou_contato = bool(_RE_CONTATO.search(msg_colaborador))

    faltando = [] if tem_doc else ["arquivo"]
    saudacao = f"Entendi, {primeiro_nome}." if primeiro_nome else "Entendi."

    if alterou_contato:
        texto_resp = (
            f"{saudacao} Vou encaminhar ao RH o pedido de atualização do seu contato."
        )
        if not tem_doc:
            texto_resp += " Quando puder, envie também uma foto ou PDF do atestado."
        motivo = (
            "solicitacao de alteracao de contato e falta o arquivo"
            if not tem_doc
            else "solicitacao de alteracao de contato"
        )
    elif not tem_doc:
        texto_resp = f"{saudacao} Pode me enviar uma foto do atestado?"
        motivo = "falta o arquivo"
    else:
        texto_resp = f"{saudacao} Já recebi seu documento e vou prosseguir com a análise."
        motivo = "documento ja anexado"

    return AnaliseConversaA1(
        faltando=faltando,
        texto_resposta=texto_resp,
        motivo=motivo,
        confianca=0.95,
        solicitou_alteracao_contato=alterou_contato,
    )


class _RunnableEstruturadoMock[TModel: BaseModel]:
    """Adapter compatível com `.invoke()` retornado por `MockLLM.with_structured_output`."""

    def __init__(
        self,
        schema: type[TModel],
        mock_llm: MockLLM,
    ) -> None:
        self._schema = schema
        self._mock_llm = mock_llm

    def invoke(self, mensagens: Any, **_kwargs: Any) -> TModel:
        if issubclass(self._schema, AtestadoExtraido):
            handler = self._mock_llm.extracao_padrao
            if callable(handler):
                res = handler(mensagens)
            elif handler is not None:
                res = handler.model_copy(deep=True)
            else:
                res = _extrair_atestado_deterministico(mensagens)
            return self._schema.model_validate(res)

        if issubclass(self._schema, AnaliseConversaA1):
            handler = self._mock_llm.conversa_padrao
            if callable(handler):
                res = handler(mensagens)
            elif handler is not None:
                res = handler.model_copy(deep=True)
            else:
                res = _analisar_conversa_deterministica(mensagens)
            return self._schema.model_validate(res)

        raise TypeError(f"Schema não suportado pelo MockLLM: {self._schema!r}")


class MockLLM:
    """Cliente LLM injetável para testes offline e contratos do Anexo F."""

    def __init__(
        self,
        *,
        nome_modelo: str = "mock/sinergia-a1",
        extracao_padrao: AtestadoExtraido | Callable[[Any], AtestadoExtraido] | None = None,
        conversa_padrao: AnaliseConversaA1 | Callable[[Any], AnaliseConversaA1] | None = None,
    ) -> None:
        self.nome_modelo = nome_modelo
        self.extracao_padrao = extracao_padrao
        self.conversa_padrao = conversa_padrao
        self.chamadas: int = 0

    def with_structured_output[TModel: BaseModel](
        self,
        schema: type[TModel],
        **_kwargs: Any,
    ) -> _RunnableEstruturadoMock[TModel]:
        """Retorna um runnable compatível com LangChain para o `schema` informado."""
        outer = self

        class _CountingRunnable(_RunnableEstruturadoMock[TModel]):
            def invoke(self, mensagens: Any, **kw: Any) -> TModel:
                outer.chamadas += 1
                return super().invoke(mensagens, **kw)

        return _CountingRunnable(schema, self)


@lru_cache(maxsize=4)
def _criar_chat_groq_cacheado(
    modelo: str,
    api_key: str,
    temperature: float,
) -> ChatGroq:
    """Cria e reutiliza o cliente `ChatGroq` uma única vez por configuração (Anexo C.2)."""
    return ChatGroq(model=modelo, api_key=api_key, temperature=temperature)


def obter_cliente_llm(
    *,
    config: A1Config | None = None,
    modelo: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> ChatGroq:
    """Retorna o cliente `ChatGroq` cacheado para o Agente A1."""
    cfg = config or obter_config_a1()
    modelo_final = (modelo or cfg.llm_model).strip()
    chave_final = (api_key if api_key is not None else cfg.groq_api_key).strip()
    if not chave_final:
        raise ValueError(
            "GROQ_API_KEY não configurada no ambiente/.env. "
            "Configure a chave ou injete um MockLLM para testes offline."
        )
    return _criar_chat_groq_cacheado(modelo_final, chave_final, float(temperature))


def _obter_nome_modelo(llm: Any, config: A1Config) -> str:
    for attr in ("model_name", "model", "nome_modelo"):
        valor = getattr(llm, attr, None)
        if isinstance(valor, str) and valor.strip():
            return valor.strip()
    return config.llm_model


def _invocar_estruturado[TModel: BaseModel](
    llm: Any,
    schema: type[TModel],
    mensagens: list[tuple[str, str]],
) -> TModel:
    """Executa `.with_structured_output` usando `json_schema` com fallback seguro."""
    try:
        runnable = llm.with_structured_output(schema, method="json_schema")
    except TypeError:
        runnable = llm.with_structured_output(schema)

    bruto = runnable.invoke(mensagens)
    if isinstance(bruto, schema):
        return bruto
    return schema.model_validate(bruto)


def _calcular_confianca_extracao(
    extraido: AtestadoExtraido,
    confianca_ocr: float | None,
) -> float:
    if extraido.legibilidade == "ilegivel":
        if confianca_ocr is not None:
            return round(max(0.0, min(1.0, float(confianca_ocr))), 4)
        return 0.0

    if extraido.confianca_campos:
        valores = [
            max(0.0, min(1.0, float(v)))
            for v in extraido.confianca_campos.values()
            if isinstance(v, (int, float))
        ]
        if valores:
            return round(sum(valores) / len(valores), 4)

    if confianca_ocr is not None:
        return round(max(0.0, min(1.0, float(confianca_ocr))), 4)
    return _CONFIANCA_PADRAO_EXTRATOR


def _tem_campos_essenciais(extraido: AtestadoExtraido) -> bool:
    return any(
        (
            extraido.periodo.inicio is not None,
            extraido.periodo.fim is not None,
            extraido.profissional.numero is not None,
        )
    )


def extrair_atestado_via_llm(
    entrada_ocr: str | ResultadoOCR,
    *,
    llm: Any | None = None,
    config: A1Config | None = None,
    extrair_cid: bool | None = None,
    limiar_confianca: float | None = None,
    confianca_media_ocr: float | None = None,
    mime: str | None = None,
) -> tuple[AtestadoExtraido, DecisaoAgente]:
    """Extrai `AtestadoExtraido` a partir do texto ou `ResultadoOCR` usando LLM.

    Aplica curto-circuito para OCR vazio/ilegível (sem gastar chamada ao LLM),
    a Política de CID (`POLITICA_EXTRAIR_CID`) e o limiar de confiança `0.8`
    (Anexo E.2), retornando também o bloco de auditoria `DecisaoAgente`.
    """
    cfg = config or obter_config_a1()
    usar_cid = cfg.politica_extrair_cid if extrair_cid is None else extrair_cid
    limiar = (
        cfg.limiar_confianca_campo if limiar_confianca is None else limiar_confianca
    )

    if isinstance(entrada_ocr, ResultadoOCR):
        texto_ocr = entrada_ocr.texto_extraido
        conf_ocr = (
            confianca_media_ocr
            if confianca_media_ocr is not None
            else entrada_ocr.confianca_media
        )
        mime_final = mime or entrada_ocr.mime
        ocr_ilegivel = entrada_ocr.legibilidade == "ilegivel"
    else:
        texto_ocr = entrada_ocr
        conf_ocr = confianca_media_ocr
        mime_final = mime
        ocr_ilegivel = False

    if (
        ocr_ilegivel
        or not texto_ocr.strip()
        or (conf_ocr is not None and conf_ocr < _LIMIAR_OCR_ILEGIVEL)
    ):
        atestado_vazio = AtestadoExtraido(legibilidade="ilegivel")
        conf_decisao = (
            round(max(0.0, min(1.0, float(conf_ocr))), 4)
            if conf_ocr is not None
            else 0.0
        )
        return atestado_vazio, DecisaoAgente(
            motivo="documento ilegivel ou vazio no OCR",
            usou_llm=False,
            confianca=conf_decisao,
            modelo=None,
        )

    cliente = llm if llm is not None else obter_cliente_llm(config=cfg)
    mensagens = construir_mensagens_extracao(
        texto_ocr,
        extrair_cid=usar_cid,
        limiar_confianca=limiar,
        confianca_media_ocr=conf_ocr,
        mime=mime_final,
    )
    extraido_bruto = _invocar_estruturado(cliente, AtestadoExtraido, mensagens)
    extraido = extraido_bruto.aplicar_politica_e_limiar(
        extrair_cid=usar_cid,
        limiar_confianca=limiar,
    )

    if extraido.legibilidade == "ok" and not _tem_campos_essenciais(extraido):
        extraido = extraido.model_copy(update={"legibilidade": "ilegivel"})

    motivo = (
        "documento ilegivel apos aplicacao do limiar de confianca"
        if extraido.legibilidade == "ilegivel"
        else "atestado extraido pelo LLM com saida estruturada"
    )
    decisao = DecisaoAgente(
        motivo=motivo,
        usou_llm=True,
        confianca=_calcular_confianca_extracao(extraido, conf_ocr),
        modelo=_obter_nome_modelo(cliente, cfg),
    )
    return extraido, decisao


def analisar_conversa_atestado(
    texto: str,
    historico_resumo: str | None = None,
    registro_inicial: RegistroInicialCaso | None = None,
    *,
    llm: Any | None = None,
    config: A1Config | None = None,
) -> tuple[AnaliseConversaA1, DecisaoAgente]:
    """Analisa uma mensagem do colaborador na ação ``a1.conversar``."""
    cfg = config or obter_config_a1()
    cliente = llm if llm is not None else obter_cliente_llm(config=cfg)
    mensagens = construir_mensagens_conversar(
        texto,
        historico_resumo=historico_resumo,
        registro_inicial=registro_inicial,
    )
    analise = _invocar_estruturado(cliente, AnaliseConversaA1, mensagens)

    faltando = list(dict.fromkeys(analise.faltando))
    precisa_arquivo = registro_inicial is None or not registro_inicial.tem_documento
    if precisa_arquivo and "arquivo" not in faltando:
        faltando.insert(0, "arquivo")

    atualizacoes: dict[str, Any] = {}
    if faltando != analise.faltando:
        atualizacoes["faltando"] = faltando
    if analise.solicitou_alteracao_contato and "rh" not in analise.texto_resposta.lower():
        atualizacoes["texto_resposta"] = (
            f"{analise.texto_resposta.rstrip()} "
            "Vou encaminhar sua solicitação de atualização de contato ao RH."
        ).strip()

    if atualizacoes:
        analise = analise.model_copy(update=atualizacoes)

    decisao = DecisaoAgente(
        motivo=analise.motivo,
        usou_llm=True,
        confianca=round(max(0.0, min(1.0, float(analise.confianca))), 4),
        modelo=_obter_nome_modelo(cliente, cfg),
    )
    return analise, decisao
