"""Fachada oficial ``executar(tarefa) -> resultado`` do Agente 1 (A1).

Implementa a Task 2A.1 do protocolo ``sinergia/1`` conforme a
``Documentação - SinergIA v2.2.docx`` (Seção 5.3 e Anexos A, B, C.2, D.1, D.2,
E.2, F e G), roteando as 4 ações oficiais:

- ``a1.conversar``
- ``a1.processar_documento``
- ``a1.confirmar``
- ``a1.registrar_baixa_fisica``
"""

from __future__ import annotations

import copy
from datetime import datetime
from functools import lru_cache
from typing import Any

from pydantic import ValidationError

from sinergia.agentes.a1_recepcao.config import A1Config, obter_config_a1
from sinergia.agentes.a1_recepcao.extrator_llm import (
    analisar_conversa_atestado,
    extrair_atestado_via_llm,
)
from sinergia.agentes.a1_recepcao.ocr import processar_arquivo_atestado
from sinergia.agentes.a1_recepcao.repositorio import (
    RegistroAtestadoDB,
    RepositorioA1Memoria,
    RepositorioA1Postgres,
    RepositorioA1Protocol,
)
from sinergia.agentes.a1_recepcao.schemas import (
    AtestadoExtraido,
    DecisaoAgente,
    DestinatarioSaida,
    ErroAgente,
    EventoEmitido,
    PayloadBaixaFisicaResultado,
    PayloadBaixaFisicaTarefa,
    PayloadConfirmarInvalidoResultado,
    PayloadConfirmarTarefa,
    PayloadConfirmarValidoResultado,
    PayloadConversarResultado,
    PayloadConversarTarefa,
    PayloadProcessarDocumentoResultado,
    PayloadProcessarDocumentoTarefa,
    PeriodoAfastamento,
    ResultadoA1,
    SaidaMensagem,
    TarefaA1,
)
from sinergia.agentes.a1_recepcao.validador import (
    detectar_divergencias,
    gerar_mensagem_confirmacao_extracao,
    mesclar_atestado_confirmado,
    validar_dados_confirmados,
)

_CACHE_IDEMPOTENCIA_MODULO: dict[str, dict[str, Any]] = {}
_EXTRACOES_PENDENTES_MODULO: dict[tuple[str | int, int, int | None], AtestadoExtraido] = {}


@lru_cache(maxsize=4)
def _criar_repositorio_padrao(database_url: str) -> RepositorioA1Protocol:
    if database_url:
        return RepositorioA1Postgres(database_url)
    return RepositorioA1Memoria()


def limpar_cache_idempotencia() -> None:
    """Limpa os caches de idempotência, repositório padrão e extrações do módulo."""
    _CACHE_IDEMPOTENCIA_MODULO.clear()
    _EXTRACOES_PENDENTES_MODULO.clear()
    _criar_repositorio_padrao.cache_clear()


def _reconstruir_base_de_pendente(pendente_db: RegistroAtestadoDB) -> AtestadoExtraido:
    """Reconstrói `AtestadoExtraido` base a partir de um registro `PENDENTE` do banco."""
    return AtestadoExtraido(
        legibilidade="ok",
        periodo=PeriodoAfastamento(
            inicio=pendente_db.inicio_afastamento.date(),
            fim=pendente_db.fim_afastamento.date(),
        ),
        data_emissao=pendente_db.data_emissao,
        cid=pendente_db.codigo_cid,
    )


def _formatar_atestado_payload(extraido: AtestadoExtraido) -> dict[str, Any]:
    """Serializa `AtestadoExtraido` no formato público do Anexo C.2 / D.2."""
    prof_dict: dict[str, Any] = {
        "conselho": extraido.profissional.conselho,
        "numero": extraido.profissional.numero,
        "uf": extraido.profissional.uf,
    }
    if extraido.profissional.nome is not None:
        prof_dict["nome"] = extraido.profissional.nome

    return {
        "legibilidade": extraido.legibilidade,
        "nome_no_documento": extraido.nome_no_documento,
        "periodo": {
            "inicio": (
                extraido.periodo.inicio.isoformat()
                if extraido.periodo.inicio is not None
                else None
            ),
            "fim": (
                extraido.periodo.fim.isoformat()
                if extraido.periodo.fim is not None
                else None
            ),
        },
        "data_emissao": (
            extraido.data_emissao.isoformat()
            if extraido.data_emissao is not None
            else None
        ),
        "profissional": prof_dict,
        "cid": extraido.cid,
        "municipio": extraido.municipio,
    }


def _serializar_saida(saida: SaidaMensagem, *, completa: bool) -> dict[str, Any]:
    """Serializa `SaidaMensagem` preservando apenas os campos relevantes ao contrato."""
    if completa:
        return saida.model_dump(mode="json")

    dado: dict[str, Any] = {
        "tipo": saida.tipo,
        "texto": saida.texto,
    }
    if saida.para is not None:
        dado["para"] = saida.para.model_dump(mode="json")
    if saida.nivel_detalhe is not None:
        dado["nivel_detalhe"] = saida.nivel_detalhe
    if saida.fontes:
        dado["fontes"] = [f.model_dump(mode="json") for f in saida.fontes]
    if saida.agendar_para is not None:
        dado["agendar_para"] = (
            saida.agendar_para.isoformat()
            if isinstance(saida.agendar_para, datetime)
            else saida.agendar_para
        )
    return dado


def _serializar_decisao(decisao: DecisaoAgente) -> dict[str, Any]:
    dado: dict[str, Any] = {
        "motivo": decisao.motivo,
        "usou_llm": decisao.usou_llm,
    }
    if decisao.confianca is not None:
        dado["confianca"] = decisao.confianca
    if decisao.modelo is not None:
        dado["modelo"] = decisao.modelo
    return dado


def _serializar_resultado(
    resultado: ResultadoA1,
    *,
    saida_completa: bool = False,
) -> dict[str, Any]:
    """Converte `ResultadoA1` no dicionário oficial do protocolo `sinergia/1`."""
    dado: dict[str, Any] = {
        "protocolo": resultado.protocolo,
        "id_mensagem": resultado.id_mensagem,
        "tipo": resultado.tipo,
        "acao": resultado.acao,
        "origem": resultado.origem,
        "destino": resultado.destino,
        "em_resposta_a": resultado.em_resposta_a,
        "correlacao": resultado.correlacao.model_dump(mode="json"),
        "status": resultado.status,
        "estado_proposto": resultado.estado_proposto,
        "payload": resultado.payload,
        "saidas": [
            _serializar_saida(s, completa=saida_completa) for s in resultado.saidas
        ],
        "eventos": [e.model_dump(mode="json") for e in resultado.eventos],
    }
    if resultado.decisao is not None:
        dado["decisao"] = _serializar_decisao(resultado.decisao)
    if resultado.erro is not None:
        dado["erro"] = resultado.erro.model_dump(mode="json")
    return dado


def _montar_erro(
    tarefa: TarefaA1,
    *,
    codigo: str,
    mensagem: str,
    reprocessavel: bool = False,
) -> ResultadoA1:
    return ResultadoA1.a_partir_da_tarefa(
        tarefa,
        status="ERRO",
        estado_proposto=None,
        payload={},
        saidas=[],
        eventos=[],
        decisao=DecisaoAgente(motivo=mensagem, usou_llm=False),
        erro=ErroAgente(
            codigo=codigo,
            mensagem=mensagem,
            reprocessavel=reprocessavel,
        ),
    )


class ServicoA1:
    """Serviço executor das 4 ações do Agente A1 (`sinergia/1`)."""

    def __init__(
        self,
        *,
        repositorio: RepositorioA1Protocol | None = None,
        llm: Any | None = None,
        config: A1Config | None = None,
        cache_idempotencia: dict[str, dict[str, Any]] | None = None,
        extracoes_pendentes: dict[tuple[str | int, int, int | None], AtestadoExtraido]
        | None = None,
    ) -> None:
        self.config = config or obter_config_a1()
        self.repositorio: RepositorioA1Protocol = (
            repositorio
            if repositorio is not None
            else _criar_repositorio_padrao(self.config.database_url.strip())
        )
        self.llm = llm
        self._cache_idempotencia: dict[str, dict[str, Any]] = (
            cache_idempotencia if cache_idempotencia is not None else {}
        )
        self._extracoes_pendentes: dict[
            tuple[str | int, int, int | None], AtestadoExtraido
        ] = extracoes_pendentes if extracoes_pendentes is not None else {}

    def _identificador_repositorio(self) -> str | int:
        db_url = getattr(self.repositorio, "_database_url", None)
        if isinstance(db_url, str) and db_url.strip():
            return db_url.strip()
        return id(self.repositorio)

    def _chave_pendente(self, tarefa: TarefaA1) -> tuple[str | int, int, int | None]:
        return (
            self._identificador_repositorio(),
            tarefa.contexto.id_colaborador,
            tarefa.correlacao.id_caso,
        )

    def _guardar_extracao_pendente(
        self, tarefa: TarefaA1, extraido: AtestadoExtraido
    ) -> None:
        self._extracoes_pendentes[self._chave_pendente(tarefa)] = extraido.model_copy(
            deep=True
        )

    def _obter_extracao_pendente(self, tarefa: TarefaA1) -> AtestadoExtraido | None:
        chave = self._chave_pendente(tarefa)
        if chave in self._extracoes_pendentes:
            return self._extracoes_pendentes[chave]
        chave_colab = (
            self._identificador_repositorio(),
            tarefa.contexto.id_colaborador,
            None,
        )
        return self._extracoes_pendentes.get(chave_colab)

    def _resolver_id_profissional(
        self,
        extraido: AtestadoExtraido,
        *,
        uf_fallback: str | None = None,
    ) -> int | None:
        prof = extraido.profissional
        if prof.conselho and prof.numero and prof.uf:
            if self.repositorio.validar_conselho_profissional(prof.conselho, prof.uf):
                return self.repositorio.obter_ou_criar_profissional_saude(
                    prof.conselho, prof.numero, prof.uf, prof.nome
                )
            return None

        if uf_fallback and self.repositorio.validar_conselho_profissional(
            "CRM", uf_fallback
        ):
            return self.repositorio.obter_ou_criar_profissional_saude(
                "CRM", "000000-PENDENTE", uf_fallback, None
            )
        return None

    def _executar_conversar(
        self, tarefa: TarefaA1, payload: PayloadConversarTarefa
    ) -> tuple[ResultadoA1, bool]:
        analise, decisao = analisar_conversa_atestado(
            payload.texto,
            historico_resumo=payload.historico_resumo,
            registro_inicial=tarefa.registro_inicial,
            llm=self.llm,
            config=self.config,
        )
        resultado = ResultadoA1.a_partir_da_tarefa(
            tarefa,
            status="OK",
            estado_proposto="NOVO",
            payload=PayloadConversarResultado(faltando=analise.faltando),
            saidas=[
                SaidaMensagem(
                    tipo="mensagem",
                    para=DestinatarioSaida(
                        perfil=tarefa.contexto.perfil,
                        id_colaborador=tarefa.contexto.id_colaborador,
                    ),
                    nivel_detalhe="individual",
                    texto=analise.texto_resposta,
                    fontes=[],
                    agendar_para=None,
                )
            ],
            eventos=[],
            decisao=decisao,
        )
        return resultado, True

    def _executar_processar_documento(
        self, tarefa: TarefaA1, payload: PayloadProcessarDocumentoTarefa
    ) -> tuple[ResultadoA1, bool]:
        id_colaborador = tarefa.contexto.id_colaborador
        id_caso = (
            tarefa.correlacao.id_caso
            or (tarefa.registro_inicial.id_caso if tarefa.registro_inicial else None)
        )

        duplicado = self.repositorio.verificar_duplicidade_arquivo(
            payload.arquivo_uri, id_colaborador=id_colaborador
        )
        if duplicado is not None:
            atestado_dup = AtestadoExtraido(
                legibilidade="ok",
                periodo={
                    "inicio": duplicado.inicio_afastamento.date(),
                    "fim": duplicado.fim_afastamento.date(),
                },
                data_emissao=duplicado.data_emissao,
                cid=duplicado.codigo_cid,
            )
            resultado = ResultadoA1.a_partir_da_tarefa(
                tarefa,
                status="PRECISA_HUMANO",
                estado_proposto="RECEBIDO",
                payload={
                    "atestado_extraido": _formatar_atestado_payload(atestado_dup),
                    "divergencias": [],
                },
                saidas=[
                    SaidaMensagem(
                        tipo="mensagem",
                        texto=(
                            "Este arquivo de atestado já foi recebido anteriormente "
                            f"(registro #{duplicado.id_atestado}). Deseja confirmar "
                            "os dados ou enviar outro documento?"
                        ),
                    )
                ],
                eventos=[],
                decisao=DecisaoAgente(
                    motivo="arquivo duplicado ja recebido anteriormente",
                    usou_llm=False,
                ),
            )
            return resultado, False

        res_ocr = processar_arquivo_atestado(
            payload.arquivo_uri,
            mime=payload.mime,
            sha256_esperado=payload.sha256,
        )
        extraido, decisao = extrair_atestado_via_llm(
            res_ocr, llm=self.llm, config=self.config
        )

        if extraido.legibilidade == "ilegivel":
            resultado = ResultadoA1.a_partir_da_tarefa(
                tarefa,
                status="PRECISA_HUMANO",
                estado_proposto="NOVO",
                payload={
                    "atestado_extraido": _formatar_atestado_payload(extraido),
                    "divergencias": [],
                },
                saidas=[
                    SaidaMensagem(
                        tipo="mensagem",
                        texto=gerar_mensagem_confirmacao_extracao(extraido, []),
                    )
                ],
                eventos=[],
                decisao=decisao,
            )
            return resultado, False

        divergencias = detectar_divergencias(extraido, tarefa.registro_inicial)
        self._guardar_extracao_pendente(tarefa, extraido)

        inicio = extraido.periodo.inicio
        fim = extraido.periodo.fim
        if inicio is not None and fim is not None and fim >= inicio:
            id_prof = self._resolver_id_profissional(
                extraido,
                uf_fallback=(
                    tarefa.registro_inicial.uf if tarefa.registro_inicial else "SP"
                ),
            )
            uf_busca = extraido.profissional.uf or (
                tarefa.registro_inicial.uf if tarefa.registro_inicial else None
            )
            id_municipio = self.repositorio.buscar_id_municipio(
                extraido.municipio,
                uf_busca,
                id_colaborador_fallback=id_colaborador,
            )
            cid_valido = self.repositorio.validar_cid(
                extraido.cid, politica_extrair_cid=self.config.politica_extrair_cid
            )
            if id_prof is not None and id_municipio is not None:
                self.repositorio.salvar_atestado_pendente(
                    id_colaborador=id_colaborador,
                    id_profissional=id_prof,
                    id_municipio=id_municipio,
                    data_emissao=extraido.data_emissao or inicio,
                    inicio_afastamento=inicio,
                    fim_afastamento=fim,
                    arquivo_uri=payload.arquivo_uri,
                    codigo_cid=cid_valido,
                    origem_dados="IA",
                )

        motivo_decisao = "documento legivel extraido pelo LLM"
        if divergencias:
            campos = ", ".join(d.campo for d in divergencias)
            motivo_decisao = (
                f"documento legivel extraido pelo LLM com divergencia de {campos}"
            )
        decisao_final = decisao.model_copy(update={"motivo": motivo_decisao})

        payload_dict = {
            "atestado_extraido": _formatar_atestado_payload(extraido),
            "divergencias": [d.model_dump(mode="json") for d in divergencias],
        }
        PayloadProcessarDocumentoResultado.model_validate(payload_dict)

        dados_evento: dict[str, Any] = {}
        if id_caso is not None:
            dados_evento["id_caso"] = id_caso

        resultado = ResultadoA1.a_partir_da_tarefa(
            tarefa,
            status="PRECISA_HUMANO",
            estado_proposto="RECEBIDO",
            payload=payload_dict,
            saidas=[
                SaidaMensagem(
                    tipo="mensagem",
                    texto=gerar_mensagem_confirmacao_extracao(extraido, divergencias),
                )
            ],
            eventos=[EventoEmitido(tipo="atestado.extraido", dados=dados_evento)],
            decisao=decisao_final,
        )
        return resultado, False

    def _executar_confirmar(
        self, tarefa: TarefaA1, payload: PayloadConfirmarTarefa
    ) -> tuple[ResultadoA1, bool]:
        id_colaborador = tarefa.contexto.id_colaborador
        id_caso = (
            tarefa.correlacao.id_caso
            or (tarefa.registro_inicial.id_caso if tarefa.registro_inicial else None)
        )

        base_extraida = self._obter_extracao_pendente(tarefa)
        pendente_db = self.repositorio.obter_ultimo_atestado_pendente(id_colaborador)
        if base_extraida is None and pendente_db is not None:
            base_extraida = _reconstruir_base_de_pendente(pendente_db)

        mesclado = mesclar_atestado_confirmado(
            payload.atestado_confirmado, base_extraida
        )

        inicio = mesclado.periodo.inicio
        fim = mesclado.periodo.fim
        colab = self.repositorio.conferir_colaborador_ativo(
            id_colaborador, inicio, fim
        )
        colaborador_ativo = bool(colab and colab.ativo)
        dias_acumulados = (
            self.repositorio.calcular_dias_afastamento_ultimos_60_dias(
                id_colaborador, inicio, cid=mesclado.cid
            )
            if inicio is not None
            else 0
        )

        validacao = validar_dados_confirmados(
            mesclado,
            payload.campos_corrigidos,
            colaborador_ativo=colaborador_ativo,
            conselhos_aceitos=tuple(
                c
                for c in ("CRM", "CRO", "CRP")
                if self.repositorio.validar_conselho_profissional(
                    c, mesclado.profissional.uf or "SP"
                )
            )
            or ("CRM", "CRO", "CRP"),
            politica_extrair_cid=self.config.politica_extrair_cid,
            dias_uteis_entrega_fisica=self.config.prazo_entrega_fisica_dias_uteis,
            dias_acumulados_60d=dias_acumulados,
        )

        if not validacao.valido:
            motivo_decisao = (
                "regra de datas falhou"
                if validacao.codigo_evento
                in {"fim < inicio", "periodo_incompleto", "emissao_futura"}
                else (validacao.motivo or "validacao de confirmacao falhou")
            )
            dados_evento_inv: dict[str, Any] = {
                "motivo": validacao.codigo_evento or validacao.motivo,
            }
            if id_caso is not None:
                dados_evento_inv["id_caso"] = id_caso

            resultado = ResultadoA1.a_partir_da_tarefa(
                tarefa,
                status="OK",
                estado_proposto="NOVO",
                payload=PayloadConfirmarInvalidoResultado(
                    motivo=validacao.motivo or "dados inválidos",
                ),
                saidas=[
                    SaidaMensagem(
                        tipo="mensagem", texto=validacao.mensagem_usuario
                    )
                ],
                eventos=[
                    EventoEmitido(tipo="atestado.invalido", dados=dados_evento_inv)
                ],
                decisao=DecisaoAgente(motivo=motivo_decisao, usou_llm=False),
            )
            return resultado, False

        assert inicio is not None
        assert fim is not None
        assert validacao.origem is not None
        assert validacao.prazo_entrega_fisica is not None

        data_emissao = mesclado.data_emissao or inicio
        uf_fallback = (
            mesclado.profissional.uf
            or (tarefa.registro_inicial.uf if tarefa.registro_inicial else None)
            or (colab.sigla_uf if colab else "SP")
        )
        tem_prof_mesclado = all(
            (
                mesclado.profissional.conselho,
                mesclado.profissional.numero,
                mesclado.profissional.uf,
            )
        )
        id_prof = (
            self._resolver_id_profissional(mesclado, uf_fallback=uf_fallback)
            if (tem_prof_mesclado or pendente_db is None)
            else pendente_db.id_profissional
        )
        id_municipio = (
            self.repositorio.buscar_id_municipio(
                mesclado.municipio,
                uf_fallback,
                id_colaborador_fallback=id_colaborador,
            )
            if (mesclado.municipio is not None or pendente_db is None)
            else pendente_db.id_municipio
        )
        cid_valido = self.repositorio.validar_cid(
            mesclado.cid, politica_extrair_cid=self.config.politica_extrair_cid
        )

        if pendente_db is None:
            if id_prof is None or id_municipio is None:
                raise ValueError(
                    "Não foi possível resolver profissional ou município para registrar o atestado."
                )
            pendente_db = self.repositorio.salvar_atestado_pendente(
                id_colaborador=id_colaborador,
                id_profissional=id_prof,
                id_municipio=id_municipio,
                data_emissao=data_emissao,
                inicio_afastamento=inicio,
                fim_afastamento=fim,
                arquivo_uri=f"local://caso-{id_caso or id_colaborador}-confirmado",
                codigo_cid=cid_valido,
                origem_dados=validacao.origem,
            )

        codigo_cid_final = cid_valido if self.config.politica_extrair_cid else None
        confirmado = self.repositorio.confirmar_atestado_no_banco(
            pendente_db.id_atestado,
            inicio_afastamento=inicio,
            fim_afastamento=fim,
            data_emissao=data_emissao,
            origem_dados=validacao.origem,
            prazo_entrega_fisica=validacao.prazo_entrega_fisica,
            id_profissional=id_prof if id_prof is not None else pendente_db.id_profissional,
            codigo_cid=codigo_cid_final,
            id_municipio=id_municipio if id_municipio is not None else pendente_db.id_municipio,
        )

        self._extracoes_pendentes.pop(self._chave_pendente(tarefa), None)

        motivo_ok = (
            "regras ok; colaborador confirmou o período do documento"
            if validacao.origem == "IA"
            else "regras ok; colaborador corrigiu campos manualmente"
        )
        dados_evento_ok: dict[str, Any] = {
            "id_atestado": confirmado.id_atestado,
            "dias": validacao.dias,
        }
        if id_caso is not None:
            dados_evento_ok["id_caso"] = id_caso

        resultado = ResultadoA1.a_partir_da_tarefa(
            tarefa,
            status="OK",
            estado_proposto="VALIDADO",
            payload=PayloadConfirmarValidoResultado(
                id_atestado=confirmado.id_atestado,
                status_atestado="CONFIRMADO",
                origem=validacao.origem,
                prazo_entrega_fisica=validacao.prazo_entrega_fisica,
            ),
            saidas=[
                SaidaMensagem(tipo="mensagem", texto=validacao.mensagem_usuario)
            ],
            eventos=[
                EventoEmitido(tipo="atestado.registrado", dados=dados_evento_ok)
            ],
            decisao=DecisaoAgente(motivo=motivo_ok, usou_llm=False),
        )
        return resultado, False

    def _executar_registrar_baixa_fisica(
        self, tarefa: TarefaA1, payload: PayloadBaixaFisicaTarefa
    ) -> tuple[ResultadoA1, bool]:
        self.repositorio.registrar_baixa_fisica_no_banco(
            payload.id_atestado,
            payload.recebido_fisico_por,
            payload.recebido_fisico_em,
        )

        recebido_em_str = (
            payload.recebido_fisico_em.isoformat()
            if isinstance(payload.recebido_fisico_em, datetime)
            else payload.recebido_fisico_em
        )
        resultado = ResultadoA1.a_partir_da_tarefa(
            tarefa,
            status="OK",
            estado_proposto=None,
            payload=PayloadBaixaFisicaResultado(
                id_atestado=payload.id_atestado,
                recebido_fisico_por=payload.recebido_fisico_por,
                recebido_fisico_em=recebido_em_str,
            ),
            saidas=[
                SaidaMensagem(
                    tipo="mensagem",
                    texto="O RH recebeu o original do seu atestado. Obrigado!",
                )
            ],
            eventos=[
                EventoEmitido(
                    tipo="atestado.baixa_registrada",
                    dados={"id_atestado": payload.id_atestado},
                )
            ],
            decisao=DecisaoAgente(
                motivo="baixa fisica registrada pelo RH",
                usou_llm=False,
            ),
        )
        return resultado, False

    def executar(self, tarefa: dict[str, Any] | TarefaA1) -> dict[str, Any]:
        """Executa uma `TAREFA` do protocolo `sinergia/1` garantindo idempotência."""
        tarefa_obj = (
            tarefa
            if isinstance(tarefa, TarefaA1)
            else TarefaA1.model_validate(tarefa)
        )

        if tarefa_obj.id_mensagem in self._cache_idempotencia:
            return copy.deepcopy(self._cache_idempotencia[tarefa_obj.id_mensagem])

        try:
            payload_validado = tarefa_obj.validar_payload_acao()
        except ValidationError as exc:
            erro_res = _montar_erro(
                tarefa_obj,
                codigo="PAYLOAD_INVALIDO",
                mensagem=f"Payload inválido para {tarefa_obj.acao}: {exc}",
                reprocessavel=False,
            )
            serializado = _serializar_resultado(erro_res)
            self._cache_idempotencia[tarefa_obj.id_mensagem] = copy.deepcopy(serializado)
            return serializado

        try:
            if isinstance(payload_validado, PayloadConversarTarefa):
                resultado, saida_completa = self._executar_conversar(
                    tarefa_obj, payload_validado
                )
            elif isinstance(payload_validado, PayloadProcessarDocumentoTarefa):
                resultado, saida_completa = self._executar_processar_documento(
                    tarefa_obj, payload_validado
                )
            elif isinstance(payload_validado, PayloadConfirmarTarefa):
                resultado, saida_completa = self._executar_confirmar(
                    tarefa_obj, payload_validado
                )
            elif isinstance(payload_validado, PayloadBaixaFisicaTarefa):
                resultado, saida_completa = self._executar_registrar_baixa_fisica(
                    tarefa_obj, payload_validado
                )
            else:  # pragma: no cover - protegido por Literal em TarefaA1
                raise TypeError(f"Ação não suportada: {tarefa_obj.acao}")
        except (FileNotFoundError, ValueError) as exc:
            erro_res = _montar_erro(
                tarefa_obj,
                codigo="ERRO_VALIDACAO_A1",
                mensagem=str(exc),
                reprocessavel=False,
            )
            serializado = _serializar_resultado(erro_res)
            self._cache_idempotencia[tarefa_obj.id_mensagem] = copy.deepcopy(serializado)
            return serializado
        except Exception as exc:  # noqa: BLE001
            erro_res = _montar_erro(
                tarefa_obj,
                codigo="ERRO_INTERNO_A1",
                mensagem=str(exc),
                reprocessavel=True,
            )
            return _serializar_resultado(erro_res)

        serializado = _serializar_resultado(resultado, saida_completa=saida_completa)
        self._cache_idempotencia[tarefa_obj.id_mensagem] = copy.deepcopy(serializado)
        return serializado


def executar(
    tarefa: dict[str, Any] | TarefaA1,
    *,
    repositorio: RepositorioA1Protocol | None = None,
    llm: Any | None = None,
    config: A1Config | None = None,
) -> dict[str, Any]:
    """Ponto de entrada único do Agente 1 no protocolo ``sinergia/1``."""
    servico = ServicoA1(
        repositorio=repositorio,
        llm=llm,
        config=config,
        cache_idempotencia=_CACHE_IDEMPOTENCIA_MODULO,
        extracoes_pendentes=_EXTRACOES_PENDENTES_MODULO,
    )
    return servico.executar(tarefa)
