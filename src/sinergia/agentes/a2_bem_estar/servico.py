"""Fachada oficial ``executar(tarefa) -> resultado`` do Agente 2 (A2 — Bem-Estar).

Implementa o protocolo ``sinergia/1`` conforme a ``Documentação - SinergIA v2.2.docx``
(Seção 5.1 e Anexos A, C.3 e D.3), garantindo idempotência por ``id_mensagem`` e
roteamento completo da ação conversacional ``a2.responder`` via CrewAI + Groq.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from typing import Any

from pydantic import ValidationError

from sinergia.agentes.a2_bem_estar.config import A2Config, obter_config_a2
from sinergia.agentes.a2_bem_estar.crew import (
    ExecutorCrewA2,
    TelemetriaExecucaoA2,
)
from sinergia.agentes.a2_bem_estar.repositorio import (
    RepositorioA2Memoria,
    RepositorioA2Postgres,
    RepositorioA2Protocol,
)
from sinergia.agentes.a2_bem_estar.schemas import (
    DecisaoAgente,
    DestinatarioSaida,
    ErroAgente,
    EventoEmitido,
    PayloadResponderResultado,
    PayloadResponderTarefa,
    ResultadoA2,
    SaidaMensagem,
    TarefaA2,
)

_CACHE_IDEMPOTENCIA_MODULO: dict[str, dict[str, Any]] = {}
_CACHE_TELEMETRIA_MODULO: dict[str, TelemetriaExecucaoA2 | None] = {}


def _postgres_a2_disponivel(database_url: str) -> bool:
    try:
        import psycopg
    except ImportError:
        return False
    try:
        with (
            psycopg.connect(database_url, connect_timeout=1) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT 1 FROM apoio.epi LIMIT 1")
            cur.fetchone()
        return True
    except (psycopg.Error, OSError):
        return False


@lru_cache(maxsize=4)
def _criar_repositorio_padrao(database_url: str) -> RepositorioA2Protocol:
    url = database_url.strip()
    if url and _postgres_a2_disponivel(url):
        return RepositorioA2Postgres(url)
    return RepositorioA2Memoria()


def limpar_cache_idempotencia() -> None:
    """Limpa os caches de idempotência, telemetria e repositório padrão do módulo."""
    _CACHE_IDEMPOTENCIA_MODULO.clear()
    _CACHE_TELEMETRIA_MODULO.clear()
    _criar_repositorio_padrao.cache_clear()


def _serializar_saida(saida: SaidaMensagem) -> dict[str, Any]:
    dado: dict[str, Any] = {
        "tipo": saida.tipo,
        "para": saida.para.model_dump(mode="json") if saida.para is not None else None,
        "nivel_detalhe": saida.nivel_detalhe,
        "texto": saida.texto,
        "fontes": [f.model_dump(mode="json") for f in saida.fontes],
        "agendar_para": (
            saida.agendar_para.isoformat()
            if isinstance(saida.agendar_para, datetime)
            else saida.agendar_para
        ),
    }
    if saida.canal is not None:
        dado["canal"] = saida.canal
    if dado["para"] is None:
        dado.pop("para")
    if dado["nivel_detalhe"] is None:
        dado.pop("nivel_detalhe")
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


def _serializar_resultado(resultado: ResultadoA2) -> dict[str, Any]:
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
        "saidas": [_serializar_saida(s) for s in resultado.saidas],
        "eventos": [e.model_dump(mode="json") for e in resultado.eventos],
    }
    if resultado.decisao is not None:
        dado["decisao"] = _serializar_decisao(resultado.decisao)
    if resultado.erro is not None:
        dado["erro"] = resultado.erro.model_dump(mode="json")
    return dado


def _montar_erro(
    tarefa: TarefaA2,
    *,
    codigo: str,
    mensagem: str,
    reprocessavel: bool = False,
) -> ResultadoA2:
    return ResultadoA2.a_partir_da_tarefa(
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


class ServicoA2:
    """Serviço executor do Agente 2 (A2 — Bem-Estar) sob protocolo ``sinergia/1``."""

    def __init__(
        self,
        *,
        repositorio: RepositorioA2Protocol | None = None,
        llm: Any | None = None,
        config: A2Config | None = None,
        buscador: Any | None = None,
        servico_llm: Any | None = None,
        cache_idempotencia: dict[str, dict[str, Any]] | None = None,
        cache_telemetria: dict[str, TelemetriaExecucaoA2 | None] | None = None,
    ) -> None:
        self.config = config or obter_config_a2()
        self.repositorio: RepositorioA2Protocol = (
            repositorio
            if repositorio is not None
            else _criar_repositorio_padrao(self.config.database_url)
        )
        self.llm = llm
        self.buscador = buscador
        self.servico_llm = servico_llm
        self._cache_idempotencia: dict[str, dict[str, Any]] = (
            cache_idempotencia if cache_idempotencia is not None else {}
        )
        self._cache_telemetria: dict[str, TelemetriaExecucaoA2 | None] = (
            cache_telemetria if cache_telemetria is not None else {}
        )
        self._executor_crew = ExecutorCrewA2(
            config=self.config,
            repositorio=self.repositorio,
            buscador=self.buscador,
            servico_llm=self.servico_llm,
            llm=self.llm,
        )

    def _executar_responder(
        self,
        tarefa: TarefaA2,
        payload: PayloadResponderTarefa,
        *,
        ao_avancar_etapa: Callable[[str], None] | None = None,
        ao_receber_reasoning: Callable[[str], None] | None = None,
        ao_receber_conteudo: Callable[[str], None] | None = None,
    ) -> tuple[ResultadoA2, TelemetriaExecucaoA2]:
        exec_res = self._executor_crew.executar_responder(
            pergunta=payload.pergunta,
            id_colaborador=tarefa.contexto.id_colaborador,
            intencao=payload.intencao,
            perfil=payload.perfil,
            historico_resumo=payload.historico_resumo,
            historico=payload.historico,
            modelo=payload.modelo,
            buscador=self.buscador,
            servico_llm=self.servico_llm,
            llm=self.llm,
            ao_avancar_etapa=ao_avancar_etapa,
            ao_receber_reasoning=ao_receber_reasoning,
            ao_receber_conteudo=ao_receber_conteudo,
        )

        payload_saida = PayloadResponderResultado(
            base=exec_res.base_slug,
            trechos_usados=exec_res.trechos_usados,
            id_solicitacao_epi=exec_res.id_solicitacao_epi,
        )
        eventos: list[EventoEmitido] = []
        if exec_res.id_solicitacao_epi is not None:
            eventos.append(
                EventoEmitido(
                    tipo="solicitacao_epi.aberta",
                    dados={"id_solicitacao": exec_res.id_solicitacao_epi},
                )
            )

        if not exec_res.telemetria.roteamento.usar_rag:
            motivo_decisao = "guardrail conversacional sem busca vetorial"
            confianca = 0.92
        elif exec_res.id_solicitacao_epi is not None:
            motivo_decisao = (
                f"roteiro NR-6 com abertura de solicitacao_epi #{exec_res.id_solicitacao_epi} "
                f"e {exec_res.trechos_usados} trecho(s) vigente(s)"
            )
            confianca = 0.90
        else:
            motivo_decisao = f"resposta RAG com {exec_res.trechos_usados} trechos vigentes"
            confianca = 0.88 if exec_res.trechos_usados > 0 else 0.70

        resultado = ResultadoA2.a_partir_da_tarefa(
            tarefa,
            status="OK",
            estado_proposto=None,
            payload=payload_saida,
            saidas=[
                SaidaMensagem(
                    tipo="mensagem",
                    para=DestinatarioSaida(
                        perfil=tarefa.contexto.perfil,
                        id_colaborador=tarefa.contexto.id_colaborador,
                    ),
                    nivel_detalhe="individual",
                    texto=exec_res.resposta,
                    fontes=exec_res.fontes,
                    agendar_para=None,
                )
            ],
            eventos=eventos,
            decisao=DecisaoAgente(
                motivo=motivo_decisao,
                usou_llm=True,
                confianca=confianca,
                modelo=exec_res.telemetria.metricas_llm.modelo,
            ),
        )
        return resultado, exec_res.telemetria

    def executar_com_telemetria(
        self,
        tarefa: dict[str, Any] | TarefaA2,
        *,
        ao_avancar_etapa: Callable[[str], None] | None = None,
        ao_receber_reasoning: Callable[[str], None] | None = None,
        ao_receber_conteudo: Callable[[str], None] | None = None,
    ) -> tuple[dict[str, Any], TelemetriaExecucaoA2 | None]:
        """Executa uma `TAREFA` do protocolo `sinergia/1` e devolve também a telemetria do A2."""
        tarefa_obj = (
            tarefa
            if isinstance(tarefa, TarefaA2)
            else TarefaA2.model_validate(tarefa)
        )

        if tarefa_obj.id_mensagem in self._cache_idempotencia:
            return (
                copy.deepcopy(self._cache_idempotencia[tarefa_obj.id_mensagem]),
                copy.deepcopy(self._cache_telemetria.get(tarefa_obj.id_mensagem)),
            )

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
            self._cache_telemetria[tarefa_obj.id_mensagem] = None
            return serializado, None

        telemetria: TelemetriaExecucaoA2 | None = None
        try:
            if isinstance(payload_validado, PayloadResponderTarefa):
                resultado, telemetria = self._executar_responder(
                    tarefa_obj,
                    payload_validado,
                    ao_avancar_etapa=ao_avancar_etapa,
                    ao_receber_reasoning=ao_receber_reasoning,
                    ao_receber_conteudo=ao_receber_conteudo,
                )
            else:
                erro_res = _montar_erro(
                    tarefa_obj,
                    codigo="ACAO_NAO_IMPLEMENTADA",
                    mensagem=f"Ação {tarefa_obj.acao} será implementada na etapa posterior.",
                    reprocessavel=False,
                )
                serializado = _serializar_resultado(erro_res)
                self._cache_idempotencia[tarefa_obj.id_mensagem] = copy.deepcopy(serializado)
                self._cache_telemetria[tarefa_obj.id_mensagem] = None
                return serializado, None
        except ValueError as exc:
            erro_res = _montar_erro(
                tarefa_obj,
                codigo="ERRO_VALIDACAO_A2",
                mensagem=str(exc),
                reprocessavel=False,
            )
            serializado = _serializar_resultado(erro_res)
            self._cache_idempotencia[tarefa_obj.id_mensagem] = copy.deepcopy(serializado)
            self._cache_telemetria[tarefa_obj.id_mensagem] = None
            return serializado, None
        except Exception as exc:  # noqa: BLE001
            erro_res = _montar_erro(
                tarefa_obj,
                codigo="ERRO_INTERNO_A2",
                mensagem=str(exc),
                reprocessavel=True,
            )
            return _serializar_resultado(erro_res), None

        serializado = _serializar_resultado(resultado)
        self._cache_idempotencia[tarefa_obj.id_mensagem] = copy.deepcopy(serializado)
        self._cache_telemetria[tarefa_obj.id_mensagem] = copy.deepcopy(telemetria)
        return serializado, telemetria

    def executar(self, tarefa: dict[str, Any] | TarefaA2) -> dict[str, Any]:
        """Executa uma `TAREFA` do protocolo `sinergia/1` garantindo idempotência."""
        serializado, _ = self.executar_com_telemetria(tarefa)
        return serializado


def executar(
    tarefa: dict[str, Any] | TarefaA2,
    *,
    repositorio: RepositorioA2Protocol | None = None,
    llm: Any | None = None,
    config: A2Config | None = None,
    buscador: Any | None = None,
    servico_llm: Any | None = None,
) -> dict[str, Any]:
    """Ponto de entrada único do Agente 2 no protocolo ``sinergia/1``."""
    servico = ServicoA2(
        repositorio=repositorio,
        llm=llm,
        config=config,
        buscador=buscador,
        servico_llm=servico_llm,
        cache_idempotencia=_CACHE_IDEMPOTENCIA_MODULO,
        cache_telemetria=_CACHE_TELEMETRIA_MODULO,
    )
    return servico.executar(tarefa)


def executar_com_telemetria(
    tarefa: dict[str, Any] | TarefaA2,
    *,
    repositorio: RepositorioA2Protocol | None = None,
    llm: Any | None = None,
    config: A2Config | None = None,
    buscador: Any | None = None,
    servico_llm: Any | None = None,
    ao_avancar_etapa: Callable[[str], None] | None = None,
    ao_receber_reasoning: Callable[[str], None] | None = None,
    ao_receber_conteudo: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], TelemetriaExecucaoA2 | None]:
    """Executa uma tarefa do Agente 2 retornando também a telemetria instrumentada para o demo."""
    servico = ServicoA2(
        repositorio=repositorio,
        llm=llm,
        config=config,
        buscador=buscador,
        servico_llm=servico_llm,
        cache_idempotencia=_CACHE_IDEMPOTENCIA_MODULO,
        cache_telemetria=_CACHE_TELEMETRIA_MODULO,
    )
    return servico.executar_com_telemetria(
        tarefa,
        ao_avancar_etapa=ao_avancar_etapa,
        ao_receber_reasoning=ao_receber_reasoning,
        ao_receber_conteudo=ao_receber_conteudo,
    )
