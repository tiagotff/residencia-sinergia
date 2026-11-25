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
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime
from functools import lru_cache
from typing import Any

from pydantic import ValidationError

from sinergia.agentes.a1_recepcao.cliente_cid_who import (
    ClienteCidProtocol,
    ClienteCidWHO,
)
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
    CampoMetadado,
    DecisaoAgente,
    DestinatarioSaida,
    DivergenciaCampo,
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
    ProfissionalExtraido,
    ResultadoA1,
    SaidaMensagem,
    TarefaA1,
)
from sinergia.agentes.a1_recepcao.validador import (
    cid_sintaxe_valida,
    construir_quadro_revisao_atestado,
    detectar_divergencias,
    gerar_mensagem_confirmacao_extracao,
    mesclar_atestado_confirmado,
    validar_dados_confirmados,
)

_LOGGER = logging.getLogger(__name__)


class _CacheIdempotenciaLimitado:
    """Cache LRU thread-safe para idempotência por `id_mensagem`."""

    def __init__(self, max_entries: int = 2048) -> None:
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._dados: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._dados

    def __getitem__(self, key: str) -> dict[str, Any]:
        with self._lock:
            valor = self._dados[key]
            self._dados.move_to_end(key)
            return valor

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            if key in self._dados:
                self._dados.move_to_end(key)
            self._dados[key] = value
            while len(self._dados) > self._max_entries:
                self._dados.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._dados.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._dados)


_CACHE_IDEMPOTENCIA_MODULO = _CacheIdempotenciaLimitado(max_entries=2048)
_EXTRACOES_PENDENTES_MODULO: dict[tuple[str | int, int, int | None], AtestadoExtraido] = {}
_LOCK_LOCKS_MSG = threading.Lock()
_LOCKS_POR_MSG: dict[str, threading.RLock] = {}


def _obter_lock_mensagem(id_mensagem: str) -> threading.RLock:
    with _LOCK_LOCKS_MSG:
        if len(_LOCKS_POR_MSG) > 4096:
            _LOCKS_POR_MSG.clear()
        lock = _LOCKS_POR_MSG.get(id_mensagem)
        if lock is None:
            lock = threading.RLock()
            _LOCKS_POR_MSG[id_mensagem] = lock
        return lock


def _sanitizar_mensagem_erro(texto: str) -> str:
    """Remove credenciais, tokens e excesso de detalhes de mensagens de erro."""
    sanitizado = re.sub(r"://[^/\s:@]+:[^/\s@]+@", "://***:***@", texto)
    sanitizado = re.sub(r"gsk_[A-Za-z0-9_-]+", "gsk_***", sanitizado)
    sanitizado = re.sub(r"(?:Bearer|Basic)\s+\S+", "Bearer ***", sanitizado, flags=re.IGNORECASE)
    sanitizado = re.sub(
        r"(client_secret|WHO_ICD_CLIENT_SECRET)\s*[:=]\s*\S+",
        r"\1=***",
        sanitizado,
        flags=re.IGNORECASE,
    )
    return sanitizado[:300]


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
    with _LOCK_LOCKS_MSG:
        _LOCKS_POR_MSG.clear()


def _reconstruir_base_de_pendente(
    pendente_db: RegistroAtestadoDB,
    repositorio: RepositorioA1Protocol | None = None,
) -> AtestadoExtraido:
    """Reconstrói `AtestadoExtraido` base a partir do banco (`atestado` + `atestado_campo` + `cid_restrito`)."""
    profissional = ProfissionalExtraido()
    municipio: str | None = None
    nome_no_documento: str | None = None
    tipo_documento = pendente_db.tipo_documento
    cid = pendente_db.codigo_cid
    campos_meta: dict[str, CampoMetadado] = {}

    if repositorio is not None:
        if hasattr(repositorio, "obter_profissional"):
            prof_dict = repositorio.obter_profissional(pendente_db.id_profissional)
            if prof_dict:
                profissional = ProfissionalExtraido(**prof_dict)
        if hasattr(repositorio, "obter_municipio"):
            mun_tupla = repositorio.obter_municipio(pendente_db.id_municipio)
            if mun_tupla:
                municipio = mun_tupla[0]
        if cid is None and hasattr(repositorio, "obter_cid_restrito"):
            cid_reg = repositorio.obter_cid_restrito(pendente_db.id_atestado)
            if cid_reg is not None:
                cid = cid_reg.codigo_cid
        if hasattr(repositorio, "listar_campos_atestado"):
            for reg_campo in repositorio.listar_campos_atestado(pendente_db.id_atestado):
                valor = (
                    reg_campo.valor_confirmado
                    if reg_campo.valor_confirmado is not None
                    else reg_campo.valor_extraido
                )
                if reg_campo.nome_campo == "nome_no_documento" and valor is not None:
                    nome_no_documento = valor
                elif reg_campo.nome_campo == "tipo_documento" and valor in {
                    "atestado_medico",
                    "declaracao_comparecimento",
                    "outro",
                }:
                    tipo_documento = valor  # type: ignore[assignment]
                elif reg_campo.nome_campo == "municipio" and valor is not None:
                    municipio = valor
                elif reg_campo.nome_campo == "cid" and valor is not None:
                    cid = valor
                elif reg_campo.nome_campo == "profissional.nome" and valor is not None:
                    profissional.nome = valor

                if reg_campo.vazio_por_baixa_confianca:
                    campos_meta[reg_campo.nome_campo] = CampoMetadado(
                        confianca=reg_campo.confianca,
                        origem=reg_campo.origem,
                        vazio_por_baixa_confianca=True,
                    )

    return AtestadoExtraido(
        tipo_documento=tipo_documento,
        legibilidade="ok",
        nome_no_documento=nome_no_documento,
        periodo=PeriodoAfastamento(
            inicio=pendente_db.inicio_afastamento.date(),
            fim=pendente_db.fim_afastamento.date(),
        ),
        data_emissao=pendente_db.data_emissao,
        profissional=profissional,
        cid=cid,
        municipio=municipio,
        campos=campos_meta,
    )


def _valores_campos_atestado(
    atestado: AtestadoExtraido, *, politica_extrair_cid: bool
) -> list[tuple[str, str | None]]:
    dias = atestado.periodo.dias_calculados
    pares: list[tuple[str, str | None]] = [
        ("tipo_documento", atestado.tipo_documento),
        ("nome_no_documento", atestado.nome_no_documento),
        (
            "periodo.inicio",
            atestado.periodo.inicio.isoformat()
            if atestado.periodo.inicio is not None
            else None,
        ),
        (
            "periodo.fim",
            atestado.periodo.fim.isoformat()
            if atestado.periodo.fim is not None
            else None,
        ),
        ("periodo.dias", str(dias) if dias is not None else None),
        (
            "data_emissao",
            atestado.data_emissao.isoformat()
            if atestado.data_emissao is not None
            else None,
        ),
        ("profissional.conselho", atestado.profissional.conselho),
        ("profissional.numero", atestado.profissional.numero),
        ("profissional.uf", atestado.profissional.uf),
        ("profissional.nome", atestado.profissional.nome),
        ("municipio", atestado.municipio),
    ]
    if politica_extrair_cid:
        pares.append(("cid", atestado.cid))
    return pares


def _montar_campos_extracao(
    extraido: AtestadoExtraido, *, politica_extrair_cid: bool
) -> list[dict[str, Any]]:
    conf_map = extraido.confianca_campos or {}
    registros: list[dict[str, Any]] = []
    for nome_campo, valor in _valores_campos_atestado(
        extraido, politica_extrair_cid=politica_extrair_cid
    ):
        meta = extraido.campos.get(nome_campo)
        conf = meta.confianca if meta is not None else conf_map.get(nome_campo)
        if conf is not None:
            conf = round(max(0.0, min(1.0, float(conf))), 4)
        registros.append(
            {
                "nome_campo": nome_campo,
                "valor_extraido": valor,
                "confianca": conf,
                "origem": "IA",
                "vazio_por_baixa_confianca": bool(
                    meta.vazio_por_baixa_confianca if meta is not None else False
                ),
            }
        )
    return registros


def _montar_campos_confirmacao(
    confirmado: AtestadoExtraido,
    campos_corrigidos: Sequence[str],
    *,
    politica_extrair_cid: bool,
) -> list[dict[str, Any]]:
    alias_para_canonico = {
        "inicio": "periodo.inicio",
        "fim": "periodo.fim",
        "dias": "periodo.dias",
        "conselho": "profissional.conselho",
        "numero": "profissional.numero",
        "uf": "profissional.uf",
    }
    corrigidos = {
        alias_para_canonico.get(c.strip(), c.strip())
        for c in campos_corrigidos
        if c and c.strip()
    }
    if "periodo.inicio" in corrigidos or "periodo.fim" in corrigidos:
        corrigidos.add("periodo.dias")

    registros: list[dict[str, Any]] = []
    for nome_campo, valor in _valores_campos_atestado(
        confirmado, politica_extrair_cid=politica_extrair_cid
    ):
        registros.append(
            {
                "nome_campo": nome_campo,
                "valor_confirmado": valor,
                "origem": "MANUAL" if nome_campo in corrigidos else "IA",
                "vazio_por_baixa_confianca": False,
            }
        )
    return registros


def _formatar_atestado_payload(extraido: AtestadoExtraido) -> dict[str, Any]:
    """Serializa `AtestadoExtraido` no formato público do Anexo C.2 / D.2 e Guia A1 v0.7."""
    prof_dict: dict[str, Any] = {
        "conselho": extraido.profissional.conselho,
        "numero": extraido.profissional.numero,
        "uf": extraido.profissional.uf,
    }
    if extraido.profissional.nome is not None:
        prof_dict["nome"] = extraido.profissional.nome

    return {
        "tipo_documento": extraido.tipo_documento,
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
            "dias": extraido.periodo.dias_calculados,
        },
        "data_emissao": (
            extraido.data_emissao.isoformat()
            if extraido.data_emissao is not None
            else None
        ),
        "profissional": prof_dict,
        "cid": extraido.cid,
        "municipio": extraido.municipio,
        "campos": {
            chave: meta.model_dump(mode="json")
            for chave, meta in extraido.campos.items()
        },
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
    if saida.sensivel:
        dado["sensivel"] = True
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
        cliente_cid: ClienteCidProtocol | None = None,
        cache_idempotencia: Any | None = None,
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
        if cliente_cid is not None:
            self.cliente_cid: ClienteCidProtocol | None = cliente_cid
        elif self.config.who_icd_enabled:
            candidato = ClienteCidWHO(config=self.config)
            self.cliente_cid = candidato if candidato.credenciais_configuradas else None
        else:
            self.cliente_cid = None
        self.ultimo_diagnostico_cid: dict[str, Any] | None = None
        self._cache_idempotencia: Any = (
            cache_idempotencia
            if cache_idempotencia is not None
            else _CacheIdempotenciaLimitado(max_entries=2048)
        )
        self._extracoes_pendentes: dict[
            tuple[str | int, int, int | None], AtestadoExtraido
        ] = extracoes_pendentes if extracoes_pendentes is not None else {}

    def _resolver_cid_com_fallback(
        self, codigo_cid: str | None
    ) -> tuple[str | None, bool | None]:
        """Valida o CID primeiro em ``afastamento.cid``; se ausente, consulta a API externa WHO.

        Retorna ``(codigo_normalizado_para_persistencia, existe_no_catalogo)``, onde
        ``existe_no_catalogo`` pode ser ``True`` (encontrado no banco ou na API WHO),
        ``False`` (sintaxe inválida ou 404 na API WHO) ou ``None`` (política desativada,
        CID ausente ou API indisponível/não configurada — degradação graciosa).
        """
        self.ultimo_diagnostico_cid = None
        if not self.config.politica_extrair_cid or not codigo_cid or not codigo_cid.strip():
            return None, None

        codigo_norm = codigo_cid.strip().upper()
        if not cid_sintaxe_valida(codigo_norm):
            self.ultimo_diagnostico_cid = {
                "codigo_cid": codigo_norm,
                "origem": "SINTAXE_INVALIDA",
                "valido": False,
                "nome_doenca": None,
                "descricao": None,
            }
            return None, False

        # 1. Consulta local no banco de dados (afastamento.cid)
        cid_local = self.repositorio.validar_cid(
            codigo_norm, politica_extrair_cid=True
        )
        if cid_local is not None:
            info_local = (
                self.repositorio.obter_cid(cid_local)
                if hasattr(self.repositorio, "obter_cid")
                else None
            )
            nome_doenca_local = (
                (info_local.get("nome_doenca") or info_local.get("descricao"))
                if info_local
                else None
            )
            self.ultimo_diagnostico_cid = {
                "codigo_cid": cid_local,
                "origem": "BANCO_LOCAL",
                "valido": True,
                "nome_doenca": nome_doenca_local,
                "descricao": info_local.get("descricao") if info_local else None,
                "capitulo": info_local.get("capitulo") if info_local else None,
            }
            return cid_local, True

        # 2. Fallback na API externa da OMS (https://id.who.int) + cache-aside no banco
        if self.cliente_cid is None:
            self.ultimo_diagnostico_cid = {
                "codigo_cid": codigo_norm,
                "origem": "NAO_ENCONTRADO_SEM_API",
                "valido": None,
                "nome_doenca": None,
                "descricao": None,
                "detalhe": "CID ausente no banco local e API WHO não configurada.",
            }
            return None, None

        res_who = self.cliente_cid.consultar_cid(codigo_norm)
        if res_who.status == "ENCONTRADO" and res_who.dados is not None:
            dados = res_who.dados
            nome_doenca_who = dados.nome_doenca or dados.descricao
            codigo_salvo = dados.codigo_cid
            if hasattr(self.repositorio, "salvar_cid"):
                codigo_salvo = self.repositorio.salvar_cid(
                    dados.codigo_cid,
                    dados.descricao,
                    dados.capitulo,
                    nome_doenca=nome_doenca_who,
                )
            origem_diag = (
                "CONTAINER_WHO_LOCAL_CACHE_ASIDE"
                if getattr(res_who, "origem_consulta", None) == "CONTAINER_LOCAL"
                else "API_WHO_EXTERNA_CACHE_ASIDE"
            )
            self.ultimo_diagnostico_cid = {
                "codigo_cid": codigo_salvo,
                "origem": origem_diag,
                "valido": True,
                "nome_doenca": nome_doenca_who,
                "descricao": dados.descricao,
                "capitulo": dados.capitulo,
                "versao_cid": dados.versao_cid,
            }
            return codigo_salvo, True

        if res_who.status == "NAO_ENCONTRADO":
            self.ultimo_diagnostico_cid = {
                "codigo_cid": codigo_norm,
                "origem": "API_WHO_NAO_ENCONTRADO",
                "valido": False,
                "nome_doenca": None,
                "descricao": None,
                "detalhe": res_who.detalhe,
            }
            return None, False

        _LOGGER.warning(
            "Fallback WHO ICD indisponível para %s: %s",
            codigo_norm,
            res_who.detalhe or "sem detalhes",
        )
        self.ultimo_diagnostico_cid = {
            "codigo_cid": codigo_norm,
            "origem": "API_WHO_INDISPONIVEL",
            "valido": None,
            "nome_doenca": None,
            "descricao": None,
            "detalhe": res_who.detalhe,
        }
        return None, None

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

    def _resposta_duplicada(
        self, tarefa: TarefaA1, duplicado: RegistroAtestadoDB
    ) -> ResultadoA1:
        atestado_dup = _reconstruir_base_de_pendente(
            duplicado, repositorio=self.repositorio
        )
        quadro_dup = construir_quadro_revisao_atestado(
            atestado_dup,
            [],
            politica_extrair_cid=self.config.politica_extrair_cid,
        )
        payload_dup = {
            "atestado_extraido": _formatar_atestado_payload(atestado_dup),
            "divergencias": [],
            "revisao_colaborador": quadro_dup.model_dump(mode="json"),
        }
        PayloadProcessarDocumentoResultado.model_validate(payload_dup)
        return ResultadoA1.a_partir_da_tarefa(
            tarefa,
            status="PRECISA_HUMANO",
            estado_proposto="RECEBIDO",
            payload=payload_dup,
            saidas=[
                SaidaMensagem(
                    tipo="mensagem",
                    texto=(
                        "Este arquivo de atestado já foi recebido anteriormente "
                        f"(registro #{duplicado.id_atestado}). Deseja confirmar "
                        "os dados ou enviar outro documento?\n\n"
                        f"{quadro_dup.resumo_markdown}"
                    ),
                )
            ],
            eventos=[],
            decisao=DecisaoAgente(
                motivo="arquivo duplicado ja recebido anteriormente",
                usou_llm=False,
            ),
        )

    def _executar_processar_documento(
        self, tarefa: TarefaA1, payload: PayloadProcessarDocumentoTarefa
    ) -> tuple[ResultadoA1, bool]:
        id_colaborador = tarefa.contexto.id_colaborador
        id_caso = (
            tarefa.correlacao.id_caso
            or (tarefa.registro_inicial.id_caso if tarefa.registro_inicial else None)
        )

        duplicado = self.repositorio.verificar_duplicidade_arquivo(
            payload.arquivo_uri, id_colaborador=id_colaborador, sha256=payload.sha256
        )
        if duplicado is not None:
            return self._resposta_duplicada(tarefa, duplicado), False

        res_ocr = processar_arquivo_atestado(
            payload.arquivo_uri,
            mime=payload.mime,
            sha256_esperado=payload.sha256,
        )
        if res_ocr.sha256:
            duplicado_sha = self.repositorio.verificar_duplicidade_arquivo(
                payload.arquivo_uri,
                id_colaborador=id_colaborador,
                sha256=res_ocr.sha256,
            )
            if duplicado_sha is not None:
                return self._resposta_duplicada(tarefa, duplicado_sha), False

        extraido, decisao = extrair_atestado_via_llm(
            res_ocr, llm=self.llm, config=self.config
        )

        if extraido.legibilidade == "ilegivel":
            quadro_ilegivel = construir_quadro_revisao_atestado(
                extraido,
                [],
                politica_extrair_cid=self.config.politica_extrair_cid,
            )
            payload_ileg = {
                "atestado_extraido": _formatar_atestado_payload(extraido),
                "divergencias": [],
                "revisao_colaborador": quadro_ilegivel.model_dump(mode="json"),
            }
            PayloadProcessarDocumentoResultado.model_validate(payload_ileg)
            resultado = ResultadoA1.a_partir_da_tarefa(
                tarefa,
                status="PRECISA_HUMANO",
                estado_proposto="NOVO",
                payload=payload_ileg,
                saidas=[
                    SaidaMensagem(
                        tipo="mensagem",
                        texto=gerar_mensagem_confirmacao_extracao(
                            extraido,
                            [],
                            politica_extrair_cid=self.config.politica_extrair_cid,
                        ),
                    )
                ],
                eventos=[],
                decisao=decisao,
            )
            return resultado, False

        divergencias = detectar_divergencias(extraido, tarefa.registro_inicial)
        cid_valido, cid_existe = self._resolver_cid_com_fallback(extraido.cid)
        nome_doenca_cid = (
            self.ultimo_diagnostico_cid.get("nome_doenca")
            or self.ultimo_diagnostico_cid.get("descricao")
            if cid_valido and isinstance(self.ultimo_diagnostico_cid, dict)
            else None
        )
        if cid_existe is False and extraido.cid:
            divergencias.append(
                DivergenciaCampo(
                    campo="cid",
                    no_documento=extraido.cid,
                    informado="não encontrado na base local nem na API oficial da OMS",
                )
            )
        self._guardar_extracao_pendente(tarefa, extraido)

        inicio = extraido.periodo.inicio
        fim = extraido.periodo.fim
        if inicio is not None and fim is not None and fim >= inicio:
            sobrepostos = [
                at
                for at in self.repositorio.listar_atestados_sobrepostos(
                    id_colaborador, inicio, fim
                )
                if at.status == "CONFIRMADO"
            ]
            for sobreposto in sobrepostos:
                divergencias.append(
                    DivergenciaCampo(
                        campo="sobreposicao",
                        no_documento=f"{inicio.isoformat()} a {fim.isoformat()}",
                        informado=(
                            f"atestado #{sobreposto.id_atestado} "
                            f"({sobreposto.inicio_afastamento.date().isoformat()} a "
                            f"{sobreposto.fim_afastamento.date().isoformat()})"
                        ),
                    )
                )

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
            if id_prof is not None and id_municipio is not None:
                reg_pendente = self.repositorio.salvar_atestado_pendente(
                    id_colaborador=id_colaborador,
                    id_profissional=id_prof,
                    id_municipio=id_municipio,
                    data_emissao=extraido.data_emissao or inicio,
                    inicio_afastamento=inicio,
                    fim_afastamento=fim,
                    arquivo_uri=payload.arquivo_uri,
                    codigo_cid=cid_valido,
                    nome_doenca=nome_doenca_cid,
                    origem_dados="IA",
                    arquivo_sha256=res_ocr.sha256,
                    id_caso=id_caso,
                    tipo_documento=extraido.tipo_documento,
                    arquivo_mime=res_ocr.mime or payload.mime,
                    politica_cid_restrito=True,
                )
                if hasattr(self.repositorio, "salvar_campos_atestado"):
                    self.repositorio.salvar_campos_atestado(
                        reg_pendente.id_atestado,
                        _montar_campos_extracao(
                            extraido,
                            politica_extrair_cid=self.config.politica_extrair_cid,
                        ),
                        id_caso=id_caso,
                    )

        motivo_decisao = "documento legivel extraido pelo LLM"
        if divergencias:
            campos = ", ".join(d.campo for d in divergencias)
            motivo_decisao = (
                f"documento legivel extraido pelo LLM com divergencia de {campos}"
            )
        decisao_final = decisao.model_copy(update={"motivo": motivo_decisao})

        quadro_revisao = construir_quadro_revisao_atestado(
            extraido,
            divergencias,
            politica_extrair_cid=self.config.politica_extrair_cid,
        )
        payload_dict = {
            "atestado_extraido": _formatar_atestado_payload(extraido),
            "divergencias": [d.model_dump(mode="json") for d in divergencias],
            "revisao_colaborador": quadro_revisao.model_dump(mode="json"),
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
                    texto=gerar_mensagem_confirmacao_extracao(
                        extraido,
                        divergencias,
                        politica_extrair_cid=self.config.politica_extrair_cid,
                    ),
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
        pendente_db = self.repositorio.obter_ultimo_atestado_pendente(
            id_colaborador, id_caso=id_caso
        )
        if base_extraida is None and pendente_db is not None:
            base_extraida = _reconstruir_base_de_pendente(
                pendente_db, repositorio=self.repositorio
            )

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

        conselhos_aceitos = (
            self.repositorio.listar_conselhos_validos(mesclado.profissional.uf or "SP")
            if hasattr(self.repositorio, "listar_conselhos_validos")
            else ()
        ) or ("CRM", "CRO", "CRP")

        cid_valido, cid_existe = self._resolver_cid_com_fallback(mesclado.cid)
        nome_doenca_cid = (
            self.ultimo_diagnostico_cid.get("nome_doenca")
            or self.ultimo_diagnostico_cid.get("descricao")
            if cid_valido and isinstance(self.ultimo_diagnostico_cid, dict)
            else None
        )

        validacao = validar_dados_confirmados(
            mesclado,
            payload.campos_corrigidos,
            colaborador_ativo=colaborador_ativo,
            conselhos_aceitos=conselhos_aceitos,
            politica_extrair_cid=self.config.politica_extrair_cid,
            cid_existe_catalogo=cid_existe,
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

            saidas_inv = [
                SaidaMensagem(
                    tipo="mensagem", texto=validacao.mensagem_usuario
                )
            ]
            if validacao.codigo_evento == "colaborador_inativo":
                saidas_inv.append(
                    SaidaMensagem(
                        tipo="mensagem",
                        para=DestinatarioSaida(perfil="rh"),
                        nivel_detalhe="individual",
                        texto=(
                            f"Colaborador #{id_colaborador} apresentou atestado com "
                            f"cadastro inativo no período ({inicio} a {fim}). "
                            "Favor verificar vínculo."
                        ),
                        sensivel=False,
                    )
                )

            resultado = ResultadoA1.a_partir_da_tarefa(
                tarefa,
                status="OK",
                estado_proposto="NOVO",
                payload=PayloadConfirmarInvalidoResultado(
                    motivo=validacao.motivo or "dados inválidos",
                ),
                saidas=saidas_inv,
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

        prazos_legais = (
            validacao.marcos_legais.codigos_prazos_legais()
            if validacao.marcos_legais is not None
            else []
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
                nome_doenca=nome_doenca_cid,
                origem_dados=validacao.origem,
                id_caso=id_caso,
                tipo_documento=mesclado.tipo_documento,
                politica_cid_restrito=True,
            )

        codigo_cid_final = cid_valido if self.config.politica_extrair_cid else None
        nome_doenca_final = nome_doenca_cid if codigo_cid_final else None
        confirmado = self.repositorio.confirmar_atestado_no_banco(
            pendente_db.id_atestado,
            inicio_afastamento=inicio,
            fim_afastamento=fim,
            data_emissao=data_emissao,
            origem_dados=validacao.origem,
            prazo_entrega_fisica=validacao.prazo_entrega_fisica,
            id_profissional=id_prof if id_prof is not None else pendente_db.id_profissional,
            codigo_cid=codigo_cid_final,
            nome_doenca=nome_doenca_final,
            id_municipio=id_municipio if id_municipio is not None else pendente_db.id_municipio,
            tipo_documento=mesclado.tipo_documento,
            prazos_legais=prazos_legais,
            politica_cid_restrito=True,
        )
        if hasattr(self.repositorio, "salvar_campos_atestado"):
            self.repositorio.salvar_campos_atestado(
                confirmado.id_atestado,
                _montar_campos_confirmacao(
                    mesclado,
                    payload.campos_corrigidos,
                    politica_extrair_cid=self.config.politica_extrair_cid,
                ),
                id_caso=id_caso,
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
            "prazos_legais": prazos_legais,
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
                prazos_legais=prazos_legais,
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
        perfil_chamador = tarefa.contexto.perfil.strip().lower()
        if perfil_chamador not in {"rh", "admin"}:
            raise PermissionError(
                f"Perfil '{tarefa.contexto.perfil}' não autorizado para registrar baixa física."
            )

        self.repositorio.registrar_baixa_fisica_no_banco(
            payload.id_atestado,
            payload.recebido_fisico_por,
            payload.recebido_fisico_em,
            perfis_permitidos=("RH", "ADMIN"),
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
                recebido_por=payload.recebido_fisico_por,
                data_hora=recebido_em_str,
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

        lock_msg = _obter_lock_mensagem(tarefa_obj.id_mensagem)
        with lock_msg:
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
            except PermissionError as exc:
                erro_res = _montar_erro(
                    tarefa_obj,
                    codigo="NAO_AUTORIZADO",
                    mensagem=str(exc),
                    reprocessavel=False,
                )
                serializado = _serializar_resultado(erro_res)
                self._cache_idempotencia[tarefa_obj.id_mensagem] = copy.deepcopy(serializado)
                return serializado
            except (FileNotFoundError, ValueError) as exc:
                erro_res = _montar_erro(
                    tarefa_obj,
                    codigo="ERRO_VALIDACAO_A1",
                    mensagem=_sanitizar_mensagem_erro(str(exc)),
                    reprocessavel=False,
                )
                serializado = _serializar_resultado(erro_res)
                self._cache_idempotencia[tarefa_obj.id_mensagem] = copy.deepcopy(serializado)
                return serializado
            except Exception as exc:
                _LOGGER.exception(
                    "Erro interno no Agente A1 (%s) [%s]",
                    tarefa_obj.acao,
                    type(exc).__name__,
                )
                erro_res = _montar_erro(
                    tarefa_obj,
                    codigo="ERRO_INTERNO_A1",
                    mensagem="Falha interna ao processar a tarefa no Agente A1.",
                    reprocessavel=True,
                )
                return _serializar_resultado(erro_res)

            serializado = _serializar_resultado(resultado, saida_completa=saida_completa)
            self._cache_idempotencia[tarefa_obj.id_mensagem] = copy.deepcopy(serializado)
            return serializado

    def executar_modelo(self, tarefa: dict[str, Any] | TarefaA1) -> ResultadoA1:
        """Executa uma `TAREFA` e devolve o envelope validado como `ResultadoA1`."""
        return ResultadoA1.model_validate(self.executar(tarefa))


class AgenteA1(ServicoA1):
    """Alias oficial do Guia A1 v0.7 (`AgenteA1(db=sessao, llm=cliente_groq)`)."""

    def __init__(
        self,
        *,
        db: Any | None = None,
        repositorio: RepositorioA1Protocol | None = None,
        llm: Any | None = None,
        config: A1Config | None = None,
        cliente_cid: ClienteCidProtocol | None = None,
        cache_idempotencia: Any | None = None,
        extracoes_pendentes: dict[tuple[str | int, int, int | None], AtestadoExtraido]
        | None = None,
        retornar_modelo: bool = False,
    ) -> None:
        cfg = config or obter_config_a1()
        repo_final: RepositorioA1Protocol | None
        if repositorio is not None:
            repo_final = repositorio
        elif db is None:
            repo_final = None
        elif isinstance(db, RepositorioA1Protocol):
            repo_final = db
        else:
            repo_final = RepositorioA1Postgres(cfg.database_url, conn=db)

        self._retornar_modelo = retornar_modelo
        super().__init__(
            repositorio=repo_final,
            llm=llm,
            config=cfg,
            cliente_cid=cliente_cid,
            cache_idempotencia=cache_idempotencia,
            extracoes_pendentes=extracoes_pendentes,
        )

    def executar(
        self, tarefa: dict[str, Any] | TarefaA1
    ) -> dict[str, Any] | ResultadoA1:
        resultado = super().executar(tarefa)
        if self._retornar_modelo:
            return ResultadoA1.model_validate(resultado)
        return resultado


def executar(
    tarefa: dict[str, Any] | TarefaA1,
    *,
    repositorio: RepositorioA1Protocol | None = None,
    llm: Any | None = None,
    config: A1Config | None = None,
    cliente_cid: ClienteCidProtocol | None = None,
) -> dict[str, Any]:
    """Ponto de entrada único do Agente 1 no protocolo ``sinergia/1``."""
    servico = ServicoA1(
        repositorio=repositorio,
        llm=llm,
        config=config,
        cliente_cid=cliente_cid,
        cache_idempotencia=_CACHE_IDEMPOTENCIA_MODULO,
        extracoes_pendentes=_EXTRACOES_PENDENTES_MODULO,
    )
    return servico.executar(tarefa)


def executar_modelo(
    tarefa: dict[str, Any] | TarefaA1,
    *,
    repositorio: RepositorioA1Protocol | None = None,
    llm: Any | None = None,
    config: A1Config | None = None,
    cliente_cid: ClienteCidProtocol | None = None,
) -> ResultadoA1:
    """Versão tipada de `executar(tarefa)` que retorna `ResultadoA1`."""
    return ResultadoA1.model_validate(
        executar(
            tarefa,
            repositorio=repositorio,
            llm=llm,
            config=config,
            cliente_cid=cliente_cid,
        )
    )
