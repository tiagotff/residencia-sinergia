"""Adaptador dos 3 modos de conexão A0 <-> A1 e inspetor das 5 tabelas v0.7."""

from __future__ import annotations

import copy
import time
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import unquote
from uuid import uuid4

import httpx
import jsonschema
from pydantic import BaseModel, Field, ValidationError

from sinergia.agentes.a1_recepcao.cliente_cid_who import ClienteCidProtocol
from sinergia.agentes.a1_recepcao.config import A1Config
from sinergia.agentes.a1_recepcao.extrator_llm import MockLLM
from sinergia.agentes.a1_recepcao.painel.diagnostico import sanitizar_url_conexao
from sinergia.agentes.a1_recepcao.repositorio import (
    RepositorioA1Memoria,
    RepositorioA1Postgres,
    RepositorioA1Protocol,
)
from sinergia.agentes.a1_recepcao.schemas import (
    OrigemPreenchimentoFormulario,
    ResultadoA1,
    TarefaA1,
    exportar_json_schemas_a1,
)
from sinergia.agentes.a1_recepcao.servico import AgenteA1, ServicoA1

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]

ModoTransporte = Literal["in_process", "http_a1", "http_a0"]
ModoRepositorio = Literal["memoria", "postgres"]
ModoLLMConexao = Literal["mock", "groq"]
PerfilInspecao = Literal["A1_RH", "GESTOR_A2"]


class RegistroTrocaProtocolo(BaseModel):
    """Registro completo de uma troca TAREFA -> RESULTADO no protocolo sinergia/1."""

    id_troca: str = Field(default_factory=lambda: uuid4().hex[:10])
    timestamp_iso: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    modo_transporte: ModoTransporte
    acao: str
    id_mensagem_tarefa: str
    id_caso: int | None = None
    thread_id: str | None = None
    latencia_ms: float = 0.0
    retornou_modelo_pydantic: bool = False
    tarefa: dict[str, Any]
    resultado: dict[str, Any]
    schema_tarefa_valido: bool
    schema_resultado_valido: bool
    erros_schema: list[str] = Field(default_factory=list)
    eh_replay_idempotencia: bool = False
    diagnostico_cid: dict[str, Any] | None = None


def validar_envelope_sinergia(
    envelope: dict[str, Any],
    tipo: Literal["TarefaA1", "ResultadoA1"],
) -> tuple[bool, list[str]]:
    """Valida um dicionário contra JSON Schema oficial e modelo Pydantic do `sinergia/1`."""
    erros: list[str] = []
    schemas = exportar_json_schemas_a1()

    try:
        jsonschema.validate(instance=envelope, schema=schemas[tipo])
    except jsonschema.ValidationError as exc:
        erros.append(f"JSON Schema ({tipo}): {exc.message}")

    try:
        if tipo == "TarefaA1":
            obj_t = TarefaA1.model_validate(envelope)
            obj_t.validar_payload_acao()
        else:
            obj_r = ResultadoA1.model_validate(envelope)
            if obj_r.status != "ERRO":
                obj_r.validar_payload_acao()
    except (ValidationError, ValueError, KeyError) as exc:
        erros.append(f"Pydantic ({tipo}): {exc}")

    return len(erros) == 0, erros


def _serializar_modelo_ou_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, BaseModel):
        return item.model_dump(mode="json")
    return dict(item)


def _construir_resultado_erro_transporte(
    tarefa: dict[str, Any],
    *,
    codigo: str,
    mensagem: str,
) -> dict[str, Any]:
    correlacao = tarefa.get("correlacao") or {
        "id_caso": None,
        "thread_id": None,
        "id_tarefa": "erro-transporte",
    }
    return {
        "protocolo": "sinergia/1",
        "id_mensagem": str(uuid4()),
        "tipo": "RESULTADO",
        "acao": tarefa.get("acao", "a1.conversar"),
        "origem": "A1",
        "destino": "A0",
        "em_resposta_a": tarefa.get("id_mensagem", "desconhecido"),
        "correlacao": correlacao,
        "status": "ERRO",
        "estado_proposto": None,
        "payload": {},
        "saidas": [],
        "eventos": [],
        "decisao": {"motivo": mensagem, "usou_llm": False},
        "erro": {"codigo": codigo, "mensagem": mensagem, "reprocessavel": True},
    }


def _executar_post_http(
    cliente: Any | None,
    destino: str,
    *,
    json_body: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float,
) -> Any:
    """Executa POST HTTP via `httpx` ou cliente injetado sem acionar deprecation do `TestClient`."""
    kwargs: dict[str, Any] = {"json": json_body}
    if headers:
        kwargs["headers"] = headers
    if cliente is not None:
        if type(cliente).__name__ != "TestClient":
            kwargs["timeout"] = timeout
        return cliente.post(destino, **kwargs)
    return httpx.post(destino, **kwargs, timeout=timeout)


class ClienteConexaoA0A1:
    """Abstrai a execução do A1 em memória/processo, HTTP FastAPI A1 ou API do A0."""

    def __init__(
        self,
        *,
        modo_transporte: ModoTransporte = "in_process",
        config: A1Config,
        modo_repo: ModoRepositorio = "memoria",
        modo_llm: ModoLLMConexao = "mock",
        retornar_modelo: bool = False,
        url_a1_http: str = "http://localhost:8001",
        url_a0_http: str = "http://localhost:8000",
        token_jwt_a0: str = "",
        repositorio_memoria: RepositorioA1Memoria | None = None,
        cache_idempotencia: dict[str, Any] | Any | None = None,
        extracoes_pendentes: dict[Any, Any] | None = None,
        http_client: Any | None = None,
        cliente_cid: ClienteCidProtocol | None = None,
    ) -> None:
        self.modo_transporte = modo_transporte
        self.config = config
        self.modo_repo = modo_repo
        self.modo_llm = modo_llm
        self.retornar_modelo = retornar_modelo
        self.url_a1_http = url_a1_http
        self.url_a0_http = url_a0_http
        self.token_jwt_a0 = token_jwt_a0
        self.repositorio_memoria = repositorio_memoria or RepositorioA1Memoria()
        self._cache_idempotencia = (
            cache_idempotencia if cache_idempotencia is not None else {}
        )
        self._extracoes_pendentes = (
            extracoes_pendentes if extracoes_pendentes is not None else {}
        )
        self._http_client = http_client
        self._cliente_cid = cliente_cid

    def _obter_repositorio(self) -> RepositorioA1Protocol:
        if self.modo_repo == "postgres":
            return RepositorioA1Postgres(self.config.database_url)
        return self.repositorio_memoria

    def _criar_agente_in_process(self) -> AgenteA1:
        llm: Any | None = MockLLM() if self.modo_llm == "mock" else None
        return AgenteA1(
            repositorio=self._obter_repositorio(),
            llm=llm,
            config=self.config,
            cliente_cid=self._cliente_cid,
            cache_idempotencia=self._cache_idempotencia,
            extracoes_pendentes=self._extracoes_pendentes,
            retornar_modelo=self.retornar_modelo,
        )

    @staticmethod
    def _extrair_diagnostico_cid_headers(headers: Any) -> dict[str, Any] | None:
        if headers is None:
            return None
        origem = headers.get("X-Sinergia-Cid-Origem")
        if not origem:
            return None
        valido_raw = headers.get("X-Sinergia-Cid-Valido")
        valido: bool | None = (
            valido_raw.lower() == "true"
            if isinstance(valido_raw, str) and valido_raw.lower() in {"true", "false"}
            else None
        )
        descricao = (
            unquote(headers["X-Sinergia-Cid-Descricao"])
            if headers.get("X-Sinergia-Cid-Descricao")
            else None
        )
        nome_doenca = (
            unquote(headers["X-Sinergia-Cid-Nome-Doenca"])
            if headers.get("X-Sinergia-Cid-Nome-Doenca")
            else descricao
        )
        diag: dict[str, Any] = {
            "codigo_cid": headers.get("X-Sinergia-Cid-Codigo"),
            "origem": str(origem),
            "valido": valido,
            "nome_doenca": nome_doenca,
            "descricao": descricao,
        }
        if headers.get("X-Sinergia-Cid-Capitulo"):
            diag["capitulo"] = unquote(headers["X-Sinergia-Cid-Capitulo"])
        if headers.get("X-Sinergia-Cid-Versao"):
            diag["versao_cid"] = str(headers["X-Sinergia-Cid-Versao"])
        if headers.get("X-Sinergia-Cid-Detalhe"):
            diag["detalhe"] = unquote(headers["X-Sinergia-Cid-Detalhe"])
        if headers.get("X-Sinergia-Cid-Status"):
            diag["status_validacao_cid"] = unquote(headers["X-Sinergia-Cid-Status"])
        if headers.get("X-Sinergia-Cid-Origem-Preenchimento"):
            diag["origem_preenchimento"] = str(
                headers["X-Sinergia-Cid-Origem-Preenchimento"]
            )
        if headers.get("X-Sinergia-Cid-Validado-Em"):
            diag["validado_em"] = str(headers["X-Sinergia-Cid-Validado-Em"])
        if headers.get("X-Sinergia-Cid-Versao-Tabela"):
            diag["versao_tabela_cid"] = unquote(headers["X-Sinergia-Cid-Versao-Tabela"])
        bloqueia_raw = headers.get("X-Sinergia-Cid-Bloqueia-Aprovacao")
        if isinstance(bloqueia_raw, str) and bloqueia_raw.lower() in {"true", "false"}:
            bloqueia = bloqueia_raw.lower() == "true"
            diag["bloqueia_aprovacao_automatica"] = bloqueia
            diag["requer_revisao_humana"] = bloqueia
        if diag.get("valido") is True:
            diag["descricao_oficial"] = descricao or nome_doenca
        return diag

    def _enviar_http_a1(
        self, tarefa: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        base = self.url_a1_http.strip().rstrip("/")
        destino = f"{base}/executar" if base else "/executar"
        timeout = max(float(self.config.llm_timeout_segundos) + 10.0, 30.0)
        try:
            resp = _executar_post_http(
                self._http_client,
                destino,
                json_body=tarefa,
                timeout=timeout,
            )
            if resp.status_code != 200:
                return (
                    _construir_resultado_erro_transporte(
                        tarefa,
                        codigo="ERRO_HTTP_A1",
                        mensagem=f"HTTP {resp.status_code} em {destino}: {resp.text[:180]}",
                    ),
                    None,
                )
            diag_cid = self._extrair_diagnostico_cid_headers(getattr(resp, "headers", None))
            return dict(resp.json()), diag_cid
        except (httpx.HTTPError, ValueError, TypeError, OSError) as exc:
            return (
                _construir_resultado_erro_transporte(
                    tarefa,
                    codigo="ERRO_CONEXAO_HTTP_A1",
                    mensagem=(
                        f"Falha ao conectar em {destino}: "
                        f"{sanitizar_url_conexao(str(exc))}"
                    ),
                ),
                None,
            )

    def enviar_tarefa(
        self,
        tarefa: dict[str, Any],
        *,
        eh_replay: bool = False,
    ) -> RegistroTrocaProtocolo:
        """Executa uma `TAREFA` pelo transporte ativo e valida todo o ciclo `sinergia/1`."""
        tarefa_copia = copy.deepcopy(tarefa)
        ok_tarefa, erros_tarefa = validar_envelope_sinergia(tarefa_copia, "TarefaA1")

        retornou_modelo = False
        diagnostico_cid: dict[str, Any] | None = None
        inicio = time.perf_counter()

        if self.modo_transporte == "http_a1":
            resultado_dict, diagnostico_cid = self._enviar_http_a1(tarefa_copia)
        else:
            agente = self._criar_agente_in_process()
            try:
                saida = agente.executar(tarefa_copia)
                if isinstance(saida, ResultadoA1):
                    retornou_modelo = True
                    resultado_dict = saida.model_dump(mode="json")
                else:
                    resultado_dict = dict(saida)
                if agente.ultimo_diagnostico_cid is not None:
                    diagnostico_cid = copy.deepcopy(agente.ultimo_diagnostico_cid)
            except (ValidationError, ValueError, RuntimeError, OSError) as exc:
                resultado_dict = _construir_resultado_erro_transporte(
                    tarefa_copia,
                    codigo="ERRO_EXECUCAO_A1",
                    mensagem=sanitizar_url_conexao(str(exc)),
                )

        latencia_ms = round((time.perf_counter() - inicio) * 1000, 2)
        ok_resultado, erros_resultado = validar_envelope_sinergia(
            resultado_dict, "ResultadoA1"
        )

        correlacao = tarefa_copia.get("correlacao") or {}
        return RegistroTrocaProtocolo(
            modo_transporte=self.modo_transporte,
            acao=str(tarefa_copia.get("acao", "")),
            id_mensagem_tarefa=str(tarefa_copia.get("id_mensagem", "")),
            id_caso=correlacao.get("id_caso"),
            thread_id=correlacao.get("thread_id"),
            latencia_ms=latencia_ms,
            retornou_modelo_pydantic=retornou_modelo,
            tarefa=tarefa_copia,
            resultado=resultado_dict,
            schema_tarefa_valido=ok_tarefa,
            schema_resultado_valido=ok_resultado,
            erros_schema=[*erros_tarefa, *erros_resultado],
            eh_replay_idempotencia=eh_replay,
            diagnostico_cid=diagnostico_cid,
        )

    def consultar_cid(
        self,
        codigo_cid: str,
        *,
        origem_preenchimento: OrigemPreenchimentoFormulario = "MANUAL",
    ) -> dict[str, Any]:
        """Valida um CID no fluxo oficial e retorna o diagnóstico 4A/4B com auditoria."""
        cfg_consulta = A1Config(
            **{**self.config.__dict__, "politica_extrair_cid": True, "politica_cid_modo": "extrair_restrito"}
        )
        servico = ServicoA1(
            repositorio=self._obter_repositorio(),
            llm=MockLLM(),
            config=cfg_consulta,
            cliente_cid=self._cliente_cid,
        )
        codigo_norm, existe = servico._resolver_cid_com_fallback(
            codigo_cid, origem_preenchimento=origem_preenchimento
        )
        diag = copy.deepcopy(servico.ultimo_diagnostico_cid) or {
            "codigo_cid": codigo_cid.strip().upper() or None,
            "origem": "SINTAXE_INVALIDA",
            "valido": False,
            "nome_doenca": None,
            "descricao": None,
            "status_validacao_cid": "CID Não Encontrada / Inválida",
            "bloqueia_aprovacao_automatica": True,
            "requer_revisao_humana": True,
            "origem_preenchimento": origem_preenchimento,
            "validado_em": datetime.now(UTC).isoformat(),
            "versao_tabela_cid": servico._versao_tabela_cid_diagnostico(None),
        }
        diag.setdefault("codigo_cid", codigo_norm or codigo_cid.strip().upper())
        diag.setdefault("valido", existe)
        if not diag.get("nome_doenca"):
            diag["nome_doenca"] = diag.get("descricao")
        return diag

    def listar_catalogo_cids(
        self, filtro: str | None = None, *, limite: int = 100
    ) -> list[dict[str, str]]:
        """Lista os códigos presentes na tabela local `afastamento.cid` (memória ou Postgres)."""
        erros_esperados: tuple[type[BaseException], ...] = (
            (OSError, ValueError, RuntimeError, psycopg.Error)
            if psycopg is not None
            else (OSError, ValueError, RuntimeError)
        )
        try:
            repo = self._obter_repositorio()
            return repo.listar_cids(filtro=filtro, limite=limite)
        except erros_esperados:
            return []

    def enviar_chat_a0(
        self,
        mensagem: str,
        *,
        thread_id: str,
        id_caso: int | None = None,
        arquivo_uri: str | None = None,
        dados_confirmacao: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Envia interação para o endpoint HTTP externo do Orquestrador A0 (`POST /chat`)."""
        base = self.url_a0_http.strip().rstrip("/")
        if not base and self._http_client is None:
            return {
                "ok": False,
                "erro": "URL da API do Agente 0 não configurada.",
            }
        destino = f"{base}/chat" if base else "/chat"
        headers: dict[str, str] = {}
        if self.token_jwt_a0.strip():
            headers["Authorization"] = f"Bearer {self.token_jwt_a0.strip()}"

        corpo: dict[str, Any] = {
            "mensagem": mensagem,
            "thread_id": thread_id,
        }
        if id_caso is not None:
            corpo["id_caso"] = id_caso
        if arquivo_uri:
            corpo["arquivo_uri"] = arquivo_uri
        if dados_confirmacao is not None:
            corpo["confirmacao"] = dados_confirmacao

        timeout = max(float(self.config.llm_timeout_segundos) + 10.0, 30.0)
        inicio = time.perf_counter()
        try:
            resp = _executar_post_http(
                self._http_client,
                destino,
                json_body=corpo,
                headers=headers or None,
                timeout=timeout,
            )
            latencia_ms = round((time.perf_counter() - inicio) * 1000, 2)
            try:
                conteudo = resp.json()
            except ValueError:
                conteudo = {"texto": resp.text}
            return {
                "ok": 200 <= resp.status_code < 300,
                "status_code": resp.status_code,
                "latencia_ms": latencia_ms,
                "resposta": conteudo,
            }
        except (httpx.HTTPError, ValueError, TypeError, OSError) as exc:
            return {
                "ok": False,
                "latencia_ms": round((time.perf_counter() - inicio) * 1000, 2),
                "erro": sanitizar_url_conexao(str(exc)),
            }

    def inspecionar_tabelas_afastamento(
        self,
        perfil_acesso: PerfilInspecao = "A1_RH",
    ) -> dict[str, list[dict[str, Any]]]:
        """Lê as 5 tabelas `afastamento.*` aplicando simulação de RLS LGPD Art. 11."""
        if self.modo_repo == "postgres":
            tabelas = self._inspecionar_postgres()
        else:
            tabelas = self._inspecionar_memoria()

        if perfil_acesso == "GESTOR_A2":
            # Simula a visão `afastamento.vw_atestado_sem_cid` e política RLS de `cid_restrito`
            atestados_sem_cid = []
            for row in tabelas["atestado"]:
                copia = dict(row)
                copia.pop("codigo_cid", None)
                copia.pop("nome_doenca", None)
                atestados_sem_cid.append(copia)
            tabelas["atestado"] = atestados_sem_cid
            tabelas["cid_restrito"] = []
        return tabelas

    def _inspecionar_memoria(self) -> dict[str, list[dict[str, Any]]]:
        repo = self.repositorio_memoria
        atestados = sorted(
            repo.atestados.values(), key=lambda x: x.id_atestado, reverse=True
        )
        arquivos = sorted(
            repo.arquivos_atestado.values(), key=lambda x: x.id_arquivo, reverse=True
        )
        campos = sorted(
            repo.campos_atestado.values(), key=lambda x: x.id_campo, reverse=True
        )
        baixas = sorted(
            repo.baixas_fisicas.values(), key=lambda x: x.id_baixa, reverse=True
        )
        cids = sorted(
            repo.cids_restritos.values(), key=lambda x: x.id_atestado, reverse=True
        )
        return {
            "atestado": [_serializar_modelo_ou_dict(a) for a in atestados],
            "atestado_arquivo": [_serializar_modelo_ou_dict(a) for a in arquivos],
            "atestado_campo": [_serializar_modelo_ou_dict(c) for c in campos],
            "baixa_fisica": [_serializar_modelo_ou_dict(b) for b in baixas],
            "cid_restrito": [_serializar_modelo_ou_dict(c) for c in cids],
        }

    def _inspecionar_postgres(self) -> dict[str, list[dict[str, Any]]]:
        vazio: dict[str, list[dict[str, Any]]] = {
            "atestado": [],
            "atestado_arquivo": [],
            "atestado_campo": [],
            "baixa_fisica": [],
            "cid_restrito": [],
        }
        if psycopg is None or not self.config.database_url.strip():
            return vazio

        consultas = {
            "atestado": "SELECT * FROM afastamento.atestado ORDER BY id_atestado DESC LIMIT 50",
            "atestado_arquivo": (
                "SELECT * FROM afastamento.atestado_arquivo ORDER BY id_arquivo DESC LIMIT 50"
            ),
            "atestado_campo": (
                "SELECT * FROM afastamento.atestado_campo ORDER BY id_campo DESC LIMIT 100"
            ),
            "baixa_fisica": (
                "SELECT * FROM afastamento.baixa_fisica ORDER BY id_baixa DESC LIMIT 50"
            ),
            "cid_restrito": (
                "SELECT * FROM afastamento.cid_restrito ORDER BY id_atestado DESC LIMIT 50"
            ),
        }
        resultado = dict(vazio)
        try:
            with (
                psycopg.connect(
                    self.config.database_url,
                    connect_timeout=2,
                    row_factory=dict_row,
                ) as conn,
                conn.cursor() as cur,
            ):
                for chave, sql in consultas.items():
                    try:
                        cur.execute(sql)
                        resultado[chave] = [dict(row) for row in cur.fetchall()]
                    except psycopg.Error:
                        conn.rollback()
                        resultado[chave] = []
        except (psycopg.Error, OSError):
            return vazio
        return resultado
