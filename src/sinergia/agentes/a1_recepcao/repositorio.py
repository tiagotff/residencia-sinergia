"""Repositório de dados do Agente 1 (A1 — Recepção e Processamento).

Implementa a Task 1B.2 do plano de desenvolvimento do A1 respeitando
estritamente a matriz de permissões do **Anexo C.2** da `Documentação - SinergIA v2.2.docx`
e o modelo físico em `db/schema/001_schema_inicial.sql`:

- **Leitura em `cadastro.*`:** `colaborador`, `cargo`, `setor`, `municipio`, `uf`, `usuario`.
- **Leitura em catálogos de `afastamento.*`:** `conselho_profissional`, `cid`.
- **Leitura e gravação em `afastamento.*`:** `profissional_saude` e `atestado`.
- **Proibido ao A1:** qualquer leitura ou gravação direta no schema de orquestração
  (o A0 recebe as decisões e entrega as mensagens pelo envelope `sinergia/1`).

Fornece duas implementações aderentes ao mesmo `Protocol`:
- `RepositorioA1Postgres`: queries SQL parametrizadas com `psycopg` (PostgreSQL 16).
- `RepositorioA1Memoria`: implementação determinística em memória para testes
  offline e simulação sem Docker, pré-carregada com o caso canônico do Anexo D.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sinergia.agentes.a1_recepcao.schemas import (
    OrigemDadosAtestado,
    StatusAtestadoDB,
    TipoDocumentoAfastamento,
)
from sinergia.agentes.a1_recepcao.validador import UFS_BRASIL

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - fallback caso psycopg não esteja instalado
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


_NAO_INFORMADO: Any = object()


def _normalizar_texto_busca(texto: str) -> str:
    sem_acentos = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(ch)
    )
    return " ".join(sem_acentos.strip().lower().split())


def _para_inicio_timestamptz(valor: date | datetime) -> datetime:
    if isinstance(valor, datetime):
        return valor if valor.tzinfo is not None else valor.replace(tzinfo=UTC)
    return datetime.combine(valor, time(0, 0, 0), tzinfo=UTC)


def _para_fim_timestamptz(valor: date | datetime) -> datetime:
    if isinstance(valor, datetime):
        return valor if valor.tzinfo is not None else valor.replace(tzinfo=UTC)
    return datetime.combine(valor, time(23, 59, 59), tzinfo=UTC)


def _para_data(valor: date | datetime) -> date:
    if isinstance(valor, datetime):
        return valor.date()
    return valor


def _parse_datetime_iso(valor: datetime | str) -> datetime:
    if isinstance(valor, datetime):
        return valor if valor.tzinfo is not None else valor.replace(tzinfo=UTC)
    dt = datetime.fromisoformat(valor)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


class DadosColaboradorCadastro(BaseModel):
    """Dados somente leitura do colaborador em `cadastro.*` (Anexo C.2)."""

    model_config = ConfigDict(extra="forbid")

    id_colaborador: int
    matricula: str
    nome: str
    id_cargo: int
    id_setor: int
    id_area: int
    id_municipio: int
    municipio_nome: str
    sigla_uf: str
    data_admissao: date
    ativo: bool
    cargo_nome: str | None = None
    setor_nome: str | None = None


class RegistroAtestadoDB(BaseModel):
    """Espelho tipado da tabela `afastamento.atestado` (`001` + `003` v0.7)."""

    model_config = ConfigDict(extra="forbid")

    id_atestado: int
    id_caso: int | None = None
    id_colaborador: int
    id_profissional: int
    codigo_cid: str | None = None
    nome_doenca: str | None = None
    id_municipio: int
    tipo_documento: TipoDocumentoAfastamento = "atestado_medico"
    data_emissao: date
    inicio_afastamento: datetime
    fim_afastamento: datetime
    arquivo_uri: str
    origem_dados: OrigemDadosAtestado
    status: StatusAtestadoDB
    postado_em: datetime
    prazo_entrega_fisica: date | None = None
    prazos_legais: list[str] = Field(default_factory=list)
    recebido_fisico_em: datetime | None = None
    recebido_fisico_por: int | None = None

    @field_validator("prazos_legais", mode="before")
    @classmethod
    def _normalizar_prazos_legais(cls, v: Any) -> list[str]:
        if v is None:
            return []
        return list(v)

    @model_validator(mode="after")
    def _validar_restricoes_check(self) -> Self:
        if self.fim_afastamento < self.inicio_afastamento:
            raise ValueError(
                "fim_afastamento não pode ser anterior a inicio_afastamento."
            )
        if (self.recebido_fisico_em is None) != (self.recebido_fisico_por is None):
            raise ValueError(
                "recebido_fisico_em e recebido_fisico_por devem ser preenchidos juntos."
            )
        return self


class RegistroArquivoAtestadoDB(BaseModel):
    """Espelho tipado de `afastamento.atestado_arquivo` (Guia A1 v0.7)."""

    model_config = ConfigDict(extra="forbid")

    id_arquivo: int
    id_atestado: int
    id_caso: int | None = None
    id_colaborador: int
    arquivo_uri: str
    mime: str | None = None
    sha256: str
    enviado_em: datetime


class RegistroCampoAtestadoDB(BaseModel):
    """Espelho tipado de `afastamento.atestado_campo` (Guia A1 v0.7)."""

    model_config = ConfigDict(extra="forbid")

    id_campo: int
    id_atestado: int
    id_caso: int | None = None
    nome_campo: str
    valor_extraido: str | None = None
    valor_confirmado: str | None = None
    confianca: float | None = Field(default=None, ge=0.0, le=1.0)
    origem: OrigemDadosAtestado = "IA"
    vazio_por_baixa_confianca: bool = False
    atualizado_em: datetime


class RegistroBaixaFisicaDB(BaseModel):
    """Espelho tipado de `afastamento.baixa_fisica` (Guia A1 v0.7)."""

    model_config = ConfigDict(extra="forbid")

    id_baixa: int
    id_atestado: int
    id_usuario_rh: int
    recebido_por: str
    data_hora: datetime
    registrado_em: datetime


class RegistroCidRestritoDB(BaseModel):
    """Espelho tipado de `afastamento.cid_restrito` (Guia A1 v0.7 / LGPD Art. 11)."""

    model_config = ConfigDict(extra="forbid")

    id_atestado: int
    codigo_cid: str
    nome_doenca: str | None = None
    registrado_em: datetime


@runtime_checkable
class RepositorioA1Protocol(Protocol):
    """Contrato único de persistência consumido pelo Agente 1."""

    def conferir_colaborador_ativo(
        self,
        id_colaborador: int,
        inicio: date | None = None,
        fim: date | None = None,
    ) -> DadosColaboradorCadastro | None: ...

    def buscar_id_municipio(
        self,
        nome_municipio: str | None,
        sigla_uf: str | None = None,
        id_colaborador_fallback: int | None = None,
    ) -> int | None: ...

    def validar_conselho_profissional(self, sigla: str, uf: str) -> int | None: ...

    def listar_conselhos_validos(self, uf: str) -> tuple[str, ...]: ...

    def obter_profissional(self, id_profissional: int) -> dict[str, Any] | None: ...

    def obter_municipio(self, id_municipio: int) -> tuple[str, str] | None: ...

    def validar_cid(
        self, codigo_cid: str | None, *, politica_extrair_cid: bool = True
    ) -> str | None: ...

    def obter_cid(self, codigo_cid: str | None) -> dict[str, str] | None: ...

    def salvar_cid(
        self,
        codigo_cid: str,
        descricao: str,
        capitulo: str,
        *,
        nome_doenca: str | None = None,
    ) -> str: ...

    def salvar_cids_em_lote(
        self,
        registros: Sequence[tuple[str, str, str] | tuple[str, str, str, str]],
    ) -> int: ...

    def listar_cids(
        self, filtro: str | None = None, *, limite: int = 200
    ) -> list[dict[str, str]]: ...

    def listar_atestados_recentes(
        self, *, limite: int = 50
    ) -> list[dict[str, Any]]: ...

    def resolver_id_usuario(
        self,
        identificador: int | str,
        *,
        perfis_permitidos: Collection[str] | None = None,
    ) -> int: ...

    def obter_ou_criar_profissional_saude(
        self,
        conselho: str,
        numero: str,
        uf: str,
        nome: str | None = None,
    ) -> int: ...

    def verificar_duplicidade_arquivo(
        self,
        arquivo_uri: str,
        id_colaborador: int | None = None,
        *,
        sha256: str | None = None,
    ) -> RegistroAtestadoDB | None: ...

    def listar_atestados_sobrepostos(
        self,
        id_colaborador: int,
        inicio: date,
        fim: date,
        ignorar_id_atestado: int | None = None,
    ) -> list[RegistroAtestadoDB]: ...

    def calcular_dias_afastamento_ultimos_60_dias(
        self,
        id_colaborador: int,
        data_referencia: date,
        cid: str | None = None,
    ) -> int: ...

    def salvar_atestado_pendente(
        self,
        *,
        id_colaborador: int,
        id_profissional: int,
        id_municipio: int,
        data_emissao: date,
        inicio_afastamento: date | datetime,
        fim_afastamento: date | datetime,
        arquivo_uri: str,
        codigo_cid: str | None = None,
        nome_doenca: str | None = None,
        origem_dados: OrigemDadosAtestado = "IA",
        arquivo_sha256: str | None = None,
        id_caso: int | None = None,
        tipo_documento: TipoDocumentoAfastamento = "atestado_medico",
        arquivo_mime: str | None = None,
        politica_cid_restrito: bool = True,
    ) -> RegistroAtestadoDB: ...

    def obter_atestado(self, id_atestado: int) -> RegistroAtestadoDB | None: ...

    def obter_ultimo_atestado_pendente(
        self, id_colaborador: int, *, id_caso: int | None = None
    ) -> RegistroAtestadoDB | None: ...

    def confirmar_atestado_no_banco(
        self,
        id_atestado: int,
        *,
        inicio_afastamento: date | datetime,
        fim_afastamento: date | datetime,
        data_emissao: date,
        origem_dados: OrigemDadosAtestado,
        prazo_entrega_fisica: date,
        id_profissional: int | None = _NAO_INFORMADO,
        codigo_cid: str | None = _NAO_INFORMADO,
        nome_doenca: str | None = _NAO_INFORMADO,
        id_municipio: int | None = _NAO_INFORMADO,
        tipo_documento: TipoDocumentoAfastamento | None = _NAO_INFORMADO,
        prazos_legais: Sequence[str] | None = _NAO_INFORMADO,
        politica_cid_restrito: bool = True,
    ) -> RegistroAtestadoDB: ...

    def registrar_baixa_fisica_no_banco(
        self,
        id_atestado: int,
        recebido_fisico_por: int | str,
        recebido_fisico_em: datetime | str,
        *,
        perfis_permitidos: Collection[str] | None = ("RH", "ADMIN"),
    ) -> RegistroAtestadoDB: ...

    def registrar_arquivo_atestado(
        self,
        id_atestado: int,
        *,
        id_colaborador: int,
        arquivo_uri: str,
        sha256: str,
        mime: str | None = None,
        id_caso: int | None = None,
    ) -> RegistroArquivoAtestadoDB: ...

    def listar_arquivos_atestado(
        self, id_atestado: int
    ) -> list[RegistroArquivoAtestadoDB]: ...

    def salvar_campos_atestado(
        self,
        id_atestado: int,
        campos: Sequence[dict[str, Any]],
        *,
        id_caso: int | None = None,
    ) -> list[RegistroCampoAtestadoDB]: ...

    def listar_campos_atestado(
        self, id_atestado: int
    ) -> list[RegistroCampoAtestadoDB]: ...

    def obter_baixa_fisica(
        self, id_atestado: int
    ) -> RegistroBaixaFisicaDB | None: ...

    def salvar_cid_restrito(
        self,
        id_atestado: int,
        codigo_cid: str | None,
        *,
        nome_doenca: str | None = None,
    ) -> RegistroCidRestritoDB | None: ...

    def obter_cid_restrito(
        self, id_atestado: int
    ) -> RegistroCidRestritoDB | None: ...


def _colaborador_ativo_no_periodo(
    ativo_cadastro: bool,
    data_admissao: date,
    inicio: date | None,
    fim: date | None,
) -> bool:
    if not ativo_cadastro:
        return False
    if fim is not None and fim < data_admissao:
        return False
    return not (inicio is not None and fim is None and inicio < data_admissao)


def _somar_dias_janela_60d(
    atestados: list[RegistroAtestadoDB],
    data_referencia: date,
    cid: str | None,
) -> int:
    inicio_janela = data_referencia - timedelta(days=60)
    cid_norm = cid.strip().upper() if cid else None
    dias_cobertos: set[date] = set()

    for at in atestados:
        if at.status != "CONFIRMADO":
            continue
        if cid_norm is not None and (at.codigo_cid or "").upper() != cid_norm:
            continue
        dt_ini = max(at.inicio_afastamento.date(), inicio_janela)
        dt_fim = min(at.fim_afastamento.date(), data_referencia)
        cursor = dt_ini
        while cursor <= dt_fim:
            dias_cobertos.add(cursor)
            cursor += timedelta(days=1)

    return len(dias_cobertos)


class _CursorDictAdapter:
    """Normaliza cursores DB-API/psycopg para sempre retornar linhas como `dict`."""

    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    def __enter__(self) -> Self:
        if hasattr(self._cursor, "__enter__"):
            self._cursor.__enter__()
        return self

    def __exit__(self, *args: object) -> Any:
        if hasattr(self._cursor, "__exit__"):
            return self._cursor.__exit__(*args)
        if hasattr(self._cursor, "close"):
            self._cursor.close()
        return None

    @property
    def description(self) -> Any:
        return getattr(self._cursor, "description", None)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._cursor.execute(*args, **kwargs)

    def _converter_linha(self, row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        if isinstance(row, dict):
            return row
        if isinstance(row, Mapping):
            return dict(row)
        desc = getattr(self._cursor, "description", None)
        if desc:
            colunas = [
                col.name if hasattr(col, "name") else col[0] for col in desc
            ]
            return dict(zip(colunas, row, strict=True))
        raise TypeError(
            f"Linha de cursor incompatível com mapeamento por nome de coluna: {type(row)!r}"
        )

    def fetchone(self) -> dict[str, Any] | None:
        return self._converter_linha(self._cursor.fetchone())

    def fetchall(self) -> list[dict[str, Any]]:
        linhas = self._cursor.fetchall()
        convertidas: list[dict[str, Any]] = []
        for linha in linhas:
            item = self._converter_linha(linha)
            if item is not None:
                convertidas.append(item)
        return convertidas

    def __getattr__(self, nome: str) -> Any:
        return getattr(self._cursor, nome)


def _extrair_raw_conn_sqlalchemy(sa_conn: Any) -> Any:
    """Obtém a conexão DB-API subjacente de uma Connection SQLAlchemy."""
    if callable(getattr(sa_conn, "cursor", None)):
        return sa_conn
    for attr in ("dbapi_connection", "connection"):
        candidato = getattr(sa_conn, attr, None)
        if candidato is not None and callable(getattr(candidato, "cursor", None)):
            return candidato
    raise TypeError(
        "Objeto SQLAlchemy sem conexão DB-API com método .cursor()."
    )


class _ConexaoAdaptada:
    """Adapta conexões psycopg/DB-API, SQLAlchemy Session/Connection ou Engine."""

    def __init__(self, alvo: Any) -> None:
        self._alvo = alvo
        self._raw_checked_out: Any | None = None
        self._modo_engine = False

    def __enter__(self) -> Self:
        if callable(getattr(self._alvo, "cursor", None)):
            return self
        if callable(getattr(self._alvo, "raw_connection", None)):
            self._raw_checked_out = self._alvo.raw_connection()
            self._modo_engine = True
            return self
        return self

    def __exit__(self, *args: object) -> None:
        if self._modo_engine and self._raw_checked_out is not None:
            if hasattr(self._raw_checked_out, "close"):
                self._raw_checked_out.close()
            self._raw_checked_out = None

    def cursor(self) -> _CursorDictAdapter:
        if callable(getattr(self._alvo, "cursor", None)):
            return _CursorDictAdapter(self._alvo.cursor())
        if self._modo_engine and self._raw_checked_out is not None:
            return _CursorDictAdapter(self._raw_checked_out.cursor())
        if callable(getattr(self._alvo, "connection", None)):
            sa_conn = self._alvo.connection()
            raw_conn = _extrair_raw_conn_sqlalchemy(sa_conn)
            return _CursorDictAdapter(raw_conn.cursor())
        if hasattr(self._alvo, "connection") or hasattr(self._alvo, "dbapi_connection"):
            raw_conn = _extrair_raw_conn_sqlalchemy(self._alvo)
            return _CursorDictAdapter(raw_conn.cursor())
        raise TypeError(
            "Conexão de banco incompatível com RepositorioA1Postgres: "
            f"{type(self._alvo)!r}."
        )

    def commit(self) -> None:
        if self._modo_engine and self._raw_checked_out is not None:
            if hasattr(self._raw_checked_out, "commit"):
                self._raw_checked_out.commit()
            return
        if callable(getattr(self._alvo, "commit", None)):
            self._alvo.commit()


class RepositorioA1Postgres:
    """Repositório PostgreSQL 16 estritamente restrito às tabelas do Anexo C.2."""

    def __init__(self, database_url: str = "", *, conn: Any | None = None) -> None:
        if conn is None and psycopg is None:
            raise RuntimeError(
                "Driver psycopg não encontrado. Instale com: pip install 'psycopg[binary]'"
            )
        self._database_url = database_url
        self._conn = conn
        self._sha256_por_chave: dict[tuple[int, str], int] = {}
        self._sha256_global: dict[str, int] = {}

    def _conectar(self) -> Any:
        if self._conn is not None:
            return _ConexaoAdaptada(self._conn)
        return psycopg.connect(self._database_url, row_factory=dict_row)

    def conferir_colaborador_ativo(
        self,
        id_colaborador: int,
        inicio: date | None = None,
        fim: date | None = None,
    ) -> DadosColaboradorCadastro | None:
        sql = """
            SELECT c.id_colaborador,
                   c.matricula,
                   c.nome,
                   c.id_cargo,
                   c.id_setor,
                   s.id_area,
                   c.id_municipio,
                   m.nome AS municipio_nome,
                   m.sigla_uf,
                   c.data_admissao,
                   c.ativo,
                   cg.nome AS cargo_nome,
                   s.nome AS setor_nome
            FROM cadastro.colaborador c
            JOIN cadastro.cargo cg     ON cg.id_cargo = c.id_cargo
            JOIN cadastro.setor s      ON s.id_setor = c.id_setor
            JOIN cadastro.municipio m  ON m.id_municipio = c.id_municipio
            WHERE c.id_colaborador = %s
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_colaborador,))
            row = cur.fetchone()
            if row is None:
                return None
            ativo_periodo = _colaborador_ativo_no_periodo(
                bool(row["ativo"]), row["data_admissao"], inicio, fim
            )
            return DadosColaboradorCadastro(**{**row, "ativo": ativo_periodo})

    def buscar_id_municipio(
        self,
        nome_municipio: str | None,
        sigla_uf: str | None = None,
        id_colaborador_fallback: int | None = None,
    ) -> int | None:
        with self._conectar() as conn, conn.cursor() as cur:
            if nome_municipio and nome_municipio.strip():
                nome_limpo = nome_municipio.strip()
                if sigla_uf and sigla_uf.strip():
                    cur.execute(
                        """
                        SELECT id_municipio
                        FROM cadastro.municipio
                        WHERE LOWER(TRIM(nome)) = LOWER(TRIM(%s))
                          AND sigla_uf = UPPER(TRIM(%s))
                        LIMIT 1
                        """,
                        (nome_limpo, sigla_uf),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id_municipio
                        FROM cadastro.municipio
                        WHERE LOWER(TRIM(nome)) = LOWER(TRIM(%s))
                        LIMIT 1
                        """,
                        (nome_limpo,),
                    )
                row = cur.fetchone()
                if row is not None:
                    return int(row["id_municipio"])

                alvo_norm = _normalizar_texto_busca(nome_limpo)
                if sigla_uf and sigla_uf.strip():
                    cur.execute(
                        """
                        SELECT id_municipio, nome
                        FROM cadastro.municipio
                        WHERE sigla_uf = UPPER(TRIM(%s))
                        """,
                        (sigla_uf,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id_municipio, nome
                        FROM cadastro.municipio
                        """
                    )
                for candidato in cur.fetchall():
                    if _normalizar_texto_busca(str(candidato["nome"])) == alvo_norm:
                        return int(candidato["id_municipio"])

            if id_colaborador_fallback is not None:
                cur.execute(
                    """
                    SELECT id_municipio
                    FROM cadastro.colaborador
                    WHERE id_colaborador = %s
                    """,
                    (id_colaborador_fallback,),
                )
                row = cur.fetchone()
                if row is not None:
                    return int(row["id_municipio"])
        return None

    def validar_conselho_profissional(self, sigla: str, uf: str) -> int | None:
        sigla_norm = sigla.strip().upper()
        uf_norm = uf.strip().upper()
        if not sigla_norm or not uf_norm:
            return None
        sql = """
            SELECT cp.id_conselho
            FROM afastamento.conselho_profissional cp
            JOIN cadastro.uf u ON u.sigla_uf = %s
            WHERE UPPER(cp.sigla) = %s
            LIMIT 1
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (uf_norm, sigla_norm))
            row = cur.fetchone()
            return int(row["id_conselho"]) if row is not None else None

    def listar_conselhos_validos(self, uf: str) -> tuple[str, ...]:
        uf_norm = uf.strip().upper()
        if not uf_norm:
            return ()
        sql = """
            SELECT DISTINCT UPPER(cp.sigla) AS sigla
            FROM afastamento.conselho_profissional cp
            JOIN cadastro.uf u ON u.sigla_uf = %s
            ORDER BY sigla
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (uf_norm,))
            rows = cur.fetchall()
            return tuple(str(r["sigla"]) for r in rows if r.get("sigla"))

    def obter_profissional(self, id_profissional: int) -> dict[str, Any] | None:
        sql = """
            SELECT cp.sigla AS conselho,
                   ps.numero_registro AS numero,
                   ps.sigla_uf AS uf,
                   ps.nome
            FROM afastamento.profissional_saude ps
            JOIN afastamento.conselho_profissional cp ON cp.id_conselho = ps.id_conselho
            WHERE ps.id_profissional = %s
            LIMIT 1
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_profissional,))
            row = cur.fetchone()
            return dict(row) if row is not None else None

    def obter_municipio(self, id_municipio: int) -> tuple[str, str] | None:
        sql = """
            SELECT nome, sigla_uf
            FROM cadastro.municipio
            WHERE id_municipio = %s
            LIMIT 1
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_municipio,))
            row = cur.fetchone()
            if row is None:
                return None
            return str(row["nome"]), str(row["sigla_uf"])

    def validar_cid(
        self, codigo_cid: str | None, *, politica_extrair_cid: bool = True
    ) -> str | None:
        if not politica_extrair_cid or not codigo_cid or not codigo_cid.strip():
            return None
        codigo_norm = codigo_cid.strip().upper()
        sql = """
            SELECT codigo_cid
            FROM afastamento.cid
            WHERE UPPER(codigo_cid) = %s
            LIMIT 1
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (codigo_norm,))
            row = cur.fetchone()
            return str(row["codigo_cid"]) if row is not None else None

    def _resolver_nome_doenca_cid(
        self,
        cur: Any,
        codigo_cid: str | None,
        nome_doenca: str | None = None,
    ) -> str | None:
        if not codigo_cid or not codigo_cid.strip():
            return None
        if nome_doenca is not None and nome_doenca.strip():
            return " ".join(nome_doenca.split()).strip()[:255]
        codigo_norm = codigo_cid.strip().upper()[:8]
        cur.execute(
            """
            SELECT COALESCE(NULLIF(btrim(nome_doenca), ''), descricao) AS nome_doenca
            FROM afastamento.cid
            WHERE UPPER(codigo_cid) = %s
            LIMIT 1
            """,
            (codigo_norm,),
        )
        row = cur.fetchone()
        if row is not None and row.get("nome_doenca"):
            return str(row["nome_doenca"])[:255]
        return None

    def obter_cid(self, codigo_cid: str | None) -> dict[str, str] | None:
        if not codigo_cid or not codigo_cid.strip():
            return None
        codigo_norm = codigo_cid.strip().upper()
        sql = """
            SELECT codigo_cid,
                   COALESCE(NULLIF(btrim(nome_doenca), ''), descricao) AS nome_doenca,
                   descricao,
                   capitulo
            FROM afastamento.cid
            WHERE UPPER(codigo_cid) = %s
            LIMIT 1
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (codigo_norm,))
            row = cur.fetchone()
            if row is None:
                return None
            nome = str(row.get("nome_doenca") or row["descricao"])
            return {
                "codigo_cid": str(row["codigo_cid"]),
                "nome_doenca": nome,
                "descricao": str(row["descricao"]),
                "capitulo": str(row["capitulo"]),
            }

    def salvar_cid(
        self,
        codigo_cid: str,
        descricao: str,
        capitulo: str,
        *,
        nome_doenca: str | None = None,
    ) -> str:
        codigo_norm = codigo_cid.strip().upper()[:8]
        if not codigo_norm:
            raise ValueError("codigo_cid é obrigatório.")
        desc_limpa = " ".join(descricao.split()).strip() or f"CID {codigo_norm}"
        nome_limpo = (
            " ".join((nome_doenca or desc_limpa).split()).strip() or desc_limpa
        )[:255]
        cap_limpo = (capitulo.strip() or codigo_norm[:1])[:10]
        sql = """
            INSERT INTO afastamento.cid (codigo_cid, nome_doenca, descricao, capitulo)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (codigo_cid) DO UPDATE
            SET nome_doenca = EXCLUDED.nome_doenca,
                descricao = EXCLUDED.descricao,
                capitulo = EXCLUDED.capitulo
            RETURNING codigo_cid
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (codigo_norm, nome_limpo, desc_limpa, cap_limpo))
            row = cur.fetchone()
            conn.commit()
            return str(row["codigo_cid"]) if row is not None else codigo_norm

    def salvar_cids_em_lote(
        self,
        registros: Sequence[tuple[str, str, str] | tuple[str, str, str, str]],
    ) -> int:
        if not registros:
            return 0
        normalizados = []
        for item in registros:
            if len(item) == 4:
                codigo, nome_doenca, descricao, capitulo = item
            else:
                codigo, descricao, capitulo = item
                nome_doenca = descricao
            codigo_norm = codigo.strip().upper()[:8]
            if not codigo_norm:
                continue
            desc_limpa = " ".join(descricao.split()).strip() or f"CID {codigo_norm}"
            nome_limpo = (
                " ".join((nome_doenca or desc_limpa).split()).strip() or desc_limpa
            )[:255]
            cap_limpo = (capitulo.strip() or codigo_norm[:1])[:10]
            normalizados.append((codigo_norm, nome_limpo, desc_limpa, cap_limpo))
        if not normalizados:
            return 0

        sql = """
            INSERT INTO afastamento.cid (codigo_cid, nome_doenca, descricao, capitulo)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (codigo_cid) DO UPDATE
            SET nome_doenca = EXCLUDED.nome_doenca,
                descricao = EXCLUDED.descricao,
                capitulo = EXCLUDED.capitulo
        """
        with self._conectar() as conn, conn.cursor() as cur:
            for idx in range(0, len(normalizados), 500):
                cur.executemany(sql, normalizados[idx : idx + 500])
            conn.commit()
        return len(normalizados)

    def listar_cids(
        self, filtro: str | None = None, *, limite: int = 200
    ) -> list[dict[str, str]]:
        limite_seguro = max(1, min(int(limite), 5000))
        with self._conectar() as conn, conn.cursor() as cur:
            if filtro and filtro.strip():
                termo = f"%{filtro.strip()}%"
                cur.execute(
                    """
                    SELECT codigo_cid,
                           COALESCE(NULLIF(btrim(nome_doenca), ''), descricao) AS nome_doenca,
                           descricao,
                           capitulo
                    FROM afastamento.cid
                    WHERE codigo_cid ILIKE %s
                       OR nome_doenca ILIKE %s
                       OR descricao ILIKE %s
                    ORDER BY codigo_cid ASC
                    LIMIT %s
                    """,
                    (termo, termo, termo, limite_seguro),
                )
            else:
                cur.execute(
                    """
                    SELECT codigo_cid,
                           COALESCE(NULLIF(btrim(nome_doenca), ''), descricao) AS nome_doenca,
                           descricao,
                           capitulo
                    FROM afastamento.cid
                    ORDER BY codigo_cid ASC
                    LIMIT %s
                    """,
                    (limite_seguro,),
                )
            return [
                {
                    "codigo_cid": str(r["codigo_cid"]),
                    "nome_doenca": str(r.get("nome_doenca") or r["descricao"]),
                    "descricao": str(r["descricao"]),
                    "capitulo": str(r["capitulo"]),
                }
                for r in cur.fetchall()
            ]

    def listar_atestados_recentes(
        self, *, limite: int = 50
    ) -> list[dict[str, Any]]:
        limite_seguro = max(1, min(int(limite), 500))
        sql = """
            SELECT a.id_atestado,
                   a.id_caso,
                   a.id_colaborador,
                   c.nome AS colaborador_nome,
                   a.tipo_documento,
                   a.status,
                   a.origem_dados,
                   a.data_emissao,
                   (a.inicio_afastamento AT TIME ZONE 'UTC')::date AS inicio,
                   (a.fim_afastamento AT TIME ZONE 'UTC')::date AS fim,
                   a.prazo_entrega_fisica,
                   a.prazos_legais,
                   cp.sigla AS conselho,
                   ps.numero_registro AS registro_profissional,
                   ps.sigla_uf AS uf_profissional,
                   COALESCE(a.codigo_cid, cr.codigo_cid) AS codigo_cid_restrito,
                   COALESCE(
                       NULLIF(btrim(a.nome_doenca), ''),
                       NULLIF(btrim(cr.nome_doenca), ''),
                       NULLIF(btrim(cid.nome_doenca), ''),
                       cid.descricao
                   ) AS nome_doenca,
                   cid.descricao AS cid_descricao,
                   a.postado_em
            FROM afastamento.atestado a
            JOIN cadastro.colaborador c ON c.id_colaborador = a.id_colaborador
            JOIN afastamento.profissional_saude ps ON ps.id_profissional = a.id_profissional
            JOIN afastamento.conselho_profissional cp ON cp.id_conselho = ps.id_conselho
            LEFT JOIN afastamento.cid_restrito cr ON cr.id_atestado = a.id_atestado
            LEFT JOIN afastamento.cid cid ON cid.codigo_cid = COALESCE(a.codigo_cid, cr.codigo_cid)
            ORDER BY a.postado_em DESC, a.id_atestado DESC
            LIMIT %s
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (limite_seguro,))
            return [dict(r) for r in cur.fetchall()]

    def _validar_perfil_usuario(
        self,
        row: dict[str, Any],
        identificador: int | str,
        perfis_permitidos: Collection[str] | None,
    ) -> None:
        if perfis_permitidos is None:
            return
        perfil_row = row.get("perfil")
        if perfil_row is None:
            return
        permitidos = {p.strip().upper() for p in perfis_permitidos if p.strip()}
        if str(perfil_row).strip().upper() not in permitidos:
            raise PermissionError(
                f"Usuário {identificador!r} (perfil={perfil_row}) não autorizado para esta operação."
            )

    def resolver_id_usuario(
        self,
        identificador: int | str,
        *,
        perfis_permitidos: Collection[str] | None = None,
    ) -> int:
        with self._conectar() as conn, conn.cursor() as cur:
            if isinstance(identificador, int) and not isinstance(identificador, bool):
                cur.execute(
                    """
                    SELECT id_usuario, perfil
                    FROM cadastro.usuario
                    WHERE id_usuario = %s AND ativo = TRUE
                    """,
                    (identificador,),
                )
                row = cur.fetchone()
                if row is not None:
                    self._validar_perfil_usuario(row, identificador, perfis_permitidos)
                    return int(row["id_usuario"])
                raise ValueError(f"Usuário não encontrado ou inativo: {identificador!r}")

            texto = str(identificador).strip()
            if not texto:
                raise ValueError("Identificador de usuário vazio.")

            if texto.isdigit():
                cur.execute(
                    """
                    SELECT id_usuario, perfil
                    FROM cadastro.usuario
                    WHERE id_usuario = %s AND ativo = TRUE
                    """,
                    (int(texto),),
                )
                row = cur.fetchone()
                if row is not None:
                    self._validar_perfil_usuario(row, identificador, perfis_permitidos)
                    return int(row["id_usuario"])

            cur.execute(
                """
                SELECT id_usuario, perfil
                FROM cadastro.usuario
                WHERE LOWER(TRIM(login)) = LOWER(TRIM(%s)) AND ativo = TRUE
                LIMIT 1
                """,
                (texto,),
            )
            row = cur.fetchone()
            if row is not None:
                self._validar_perfil_usuario(row, identificador, perfis_permitidos)
                return int(row["id_usuario"])

        raise ValueError(f"Usuário não encontrado ou inativo: {identificador!r}")

    def obter_ou_criar_profissional_saude(
        self,
        conselho: str,
        numero: str,
        uf: str,
        nome: str | None = None,
    ) -> int:
        numero_limpo = numero.strip()
        uf_norm = uf.strip().upper()
        if not numero_limpo:
            raise ValueError("Número de registro profissional é obrigatório.")

        id_conselho = self.validar_conselho_profissional(conselho, uf_norm)
        if id_conselho is None:
            raise ValueError(
                f"Conselho ou UF não reconhecido no catálogo: {conselho!r}/{uf_norm!r}"
            )

        nome_limpo = nome.strip() if nome and nome.strip() else None
        sql = """
            INSERT INTO afastamento.profissional_saude (
                id_conselho, numero_registro, sigla_uf, nome
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id_conselho, numero_registro, sigla_uf)
            DO UPDATE SET nome = COALESCE(afastamento.profissional_saude.nome, EXCLUDED.nome)
            RETURNING id_profissional
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_conselho, numero_limpo, uf_norm, nome_limpo))
            row = cur.fetchone()
            conn.commit()
            return int(row["id_profissional"])

    def verificar_duplicidade_arquivo(
        self,
        arquivo_uri: str,
        id_colaborador: int | None = None,
        *,
        sha256: str | None = None,
    ) -> RegistroAtestadoDB | None:
        if sha256 and sha256.strip():
            sha_norm = sha256.strip().lower()
            with self._conectar() as conn, conn.cursor() as cur:
                if id_colaborador is not None:
                    cur.execute(
                        """
                        SELECT a.*
                        FROM afastamento.atestado_arquivo aa
                        JOIN afastamento.atestado a ON a.id_atestado = aa.id_atestado
                        WHERE aa.sha256 = %s
                          AND aa.id_colaborador = %s
                          AND a.status <> 'REJEITADO'
                        ORDER BY aa.enviado_em DESC, aa.id_arquivo DESC
                        LIMIT 1
                        """,
                        (sha_norm, id_colaborador),
                    )
                else:
                    cur.execute(
                        """
                        SELECT a.*
                        FROM afastamento.atestado_arquivo aa
                        JOIN afastamento.atestado a ON a.id_atestado = aa.id_atestado
                        WHERE aa.sha256 = %s
                          AND a.status <> 'REJEITADO'
                        ORDER BY aa.enviado_em DESC, aa.id_arquivo DESC
                        LIMIT 1
                        """,
                        (sha_norm,),
                    )
                row_sha = cur.fetchone()
                if row_sha is not None:
                    return RegistroAtestadoDB(**row_sha)

            id_at = (
                self._sha256_por_chave.get((id_colaborador, sha_norm))
                if id_colaborador is not None
                else self._sha256_global.get(sha_norm)
            )
            if id_at is not None:
                reg = self.obter_atestado(id_at)
                if reg is not None and reg.status != "REJEITADO":
                    return reg

        uri_limpa = arquivo_uri.strip()
        if not uri_limpa:
            return None
        with self._conectar() as conn, conn.cursor() as cur:
            if id_colaborador is not None:
                cur.execute(
                    """
                    SELECT *
                    FROM afastamento.atestado
                    WHERE arquivo_uri = %s
                      AND id_colaborador = %s
                      AND status <> 'REJEITADO'
                    ORDER BY postado_em DESC
                    LIMIT 1
                    """,
                    (uri_limpa, id_colaborador),
                )
            else:
                cur.execute(
                    """
                    SELECT *
                    FROM afastamento.atestado
                    WHERE arquivo_uri = %s
                      AND status <> 'REJEITADO'
                    ORDER BY postado_em DESC
                    LIMIT 1
                    """,
                    (uri_limpa,),
                )
            row = cur.fetchone()
            return RegistroAtestadoDB(**row) if row is not None else None

    def listar_atestados_sobrepostos(
        self,
        id_colaborador: int,
        inicio: date,
        fim: date,
        ignorar_id_atestado: int | None = None,
    ) -> list[RegistroAtestadoDB]:
        sql = """
            SELECT *
            FROM afastamento.atestado
            WHERE id_colaborador = %s
              AND status <> 'REJEITADO'
              AND (inicio_afastamento AT TIME ZONE 'UTC')::date <= %s
              AND (fim_afastamento AT TIME ZONE 'UTC')::date >= %s
              AND (%s::bigint IS NULL OR id_atestado <> %s)
            ORDER BY inicio_afastamento ASC
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    id_colaborador,
                    fim,
                    inicio,
                    ignorar_id_atestado,
                    ignorar_id_atestado,
                ),
            )
            return [RegistroAtestadoDB(**row) for row in cur.fetchall()]

    def calcular_dias_afastamento_ultimos_60_dias(
        self,
        id_colaborador: int,
        data_referencia: date,
        cid: str | None = None,
    ) -> int:
        inicio_janela = data_referencia - timedelta(days=60)
        sql = """
            SELECT a.*, COALESCE(cr.codigo_cid, a.codigo_cid) AS codigo_cid_efetivo
            FROM afastamento.atestado a
            LEFT JOIN afastamento.cid_restrito cr ON cr.id_atestado = a.id_atestado
            WHERE a.id_colaborador = %s
              AND a.status = 'CONFIRMADO'
              AND (a.fim_afastamento AT TIME ZONE 'UTC')::date >= %s
              AND (a.inicio_afastamento AT TIME ZONE 'UTC')::date <= %s
        """
        atestados: list[RegistroAtestadoDB] = []
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_colaborador, inicio_janela, data_referencia))
            for row in cur.fetchall():
                dados = dict(row)
                cid_efetivo = dados.pop("codigo_cid_efetivo", dados.get("codigo_cid"))
                dados["codigo_cid"] = cid_efetivo
                atestados.append(RegistroAtestadoDB(**dados))
        return _somar_dias_janela_60d(atestados, data_referencia, cid)

    def salvar_atestado_pendente(
        self,
        *,
        id_colaborador: int,
        id_profissional: int,
        id_municipio: int,
        data_emissao: date,
        inicio_afastamento: date | datetime,
        fim_afastamento: date | datetime,
        arquivo_uri: str,
        codigo_cid: str | None = None,
        nome_doenca: str | None = None,
        origem_dados: OrigemDadosAtestado = "IA",
        arquivo_sha256: str | None = None,
        id_caso: int | None = None,
        tipo_documento: TipoDocumentoAfastamento = "atestado_medico",
        arquivo_mime: str | None = None,
        politica_cid_restrito: bool = True,
    ) -> RegistroAtestadoDB:
        uri_limpa = arquivo_uri.strip()
        if not uri_limpa:
            raise ValueError("arquivo_uri é obrigatório.")
        dt_inicio = _para_inicio_timestamptz(inicio_afastamento)
        dt_fim = _para_fim_timestamptz(fim_afastamento)
        if dt_fim < dt_inicio or _para_data(dt_fim) < _para_data(dt_inicio):
            raise ValueError("fim_afastamento não pode ser anterior a inicio_afastamento.")

        cid_norm = codigo_cid.strip().upper()[:8] if codigo_cid and codigo_cid.strip() else None
        sql = """
            INSERT INTO afastamento.atestado (
                id_caso,
                id_colaborador,
                id_profissional,
                codigo_cid,
                nome_doenca,
                id_municipio,
                tipo_documento,
                data_emissao,
                inicio_afastamento,
                fim_afastamento,
                arquivo_uri,
                origem_dados,
                status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDENTE')
            RETURNING *
        """
        with self._conectar() as conn, conn.cursor() as cur:
            nome_doenca_final = (
                self._resolver_nome_doenca_cid(cur, cid_norm, nome_doenca)
                if cid_norm is not None
                else None
            )
            cur.execute(
                sql,
                (
                    id_caso,
                    id_colaborador,
                    id_profissional,
                    cid_norm,
                    nome_doenca_final,
                    id_municipio,
                    tipo_documento,
                    data_emissao,
                    dt_inicio,
                    dt_fim,
                    uri_limpa,
                    origem_dados,
                ),
            )
            row = cur.fetchone()
            registro = RegistroAtestadoDB(**row)

            if politica_cid_restrito and cid_norm is not None:
                cur.execute(
                    """
                    INSERT INTO afastamento.cid_restrito (id_atestado, codigo_cid, nome_doenca)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id_atestado)
                    DO UPDATE SET codigo_cid = EXCLUDED.codigo_cid,
                                  nome_doenca = EXCLUDED.nome_doenca,
                                  registrado_em = now()
                    """,
                    (registro.id_atestado, cid_norm, nome_doenca_final),
                )

            if arquivo_sha256 and arquivo_sha256.strip():
                sha_norm = arquivo_sha256.strip().lower()
                cur.execute(
                    """
                    INSERT INTO afastamento.atestado_arquivo (
                        id_atestado, id_caso, id_colaborador, arquivo_uri, mime, sha256
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        registro.id_atestado,
                        id_caso,
                        id_colaborador,
                        uri_limpa,
                        arquivo_mime,
                        sha_norm,
                    ),
                )
                self._sha256_por_chave[(id_colaborador, sha_norm)] = registro.id_atestado
                self._sha256_global[sha_norm] = registro.id_atestado

            conn.commit()
            return registro

    def obter_atestado(self, id_atestado: int) -> RegistroAtestadoDB | None:
        sql = "SELECT * FROM afastamento.atestado WHERE id_atestado = %s"
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_atestado,))
            row = cur.fetchone()
            return RegistroAtestadoDB(**row) if row is not None else None

    def obter_ultimo_atestado_pendente(
        self, id_colaborador: int, *, id_caso: int | None = None
    ) -> RegistroAtestadoDB | None:
        with self._conectar() as conn, conn.cursor() as cur:
            if id_caso is not None:
                cur.execute(
                    """
                    SELECT *
                    FROM afastamento.atestado
                    WHERE id_colaborador = %s
                      AND id_caso = %s
                      AND status = 'PENDENTE'
                    ORDER BY postado_em DESC, id_atestado DESC
                    LIMIT 1
                    """,
                    (id_colaborador, id_caso),
                )
                row = cur.fetchone()
                if row is not None:
                    return RegistroAtestadoDB(**row)

                cur.execute(
                    """
                    SELECT *
                    FROM afastamento.atestado
                    WHERE id_colaborador = %s
                      AND id_caso IS NULL
                      AND status = 'PENDENTE'
                    ORDER BY postado_em DESC, id_atestado DESC
                    LIMIT 1
                    """,
                    (id_colaborador,),
                )
                row_legado = cur.fetchone()
                return (
                    RegistroAtestadoDB(**row_legado)
                    if row_legado is not None
                    else None
                )

            cur.execute(
                """
                SELECT *
                FROM afastamento.atestado
                WHERE id_colaborador = %s AND status = 'PENDENTE'
                ORDER BY postado_em DESC, id_atestado DESC
                LIMIT 1
                """,
                (id_colaborador,),
            )
            row = cur.fetchone()
            return RegistroAtestadoDB(**row) if row is not None else None

    def confirmar_atestado_no_banco(
        self,
        id_atestado: int,
        *,
        inicio_afastamento: date | datetime,
        fim_afastamento: date | datetime,
        data_emissao: date,
        origem_dados: OrigemDadosAtestado,
        prazo_entrega_fisica: date,
        id_profissional: int | None = _NAO_INFORMADO,
        codigo_cid: str | None = _NAO_INFORMADO,
        nome_doenca: str | None = _NAO_INFORMADO,
        id_municipio: int | None = _NAO_INFORMADO,
        tipo_documento: TipoDocumentoAfastamento | None = _NAO_INFORMADO,
        prazos_legais: Sequence[str] | None = _NAO_INFORMADO,
        politica_cid_restrito: bool = True,
    ) -> RegistroAtestadoDB:
        dt_inicio = _para_inicio_timestamptz(inicio_afastamento)
        dt_fim = _para_fim_timestamptz(fim_afastamento)
        if dt_fim < dt_inicio or _para_data(dt_fim) < _para_data(dt_inicio):
            raise ValueError("fim_afastamento não pode ser anterior a inicio_afastamento.")

        set_clauses = [
            "inicio_afastamento = %s",
            "fim_afastamento = %s",
            "data_emissao = %s",
            "origem_dados = %s",
            "prazo_entrega_fisica = %s",
            "status = 'CONFIRMADO'",
        ]
        params: list[Any] = [
            dt_inicio,
            dt_fim,
            data_emissao,
            origem_dados,
            prazo_entrega_fisica,
        ]
        if id_profissional is not _NAO_INFORMADO and id_profissional is not None:
            set_clauses.append("id_profissional = %s")
            params.append(id_profissional)
        with self._conectar() as conn, conn.cursor() as cur:
            cid_norm_confirmado: str | None = None
            nome_doenca_final: str | None = None
            if codigo_cid is not _NAO_INFORMADO:
                cid_norm_confirmado = (
                    codigo_cid.strip().upper()[:8]
                    if codigo_cid and str(codigo_cid).strip()
                    else None
                )
                if cid_norm_confirmado is not None:
                    nome_arg = None if nome_doenca is _NAO_INFORMADO else nome_doenca
                    nome_doenca_final = self._resolver_nome_doenca_cid(
                        cur, cid_norm_confirmado, nome_arg
                    )
                set_clauses.append("codigo_cid = %s")
                params.append(cid_norm_confirmado)
                set_clauses.append("nome_doenca = %s")
                params.append(nome_doenca_final)
            elif nome_doenca is not _NAO_INFORMADO:
                set_clauses.append("nome_doenca = %s")
                params.append(
                    " ".join(str(nome_doenca).split()).strip()[:255]
                    if nome_doenca is not None and str(nome_doenca).strip()
                    else None
                )
            if id_municipio is not _NAO_INFORMADO and id_municipio is not None:
                set_clauses.append("id_municipio = %s")
                params.append(id_municipio)
            if tipo_documento is not _NAO_INFORMADO and tipo_documento is not None:
                set_clauses.append("tipo_documento = %s")
                params.append(tipo_documento)
            if prazos_legais is not _NAO_INFORMADO and prazos_legais is not None:
                set_clauses.append("prazos_legais = %s")
                params.append(list(prazos_legais))

            params.append(id_atestado)
            sql = f"""
                UPDATE afastamento.atestado
                SET {', '.join(set_clauses)}
                WHERE id_atestado = %s
                RETURNING *
            """
            cur.execute(sql, tuple(params))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Atestado {id_atestado} não encontrado para confirmação.")

            if codigo_cid is not _NAO_INFORMADO and politica_cid_restrito:
                if cid_norm_confirmado is not None:
                    cur.execute(
                        """
                        INSERT INTO afastamento.cid_restrito (id_atestado, codigo_cid, nome_doenca)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (id_atestado)
                        DO UPDATE SET codigo_cid = EXCLUDED.codigo_cid,
                                      nome_doenca = EXCLUDED.nome_doenca,
                                      registrado_em = now()
                        """,
                        (id_atestado, cid_norm_confirmado, nome_doenca_final),
                    )
                else:
                    cur.execute(
                        "DELETE FROM afastamento.cid_restrito WHERE id_atestado = %s",
                        (id_atestado,),
                    )

            conn.commit()
            return RegistroAtestadoDB(**row)

    def registrar_baixa_fisica_no_banco(
        self,
        id_atestado: int,
        recebido_fisico_por: int | str,
        recebido_fisico_em: datetime | str,
        *,
        perfis_permitidos: Collection[str] | None = ("RH", "ADMIN"),
    ) -> RegistroAtestadoDB:
        id_usuario_rh = self.resolver_id_usuario(
            recebido_fisico_por, perfis_permitidos=perfis_permitidos
        )
        dt_recebimento = _parse_datetime_iso(recebido_fisico_em)
        recebido_por_texto = str(recebido_fisico_por).strip() or str(id_usuario_rh)

        sql = """
            UPDATE afastamento.atestado
            SET recebido_fisico_em = %s,
                recebido_fisico_por = %s
            WHERE id_atestado = %s
              AND status = 'CONFIRMADO'
            RETURNING *
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (dt_recebimento, id_usuario_rh, id_atestado))
            row = cur.fetchone()
            if row is None:
                raise ValueError(
                    f"Atestado {id_atestado} não encontrado ou ainda não confirmado."
                )
            cur.execute(
                """
                INSERT INTO afastamento.baixa_fisica (
                    id_atestado, id_usuario_rh, recebido_por, data_hora
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id_atestado)
                DO UPDATE SET id_usuario_rh = EXCLUDED.id_usuario_rh,
                              recebido_por = EXCLUDED.recebido_por,
                              data_hora = EXCLUDED.data_hora,
                              registrado_em = now()
                """,
                (id_atestado, id_usuario_rh, recebido_por_texto, dt_recebimento),
            )
            conn.commit()
            return RegistroAtestadoDB(**row)

    def registrar_arquivo_atestado(
        self,
        id_atestado: int,
        *,
        id_colaborador: int,
        arquivo_uri: str,
        sha256: str,
        mime: str | None = None,
        id_caso: int | None = None,
    ) -> RegistroArquivoAtestadoDB:
        sha_norm = sha256.strip().lower()
        uri_limpa = arquivo_uri.strip()
        sql = """
            INSERT INTO afastamento.atestado_arquivo (
                id_atestado, id_caso, id_colaborador, arquivo_uri, mime, sha256
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(
                sql, (id_atestado, id_caso, id_colaborador, uri_limpa, mime, sha_norm)
            )
            row = cur.fetchone()
            conn.commit()
            self._sha256_por_chave[(id_colaborador, sha_norm)] = id_atestado
            self._sha256_global[sha_norm] = id_atestado
            return RegistroArquivoAtestadoDB(**row)

    def listar_arquivos_atestado(
        self, id_atestado: int
    ) -> list[RegistroArquivoAtestadoDB]:
        sql = """
            SELECT *
            FROM afastamento.atestado_arquivo
            WHERE id_atestado = %s
            ORDER BY enviado_em ASC, id_arquivo ASC
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_atestado,))
            return [RegistroArquivoAtestadoDB(**row) for row in cur.fetchall()]

    def salvar_campos_atestado(
        self,
        id_atestado: int,
        campos: Sequence[dict[str, Any]],
        *,
        id_caso: int | None = None,
    ) -> list[RegistroCampoAtestadoDB]:
        if not campos:
            return []
        sql = """
            INSERT INTO afastamento.atestado_campo (
                id_atestado,
                id_caso,
                nome_campo,
                valor_extraido,
                valor_confirmado,
                confianca,
                origem,
                vazio_por_baixa_confianca
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id_atestado, nome_campo)
            DO UPDATE SET
                id_caso = COALESCE(EXCLUDED.id_caso, afastamento.atestado_campo.id_caso),
                valor_extraido = COALESCE(
                    EXCLUDED.valor_extraido, afastamento.atestado_campo.valor_extraido
                ),
                valor_confirmado = COALESCE(
                    EXCLUDED.valor_confirmado, afastamento.atestado_campo.valor_confirmado
                ),
                confianca = COALESCE(
                    EXCLUDED.confianca, afastamento.atestado_campo.confianca
                ),
                origem = EXCLUDED.origem,
                vazio_por_baixa_confianca = EXCLUDED.vazio_por_baixa_confianca,
                atualizado_em = now()
            RETURNING *
        """
        salvos: list[RegistroCampoAtestadoDB] = []
        with self._conectar() as conn, conn.cursor() as cur:
            for item in campos:
                cur.execute(
                    sql,
                    (
                        id_atestado,
                        item.get("id_caso", id_caso),
                        str(item["nome_campo"]),
                        item.get("valor_extraido"),
                        item.get("valor_confirmado"),
                        item.get("confianca"),
                        item.get("origem", "IA"),
                        bool(item.get("vazio_por_baixa_confianca", False)),
                    ),
                )
                row = cur.fetchone()
                if row is not None:
                    salvos.append(RegistroCampoAtestadoDB(**row))
            conn.commit()
        return salvos

    def listar_campos_atestado(
        self, id_atestado: int
    ) -> list[RegistroCampoAtestadoDB]:
        sql = """
            SELECT *
            FROM afastamento.atestado_campo
            WHERE id_atestado = %s
            ORDER BY id_campo ASC
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_atestado,))
            return [RegistroCampoAtestadoDB(**row) for row in cur.fetchall()]

    def obter_baixa_fisica(self, id_atestado: int) -> RegistroBaixaFisicaDB | None:
        sql = "SELECT * FROM afastamento.baixa_fisica WHERE id_atestado = %s LIMIT 1"
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_atestado,))
            row = cur.fetchone()
            return RegistroBaixaFisicaDB(**row) if row is not None else None

    def salvar_cid_restrito(
        self,
        id_atestado: int,
        codigo_cid: str | None,
        *,
        nome_doenca: str | None = None,
    ) -> RegistroCidRestritoDB | None:
        with self._conectar() as conn, conn.cursor() as cur:
            if codigo_cid is None or not str(codigo_cid).strip():
                cur.execute(
                    "DELETE FROM afastamento.cid_restrito WHERE id_atestado = %s",
                    (id_atestado,),
                )
                conn.commit()
                return None
            cid_norm = str(codigo_cid).strip().upper()[:8]
            nome_doenca_final = self._resolver_nome_doenca_cid(
                cur, cid_norm, nome_doenca
            )
            cur.execute(
                """
                INSERT INTO afastamento.cid_restrito (id_atestado, codigo_cid, nome_doenca)
                VALUES (%s, %s, %s)
                ON CONFLICT (id_atestado)
                DO UPDATE SET codigo_cid = EXCLUDED.codigo_cid,
                              nome_doenca = EXCLUDED.nome_doenca,
                              registrado_em = now()
                RETURNING *
                """,
                (id_atestado, cid_norm, nome_doenca_final),
            )
            row = cur.fetchone()
            conn.commit()
            return RegistroCidRestritoDB(**row) if row is not None else None

    def obter_cid_restrito(self, id_atestado: int) -> RegistroCidRestritoDB | None:
        sql = "SELECT * FROM afastamento.cid_restrito WHERE id_atestado = %s LIMIT 1"
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_atestado,))
            row = cur.fetchone()
            return RegistroCidRestritoDB(**row) if row is not None else None


class RepositorioA1Memoria:
    """Repositório em memória para testes unitários, contrato e simulação offline.

    Espelha as mesmas restrições de integridade do `001_schema_inicial.sql` e
    vem pré-populado com os dados do caso canônico do Anexo D (Carla Souza /
    Campinas-SP / usuária RH ``rh.ana`` / conselhos CRM, CRO, CRP).
    """

    def __init__(self, *, popular_padrao: bool = True) -> None:
        self.ufs: set[str] = set(UFS_BRASIL)
        self.municipios: dict[int, tuple[str, str]] = {}
        self.colaboradores: dict[int, DadosColaboradorCadastro] = {}
        self.usuarios_por_id: dict[int, dict[str, Any]] = {}
        self.usuarios_por_login: dict[str, int] = {}
        self.conselhos: dict[str, int] = {}
        self.cids: dict[str, str] = {}
        self.cids_detalhes: dict[str, dict[str, str]] = {}
        self.profissionais: dict[int, dict[str, Any]] = {}
        self._chave_profissional: dict[tuple[int, str, str], int] = {}
        self.atestados: dict[int, RegistroAtestadoDB] = {}
        self.arquivos_atestado: dict[int, RegistroArquivoAtestadoDB] = {}
        self.campos_atestado: dict[tuple[int, str], RegistroCampoAtestadoDB] = {}
        self.baixas_fisicas: dict[int, RegistroBaixaFisicaDB] = {}
        self.cids_restritos: dict[int, RegistroCidRestritoDB] = {}
        self._sha256_por_chave: dict[tuple[int, str], int] = {}
        self._sha256_global: dict[str, int] = {}
        self._seq_profissional = 1
        self._seq_atestado = 311
        self._seq_arquivo = 1
        self._seq_campo = 1
        self._seq_baixa = 1

        if popular_padrao:
            self._popular_dados_canonicos()

    def _popular_dados_canonicos(self) -> None:
        self.municipios = {
            3509502: ("Campinas", "SP"),
            3550308: ("São Paulo", "SP"),
            3304557: ("Rio de Janeiro", "RJ"),
        }
        self.conselhos = {"CRM": 1, "CRO": 2, "CRP": 3}
        self.cids = {
            "M54.5": "Dor localizada na região lombar inferior (lombalgia não especificada).",
            "F41.1": "Transtorno caracterizado por ansiedade e preocupação persistentes e excessivas.",
            "J06.9": "Infecção aguda das vias aéreas superiores não especificada (resfriado comum / IVAS).",
            "Z00.0": "Consulta médica para exame clínico geral ou rastreamento sem queixas específicas.",
        }
        self.cids_detalhes = {
            "M54.5": {
                "codigo_cid": "M54.5",
                "nome_doenca": "Dor lombar baixa",
                "descricao": "Dor localizada na região lombar inferior (lombalgia não especificada).",
                "capitulo": "M",
            },
            "F41.1": {
                "codigo_cid": "F41.1",
                "nome_doenca": "Ansiedade generalizada",
                "descricao": "Transtorno caracterizado por ansiedade e preocupação persistentes e excessivas.",
                "capitulo": "F",
            },
            "J06.9": {
                "codigo_cid": "J06.9",
                "nome_doenca": "Infecção aguda das vias aéreas superiores",
                "descricao": "Infecção aguda das vias aéreas superiores não especificada (resfriado comum / IVAS).",
                "capitulo": "J",
            },
            "Z00.0": {
                "codigo_cid": "Z00.0",
                "nome_doenca": "Exame médico geral",
                "descricao": "Consulta médica para exame clínico geral ou rastreamento sem queixas específicas.",
                "capitulo": "Z",
            },
        }
        self.colaboradores[42] = DadosColaboradorCadastro(
            id_colaborador=42,
            matricula="00123",
            nome="Carla Souza",
            id_cargo=15,
            id_setor=7,
            id_area=2,
            id_municipio=3509502,
            municipio_nome="Campinas",
            sigla_uf="SP",
            data_admissao=date(2024, 3, 1),
            ativo=True,
            cargo_nome="Analista de Operações",
            setor_nome="Logística",
        )
        self.registrar_usuario(
            id_usuario=42, id_colaborador=42, login="carla.souza", perfil="COLABORADOR"
        )
        self.registrar_usuario(
            id_usuario=7, id_colaborador=None, login="rh.ana", perfil="RH"
        )

    def registrar_usuario(
        self,
        *,
        id_usuario: int,
        login: str,
        perfil: str,
        id_colaborador: int | None = None,
        ativo: bool = True,
    ) -> None:
        self.usuarios_por_id[id_usuario] = {
            "id_usuario": id_usuario,
            "id_colaborador": id_colaborador,
            "login": login,
            "perfil": perfil,
            "ativo": ativo,
        }
        self.usuarios_por_login[login.strip().lower()] = id_usuario

    def conferir_colaborador_ativo(
        self,
        id_colaborador: int,
        inicio: date | None = None,
        fim: date | None = None,
    ) -> DadosColaboradorCadastro | None:
        colab = self.colaboradores.get(id_colaborador)
        if colab is None:
            return None
        ativo_periodo = _colaborador_ativo_no_periodo(
            colab.ativo, colab.data_admissao, inicio, fim
        )
        return colab.model_copy(update={"ativo": ativo_periodo})

    def buscar_id_municipio(
        self,
        nome_municipio: str | None,
        sigla_uf: str | None = None,
        id_colaborador_fallback: int | None = None,
    ) -> int | None:
        if nome_municipio and nome_municipio.strip():
            alvo = _normalizar_texto_busca(nome_municipio)
            uf_alvo = sigla_uf.strip().upper() if sigla_uf and sigla_uf.strip() else None
            for id_mun, (nome, uf) in self.municipios.items():
                if _normalizar_texto_busca(nome) == alvo and (
                    uf_alvo is None or uf.upper() == uf_alvo
                ):
                    return id_mun

        if id_colaborador_fallback is not None:
            colab = self.colaboradores.get(id_colaborador_fallback)
            if colab is not None:
                return colab.id_municipio
        return None

    def validar_conselho_profissional(self, sigla: str, uf: str) -> int | None:
        sigla_norm = sigla.strip().upper()
        uf_norm = uf.strip().upper()
        if uf_norm not in self.ufs:
            return None
        return self.conselhos.get(sigla_norm)

    def listar_conselhos_validos(self, uf: str) -> tuple[str, ...]:
        uf_norm = uf.strip().upper()
        if uf_norm not in self.ufs:
            return ()
        return tuple(sorted(self.conselhos.keys()))

    def obter_profissional(self, id_profissional: int) -> dict[str, Any] | None:
        prof = self.profissionais.get(id_profissional)
        if prof is None:
            return None
        sigla_conselho = next(
            (sig for sig, cid in self.conselhos.items() if cid == prof["id_conselho"]),
            None,
        )
        if sigla_conselho is None:
            return None
        return {
            "conselho": sigla_conselho,
            "numero": prof["numero_registro"],
            "uf": prof["sigla_uf"],
            "nome": prof.get("nome"),
        }

    def obter_municipio(self, id_municipio: int) -> tuple[str, str] | None:
        return self.municipios.get(id_municipio)

    def validar_cid(
        self, codigo_cid: str | None, *, politica_extrair_cid: bool = True
    ) -> str | None:
        if not politica_extrair_cid or not codigo_cid or not codigo_cid.strip():
            return None
        codigo_norm = codigo_cid.strip().upper()
        return codigo_norm if codigo_norm in self.cids else None

    def _resolver_nome_doenca_memoria(
        self,
        codigo_cid: str | None,
        nome_doenca: str | None = None,
    ) -> str | None:
        if not codigo_cid or not codigo_cid.strip():
            return None
        if nome_doenca is not None and nome_doenca.strip():
            return " ".join(nome_doenca.split()).strip()[:255]
        codigo_norm = codigo_cid.strip().upper()[:8]
        detalhe = self.cids_detalhes.get(codigo_norm)
        if detalhe:
            nome = detalhe.get("nome_doenca") or detalhe.get("descricao")
            if nome:
                return nome[:255]
        desc = self.cids.get(codigo_norm)
        return desc[:255] if desc else None

    @staticmethod
    def _capitulo_padrao_cid(codigo_norm: str) -> str:
        return (
            codigo_norm[:2]
            if len(codigo_norm) >= 4 and "." not in codigo_norm[:4]
            else codigo_norm[:1]
        )

    def obter_cid(self, codigo_cid: str | None) -> dict[str, str] | None:
        if not codigo_cid or not codigo_cid.strip():
            return None
        codigo_norm = codigo_cid.strip().upper()
        detalhe = self.cids_detalhes.get(codigo_norm)
        if detalhe is not None:
            return dict(detalhe)
        descricao = self.cids.get(codigo_norm)
        if descricao is None:
            return None
        return {
            "codigo_cid": codigo_norm,
            "nome_doenca": descricao,
            "descricao": descricao,
            "capitulo": self._capitulo_padrao_cid(codigo_norm),
        }

    def salvar_cid(
        self,
        codigo_cid: str,
        descricao: str,
        capitulo: str,
        *,
        nome_doenca: str | None = None,
    ) -> str:
        codigo_norm = codigo_cid.strip().upper()[:8]
        if not codigo_norm:
            raise ValueError("codigo_cid é obrigatório.")
        desc_limpa = " ".join(descricao.split()).strip() or f"CID {codigo_norm}"
        nome_limpo = (
            " ".join((nome_doenca or desc_limpa).split()).strip() or desc_limpa
        )[:255]
        cap_limpo = (capitulo.strip() or self._capitulo_padrao_cid(codigo_norm))[:10]
        self.cids[codigo_norm] = desc_limpa
        self.cids_detalhes[codigo_norm] = {
            "codigo_cid": codigo_norm,
            "nome_doenca": nome_limpo,
            "descricao": desc_limpa,
            "capitulo": cap_limpo,
        }
        return codigo_norm

    def salvar_cids_em_lote(
        self,
        registros: Sequence[tuple[str, str, str] | tuple[str, str, str, str]],
    ) -> int:
        total = 0
        for item in registros:
            if len(item) == 4:
                codigo, nome_doenca, descricao, capitulo = item
            else:
                codigo, descricao, capitulo = item
                nome_doenca = descricao
            if codigo and codigo.strip():
                self.salvar_cid(
                    codigo,
                    descricao,
                    capitulo,
                    nome_doenca=nome_doenca,
                )
                total += 1
        return total

    def listar_cids(
        self, filtro: str | None = None, *, limite: int = 200
    ) -> list[dict[str, str]]:
        termo = filtro.strip().lower() if filtro and filtro.strip() else None
        itens: list[dict[str, str]] = []
        for codigo in sorted(self.cids.keys()):
            det = self.cids_detalhes.get(codigo)
            desc = det["descricao"] if det else self.cids[codigo]
            nome = det["nome_doenca"] if det else desc
            cap = det["capitulo"] if det else self._capitulo_padrao_cid(codigo)
            if (
                termo
                and termo not in codigo.lower()
                and termo not in desc.lower()
                and termo not in nome.lower()
            ):
                continue
            itens.append(
                {
                    "codigo_cid": codigo,
                    "nome_doenca": nome,
                    "descricao": desc,
                    "capitulo": cap,
                }
            )
            if len(itens) >= limite:
                break
        return itens

    def listar_atestados_recentes(
        self, *, limite: int = 50
    ) -> list[dict[str, Any]]:
        ordenados = sorted(
            self.atestados.values(),
            key=lambda a: (a.postado_em, a.id_atestado),
            reverse=True,
        )[: max(1, int(limite))]
        saida: list[dict[str, Any]] = []
        for at in ordenados:
            colab = self.colaboradores.get(at.id_colaborador)
            prof = self.obter_profissional(at.id_profissional) or {}
            cid_reg = self.cids_restritos.get(at.id_atestado)
            cid_cod = at.codigo_cid or (cid_reg.codigo_cid if cid_reg is not None else None)
            nome_doenca = (
                at.nome_doenca
                or (cid_reg.nome_doenca if cid_reg is not None else None)
                or self._resolver_nome_doenca_memoria(cid_cod)
            )
            saida.append(
                {
                    "id_atestado": at.id_atestado,
                    "id_caso": at.id_caso,
                    "id_colaborador": at.id_colaborador,
                    "colaborador_nome": colab.nome if colab else str(at.id_colaborador),
                    "tipo_documento": at.tipo_documento,
                    "status": at.status,
                    "origem_dados": at.origem_dados,
                    "data_emissao": at.data_emissao,
                    "inicio": at.inicio_afastamento.date(),
                    "fim": at.fim_afastamento.date(),
                    "prazo_entrega_fisica": at.prazo_entrega_fisica,
                    "prazos_legais": list(at.prazos_legais),
                    "conselho": prof.get("conselho"),
                    "registro_profissional": prof.get("numero"),
                    "uf_profissional": prof.get("uf"),
                    "codigo_cid_restrito": cid_cod,
                    "nome_doenca": nome_doenca,
                    "cid_descricao": self.cids.get(cid_cod) if cid_cod else None,
                    "postado_em": at.postado_em,
                }
            )
        return saida

    def _validar_perfil_usuario(
        self,
        user: dict[str, Any],
        identificador: int | str,
        perfis_permitidos: Collection[str] | None,
    ) -> None:
        if perfis_permitidos is None:
            return
        permitidos = {p.strip().upper() for p in perfis_permitidos if p.strip()}
        if str(user["perfil"]).strip().upper() not in permitidos:
            raise PermissionError(
                f"Usuário {identificador!r} (perfil={user['perfil']}) não autorizado para esta operação."
            )

    def resolver_id_usuario(
        self,
        identificador: int | str,
        *,
        perfis_permitidos: Collection[str] | None = None,
    ) -> int:
        if isinstance(identificador, int) and not isinstance(identificador, bool):
            user = self.usuarios_por_id.get(identificador)
            if user and user["ativo"]:
                self._validar_perfil_usuario(user, identificador, perfis_permitidos)
                return int(user["id_usuario"])
            raise ValueError(f"Usuário não encontrado ou inativo: {identificador!r}")

        texto = str(identificador).strip()
        if not texto:
            raise ValueError("Identificador de usuário vazio.")

        if texto.isdigit():
            user = self.usuarios_por_id.get(int(texto))
            if user and user["ativo"]:
                self._validar_perfil_usuario(user, identificador, perfis_permitidos)
                return int(user["id_usuario"])

        uid = self.usuarios_por_login.get(texto.lower())
        if uid is not None:
            user = self.usuarios_por_id.get(uid)
            if user and user["ativo"]:
                self._validar_perfil_usuario(user, identificador, perfis_permitidos)
                return uid

        raise ValueError(f"Usuário não encontrado ou inativo: {identificador!r}")

    def obter_ou_criar_profissional_saude(
        self,
        conselho: str,
        numero: str,
        uf: str,
        nome: str | None = None,
    ) -> int:
        numero_limpo = numero.strip()
        uf_norm = uf.strip().upper()
        if not numero_limpo:
            raise ValueError("Número de registro profissional é obrigatório.")

        id_conselho = self.validar_conselho_profissional(conselho, uf_norm)
        if id_conselho is None:
            raise ValueError(
                f"Conselho ou UF não reconhecido no catálogo: {conselho!r}/{uf_norm!r}"
            )

        chave = (id_conselho, numero_limpo, uf_norm)
        if chave in self._chave_profissional:
            id_prof = self._chave_profissional[chave]
            if nome and nome.strip() and not self.profissionais[id_prof]["nome"]:
                self.profissionais[id_prof]["nome"] = nome.strip()
            return id_prof

        id_prof = self._seq_profissional
        self._seq_profissional += 1
        self._chave_profissional[chave] = id_prof
        self.profissionais[id_prof] = {
            "id_profissional": id_prof,
            "id_conselho": id_conselho,
            "numero_registro": numero_limpo,
            "sigla_uf": uf_norm,
            "nome": nome.strip() if nome and nome.strip() else None,
            "verificado": False,
        }
        return id_prof

    def verificar_duplicidade_arquivo(
        self,
        arquivo_uri: str,
        id_colaborador: int | None = None,
        *,
        sha256: str | None = None,
    ) -> RegistroAtestadoDB | None:
        if sha256 and sha256.strip():
            sha_norm = sha256.strip().lower()
            arquivos_match = [
                arq
                for arq in self.arquivos_atestado.values()
                if arq.sha256 == sha_norm
                and (id_colaborador is None or arq.id_colaborador == id_colaborador)
            ]
            arquivos_match.sort(
                key=lambda arq: (arq.enviado_em, arq.id_arquivo), reverse=True
            )
            for arq in arquivos_match:
                reg = self.atestados.get(arq.id_atestado)
                if reg is not None and reg.status != "REJEITADO":
                    return reg.model_copy(deep=True)

            id_at = (
                self._sha256_por_chave.get((id_colaborador, sha_norm))
                if id_colaborador is not None
                else self._sha256_global.get(sha_norm)
            )
            if id_at is not None:
                reg = self.atestados.get(id_at)
                if reg is not None and reg.status != "REJEITADO":
                    return reg.model_copy(deep=True)

        uri_limpa = arquivo_uri.strip()
        if not uri_limpa:
            return None
        candidatos = [
            at
            for at in self.atestados.values()
            if at.arquivo_uri == uri_limpa
            and at.status != "REJEITADO"
            and (id_colaborador is None or at.id_colaborador == id_colaborador)
        ]
        if not candidatos:
            return None
        candidatos.sort(key=lambda at: (at.postado_em, at.id_atestado), reverse=True)
        return candidatos[0].model_copy(deep=True)

    def listar_atestados_sobrepostos(
        self,
        id_colaborador: int,
        inicio: date,
        fim: date,
        ignorar_id_atestado: int | None = None,
    ) -> list[RegistroAtestadoDB]:
        sobrepostos = [
            at.model_copy(deep=True)
            for at in self.atestados.values()
            if at.id_colaborador == id_colaborador
            and at.status != "REJEITADO"
            and at.inicio_afastamento.date() <= fim
            and at.fim_afastamento.date() >= inicio
            and (ignorar_id_atestado is None or at.id_atestado != ignorar_id_atestado)
        ]
        sobrepostos.sort(key=lambda at: at.inicio_afastamento)
        return sobrepostos

    def calcular_dias_afastamento_ultimos_60_dias(
        self,
        id_colaborador: int,
        data_referencia: date,
        cid: str | None = None,
    ) -> int:
        do_colaborador: list[RegistroAtestadoDB] = []
        for at in self.atestados.values():
            if at.id_colaborador != id_colaborador:
                continue
            cid_restrito = self.cids_restritos.get(at.id_atestado)
            if cid_restrito is not None and at.codigo_cid is None:
                do_colaborador.append(
                    at.model_copy(update={"codigo_cid": cid_restrito.codigo_cid})
                )
            else:
                do_colaborador.append(at)
        return _somar_dias_janela_60d(do_colaborador, data_referencia, cid)

    def salvar_atestado_pendente(
        self,
        *,
        id_colaborador: int,
        id_profissional: int,
        id_municipio: int,
        data_emissao: date,
        inicio_afastamento: date | datetime,
        fim_afastamento: date | datetime,
        arquivo_uri: str,
        codigo_cid: str | None = None,
        nome_doenca: str | None = None,
        origem_dados: OrigemDadosAtestado = "IA",
        arquivo_sha256: str | None = None,
        id_caso: int | None = None,
        tipo_documento: TipoDocumentoAfastamento = "atestado_medico",
        arquivo_mime: str | None = None,
        politica_cid_restrito: bool = True,
    ) -> RegistroAtestadoDB:
        if id_colaborador not in self.colaboradores:
            raise ValueError(f"Colaborador {id_colaborador} inexistente.")
        if id_profissional not in self.profissionais:
            raise ValueError(f"Profissional {id_profissional} inexistente.")
        if id_municipio not in self.municipios:
            raise ValueError(f"Município {id_municipio} inexistente.")
        if codigo_cid is not None and codigo_cid.upper() not in self.cids:
            raise ValueError(f"CID {codigo_cid!r} inexistente no catálogo.")

        uri_limpa = arquivo_uri.strip()
        if not uri_limpa:
            raise ValueError("arquivo_uri é obrigatório.")

        dt_inicio = _para_inicio_timestamptz(inicio_afastamento)
        dt_fim = _para_fim_timestamptz(fim_afastamento)
        if dt_fim < dt_inicio or _para_data(dt_fim) < _para_data(dt_inicio):
            raise ValueError("fim_afastamento não pode ser anterior a inicio_afastamento.")

        id_atestado = self._seq_atestado
        self._seq_atestado += 1
        cid_norm = codigo_cid.strip().upper()[:8] if codigo_cid and codigo_cid.strip() else None
        nome_doenca_final = self._resolver_nome_doenca_memoria(cid_norm, nome_doenca)
        registro = RegistroAtestadoDB(
            id_atestado=id_atestado,
            id_caso=id_caso,
            id_colaborador=id_colaborador,
            id_profissional=id_profissional,
            codigo_cid=cid_norm,
            nome_doenca=nome_doenca_final,
            id_municipio=id_municipio,
            tipo_documento=tipo_documento,
            data_emissao=data_emissao,
            inicio_afastamento=dt_inicio,
            fim_afastamento=dt_fim,
            arquivo_uri=uri_limpa,
            origem_dados=origem_dados,
            status="PENDENTE",
            postado_em=datetime.now(UTC),
        )
        self.atestados[id_atestado] = registro
        if politica_cid_restrito and cid_norm is not None:
            self.salvar_cid_restrito(
                id_atestado, cid_norm, nome_doenca=nome_doenca_final
            )
        if arquivo_sha256 and arquivo_sha256.strip():
            self.registrar_arquivo_atestado(
                id_atestado,
                id_colaborador=id_colaborador,
                arquivo_uri=uri_limpa,
                sha256=arquivo_sha256,
                mime=arquivo_mime,
                id_caso=id_caso,
            )
        return registro.model_copy(deep=True)

    def obter_atestado(self, id_atestado: int) -> RegistroAtestadoDB | None:
        reg = self.atestados.get(id_atestado)
        return reg.model_copy(deep=True) if reg is not None else None

    def obter_ultimo_atestado_pendente(
        self, id_colaborador: int, *, id_caso: int | None = None
    ) -> RegistroAtestadoDB | None:
        if id_caso is not None:
            pendentes_caso = [
                at
                for at in self.atestados.values()
                if at.id_colaborador == id_colaborador
                and at.id_caso == id_caso
                and at.status == "PENDENTE"
            ]
            if pendentes_caso:
                pendentes_caso.sort(
                    key=lambda at: (at.postado_em, at.id_atestado), reverse=True
                )
                return pendentes_caso[0].model_copy(deep=True)

            pendentes_legados = [
                at
                for at in self.atestados.values()
                if at.id_colaborador == id_colaborador
                and at.id_caso is None
                and at.status == "PENDENTE"
            ]
            if not pendentes_legados:
                return None
            pendentes_legados.sort(
                key=lambda at: (at.postado_em, at.id_atestado), reverse=True
            )
            return pendentes_legados[0].model_copy(deep=True)

        pendentes = [
            at
            for at in self.atestados.values()
            if at.id_colaborador == id_colaborador and at.status == "PENDENTE"
        ]
        if not pendentes:
            return None
        pendentes.sort(key=lambda at: (at.postado_em, at.id_atestado), reverse=True)
        return pendentes[0].model_copy(deep=True)

    def confirmar_atestado_no_banco(
        self,
        id_atestado: int,
        *,
        inicio_afastamento: date | datetime,
        fim_afastamento: date | datetime,
        data_emissao: date,
        origem_dados: OrigemDadosAtestado,
        prazo_entrega_fisica: date,
        id_profissional: int | None = _NAO_INFORMADO,
        codigo_cid: str | None = _NAO_INFORMADO,
        nome_doenca: str | None = _NAO_INFORMADO,
        id_municipio: int | None = _NAO_INFORMADO,
        tipo_documento: TipoDocumentoAfastamento | None = _NAO_INFORMADO,
        prazos_legais: Sequence[str] | None = _NAO_INFORMADO,
        politica_cid_restrito: bool = True,
    ) -> RegistroAtestadoDB:
        atual = self.atestados.get(id_atestado)
        if atual is None:
            raise ValueError(f"Atestado {id_atestado} não encontrado para confirmação.")

        dt_inicio = _para_inicio_timestamptz(inicio_afastamento)
        dt_fim = _para_fim_timestamptz(fim_afastamento)
        if dt_fim < dt_inicio or _para_data(dt_fim) < _para_data(dt_inicio):
            raise ValueError("fim_afastamento não pode ser anterior a inicio_afastamento.")

        updates: dict[str, Any] = {
            "inicio_afastamento": dt_inicio,
            "fim_afastamento": dt_fim,
            "data_emissao": data_emissao,
            "origem_dados": origem_dados,
            "prazo_entrega_fisica": prazo_entrega_fisica,
            "status": "CONFIRMADO",
        }
        if id_profissional is not _NAO_INFORMADO and id_profissional is not None:
            if id_profissional not in self.profissionais:
                raise ValueError(f"Profissional {id_profissional} inexistente.")
            updates["id_profissional"] = id_profissional
        if codigo_cid is not _NAO_INFORMADO:
            if codigo_cid is not None and codigo_cid.upper() not in self.cids:
                raise ValueError(f"CID {codigo_cid!r} inexistente no catálogo.")
            cid_norm = codigo_cid.strip().upper()[:8] if codigo_cid and codigo_cid.strip() else None
            nome_arg = None if nome_doenca is _NAO_INFORMADO else nome_doenca
            nome_doenca_final = self._resolver_nome_doenca_memoria(cid_norm, nome_arg)
            updates["codigo_cid"] = cid_norm
            updates["nome_doenca"] = nome_doenca_final
            if politica_cid_restrito:
                self.salvar_cid_restrito(
                    id_atestado, cid_norm, nome_doenca=nome_doenca_final
                )
        elif nome_doenca is not _NAO_INFORMADO:
            updates["nome_doenca"] = (
                " ".join(str(nome_doenca).split()).strip()[:255]
                if nome_doenca is not None and str(nome_doenca).strip()
                else None
            )
        if id_municipio is not _NAO_INFORMADO and id_municipio is not None:
            if id_municipio not in self.municipios:
                raise ValueError(f"Município {id_municipio} inexistente.")
            updates["id_municipio"] = id_municipio
        if tipo_documento is not _NAO_INFORMADO and tipo_documento is not None:
            updates["tipo_documento"] = tipo_documento
        if prazos_legais is not _NAO_INFORMADO and prazos_legais is not None:
            updates["prazos_legais"] = list(prazos_legais)

        atualizado = atual.model_copy(update=updates)
        self.atestados[id_atestado] = atualizado
        return atualizado.model_copy(deep=True)

    def registrar_baixa_fisica_no_banco(
        self,
        id_atestado: int,
        recebido_fisico_por: int | str,
        recebido_fisico_em: datetime | str,
        *,
        perfis_permitidos: Collection[str] | None = ("RH", "ADMIN"),
    ) -> RegistroAtestadoDB:
        atual = self.atestados.get(id_atestado)
        if atual is None or atual.status != "CONFIRMADO":
            raise ValueError(
                f"Atestado {id_atestado} não encontrado ou ainda não confirmado."
            )

        id_usuario_rh = self.resolver_id_usuario(
            recebido_fisico_por, perfis_permitidos=perfis_permitidos
        )
        dt_recebimento = _parse_datetime_iso(recebido_fisico_em)
        atualizado = atual.model_copy(
            update={
                "recebido_fisico_em": dt_recebimento,
                "recebido_fisico_por": id_usuario_rh,
            }
        )
        self.atestados[id_atestado] = atualizado

        recebido_por_texto = str(recebido_fisico_por).strip() or str(id_usuario_rh)
        existente = self.baixas_fisicas.get(id_atestado)
        id_baixa = existente.id_baixa if existente is not None else self._seq_baixa
        if existente is None:
            self._seq_baixa += 1
        self.baixas_fisicas[id_atestado] = RegistroBaixaFisicaDB(
            id_baixa=id_baixa,
            id_atestado=id_atestado,
            id_usuario_rh=id_usuario_rh,
            recebido_por=recebido_por_texto,
            data_hora=dt_recebimento,
            registrado_em=datetime.now(UTC),
        )
        return atualizado.model_copy(deep=True)

    def registrar_arquivo_atestado(
        self,
        id_atestado: int,
        *,
        id_colaborador: int,
        arquivo_uri: str,
        sha256: str,
        mime: str | None = None,
        id_caso: int | None = None,
    ) -> RegistroArquivoAtestadoDB:
        sha_norm = sha256.strip().lower()
        id_arquivo = self._seq_arquivo
        self._seq_arquivo += 1
        registro = RegistroArquivoAtestadoDB(
            id_arquivo=id_arquivo,
            id_atestado=id_atestado,
            id_caso=id_caso,
            id_colaborador=id_colaborador,
            arquivo_uri=arquivo_uri.strip(),
            mime=mime,
            sha256=sha_norm,
            enviado_em=datetime.now(UTC),
        )
        self.arquivos_atestado[id_arquivo] = registro
        self._sha256_por_chave[(id_colaborador, sha_norm)] = id_atestado
        self._sha256_global[sha_norm] = id_atestado
        return registro.model_copy(deep=True)

    def listar_arquivos_atestado(
        self, id_atestado: int
    ) -> list[RegistroArquivoAtestadoDB]:
        itens = [
            arq.model_copy(deep=True)
            for arq in self.arquivos_atestado.values()
            if arq.id_atestado == id_atestado
        ]
        itens.sort(key=lambda arq: (arq.enviado_em, arq.id_arquivo))
        return itens

    def salvar_campos_atestado(
        self,
        id_atestado: int,
        campos: Sequence[dict[str, Any]],
        *,
        id_caso: int | None = None,
    ) -> list[RegistroCampoAtestadoDB]:
        salvos: list[RegistroCampoAtestadoDB] = []
        agora = datetime.now(UTC)
        for item in campos:
            nome_campo = str(item["nome_campo"])
            chave = (id_atestado, nome_campo)
            existente = self.campos_atestado.get(chave)
            if existente is not None:
                registro = existente.model_copy(
                    update={
                        "id_caso": item.get("id_caso", id_caso) or existente.id_caso,
                        "valor_extraido": (
                            item["valor_extraido"]
                            if item.get("valor_extraido") is not None
                            else existente.valor_extraido
                        ),
                        "valor_confirmado": (
                            item["valor_confirmado"]
                            if item.get("valor_confirmado") is not None
                            else existente.valor_confirmado
                        ),
                        "confianca": (
                            item["confianca"]
                            if item.get("confianca") is not None
                            else existente.confianca
                        ),
                        "origem": item.get("origem", "IA"),
                        "vazio_por_baixa_confianca": bool(
                            item.get("vazio_por_baixa_confianca", False)
                        ),
                        "atualizado_em": agora,
                    }
                )
            else:
                id_campo = self._seq_campo
                self._seq_campo += 1
                registro = RegistroCampoAtestadoDB(
                    id_campo=id_campo,
                    id_atestado=id_atestado,
                    id_caso=item.get("id_caso", id_caso),
                    nome_campo=nome_campo,
                    valor_extraido=item.get("valor_extraido"),
                    valor_confirmado=item.get("valor_confirmado"),
                    confianca=item.get("confianca"),
                    origem=item.get("origem", "IA"),
                    vazio_por_baixa_confianca=bool(
                        item.get("vazio_por_baixa_confianca", False)
                    ),
                    atualizado_em=agora,
                )
            self.campos_atestado[chave] = registro
            salvos.append(registro.model_copy(deep=True))
        return salvos

    def listar_campos_atestado(
        self, id_atestado: int
    ) -> list[RegistroCampoAtestadoDB]:
        itens = [
            campo.model_copy(deep=True)
            for campo in self.campos_atestado.values()
            if campo.id_atestado == id_atestado
        ]
        itens.sort(key=lambda c: c.id_campo)
        return itens

    def obter_baixa_fisica(self, id_atestado: int) -> RegistroBaixaFisicaDB | None:
        reg = self.baixas_fisicas.get(id_atestado)
        return reg.model_copy(deep=True) if reg is not None else None

    def salvar_cid_restrito(
        self,
        id_atestado: int,
        codigo_cid: str | None,
        *,
        nome_doenca: str | None = None,
    ) -> RegistroCidRestritoDB | None:
        if codigo_cid is None or not str(codigo_cid).strip():
            self.cids_restritos.pop(id_atestado, None)
            return None
        cid_norm = str(codigo_cid).strip().upper()[:8]
        if cid_norm not in self.cids:
            raise ValueError(f"CID {codigo_cid!r} inexistente no catálogo.")
        nome_doenca_final = self._resolver_nome_doenca_memoria(cid_norm, nome_doenca)
        registro = RegistroCidRestritoDB(
            id_atestado=id_atestado,
            codigo_cid=cid_norm,
            nome_doenca=nome_doenca_final,
            registrado_em=datetime.now(UTC),
        )
        self.cids_restritos[id_atestado] = registro
        return registro.model_copy(deep=True)

    def obter_cid_restrito(self, id_atestado: int) -> RegistroCidRestritoDB | None:
        reg = self.cids_restritos.get(id_atestado)
        return reg.model_copy(deep=True) if reg is not None else None
