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
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, model_validator

from sinergia.agentes.a1_recepcao.schemas import OrigemDadosAtestado, StatusAtestadoDB
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
    """Espelho tipado da tabela `afastamento.atestado` (`001_schema_inicial.sql`)."""

    model_config = ConfigDict(extra="forbid")

    id_atestado: int
    id_colaborador: int
    id_profissional: int
    codigo_cid: str | None = None
    id_municipio: int
    data_emissao: date
    inicio_afastamento: datetime
    fim_afastamento: datetime
    arquivo_uri: str
    origem_dados: OrigemDadosAtestado
    status: StatusAtestadoDB
    postado_em: datetime
    prazo_entrega_fisica: date | None = None
    recebido_fisico_em: datetime | None = None
    recebido_fisico_por: int | None = None

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

    def validar_cid(
        self, codigo_cid: str | None, *, politica_extrair_cid: bool = True
    ) -> str | None: ...

    def resolver_id_usuario(self, identificador: int | str) -> int: ...

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
        origem_dados: OrigemDadosAtestado = "IA",
    ) -> RegistroAtestadoDB: ...

    def obter_atestado(self, id_atestado: int) -> RegistroAtestadoDB | None: ...

    def obter_ultimo_atestado_pendente(
        self, id_colaborador: int
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
        id_municipio: int | None = _NAO_INFORMADO,
    ) -> RegistroAtestadoDB: ...

    def registrar_baixa_fisica_no_banco(
        self,
        id_atestado: int,
        recebido_fisico_por: int | str,
        recebido_fisico_em: datetime | str,
    ) -> RegistroAtestadoDB: ...


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


class RepositorioA1Postgres:
    """Repositório PostgreSQL 16 estritamente restrito às tabelas do Anexo C.2."""

    def __init__(self, database_url: str) -> None:
        if psycopg is None:
            raise RuntimeError(
                "Driver psycopg não encontrado. Instale com: pip install 'psycopg[binary]'"
            )
        self._database_url = database_url

    def _conectar(self) -> Any:
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

    def resolver_id_usuario(self, identificador: int | str) -> int:
        with self._conectar() as conn, conn.cursor() as cur:
            if isinstance(identificador, int) and not isinstance(identificador, bool):
                cur.execute(
                    """
                    SELECT id_usuario
                    FROM cadastro.usuario
                    WHERE id_usuario = %s AND ativo = TRUE
                    """,
                    (identificador,),
                )
                row = cur.fetchone()
                if row is not None:
                    return int(row["id_usuario"])
                raise ValueError(f"Usuário não encontrado ou inativo: {identificador!r}")

            texto = str(identificador).strip()
            if not texto:
                raise ValueError("Identificador de usuário vazio.")

            if texto.isdigit():
                cur.execute(
                    """
                    SELECT id_usuario
                    FROM cadastro.usuario
                    WHERE id_usuario = %s AND ativo = TRUE
                    """,
                    (int(texto),),
                )
                row = cur.fetchone()
                if row is not None:
                    return int(row["id_usuario"])

            cur.execute(
                """
                SELECT id_usuario
                FROM cadastro.usuario
                WHERE LOWER(TRIM(login)) = LOWER(TRIM(%s)) AND ativo = TRUE
                LIMIT 1
                """,
                (texto,),
            )
            row = cur.fetchone()
            if row is not None:
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
    ) -> RegistroAtestadoDB | None:
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
            SELECT *
            FROM afastamento.atestado
            WHERE id_colaborador = %s
              AND status = 'CONFIRMADO'
              AND (fim_afastamento AT TIME ZONE 'UTC')::date >= %s
              AND (inicio_afastamento AT TIME ZONE 'UTC')::date <= %s
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_colaborador, inicio_janela, data_referencia))
            atestados = [RegistroAtestadoDB(**row) for row in cur.fetchall()]
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
        origem_dados: OrigemDadosAtestado = "IA",
    ) -> RegistroAtestadoDB:
        uri_limpa = arquivo_uri.strip()
        if not uri_limpa:
            raise ValueError("arquivo_uri é obrigatório.")
        dt_inicio = _para_inicio_timestamptz(inicio_afastamento)
        dt_fim = _para_fim_timestamptz(fim_afastamento)
        if dt_fim < dt_inicio or _para_data(dt_fim) < _para_data(dt_inicio):
            raise ValueError("fim_afastamento não pode ser anterior a inicio_afastamento.")

        sql = """
            INSERT INTO afastamento.atestado (
                id_colaborador,
                id_profissional,
                codigo_cid,
                id_municipio,
                data_emissao,
                inicio_afastamento,
                fim_afastamento,
                arquivo_uri,
                origem_dados,
                status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDENTE')
            RETURNING *
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    id_colaborador,
                    id_profissional,
                    codigo_cid,
                    id_municipio,
                    data_emissao,
                    dt_inicio,
                    dt_fim,
                    uri_limpa,
                    origem_dados,
                ),
            )
            row = cur.fetchone()
            conn.commit()
            return RegistroAtestadoDB(**row)

    def obter_atestado(self, id_atestado: int) -> RegistroAtestadoDB | None:
        sql = "SELECT * FROM afastamento.atestado WHERE id_atestado = %s"
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_atestado,))
            row = cur.fetchone()
            return RegistroAtestadoDB(**row) if row is not None else None

    def obter_ultimo_atestado_pendente(
        self, id_colaborador: int
    ) -> RegistroAtestadoDB | None:
        sql = """
            SELECT *
            FROM afastamento.atestado
            WHERE id_colaborador = %s AND status = 'PENDENTE'
            ORDER BY postado_em DESC, id_atestado DESC
            LIMIT 1
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_colaborador,))
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
        id_municipio: int | None = _NAO_INFORMADO,
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
        if codigo_cid is not _NAO_INFORMADO:
            set_clauses.append("codigo_cid = %s")
            params.append(codigo_cid)
        if id_municipio is not _NAO_INFORMADO and id_municipio is not None:
            set_clauses.append("id_municipio = %s")
            params.append(id_municipio)

        params.append(id_atestado)
        sql = f"""
            UPDATE afastamento.atestado
            SET {', '.join(set_clauses)}
            WHERE id_atestado = %s
            RETURNING *
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Atestado {id_atestado} não encontrado para confirmação.")
            conn.commit()
            return RegistroAtestadoDB(**row)

    def registrar_baixa_fisica_no_banco(
        self,
        id_atestado: int,
        recebido_fisico_por: int | str,
        recebido_fisico_em: datetime | str,
    ) -> RegistroAtestadoDB:
        id_usuario_rh = self.resolver_id_usuario(recebido_fisico_por)
        dt_recebimento = _parse_datetime_iso(recebido_fisico_em)

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
            conn.commit()
            return RegistroAtestadoDB(**row)


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
        self.profissionais: dict[int, dict[str, Any]] = {}
        self._chave_profissional: dict[tuple[int, str, str], int] = {}
        self.atestados: dict[int, RegistroAtestadoDB] = {}
        self._seq_profissional = 1
        self._seq_atestado = 311

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
            "M54.5": "Dor lombar baixa",
            "F41.1": "Ansiedade generalizada",
            "J06.9": "Infecção aguda das vias aéreas superiores",
            "Z00.0": "Exame médico geral",
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

    def validar_cid(
        self, codigo_cid: str | None, *, politica_extrair_cid: bool = True
    ) -> str | None:
        if not politica_extrair_cid or not codigo_cid or not codigo_cid.strip():
            return None
        codigo_norm = codigo_cid.strip().upper()
        return codigo_norm if codigo_norm in self.cids else None

    def resolver_id_usuario(self, identificador: int | str) -> int:
        if isinstance(identificador, int) and not isinstance(identificador, bool):
            user = self.usuarios_por_id.get(identificador)
            if user and user["ativo"]:
                return int(user["id_usuario"])
            raise ValueError(f"Usuário não encontrado ou inativo: {identificador!r}")

        texto = str(identificador).strip()
        if not texto:
            raise ValueError("Identificador de usuário vazio.")

        if texto.isdigit():
            user = self.usuarios_por_id.get(int(texto))
            if user and user["ativo"]:
                return int(user["id_usuario"])

        uid = self.usuarios_por_login.get(texto.lower())
        if uid is not None:
            user = self.usuarios_por_id.get(uid)
            if user and user["ativo"]:
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
    ) -> RegistroAtestadoDB | None:
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
        do_colaborador = [
            at for at in self.atestados.values() if at.id_colaborador == id_colaborador
        ]
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
        origem_dados: OrigemDadosAtestado = "IA",
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
        registro = RegistroAtestadoDB(
            id_atestado=id_atestado,
            id_colaborador=id_colaborador,
            id_profissional=id_profissional,
            codigo_cid=codigo_cid.upper() if codigo_cid else None,
            id_municipio=id_municipio,
            data_emissao=data_emissao,
            inicio_afastamento=dt_inicio,
            fim_afastamento=dt_fim,
            arquivo_uri=uri_limpa,
            origem_dados=origem_dados,
            status="PENDENTE",
            postado_em=datetime.now(UTC),
        )
        self.atestados[id_atestado] = registro
        return registro.model_copy(deep=True)

    def obter_atestado(self, id_atestado: int) -> RegistroAtestadoDB | None:
        reg = self.atestados.get(id_atestado)
        return reg.model_copy(deep=True) if reg is not None else None

    def obter_ultimo_atestado_pendente(
        self, id_colaborador: int
    ) -> RegistroAtestadoDB | None:
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
        id_municipio: int | None = _NAO_INFORMADO,
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
            updates["codigo_cid"] = codigo_cid.upper() if codigo_cid else None
        if id_municipio is not _NAO_INFORMADO and id_municipio is not None:
            if id_municipio not in self.municipios:
                raise ValueError(f"Município {id_municipio} inexistente.")
            updates["id_municipio"] = id_municipio

        atualizado = atual.model_copy(update=updates)
        self.atestados[id_atestado] = atualizado
        return atualizado.model_copy(deep=True)

    def registrar_baixa_fisica_no_banco(
        self,
        id_atestado: int,
        recebido_fisico_por: int | str,
        recebido_fisico_em: datetime | str,
    ) -> RegistroAtestadoDB:
        atual = self.atestados.get(id_atestado)
        if atual is None or atual.status != "CONFIRMADO":
            raise ValueError(
                f"Atestado {id_atestado} não encontrado ou ainda não confirmado."
            )

        id_usuario_rh = self.resolver_id_usuario(recebido_fisico_por)
        dt_recebimento = _parse_datetime_iso(recebido_fisico_em)
        atualizado = atual.model_copy(
            update={
                "recebido_fisico_em": dt_recebimento,
                "recebido_fisico_por": id_usuario_rh,
            }
        )
        self.atestados[id_atestado] = atualizado
        return atualizado.model_copy(deep=True)
