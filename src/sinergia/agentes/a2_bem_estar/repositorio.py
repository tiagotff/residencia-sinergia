"""Repositório de dados do Agente 2 (A2 — Bem-Estar).

Respeita a matriz de permissões do **Anexo C.3** da `Documentação - SinergIA v2.2.docx`
e o modelo físico em `db/schema/001_schema_inicial.sql`:

- **Leitura em `cadastro.*`:** `colaborador`, `cargo`, `setor`.
- **Leitura em `bem_estar.*`:** `beneficio`, `colaborador_beneficio` (e demais tabelas do A2).
- **Leitura em `apoio.*`:** `epi`, `cargo_epi`.
- **Gravação em `apoio.solicitacao_epi`:** abertura de pedidos de reposição com status `ABERTA` (`F5`).

Fornece duas implementações aderentes ao mesmo `Protocol`:
- `RepositorioA2Postgres`: queries SQL parametrizadas com `psycopg` (PostgreSQL 16).
- `RepositorioA2Memoria`: implementação determinística em memória para testes offline
  e demonstração, pré-carregada com os colaboradores canônicos e catálogo de EPIs.
"""

from __future__ import annotations

import unicodedata
from datetime import UTC, date, datetime, time
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sinergia.agentes.a2_bem_estar.schemas import (
    MotivoSolicitacaoEPI,
    StatusSolicitacaoEPI,
)

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - fallback caso psycopg não esteja instalado
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


def _normalizar_texto(texto: str) -> str:
    sem_acentos = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", texto or "")
        if not unicodedata.combining(ch)
    )
    return " ".join(sem_acentos.strip().lower().split())


def _formatar_hora(valor: Any) -> str:
    if isinstance(valor, time):
        return valor.strftime("%H:%M")
    return str(valor)


class PerfilColaboradorDB(BaseModel):
    """Perfil de colaborador consultado pelo Agente 2 em `cadastro.*` e `bem_estar.*`."""

    model_config = ConfigDict(extra="forbid")

    id_colaborador: int
    matricula: str
    nome: str
    id_cargo: int
    cargo: str
    id_setor: int
    setor: str
    id_area: int
    jornada_inicio: str
    jornada_fim: str
    ativo: bool = True
    beneficios: list[str] = Field(default_factory=list)


class RegistroEpiDB(BaseModel):
    """Espelho tipado da tabela `apoio.epi` (`001_schema_inicial.sql`)."""

    model_config = ConfigDict(extra="forbid")

    id_epi: int
    nome: str
    numero_ca: str
    validade_ca: date
    vida_util_dias: int


class RegistroSolicitacaoEpiDB(BaseModel):
    """Espelho tipado da tabela `apoio.solicitacao_epi` (`001_schema_inicial.sql`)."""

    model_config = ConfigDict(extra="forbid")

    id_solicitacao: int
    id_colaborador: int
    id_epi: int
    motivo: MotivoSolicitacaoEPI
    descricao: str | None = None
    status: StatusSolicitacaoEPI = "ABERTA"
    id_entrega: int | None = None
    aberta_em: datetime
    atendida_em: datetime | None = None

    @model_validator(mode="after")
    def _validar_regra_entrega(self) -> Self:
        if (self.status == "ATENDIDA") != (self.id_entrega is not None):
            raise ValueError(
                "Regra de integridade violada: status 'ATENDIDA' exige 'id_entrega' preenchido."
            )
        return self


_PALAVRAS_CHAVE_EPI: tuple[tuple[tuple[str, ...], str], ...] = (
    (("luva", "luvas", "nitrilica", "rasgou", "rasgada"), "luva"),
    (("protetor auricular", "auricular", "abafador", "plug", "ouvido"), "protetor auricular"),
    (("oculos", "viseira"), "oculos"),
    (("capacete", "jugular"), "capacete"),
    (("bota", "calcado", "sapato", "botina"), "calcado"),
)


def _selecionar_epi_por_texto(
    texto: str,
    epis_preferidos: list[RegistroEpiDB],
    todos_epis: list[RegistroEpiDB],
) -> RegistroEpiDB | None:
    """Seleciona o EPI mais compatível com o relato textual do colaborador."""
    norm = _normalizar_texto(texto)
    candidatos = [*epis_preferidos, *[e for e in todos_epis if e not in epis_preferidos]]
    if not candidatos:
        return None

    for epi in candidatos:
        epi_norm = _normalizar_texto(epi.nome)
        tokens_epi = [tk for tk in epi_norm.split() if len(tk) >= 4 and tk not in {"para", "com", "protecao", "seguranca"}]
        if any(tk in norm for tk in tokens_epi):
            return epi

    for sinonimos, termo_alvo in _PALAVRAS_CHAVE_EPI:
        if any(s in norm for s in sinonimos):
            for epi in candidatos:
                if termo_alvo in _normalizar_texto(epi.nome):
                    return epi

    return candidatos[0]


@runtime_checkable
class RepositorioA2Protocol(Protocol):
    """Contrato único de persistência consumido pelo Agente 2."""

    def obter_perfil_colaborador(
        self, id_colaborador: int
    ) -> PerfilColaboradorDB | None: ...

    def listar_epis_colaborador(
        self, id_colaborador: int
    ) -> list[RegistroEpiDB]: ...

    def resolver_id_epi_por_texto(
        self, texto: str, id_colaborador: int | None = None
    ) -> int: ...

    def abrir_solicitacao_epi(
        self,
        *,
        id_colaborador: int,
        id_epi: int,
        motivo: MotivoSolicitacaoEPI,
        descricao: str | None = None,
    ) -> RegistroSolicitacaoEpiDB: ...

    def obter_solicitacao_epi(
        self, id_solicitacao: int
    ) -> RegistroSolicitacaoEpiDB | None: ...


class RepositorioA2Postgres:
    """Repositório PostgreSQL 16 estritamente restrito às tabelas do Anexo C.3."""

    def __init__(self, database_url: str) -> None:
        if psycopg is None:
            raise RuntimeError(
                "Driver psycopg não encontrado. Instale com: pip install 'psycopg[binary]'"
            )
        self._database_url = database_url

    def _conectar(self) -> Any:
        return psycopg.connect(self._database_url, row_factory=dict_row)

    def obter_perfil_colaborador(
        self, id_colaborador: int
    ) -> PerfilColaboradorDB | None:
        sql_colab = """
            SELECT c.id_colaborador,
                   c.matricula,
                   c.nome,
                   c.id_cargo,
                   cg.nome AS cargo,
                   c.id_setor,
                   s.nome AS setor,
                   s.id_area,
                   c.jornada_inicio,
                   c.jornada_fim,
                   c.ativo
            FROM cadastro.colaborador c
            JOIN cadastro.cargo cg ON cg.id_cargo = c.id_cargo
            JOIN cadastro.setor s  ON s.id_setor = c.id_setor
            WHERE c.id_colaborador = %s
        """
        sql_beneficios = """
            SELECT b.nome
            FROM bem_estar.colaborador_beneficio cb
            JOIN bem_estar.beneficio b ON b.id_beneficio = cb.id_beneficio
            WHERE cb.id_colaborador = %s
              AND (cb.data_fim IS NULL OR cb.data_fim >= CURRENT_DATE)
            ORDER BY b.nome ASC
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql_colab, (id_colaborador,))
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute(sql_beneficios, (id_colaborador,))
            beneficios = [str(r["nome"]) for r in cur.fetchall()]

            return PerfilColaboradorDB(
                id_colaborador=int(row["id_colaborador"]),
                matricula=str(row["matricula"]),
                nome=str(row["nome"]),
                id_cargo=int(row["id_cargo"]),
                cargo=str(row["cargo"]),
                id_setor=int(row["id_setor"]),
                setor=str(row["setor"]),
                id_area=int(row["id_area"]),
                jornada_inicio=_formatar_hora(row["jornada_inicio"]),
                jornada_fim=_formatar_hora(row["jornada_fim"]),
                ativo=bool(row["ativo"]),
                beneficios=beneficios,
            )

    def listar_epis_colaborador(
        self, id_colaborador: int
    ) -> list[RegistroEpiDB]:
        sql = """
            SELECT e.id_epi, e.nome, e.numero_ca, e.validade_ca, e.vida_util_dias
            FROM cadastro.colaborador c
            JOIN apoio.cargo_epi ce ON ce.id_cargo = c.id_cargo
            JOIN apoio.epi e        ON e.id_epi = ce.id_epi
            WHERE c.id_colaborador = %s
            ORDER BY ce.obrigatorio DESC, e.id_epi ASC
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_colaborador,))
            rows = cur.fetchall()
            if not rows:
                cur.execute(
                    "SELECT id_epi, nome, numero_ca, validade_ca, vida_util_dias "
                    "FROM apoio.epi ORDER BY id_epi ASC"
                )
                rows = cur.fetchall()
            return [RegistroEpiDB(**r) for r in rows]

    def resolver_id_epi_por_texto(
        self, texto: str, id_colaborador: int | None = None
    ) -> int:
        preferidos = (
            self.listar_epis_colaborador(id_colaborador)
            if id_colaborador is not None
            else []
        )
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id_epi, nome, numero_ca, validade_ca, vida_util_dias "
                "FROM apoio.epi ORDER BY id_epi ASC"
            )
            todos = [RegistroEpiDB(**r) for r in cur.fetchall()]

        escolhido = _selecionar_epi_por_texto(texto, preferidos, todos)
        if escolhido is None:
            raise ValueError("Nenhum EPI cadastrado em `apoio.epi` para abrir solicitação.")
        return escolhido.id_epi

    def abrir_solicitacao_epi(
        self,
        *,
        id_colaborador: int,
        id_epi: int,
        motivo: MotivoSolicitacaoEPI,
        descricao: str | None = None,
    ) -> RegistroSolicitacaoEpiDB:
        sql = """
            INSERT INTO apoio.solicitacao_epi (
                id_colaborador,
                id_epi,
                motivo,
                descricao,
                status
            )
            VALUES (%s, %s, %s, %s, 'ABERTA')
            RETURNING id_solicitacao,
                      id_colaborador,
                      id_epi,
                      motivo,
                      descricao,
                      status,
                      id_entrega,
                      aberta_em,
                      atendida_em
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_colaborador, id_epi, motivo, descricao))
            row = cur.fetchone()
            conn.commit()
            return RegistroSolicitacaoEpiDB(**row)

    def obter_solicitacao_epi(
        self, id_solicitacao: int
    ) -> RegistroSolicitacaoEpiDB | None:
        sql = """
            SELECT id_solicitacao,
                   id_colaborador,
                   id_epi,
                   motivo,
                   descricao,
                   status,
                   id_entrega,
                   aberta_em,
                   atendida_em
            FROM apoio.solicitacao_epi
            WHERE id_solicitacao = %s
        """
        with self._conectar() as conn, conn.cursor() as cur:
            cur.execute(sql, (id_solicitacao,))
            row = cur.fetchone()
            return RegistroSolicitacaoEpiDB(**row) if row is not None else None


class RepositorioA2Memoria:
    """Repositório em memória para testes unitários, contrato e demonstração offline.

    Vem pré-populado com os colaboradores canônicos (`42` Carla Souza, `57` e `63`),
    catálogo de EPIs e sequência inicial `58` para `apoio.solicitacao_epi` (Anexo D.3).
    """

    def __init__(self, *, popular_padrao: bool = True) -> None:
        self.colaboradores: dict[int, PerfilColaboradorDB] = {}
        self.epis: dict[int, RegistroEpiDB] = {}
        self.cargo_epis: dict[int, list[int]] = {}
        self.solicitacoes_epi: dict[int, RegistroSolicitacaoEpiDB] = {}
        self._seq_solicitacao_epi = 58

        if popular_padrao:
            self._popular_dados_canonicos()

    def _popular_dados_canonicos(self) -> None:
        self.colaboradores = {
            42: PerfilColaboradorDB(
                id_colaborador=42,
                matricula="00123",
                nome="Carla Souza",
                id_cargo=15,
                cargo="Analista de Operações",
                id_setor=7,
                setor="Logística",
                id_area=2,
                jornada_inicio="07:00",
                jornada_fim="16:00",
                ativo=True,
                beneficios=["Vale-Alimentação", "Plano de Saúde", "Auxílio Creche"],
            ),
            57: PerfilColaboradorDB(
                id_colaborador=57,
                matricula="00157",
                nome="Bruno Lima",
                id_cargo=12,
                cargo="Operador de Produção",
                id_setor=4,
                setor="Montagem",
                id_area=1,
                jornada_inicio="08:00",
                jornada_fim="17:00",
                ativo=True,
                beneficios=["Vale-Alimentação", "Gympass"],
            ),
            63: PerfilColaboradorDB(
                id_colaborador=63,
                matricula="00163",
                nome="Mariana Alves",
                id_cargo=18,
                cargo="Técnica de Manutenção",
                id_setor=5,
                setor="Manutenção Industrial",
                id_area=1,
                jornada_inicio="07:00",
                jornada_fim="16:00",
                ativo=True,
                beneficios=["Plano Odontológico", "Vale-Transporte"],
            ),
        }

        self.epis = {
            1: RegistroEpiDB(
                id_epi=1,
                nome="Luva de proteção nitrílica",
                numero_ca="12345",
                validade_ca=date(2028, 6, 30),
                vida_util_dias=90,
            ),
            2: RegistroEpiDB(
                id_epi=2,
                nome="Protetor auricular",
                numero_ca="23456",
                validade_ca=date(2028, 12, 31),
                vida_util_dias=180,
            ),
            3: RegistroEpiDB(
                id_epi=3,
                nome="Óculos de proteção",
                numero_ca="34567",
                validade_ca=date(2029, 3, 15),
                vida_util_dias=365,
            ),
            4: RegistroEpiDB(
                id_epi=4,
                nome="Capacete de segurança",
                numero_ca="45678",
                validade_ca=date(2030, 1, 1),
                vida_util_dias=730,
            ),
            5: RegistroEpiDB(
                id_epi=5,
                nome="Calçado de segurança",
                numero_ca="56789",
                validade_ca=date(2028, 9, 10),
                vida_util_dias=365,
            ),
        }

        self.cargo_epis = {
            15: [1, 2, 3],
            12: [1, 2, 3, 4, 5],
            18: [1, 2, 3, 4, 5],
        }

    def obter_perfil_colaborador(
        self, id_colaborador: int
    ) -> PerfilColaboradorDB | None:
        return self.colaboradores.get(id_colaborador)

    def listar_epis_colaborador(
        self, id_colaborador: int
    ) -> list[RegistroEpiDB]:
        perfil = self.colaboradores.get(id_colaborador)
        if perfil is not None and perfil.id_cargo in self.cargo_epis:
            ids = self.cargo_epis[perfil.id_cargo]
            return [self.epis[i] for i in ids if i in self.epis]
        return list(self.epis.values())

    def resolver_id_epi_por_texto(
        self, texto: str, id_colaborador: int | None = None
    ) -> int:
        preferidos = (
            self.listar_epis_colaborador(id_colaborador)
            if id_colaborador is not None
            else []
        )
        todos = list(self.epis.values())
        escolhido = _selecionar_epi_por_texto(texto, preferidos, todos)
        if escolhido is None:
            raise ValueError("Nenhum EPI cadastrado em memória para abrir solicitação.")
        return escolhido.id_epi

    def abrir_solicitacao_epi(
        self,
        *,
        id_colaborador: int,
        id_epi: int,
        motivo: MotivoSolicitacaoEPI,
        descricao: str | None = None,
    ) -> RegistroSolicitacaoEpiDB:
        if id_colaborador <= 0:
            raise ValueError(f"id_colaborador inválido: {id_colaborador}")
        if id_epi not in self.epis:
            raise ValueError(f"EPI {id_epi} não encontrado no catálogo.")

        id_solicitacao = self._seq_solicitacao_epi
        self._seq_solicitacao_epi += 1

        registro = RegistroSolicitacaoEpiDB(
            id_solicitacao=id_solicitacao,
            id_colaborador=id_colaborador,
            id_epi=id_epi,
            motivo=motivo,
            descricao=descricao,
            status="ABERTA",
            id_entrega=None,
            aberta_em=datetime.now(tz=UTC),
            atendida_em=None,
        )
        self.solicitacoes_epi[id_solicitacao] = registro
        return registro

    def obter_solicitacao_epi(
        self, id_solicitacao: int
    ) -> RegistroSolicitacaoEpiDB | None:
        return self.solicitacoes_epi.get(id_solicitacao)
