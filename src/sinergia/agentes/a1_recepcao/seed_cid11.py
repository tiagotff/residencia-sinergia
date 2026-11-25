"""Script CLI de seed de CID-11 (pt-BR) a partir do container Docker ``whoicd/icd-api``.

Extrai os códigos CID-11 diretamente da instância local do container oficial da OMS
(sem OAuth2 e sem rate limit) e popula a tabela ``afastamento.cid`` no PostgreSQL
do projeto (ou gera um arquivo ``.sql`` idempotente versionável).

Exemplos de uso:
    python -m sinergia.agentes.a1_recepcao.seed_cid11 --limite 100 --dry-run
    python -m sinergia.agentes.a1_recepcao.seed_cid11
    python -m sinergia.agentes.a1_recepcao.seed_cid11 --gerar-sql db/seeds/003_seed_cid11_pt.sql
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sinergia.agentes.a1_recepcao.cliente_cid_who import ClienteCidWHO, DadosCidWHO
from sinergia.agentes.a1_recepcao.config import obter_config_a1
from sinergia.agentes.a1_recepcao.repositorio import RepositorioA1Postgres


def _escapar_sql(texto: str) -> str:
    return texto.replace("'", "''")


def gerar_sql_seed_cids(
    registros: Sequence[DadosCidWHO],
    destino: Path,
    *,
    cabecalho: str | None = None,
) -> Path:
    """Exporta os CIDs extraídos como script SQL idempotente (INSERT ... ON CONFLICT)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    comentario_topo = (
        cabecalho.strip()
        if cabecalho and cabecalho.strip()
        else "-- Seed idempotente de CID-11 (pt-BR) gerada a partir do container local whoicd/icd-api."
    )
    linhas = [
        comentario_topo,
        "-- Seguro para executar múltiplas vezes.",
        "",
    ]
    if registros:
        linhas.append(
            "INSERT INTO afastamento.cid (codigo_cid, nome_doenca, descricao, capitulo)"
        )
        linhas.append("VALUES")
        valores = []
        for r in registros:
            nome = (r.nome_doenca or r.descricao).strip() or r.descricao
            valores.append(
                "    ("
                f"'{_escapar_sql(r.codigo_cid)}', "
                f"'{_escapar_sql(nome)}', "
                f"'{_escapar_sql(r.descricao)}', "
                f"'{_escapar_sql(r.capitulo)}'"
                ")"
            )
        linhas.append(",\n".join(valores))
        linhas.append("ON CONFLICT (codigo_cid) DO UPDATE")
        linhas.append(
            "SET nome_doenca = EXCLUDED.nome_doenca, "
            "descricao = EXCLUDED.descricao, "
            "capitulo = EXCLUDED.capitulo;"
        )
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return destino


def executar_seed_cid11(
    *,
    seed_url: str,
    release_11: str = "2026-01",
    linearization: str = "mms",
    language: str = "pt",
    database_url: str | None = None,
    limite: int | None = None,
    gerar_sql: Path | None = None,
    dry_run: bool = False,
    cliente: ClienteCidWHO | None = None,
) -> tuple[list[DadosCidWHO], int]:
    """Extrai CIDs do container Docker ``whoicd/icd-api`` e persiste em ``afastamento.cid``."""
    who_client = cliente or ClienteCidWHO(
        base_url=seed_url,
        release_11=release_11,
        linearization=linearization,
        language=language,
        requer_oauth=False,
    )

    def _log_progresso(total_cids: int, nos_visitados: int) -> None:
        if total_cids % 250 == 0:
            print(f"[seed_cid11] {total_cids} CIDs coletados ({nos_visitados} nós visitados)...")

    cids = who_client.percorrer_arvore_icd11(
        limite=limite,
        progresso_cb=_log_progresso,
    )

    if gerar_sql is not None:
        gerar_sql_seed_cids(cids, gerar_sql)

    inseridos = 0
    if not dry_run and database_url:
        repo = RepositorioA1Postgres(database_url)
        tuplas = [
            (c.codigo_cid, c.nome_doenca or c.descricao, c.descricao, c.capitulo)
            for c in cids
        ]
        inseridos = repo.salvar_cids_em_lote(tuplas)

    return cids, inseridos


def construir_parser() -> argparse.ArgumentParser:
    cfg = obter_config_a1()
    parser = argparse.ArgumentParser(
        description="Popula afastamento.cid com CID-11 (pt-BR) a partir do container Docker whoicd/icd-api."
    )
    parser.add_argument(
        "--seed-url",
        default=cfg.who_icd_seed_url,
        help=f"URL base do container Docker whoicd/icd-api (padrão: {cfg.who_icd_seed_url}).",
    )
    parser.add_argument(
        "--release",
        default=cfg.who_icd_release_11,
        help=f"Release ICD-11 carregado no container (padrão: {cfg.who_icd_release_11}).",
    )
    parser.add_argument(
        "--linearization",
        default=cfg.who_icd_linearization,
        help=f"Linearização ICD-11 (padrão: {cfg.who_icd_linearization}).",
    )
    parser.add_argument(
        "--lang",
        default=cfg.who_icd_language,
        help=f"Idioma Accept-Language (padrão: {cfg.who_icd_language}).",
    )
    parser.add_argument(
        "--database-url",
        default=cfg.database_url,
        help="URL de conexão do PostgreSQL do projeto.",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Limita a quantidade máxima de CIDs extraídos (útil para testes rápidos).",
    )
    parser.add_argument(
        "--gerar-sql",
        type=Path,
        default=None,
        help="Opcional: grava também um arquivo .sql idempotente com todos os CIDs extraídos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas extrai e mostra o resumo sem gravar no PostgreSQL.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    print(
        f"[seed_cid11] Conectando ao container {args.seed_url} "
        f"(release={args.release}, lang={args.lang})..."
    )
    cids, inseridos = executar_seed_cid11(
        seed_url=args.seed_url,
        release_11=args.release,
        linearization=args.linearization,
        language=args.lang,
        database_url=args.database_url,
        limite=args.limite,
        gerar_sql=args.gerar_sql,
        dry_run=args.dry_run,
    )
    print(f"[seed_cid11] Extração concluída: {len(cids)} códigos CID-11 coletados.")
    if args.gerar_sql:
        print(f"[seed_cid11] SQL exportado em: {args.gerar_sql}")
    if args.dry_run:
        print("[seed_cid11] Modo --dry-run ativo: nenhuma gravação feita no banco.")
    else:
        print(f"[seed_cid11] {inseridos} registros gravados/atualizados em afastamento.cid.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
