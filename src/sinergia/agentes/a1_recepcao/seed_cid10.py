"""Script CLI de seed de CID-10 (pt-BR) a partir do pacote oficial do DATASUS.

Baixa (em 1 única requisição HTTP) ou lê localmente o arquivo ``CID10CSV.zip``
do Ministério da Saúde / DATASUS e popula a tabela ``afastamento.cid`` no PostgreSQL
do projeto (ou gera um arquivo ``.sql`` idempotente versionável).

Exemplos de uso:
    python -m sinergia.agentes.a1_recepcao.seed_cid10 --limite 100 --dry-run
    python -m sinergia.agentes.a1_recepcao.seed_cid10 --gerar-sql db/seeds/004_seed_cid10_pt.sql --dry-run
    python -m sinergia.agentes.a1_recepcao.seed_cid10
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from collections.abc import Sequence
from pathlib import Path

import httpx

from sinergia.agentes.a1_recepcao.cliente_cid_who import DadosCidWHO
from sinergia.agentes.a1_recepcao.config import obter_config_a1
from sinergia.agentes.a1_recepcao.repositorio import RepositorioA1Postgres
from sinergia.agentes.a1_recepcao.seed_cid11 import gerar_sql_seed_cids
from sinergia.agentes.a1_recepcao.validador import eh_cid10

URL_DATASUS_CID10_ZIP = "http://www2.datasus.gov.br/cid10/V2008/downloads/CID10CSV.zip"


def _formatar_codigo_cid10(codigo_bruto: str) -> str:
    limpo = codigo_bruto.strip().upper()
    if len(limpo) == 4 and "." not in limpo:
        return f"{limpo[:3]}.{limpo[3:]}"
    return limpo


def _carregar_faixas_capitulos(zf: zipfile.ZipFile) -> list[tuple[str, str, str]]:
    faixas: list[tuple[str, str, str]] = []
    nome_cap = next(
        (n for n in zf.namelist() if n.upper().endswith("CID-10-CAPITULOS.CSV")),
        None,
    )
    if nome_cap is None:
        return faixas

    with zf.open(nome_cap) as raw:
        leitor = csv.DictReader(io.TextIOWrapper(raw, encoding="latin-1"), delimiter=";")
        for linha in leitor:
            numcap = str(linha.get("NUMCAP") or "").strip()
            cat_ini = str(linha.get("CATINIC") or "").strip().upper()[:3]
            cat_fim = str(linha.get("CATFIM") or "").strip().upper()[:3]
            if not (numcap and cat_ini and cat_fim):
                continue
            cap_norm = numcap.zfill(2) if numcap.isdigit() else numcap[:10]
            faixas.append((cat_ini, cat_fim, cap_norm[:10]))
    return faixas


def _resolver_capitulo_cid10(
    codigo_cid: str, faixas: Sequence[tuple[str, str, str]]
) -> str:
    cat3 = codigo_cid[:3].upper()
    for cat_ini, cat_fim, cap in faixas:
        if cat_ini <= cat3 <= cat_fim:
            return cap
    return (cat3[:1] or "N/D")[:10]


def extrair_cids10_de_zip_bytes(
    conteudo_zip: bytes,
    *,
    limite: int | None = None,
) -> list[DadosCidWHO]:
    """Extrai categorias e subcategorias CID-10 em pt-BR a partir de ``CID10CSV.zip``."""
    registros: dict[str, DadosCidWHO] = {}

    with zipfile.ZipFile(io.BytesIO(conteudo_zip)) as zf:
        faixas_capitulos = _carregar_faixas_capitulos(zf)

        def _processar_csv(nome_sufixo: str, coluna_codigo: str) -> None:
            nome_arq = next(
                (n for n in zf.namelist() if n.upper().endswith(nome_sufixo)),
                None,
            )
            if nome_arq is None:
                return
            with zf.open(nome_arq) as raw:
                leitor = csv.DictReader(
                    io.TextIOWrapper(raw, encoding="latin-1"), delimiter=";"
                )
                for linha in leitor:
                    if limite is not None and len(registros) >= limite:
                        break
                    codigo_fmt = _formatar_codigo_cid10(
                        str(linha.get(coluna_codigo) or "")
                    )
                    if not codigo_fmt or len(codigo_fmt) > 8 or not eh_cid10(codigo_fmt):
                        continue
                    descricao = " ".join(
                        str(linha.get("DESCRICAO") or "").split()
                    ).strip()
                    if not descricao:
                        continue
                    capitulo = _resolver_capitulo_cid10(codigo_fmt, faixas_capitulos)
                    registros[codigo_fmt] = DadosCidWHO(
                        codigo_cid=codigo_fmt,
                        nome_doenca=descricao[:255],
                        descricao=descricao,
                        capitulo=capitulo,
                        versao_cid="CID-10",
                    )

        _processar_csv("CID-10-CATEGORIAS.CSV", "CAT")
        if limite is None or len(registros) < limite:
            _processar_csv("CID-10-SUBCATEGORIAS.CSV", "SUBCAT")

    return list(registros.values())


def baixar_zip_datasus_cid10(
    url: str = URL_DATASUS_CID10_ZIP,
    *,
    timeout_segundos: float = 30.0,
    transport: httpx.BaseTransport | None = None,
) -> bytes:
    """Baixa o pacote ``CID10CSV.zip`` do DATASUS em uma única requisição HTTP."""
    with httpx.Client(
        timeout=timeout_segundos,
        follow_redirects=True,
        transport=transport,
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def executar_seed_cid10(
    *,
    url_zip: str = URL_DATASUS_CID10_ZIP,
    arquivo_zip: Path | None = None,
    database_url: str | None = None,
    limite: int | None = None,
    gerar_sql: Path | None = None,
    dry_run: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> tuple[list[DadosCidWHO], int]:
    """Extrai CID-10 (pt-BR) do pacote DATASUS e persiste em ``afastamento.cid``."""
    if arquivo_zip is not None:
        conteudo_zip = arquivo_zip.read_bytes()
    else:
        conteudo_zip = baixar_zip_datasus_cid10(url_zip, transport=transport)

    cids = extrair_cids10_de_zip_bytes(conteudo_zip, limite=limite)

    if gerar_sql is not None:
        gerar_sql_seed_cids(
            cids,
            gerar_sql,
            cabecalho=(
                "-- Seed idempotente de CID-10 (pt-BR) gerada a partir do pacote oficial "
                "CID10CSV.zip do DATASUS."
            ),
        )

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
        description="Popula afastamento.cid com CID-10 (pt-BR) a partir do pacote oficial DATASUS."
    )
    parser.add_argument(
        "--url-zip",
        default=URL_DATASUS_CID10_ZIP,
        help=f"URL do pacote CID10CSV.zip do DATASUS (padrão: {URL_DATASUS_CID10_ZIP}).",
    )
    parser.add_argument(
        "--arquivo-zip",
        type=Path,
        default=None,
        help="Opcional: caminho local de um arquivo CID10CSV.zip já baixado.",
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
    origem = str(args.arquivo_zip) if args.arquivo_zip else args.url_zip
    print(f"[seed_cid10] Carregando CID-10 (pt-BR) de {origem}...")
    cids, inseridos = executar_seed_cid10(
        url_zip=args.url_zip,
        arquivo_zip=args.arquivo_zip,
        database_url=args.database_url,
        limite=args.limite,
        gerar_sql=args.gerar_sql,
        dry_run=args.dry_run,
    )
    print(f"[seed_cid10] Extração concluída: {len(cids)} códigos CID-10 coletados.")
    if args.gerar_sql:
        print(f"[seed_cid10] SQL exportado em: {args.gerar_sql}")
    if args.dry_run:
        print("[seed_cid10] Modo --dry-run ativo: nenhuma gravação feita no banco.")
    else:
        print(f"[seed_cid10] {inseridos} registros gravados/atualizados em afastamento.cid.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
