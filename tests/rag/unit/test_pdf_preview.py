"""Testes unitários para visualização de trechos no PDF via BBox."""

from pathlib import Path

from sinergia.rag.chunking import chunking_documento_pdf
from sinergia.rag.ingestao import MAPA_MOCKS_PADRAO
from sinergia.rag.interfaces import MetadadosTrecho
from sinergia.rag.pitch.pdf_preview import (
    CoordenadaBBox,
    extrair_bboxes_trecho,
    renderizar_visualizacao_trecho,
    resolver_caminho_pdf,
)

DIRETORIO_MOCKS = (
    Path(__file__).resolve().parents[3] / "src" / "sinergia" / "rag" / "data" / "mocks"
)


def test_resolver_caminho_pdf_todas_bases() -> None:
    """Garante que todas as bases F1..F5 resolvem para o respectivo arquivo PDF mock."""
    for nome_arquivo, info in MAPA_MOCKS_PADRAO.items():
        meta = MetadadosTrecho(
            titulo=info["titulo"],
            fonte=info["fonte"],
            versao=info["versao"],
            id_base=info["id_base"],
            pagina=1,
        )
        caminho = resolver_caminho_pdf(meta, diretorio_mocks=DIRETORIO_MOCKS)
        assert caminho is not None
        assert caminho.name == nome_arquivo
        assert caminho.is_file()


def test_extrair_bboxes_trecho_coordenadas_e_bboxes() -> None:
    """Valida a extração de BBoxes tanto de 'coordenadas' quanto de 'bboxes' brutos."""
    meta_coords = MetadadosTrecho(
        titulo="NR-17",
        fonte="NR-17",
        versao="2024.1",
        pagina=2,
        extra={
            "coordenadas": [
                {"pagina": 2, "x0": 72.0, "y0": 150.5, "x1": 510.2, "y1": 210.8},
                {"pagina": 2, "x0": 100.0, "y0": 200.0, "x1": 90.0, "y1": 220.0},
            ]
        },
    )
    bboxes = extrair_bboxes_trecho(meta_coords)
    assert bboxes == [
        CoordenadaBBox(pagina=2, x0=72.0, y0=150.5, x1=510.2, y1=210.8),
    ]

    meta_brutas = MetadadosTrecho(
        titulo="NR-17",
        fonte="NR-17",
        versao="2024.1",
        pagina=1,
        extra={"bboxes": [(1, 80.0, 120.0, 490.0, 180.0)]},
    )
    bboxes_brutas = extrair_bboxes_trecho(meta_brutas)
    assert bboxes_brutas == [
        CoordenadaBBox(pagina=1, x0=80.0, y0=120.0, x1=490.0, y1=180.0),
    ]


def test_renderizar_visualizacao_trecho_pdf_real() -> None:
    """Renderiza recorte e página inteira de um chunk real da NR-17 com assinatura PNG válida."""
    pdf_nr17 = DIRETORIO_MOCKS / "base1_nr17_ergonomia.pdf"
    chunks = chunking_documento_pdf(str(pdf_nr17), "NR-17 Ergonomia")
    assert chunks, "Esperava chunks extraídos da NR-17"

    chunk_com_bbox = next(
        (c for c in chunks if c.metadados.get("coordenadas")),
        chunks[0],
    )
    meta = MetadadosTrecho(
        titulo="NR-17 Ergonomia",
        fonte="Portaria MTP nº 423",
        versao="2024.1",
        pagina=chunk_com_bbox.pagina,
        id_base=1,
        extra=chunk_com_bbox.metadados,
    )

    vis = renderizar_visualizacao_trecho(meta, diretorio_mocks=DIRETORIO_MOCKS)
    assert vis is not None
    assert vis.nome_arquivo == "base1_nr17_ergonomia.pdf"
    assert vis.pagina >= 1
    assert vis.total_paginas >= 1
    assert vis.bboxes
    assert vis.crop_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert vis.pagina_png.startswith(b"\x89PNG\r\n\x1a\n")
