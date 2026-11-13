"""Utilitários de renderização visual de trechos em PDF a partir de coordenadas BBox via PyMuPDF."""

from pathlib import Path
from typing import Any

import pymupdf
from pydantic import BaseModel, Field

from sinergia.rag.ingestao import MAPA_MOCKS_PADRAO
from sinergia.rag.interfaces import MetadadosTrecho

DIRETORIO_MOCKS_PADRAO = Path(__file__).resolve().parent.parent / "data" / "mocks"


class CoordenadaBBox(BaseModel):
    """Coordenada espacial normalizada (em pontos PDF) de um trecho em uma página."""

    pagina: int = Field(..., ge=1, description="Número da página (1-based)")
    x0: float = Field(..., description="Coordenada esquerda")
    y0: float = Field(..., description="Coordenada superior")
    x1: float = Field(..., description="Coordenada direita")
    y1: float = Field(..., description="Coordenada inferior")


class VisualizacaoTrechoPDF(BaseModel):
    """Artefatos de imagem PNG gerados para o recorte e página inteira de um trecho no PDF."""

    caminho_pdf: str
    nome_arquivo: str
    pagina: int
    total_paginas: int
    bboxes: list[CoordenadaBBox] = Field(default_factory=list)
    crop_png: bytes
    pagina_png: bytes


def resolver_caminho_pdf(
    metadados: MetadadosTrecho,
    diretorio_mocks: Path | None = None,
) -> Path | None:
    """Resolve o caminho do PDF original a partir dos metadados do trecho."""
    pasta = diretorio_mocks or DIRETORIO_MOCKS_PADRAO

    if metadados.fonte:
        candidato = Path(metadados.fonte)
        if candidato.is_file() and candidato.suffix.lower() == ".pdf":
            return candidato
        candidato_rel = pasta / metadados.fonte
        if candidato_rel.is_file() and candidato_rel.suffix.lower() == ".pdf":
            return candidato_rel

    if not pasta.exists():
        return None

    for nome_arquivo, info in MAPA_MOCKS_PADRAO.items():
        if metadados.id_base is not None and info.get("id_base") == metadados.id_base:
            caminho = pasta / nome_arquivo
            if caminho.is_file():
                return caminho

    titulo_alvo = (metadados.titulo or "").strip().lower()
    if titulo_alvo:
        for nome_arquivo, info in MAPA_MOCKS_PADRAO.items():
            titulo_info = str(info.get("titulo") or "").strip().lower()
            if titulo_info and (titulo_info in titulo_alvo or titulo_alvo in titulo_info):
                caminho = pasta / nome_arquivo
                if caminho.is_file():
                    return caminho

    return None


def _converter_para_bbox(item: Any, pagina_padrao: int) -> CoordenadaBBox | None:
    """Converte um dicionário ou sequência de coordenadas em CoordenadaBBox válida."""
    if isinstance(item, dict):
        try:
            pag = int(item.get("pagina") or pagina_padrao)
            x0 = float(item["x0"])
            y0 = float(item["y0"])
            x1 = float(item["x1"])
            y1 = float(item["y1"])
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(item, (list, tuple)) and len(item) >= 5:
        try:
            pag = int(item[0])
            x0 = float(item[1])
            y0 = float(item[2])
            x1 = float(item[3])
            y1 = float(item[4])
        except (TypeError, ValueError):
            return None
    else:
        return None

    if pag < 1 or x1 <= x0 or y1 <= y0:
        return None
    return CoordenadaBBox(pagina=pag, x0=x0, y0=y0, x1=x1, y1=y1)


def extrair_bboxes_trecho(metadados: MetadadosTrecho) -> list[CoordenadaBBox]:
    """Extrai coordenadas BBox estruturadas a partir de metadados.extra."""
    extra = metadados.extra or {}
    pagina_padrao = metadados.pagina if metadados.pagina and metadados.pagina >= 1 else 1

    brutos = extra.get("coordenadas") or extra.get("bboxes") or []
    resultado: list[CoordenadaBBox] = []
    for item in brutos:
        bbox = _converter_para_bbox(item, pagina_padrao)
        if bbox is not None:
            resultado.append(bbox)

    if not resultado and extra.get("coordenadas") and extra.get("bboxes"):
        for item in extra.get("bboxes", []):
            bbox = _converter_para_bbox(item, pagina_padrao)
            if bbox is not None:
                resultado.append(bbox)

    return resultado


def renderizar_bboxes_em_pdf_bytes(
    caminho_pdf: str,
    pagina: int,
    bboxes_tuplas: tuple[tuple[float, float, float, float], ...],
    margem_vertical: float = 55.0,
    dpi_crop: int = 200,
    dpi_pagina: int = 150,
) -> tuple[bytes, bytes, int]:
    """Renderiza o recorte focado e a página inteira destacada em PNG via PyMuPDF."""
    doc = pymupdf.open(caminho_pdf)
    try:
        total_paginas = len(doc)
        if total_paginas == 0:
            raise ValueError(f"O PDF não possui páginas: {caminho_pdf}")

        idx_pag = max(0, min(total_paginas - 1, pagina - 1))
        page = doc[idx_pag]
        page_rect = page.rect

        rects_validos: list[pymupdf.Rect] = []
        for x0, y0, x1, y1 in bboxes_tuplas:
            rect = pymupdf.Rect(x0, y0, x1, y1) & page_rect
            if not rect.is_empty and rect.width > 0 and rect.height > 0:
                page.draw_rect(
                    rect,
                    color=(0.85, 0.35, 0.05),
                    fill=(1.0, 0.88, 0.20),
                    fill_opacity=0.32,
                    width=1.6,
                )
                rects_validos.append(rect)

        if rects_validos:
            min_y0 = min(r.y0 for r in rects_validos)
            max_y1 = max(r.y1 for r in rects_validos)
            clip_rect = pymupdf.Rect(
                page_rect.x0,
                max(page_rect.y0, min_y0 - margem_vertical),
                page_rect.x1,
                min(page_rect.y1, max_y1 + margem_vertical),
            )
        else:
            clip_rect = pymupdf.Rect(
                page_rect.x0,
                page_rect.y0,
                page_rect.x1,
                min(page_rect.y1, page_rect.y0 + page_rect.height * 0.45),
            )

        crop_png = page.get_pixmap(clip=clip_rect, dpi=dpi_crop).tobytes("png")
        pagina_png = page.get_pixmap(dpi=dpi_pagina).tobytes("png")
        return crop_png, pagina_png, total_paginas
    finally:
        doc.close()


def renderizar_visualizacao_trecho(
    metadados: MetadadosTrecho,
    diretorio_mocks: Path | None = None,
) -> VisualizacaoTrechoPDF | None:
    """Resolve o PDF, aplica highlight nas BBoxes do trecho e gera os PNGs de visualização."""
    caminho_pdf = resolver_caminho_pdf(metadados, diretorio_mocks=diretorio_mocks)
    if caminho_pdf is None or not caminho_pdf.is_file():
        return None

    bboxes = extrair_bboxes_trecho(metadados)
    pagina_alvo = metadados.pagina or (bboxes[0].pagina if bboxes else 1)
    bboxes_pagina = [b for b in bboxes if b.pagina == pagina_alvo] or bboxes
    bboxes_tuplas = tuple((b.x0, b.y0, b.x1, b.y1) for b in bboxes_pagina)

    crop_png, pagina_png, total_paginas = renderizar_bboxes_em_pdf_bytes(
        caminho_pdf=str(caminho_pdf),
        pagina=pagina_alvo,
        bboxes_tuplas=bboxes_tuplas,
    )

    return VisualizacaoTrechoPDF(
        caminho_pdf=str(caminho_pdf),
        nome_arquivo=caminho_pdf.name,
        pagina=pagina_alvo,
        total_paginas=total_paginas,
        bboxes=bboxes_pagina,
        crop_png=crop_png,
        pagina_png=pagina_png,
    )
