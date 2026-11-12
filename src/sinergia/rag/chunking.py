"""Módulo de chunking híbrido (normativo e estrutural) com injeção de metadados e breadcrumbs."""

import os
import re
import zlib
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from sinergia.rag.limpeza import sanitizar_texto

try:
    import pymupdf
except ImportError:
    pymupdf = None


class ChunkDoc(BaseModel):
    """Representação de um chunk de texto particionado com metadados estruturais."""

    conteudo: str = Field(..., description="Conteúdo textual do chunk incluindo breadcrumb injetado")
    secao: Optional[str] = Field(default=None, description="Seção, Artigo ou Item normativo")
    pagina: int = Field(default=1, description="Número da página de origem")
    ordem: int = Field(default=1, description="Ordem sequencial do chunk no documento")
    breadcrumb: Optional[str] = Field(
        default=None, description="Trilha hierárquica completa do trecho"
    )


def extrair_texto_pdf(caminho_pdf: str) -> List[Tuple[int, str]]:
    """Extrai texto e número de página de um arquivo PDF via PyMuPDF ou fallback nativo."""
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {caminho_pdf}")

    if pymupdf is not None:
        with pymupdf.open(caminho_pdf) as doc:
            paginas = []
            for i, pagina in enumerate(doc, start=1):
                texto = pagina.get_text()
                paginas.append((i, texto))
            return paginas

    # Fallback nativo: descompressão de streams PDF via zlib
    with open(caminho_pdf, "rb") as f:
        conteudo = f.read()

    paginas = []
    pos = 0
    page_num = 1
    while True:
        s_idx = conteudo.find(b"stream\n", pos)
        if s_idx == -1:
            s_idx = conteudo.find(b"stream\r\n", pos)
            if s_idx == -1:
                break
            offset = len(b"stream\r\n")
        else:
            offset = len(b"stream\n")

        e_idx = conteudo.find(b"endstream", s_idx)
        if e_idx == -1:
            break

        dados_comprimidos = conteudo[s_idx + offset : e_idx].rstrip(b"\r\n")
        try:
            dados_decomp = zlib.decompress(dados_comprimidos)
            # Extrair sequências de texto em Tj ou TJ
            padrao_texto = re.findall(rb"\((.*?)\)", dados_decomp)
            partes = [p.decode("latin1", errors="ignore") for p in padrao_texto if len(p) > 0]
            texto_extraido = " ".join(partes)
            if len(texto_extraido.strip()) > 20:
                paginas.append((page_num, texto_extraido))
                page_num += 1
        except Exception:
            pass

        pos = e_idx + len(b"endstream")

    if not paginas:
        # Se nenhum stream válido for encontrado, retornar ao menos uma página vazia
        paginas.append((1, ""))

    return paginas


def chunking_normativo(
    texto: str,
    titulo_documento: str,
    pagina: int = 1,
) -> List[ChunkDoc]:
    """Particiona textos normativos preservando artigos/itens e suas exceções/parágrafos."""
    texto_limpo = sanitizar_texto(texto)
    if not texto_limpo:
        return []

    # Regex para capturar divisões normativas: Artigos, Itens, Capítulos
    padrao_divisor = re.compile(
        r"(?=(?:\n|^)(?:Art\.\s*\d+[ºo°]?|Artigo\s*\d+[ºo°]?|Item\s*\d+(?:\.\d+)*|CAPÍTULO\s+[IVXLCDM\d]+[^\n]*))",
        re.IGNORECASE,
    )

    segmentos = [s.strip() for s in padrao_divisor.split(texto_limpo) if s.strip()]
    if not segmentos:
        segmentos = [texto_limpo]

    chunks = []
    capitulo_atual = ""

    for seg in segmentos:
        # Verificar se é cabeçalho de capítulo
        match_cap = re.match(r"^(CAPÍTULO\s+[IVXLCDM\d]+[^\n]*)", seg, re.IGNORECASE)
        if match_cap:
            capitulo_atual = match_cap.group(1).strip()
            # Se o segmento contém mais texto após o capítulo, continuará a processar
            linhas = seg.split("\n", 1)
            if len(linhas) > 1 and len(linhas[1].strip()) > 0:
                seg = linhas[1].strip()
            else:
                continue

        # Identificar seção (Artigo ou Item)
        match_art = re.search(r"\b(Art\.\s*\d+[ºo°]?|Artigo\s*\d+[ºo°]?)", seg, re.IGNORECASE)
        match_item = re.search(r"\b(Item\s*\d+(?:\.\d+)*)", seg, re.IGNORECASE)

        if match_art:
            secao = match_art.group(1).strip()
        elif match_item:
            secao = match_item.group(1).strip()
        elif capitulo_atual:
            secao = capitulo_atual
        else:
            secao = "Disposições Gerais"

        partes_breadcrumb = [titulo_documento]
        if capitulo_atual and capitulo_atual != secao:
            partes_breadcrumb.append(capitulo_atual)
        partes_breadcrumb.append(secao)
        partes_breadcrumb.append(f"pág. {pagina}")
        breadcrumb = " > ".join(partes_breadcrumb)

        conteudo_com_breadcrumb = f"[{breadcrumb}]\n{seg}"
        chunks.append(
            ChunkDoc(
                conteudo=conteudo_com_breadcrumb,
                secao=secao,
                pagina=pagina,
                breadcrumb=breadcrumb,
            )
        )

    return chunks


def chunking_estrutural(
    texto: str,
    titulo_documento: str,
    pagina: int = 1,
    max_chars: int = 800,
) -> List[ChunkDoc]:
    """Particiona manuais, POPs e procedimentos por seções e blocos lógicos."""
    texto_limpo = sanitizar_texto(texto)
    if not texto_limpo:
        return []

    # Quebra inicial por blocos de parágrafos duplos
    blocos = [b.strip() for b in re.split(r"\n\s*\n", texto_limpo) if b.strip()]
    if not blocos:
        blocos = [texto_limpo]

    chunks = []
    secao_atual = "Geral"

    for bloco in blocos:
        # Detectar possível cabeçalho de seção (ex: "1. OBJETIVO", "2. PROCEDIMENTOS")
        match_titulo = re.match(r"^(\d+\.\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ\s]{3,40})", bloco)
        if match_titulo:
            secao_atual = match_titulo.group(1).strip()

        breadcrumb = f"{titulo_documento} > {secao_atual} > pág. {pagina}"
        conteudo_com_breadcrumb = f"[{breadcrumb}]\n{bloco}"

        chunks.append(
            ChunkDoc(
                conteudo=conteudo_com_breadcrumb,
                secao=secao_atual,
                pagina=pagina,
                breadcrumb=breadcrumb,
            )
        )

    return chunks


def chunking_documento_pdf(
    caminho_pdf: str,
    titulo_documento: str,
) -> List[ChunkDoc]:
    """Carrega o PDF, extrai as páginas e executa chunking híbrido com injeção de metadados."""
    paginas = extrair_texto_pdf(caminho_pdf)
    todos_chunks: List[ChunkDoc] = []

    for num_pagina, texto_pagina in paginas:
        if not texto_pagina or not texto_pagina.strip():
            continue

        # Identificar se o texto possui marcadores normativos
        tem_artigo = bool(re.search(r"\b(?:Art\.|Artigo)\s*\d+", texto_pagina, re.IGNORECASE))
        tem_item = bool(re.search(r"\bItem\s*\d+", texto_pagina, re.IGNORECASE))

        if tem_artigo or tem_item:
            chunks_pagina = chunking_normativo(
                texto=texto_pagina,
                titulo_documento=titulo_documento,
                pagina=num_pagina,
            )
        else:
            chunks_pagina = chunking_estrutural(
                texto=texto_pagina,
                titulo_documento=titulo_documento,
                pagina=num_pagina,
            )

        todos_chunks.extend(chunks_pagina)

    # Atribuir ordem sequencial única aos chunks do documento
    for idx, chunk in enumerate(todos_chunks, start=1):
        chunk.ordem = idx

    return todos_chunks

