"""Módulo de chunking estrutural com PyMuPDF, hierarquia H1-H2-H3...Hn, coordenadas e breadcrumbs."""

import os
import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from sinergia.rag.limpeza import sanitizar_texto

try:
    import pymupdf
except ImportError:
    pymupdf = None


class ChunkDoc(BaseModel):
    """Representação de um chunk de texto particionado com metadados estruturais flexíveis."""

    conteudo: str = Field(
        ..., description="Conteúdo textual do chunk incluindo breadcrumb injetado"
    )
    secao: Optional[str] = Field(
        default=None, description="Seção mais específica ou nível hierárquico ativo"
    )
    pagina: int = Field(default=1, description="Número da página de origem")
    ordem: int = Field(default=1, description="Ordem sequencial do chunk no documento")
    breadcrumb: Optional[str] = Field(
        default=None, description="Trilha hierárquica completa do trecho"
    )
    metadados: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadados dinâmicos adicionais (hierarquia H1..Hn, coordenadas/bboxes, token_count, etc.)",
    )


def chunking_estrutural(
    documento: Optional[Union[str, Any]] = None,
    titulo_documento: str = "",
    pagina: int = 1,
    max_chars: int = 800,
    texto: Optional[str] = None,
) -> List[ChunkDoc]:
    """Particiona documentos PDF ou blocos de texto por seções e hierarquia H1-H2-H3...Hn.

    Quando `documento` (ou `texto`) é um caminho de arquivo PDF existente ou objeto
    PyMuPDF Document, utiliza nativamente `doc.to_chunks()` para segmentação
    estrutural da indústria, preservando tabelas em Markdown, coordenadas espaciais
    de cada sentença (bboxes) e construindo metadados de breadcrumb hierárquico dinâmico.

    Caso receba texto puro (`str`), opera em modo de compatibilidade dividindo
    em blocos estruturais/markdown e gerando a trilha de breadcrumbs.
    """
    alvo = documento if documento is not None else texto
    if alvo is None:
        return []

    # 1. Modo nativo PyMuPDF se for um caminho de arquivo PDF existente ou objeto Document
    is_pdf_path = isinstance(alvo, str) and (
        alvo.lower().endswith(".pdf") or os.path.exists(alvo)
    )
    is_pdf_doc = pymupdf is not None and hasattr(alvo, "to_chunks")

    if pymupdf is not None and (is_pdf_path or is_pdf_doc):
        chunks: List[ChunkDoc] = []
        doc = None
        should_close = False
        try:
            if is_pdf_doc:
                doc = alvo
            else:
                doc = pymupdf.open(alvo)
                should_close = True

            # Invocação do chunking nativo do PyMuPDF
            chunks_nativos = doc.to_chunks()

            for chunk_nativo in chunks_nativos:
                texto_chunk = getattr(chunk_nativo, "text", "") or ""
                texto_chunk = texto_chunk.strip()
                if not texto_chunk:
                    continue

                meta = getattr(chunk_nativo, "metadata", None)
                section_path: List[str] = []
                bboxes_brutas = []
                page_start = pagina
                page_end = pagina
                token_count = None
                types_list = []

                if meta is not None:
                    section_path = getattr(meta, "section_path", []) or []
                    bboxes_brutas = getattr(meta, "bboxes", []) or []
                    page_start = getattr(meta, "page_start", pagina)
                    page_end = getattr(meta, "page_end", page_start)
                    token_count = getattr(meta, "token_count", None)
                    types_list = getattr(meta, "types", []) or []

                # Identificar seção mais específica
                secao = section_path[-1].strip() if section_path else "Geral"

                # Montar breadcrumb dinâmico: TituloDocumento > H1 > H2 > ... > Hn > pág. X
                partes_breadcrumb = [titulo_documento] if titulo_documento else []
                for s in section_path:
                    s_limpo = s.strip()
                    if s_limpo and s_limpo != titulo_documento and s_limpo not in partes_breadcrumb:
                        partes_breadcrumb.append(s_limpo)
                partes_breadcrumb.append(f"pág. {page_start}")
                breadcrumb = " > ".join(partes_breadcrumb)

                # Formatar coordenadas estruturadas amigáveis a partir dos bboxes
                coordenadas = []
                for b in bboxes_brutas:
                    if isinstance(b, (list, tuple)) and len(b) >= 5:
                        coordenadas.append(
                            {
                                "pagina": int(b[0]),
                                "x0": round(float(b[1]), 2),
                                "y0": round(float(b[2]), 2),
                                "x1": round(float(b[3]), 2),
                                "y1": round(float(b[4]), 2),
                            }
                        )

                # Mapeamento dinâmico de níveis hierárquicos h1, h2, ..., hn
                secoes_map = {f"h{i+1}": s.strip() for i, s in enumerate(section_path)}

                metadados_dinamicos = {
                    "hierarquia": [s.strip() for s in section_path],
                    "secoes_map": secoes_map,
                    "bboxes": bboxes_brutas,
                    "coordenadas": coordenadas,
                    "page_start": page_start,
                    "page_end": page_end,
                    "token_count": token_count,
                    "types": types_list,
                    "breadcrumb": breadcrumb,
                }

                chunks.append(
                    ChunkDoc(
                        conteudo=texto_chunk,
                        secao=secao,
                        pagina=page_start,
                        breadcrumb=breadcrumb,
                        metadados=metadados_dinamicos,
                    )
                )

            return chunks
        finally:
            if should_close and doc is not None:
                doc.close()

    # 2. Modo compatibilidade para texto puro
    texto_limpo = sanitizar_texto(str(alvo))
    if not texto_limpo:
        return []

    # Quebra inicial por blocos de parágrafos duplos
    blocos = [b.strip() for b in re.split(r"\n\s*\n", texto_limpo) if b.strip()]
    if not blocos:
        blocos = [texto_limpo]

    chunks = []
    secao_atual = "Geral"
    hierarquia_atual: List[str] = []

    for bloco in blocos:
        # Detectar cabeçalho markdown (# H1, ## H2, etc.) ou cabeçalho numerado
        match_md = re.match(r"^(#{1,6})\s+(.+)$", bloco, re.MULTILINE)
        match_titulo = re.match(
            r"^(\d+(?:\.\d+)*\.?\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ\s]{3,60})", bloco
        )

        if match_md:
            nivel = len(match_md.group(1))
            titulo_h = match_md.group(2).strip()
            # Ajustar hierarquia dinâmica
            hierarquia_atual = hierarquia_atual[: nivel - 1] + [titulo_h]
            secao_atual = titulo_h
        elif match_titulo:
            titulo_num = match_titulo.group(1).strip()
            hierarquia_atual = [titulo_num]
            secao_atual = titulo_num

        partes_breadcrumb = [titulo_documento] if titulo_documento else []
        for h in hierarquia_atual:
            if h not in partes_breadcrumb:
                partes_breadcrumb.append(h)
        if not hierarquia_atual and secao_atual != "Geral":
            partes_breadcrumb.append(secao_atual)
        partes_breadcrumb.append(f"pág. {pagina}")
        breadcrumb = " > ".join(partes_breadcrumb)

        secoes_map = {f"h{i+1}": h for i, h in enumerate(hierarquia_atual)}
        metadados_dinamicos = {
            "hierarquia": list(hierarquia_atual),
            "secoes_map": secoes_map,
            "bboxes": [],
            "coordenadas": [],
            "page_start": pagina,
            "page_end": pagina,
            "token_count": None,
            "types": ["text"],
            "breadcrumb": breadcrumb,
        }

        chunks.append(
            ChunkDoc(
                conteudo=bloco,
                secao=secao_atual,
                pagina=pagina,
                breadcrumb=breadcrumb,
                metadados=metadados_dinamicos,
            )
        )

    return chunks


def chunking_documento_pdf(
    caminho_pdf: str,
    titulo_documento: str,
) -> List[ChunkDoc]:
    """Carrega o PDF e executa o particionamento estrutural unificado via PyMuPDF.

    Utiliza nativamente o analisador estrutural do PyMuPDF (`doc.to_chunks()`),
    gerando chunks no padrão de mercado com:
    - Hierarquia dinâmica H1-H2-H3...Hn (secoes_map e hierarquia)
    - Breadcrumbs padronizados injetados no conteúdo e nos metadados
    - Coordenadas espaciais de sentenças/blocos (bboxes e coordenadas normalizadas)
    - Ordem sequencial contínua para o documento completo
    """
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {caminho_pdf}")

    todos_chunks = chunking_estrutural(
        documento=caminho_pdf,
        titulo_documento=titulo_documento,
    )

    # Atribuir ordem sequencial única aos chunks do documento
    for idx, chunk in enumerate(todos_chunks, start=1):
        chunk.ordem = idx

    return todos_chunks
