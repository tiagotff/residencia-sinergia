"""Módulo de chunking estrutural para layout de Bases de Conhecimento SinergIA, com limite de caracteres e overlap."""

from contextlib import contextmanager
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from sinergia.rag.interfaces import MetadadosCabecalhoKB
from sinergia.rag.limpeza import normalizar_ligaturas, sanitizar_texto

try:
    import pymupdf
except ImportError:
    pymupdf = None

MAX_CHARS_PADRAO = 700
OVERLAP_PADRAO = 120

# Padrões estruturais do layout das Bases de Conhecimento em data/rag/mocks
PADRAO_SECAO_H2 = re.compile(r"^\d+\.\s+\S+")
PADRAO_PERGUNTA_KB = re.compile(r"^([A-Z0-9]+(?:-[A-Z0-9]+)*-Q\d{2,4})\s*\|\s*(.+)$")
PADRAO_TAMBEM_PERGUNTAM = re.compile(r"^tamb[ée]m\s+perguntam\s*:\s*", re.IGNORECASE)
PADRAO_FONTE_ITEM = re.compile(r"^fonte\s*:\s*", re.IGNORECASE)
PADRAO_ITEM_NUMERADO = re.compile(r"^(\d{1,2})\.\s+")

CHAVES_CABECALHO_KB: Dict[str, str] = {
    "título": "titulo",
    "titulo": "titulo",
    "código": "codigo",
    "codigo": "codigo",
    "versão": "versao",
    "versao": "versao",
    "vigência": "vigencia",
    "vigencia": "vigencia",
    "tipo de fonte": "tipo_fonte",
    "fonte": "fonte",
    "base rag": "base_rag",
    "responsável": "responsavel",
    "responsavel": "responsavel",
    "aprovação": "aprovacao",
    "aprovacao": "aprovacao",
    "revisão": "revisao",
    "revisao": "revisao",
    "convenções": "convencoes",
    "convencoes": "convencoes",
}


class ChunkDoc(BaseModel):
    """Representação de um chunk de texto particionado com metadados estruturais flexíveis."""

    conteudo: str = Field(
        ..., description="Conteúdo textual limpo do chunk (respeitando max_chars e overlap)"
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
        description="Metadados estruturados (id_pergunta, tambem_perguntam, fonte_item, coordenadas, etc.)",
    )


@contextmanager
def _abrir_documento_pdf(documento: Union[str, Any]) -> Iterator[Any]:
    """Abre um arquivo PDF via PyMuPDF ou reutiliza uma instância Document aberta."""
    if pymupdf is None:
        raise RuntimeError("Biblioteca 'pymupdf' não está instalada no ambiente.")

    if hasattr(documento, "page_count") and hasattr(documento, "load_page"):
        yield documento
        return

    doc = pymupdf.open(documento)
    try:
        yield doc
    finally:
        doc.close()


def _extrair_linhas_pagina(page: Any, numero_pagina: int) -> List[Dict[str, Any]]:
    """Extrai linhas ordenadas de uma página PyMuPDF com tamanho, flags tipográficas e bbox."""
    linhas_extraidas: List[Dict[str, Any]] = []
    blocos = page.get_text("dict").get("blocks", [])
    for bloco in blocos:
        if bloco.get("type") != 0:
            continue
        for linha in bloco.get("lines", []):
            spans = linha.get("spans", [])
            if not spans:
                continue
            texto_bruto = "".join(s.get("text", "") for s in spans)
            texto_norm = normalizar_ligaturas(texto_bruto)
            texto_limpo = re.sub(r"[ \t]+", " ", texto_norm).strip()
            if not texto_limpo:
                continue

            span_ref = next((s for s in spans if s.get("text", "").strip()), spans[0])
            size = float(span_ref.get("size", 10.0))
            flags = int(span_ref.get("flags", 0))
            bbox = linha.get("bbox", (0.0, 0.0, 0.0, 0.0))
            linhas_extraidas.append(
                {
                    "pagina": numero_pagina,
                    "texto": texto_limpo,
                    "size": size,
                    "flags": flags,
                    "is_bold": bool(flags & 16),
                    "is_italic": bool(flags & 2),
                    "x0": float(bbox[0]),
                    "y0": float(bbox[1]),
                    "x1": float(bbox[2]),
                    "y1": float(bbox[3]),
                }
            )
    return linhas_extraidas


def _montar_breadcrumb(
    titulo_documento: str,
    hierarquia: List[str],
    pagina: int,
    secao_fallback: Optional[str] = None,
) -> str:
    """Constrói a trilha de breadcrumb sem duplicações consecutivas."""
    partes: List[str] = [titulo_documento.strip()] if titulo_documento.strip() else []
    for item in hierarquia:
        item_limpo = item.strip()
        if item_limpo and item_limpo not in partes:
            partes.append(item_limpo)
    if not hierarquia and secao_fallback and secao_fallback != "Geral" and secao_fallback not in partes:
        partes.append(secao_fallback)
    partes.append(f"pág. {pagina}")
    return " > ".join(partes)


def _consolidar_coordenadas_por_pagina(
    bboxes_linhas: List[Tuple[int, float, float, float, float]],
) -> Tuple[List[Tuple[int, float, float, float, float]], List[Dict[str, Any]]]:
    """Agrupa bboxes de linhas em um retângulo envolvente por página."""
    por_pagina: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for pag, x0, y0, x1, y1 in bboxes_linhas:
        if x1 <= x0 or y1 <= y0:
            continue
        por_pagina.setdefault(pag, []).append((x0, y0, x1, y1))

    bboxes_consolidadas: List[Tuple[int, float, float, float, float]] = []
    coordenadas: List[Dict[str, Any]] = []
    for pag in sorted(por_pagina.keys()):
        rects = por_pagina[pag]
        mx0 = round(min(r[0] for r in rects), 2)
        my0 = round(min(r[1] for r in rects), 2)
        mx1 = round(max(r[2] for r in rects), 2)
        my1 = round(max(r[3] for r in rects), 2)
        bboxes_consolidadas.append((pag, mx0, my0, mx1, my1))
        coordenadas.append(
            {
                "pagina": pag,
                "x0": mx0,
                "y0": my0,
                "x1": mx1,
                "y1": my1,
            }
        )
    return bboxes_consolidadas, coordenadas


def _juntar_linha_texto(acumulado: str, nova_linha: str) -> str:
    """Concatena uma linha de continuação tratando hifenização de quebra de linha."""
    if not acumulado:
        return nova_linha
    if re.search(r"[a-zA-ZÀ-ÿ]-$", acumulado) and re.match(r"^[a-zà-ÿ]", nova_linha):
        return f"{acumulado[:-1]}{nova_linha}"
    return f"{acumulado} {nova_linha}"


def _escolher_ponto_corte(texto: str, inicio: int, fim_max: int) -> int:
    """Escolhe o melhor índice de corte até `fim_max` priorizando quebras naturais."""
    min_corte = inicio + max(1, (fim_max - inicio) // 3)
    janela = texto[inicio:fim_max]

    # 1. Quebra de parágrafo ou item de lista
    for sep in ("\n• ", "\n"):
        pos = janela.rfind(sep, min_corte - inicio)
        if pos > 0:
            return inicio + pos

    # 2. Fim de sentença
    for sep in (". ", "? ", "! "):
        pos = janela.rfind(sep, min_corte - inicio)
        if pos > 0:
            return inicio + pos + 1

    # 3. Pontuação intermediária
    for sep in ("; ", ", "):
        pos = janela.rfind(sep, min_corte - inicio)
        if pos > 0:
            return inicio + pos + 1

    # 4. Limite de palavra
    pos_espaco = janela.rfind(" ", min_corte - inicio)
    if pos_espaco > 0:
        return inicio + pos_espaco

    return fim_max


def _alinhar_inicio_overlap(texto: str, inicio_anterior: int, corte: int, overlap: int) -> int:
    """Calcula o início da próxima janela garantindo avanço e alinhamento em início de palavra."""
    if overlap <= 0:
        proximo = corte
    else:
        alvo = max(inicio_anterior + 1, corte - overlap)
        proximo = alvo
        # Avança até o próximo início de palavra se cair no meio de uma palavra
        if proximo > 0 and proximo < corte and not texto[proximo - 1].isspace() and not texto[proximo].isspace():
            prox_espaco = texto.find(" ", proximo, corte)
            prox_quebra = texto.find("\n", proximo, corte)
            candidatos = [p for p in (prox_espaco, prox_quebra) if p != -1]
            if candidatos:
                proximo = min(candidatos) + 1
            else:
                proximo = alvo

    while proximo < len(texto) and texto[proximo].isspace():
        proximo += 1

    if proximo <= inicio_anterior:
        proximo = min(len(texto), max(inicio_anterior + 1, corte))
        while proximo < len(texto) and texto[proximo].isspace():
            proximo += 1

    return proximo


def dividir_texto_com_overlap(
    texto: str,
    max_chars: int = MAX_CHARS_PADRAO,
    overlap: int = OVERLAP_PADRAO,
) -> List[str]:
    """Divide um texto em segmentos de até `max_chars` caracteres com sobreposição (`overlap`)."""
    if max_chars <= 0:
        raise ValueError("O parâmetro 'max_chars' deve ser maior que zero.")

    texto_limpo = texto.strip()
    if not texto_limpo:
        return []
    if len(texto_limpo) <= max_chars:
        return [texto_limpo]

    overlap_efetivo = max(0, overlap)
    if overlap_efetivo >= max_chars:
        overlap_efetivo = max(0, max_chars // 4)

    segmentos: List[str] = []
    inicio = 0
    total = len(texto_limpo)

    while inicio < total:
        fim_max = min(total, inicio + max_chars)
        if fim_max >= total:
            cauda = texto_limpo[inicio:].strip()
            if cauda:
                segmentos.append(cauda)
            break

        corte = _escolher_ponto_corte(texto_limpo, inicio, fim_max)
        segmento = texto_limpo[inicio:corte].strip()
        if segmento:
            segmentos.append(segmento)

        inicio = _alinhar_inicio_overlap(texto_limpo, inicio, corte, overlap_efetivo)

    return segmentos


def extrair_metadados_cabecalho_pdf(
    documento: Union[str, Any],
) -> Optional[MetadadosCabecalhoKB]:
    """Extrai os metadados estruturados da tabela de governança da página 1 no layout KB."""
    if pymupdf is None:
        return None
    if isinstance(documento, str) and not os.path.exists(documento):
        return None

    with _abrir_documento_pdf(documento) as doc:
        if len(doc) == 0:
            return None
        linhas_p1 = _extrair_linhas_pagina(doc[0], 1)

    titulo_h1_partes: List[str] = []
    campos: Dict[str, str] = {}
    chave_atual: Optional[str] = None

    for linha in linhas_p1:
        txt = linha["texto"]
        size = linha["size"]
        is_bold = linha["is_bold"]

        # Encerra leitura do cabeçalho ao atingir a primeira seção H2 ou primeira pergunta
        if (size >= 13.0 and is_bold and PADRAO_SECAO_H2.match(txt)) or PADRAO_PERGUNTA_KB.match(txt):
            break

        if size >= 16.0 and is_bold:
            titulo_h1_partes.append(txt)
            continue

        if size <= 9.8:
            chave_norm = txt.lower().rstrip(":")
            if is_bold and linha["x0"] < 140.0 and chave_norm in CHAVES_CABECALHO_KB:
                chave_atual = CHAVES_CABECALHO_KB[chave_norm]
                campos.setdefault(chave_atual, "")
                continue

            if chave_atual is not None:
                campos[chave_atual] = _juntar_linha_texto(campos[chave_atual], txt)

    if len(campos) < 2:
        return None

    if not campos.get("titulo") and titulo_h1_partes:
        campos["titulo"] = " ".join(titulo_h1_partes).strip()

    return MetadadosCabecalhoKB(**campos)


def _classificar_natureza_fonte(fonte_item: str) -> List[str]:
    """Extrai os marcadores de natureza normativa da fonte (OFICIAL / INTERNO)."""
    naturezas: List[str] = []
    texto_upper = fonte_item.upper()
    if "(OFICIAL)" in texto_upper:
        naturezas.append("OFICIAL")
    if "(INTERNO)" in texto_upper:
        naturezas.append("INTERNO")
    return naturezas


def _materializar_chunks_qa(
    bloco_qa: Dict[str, Any],
    titulo_documento: str,
    codigo_documento: Optional[str],
    max_chars: int,
    overlap: int,
) -> List[ChunkDoc]:
    """Converte uma unidade Q&A extraída em um ou mais ChunkDocs respeitando max_chars e overlap."""
    id_pergunta = bloco_qa["id_pergunta"]
    pergunta = bloco_qa["pergunta"]
    secao_h2 = bloco_qa["secao"]
    tambem_perguntam: List[str] = bloco_qa["tambem_perguntam"]
    corpo_itens: List[str] = bloco_qa["corpo_itens"]
    fonte_item = bloco_qa["fonte_item"]
    page_start = bloco_qa["page_start"]
    page_end = bloco_qa["page_end"]

    titulo_pergunta = f"{id_pergunta} | {pergunta}"
    linha_tp = f"Também perguntam: {' | '.join(tambem_perguntam)}" if tambem_perguntam else ""
    corpo = "\n".join(item for item in corpo_itens if item.strip()).strip()
    linha_fonte = f"Fonte: {fonte_item}" if fonte_item else ""

    partes_completas = [titulo_pergunta]
    if linha_tp:
        partes_completas.append(linha_tp)
    if corpo:
        partes_completas.append(corpo)
    if linha_fonte:
        partes_completas.append(linha_fonte)

    texto_completo = "\n".join(partes_completas).strip()
    if not texto_completo:
        return []

    if len(texto_completo) <= max_chars:
        textos_partes = [texto_completo]
    else:
        prefixo_ancora = f"{titulo_pergunta}\n"
        sufixo_ancora = f"\n{linha_fonte}" if linha_fonte else ""
        miolo_partes = [p for p in (linha_tp, corpo) if p]
        miolo = "\n".join(miolo_partes).strip()
        orcamento_miolo = max_chars - len(prefixo_ancora) - len(sufixo_ancora)

        # Preserva título da pergunta e fonte em cada parte quando o orçamento comportar
        if miolo and orcamento_miolo >= max(80, max_chars // 3):
            sub_miolos = dividir_texto_com_overlap(
                miolo,
                max_chars=orcamento_miolo,
                overlap=min(overlap, max(0, orcamento_miolo - 1)),
            )
            textos_partes = [f"{prefixo_ancora}{sub}{sufixo_ancora}" for sub in sub_miolos]
        else:
            textos_partes = dividir_texto_com_overlap(
                texto_completo,
                max_chars=max_chars,
                overlap=overlap,
            )

    bboxes_consolidadas, coordenadas = _consolidar_coordenadas_por_pagina(bloco_qa["bboxes"])
    hierarquia = [secao_h2, titulo_pergunta] if secao_h2 and secao_h2 != "Geral" else [titulo_pergunta]
    breadcrumb = _montar_breadcrumb(titulo_documento, hierarquia, page_start, secao_fallback=titulo_pergunta)
    natureza_fonte = _classificar_natureza_fonte(fonte_item)
    total_partes = len(textos_partes)

    chunks_gerados: List[ChunkDoc] = []
    for idx_parte, texto_parte in enumerate(textos_partes, start=1):
        secoes_map = {f"h{i + 1}": item for i, item in enumerate(hierarquia)}
        metadados: Dict[str, Any] = {
            "secao": titulo_pergunta,
            "hierarquia": list(hierarquia),
            "secoes_map": secoes_map,
            "bboxes": list(bboxes_consolidadas),
            "coordenadas": list(coordenadas),
            "page_start": page_start,
            "page_end": page_end,
            "breadcrumb": breadcrumb,
            "id_pergunta": id_pergunta,
            "pergunta": pergunta,
            "tambem_perguntam": list(tambem_perguntam),
            "fonte_item": fonte_item,
            "natureza_fonte": natureza_fonte,
            "parte_chunk": idx_parte,
            "total_partes": total_partes,
        }
        if codigo_documento:
            metadados["codigo_documento"] = codigo_documento

        chunks_gerados.append(
            ChunkDoc(
                conteudo=texto_parte,
                secao=titulo_pergunta,
                pagina=page_start,
                breadcrumb=breadcrumb,
                metadados=metadados,
            )
        )

    return chunks_gerados


def _finalizar_bloco_qa(bloco_bruto: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza campos acumulados de uma unidade Q&A antes de materializar chunks."""
    tp_texto = " ".join(bloco_bruto["tambem_perguntam_partes"]).strip()
    tambem_perguntam = [v.strip() for v in tp_texto.split("|") if v.strip()] if tp_texto else []
    fonte_item = " ".join(bloco_bruto["fonte_partes"]).strip()
    return {
        "id_pergunta": bloco_bruto["id_pergunta"],
        "pergunta": bloco_bruto["pergunta"],
        "secao": bloco_bruto["secao"],
        "tambem_perguntam": tambem_perguntam,
        "corpo_itens": bloco_bruto["corpo_itens"],
        "fonte_item": fonte_item,
        "page_start": bloco_bruto["page_start"],
        "page_end": bloco_bruto["page_end"],
        "bboxes": bloco_bruto["bboxes"],
    }


def _extrair_chunks_layout_kb_pdf(
    doc: Any,
    titulo_documento: str,
    codigo_documento: Optional[str],
    max_chars: int,
    overlap: int,
) -> List[ChunkDoc]:
    """Executa o parser específico para o layout KB (Seções H2 + Unidades Q&A) via PyMuPDF."""
    blocos_qa: List[Dict[str, Any]] = []
    secao_atual = "Geral"
    bloco_atual: Optional[Dict[str, Any]] = None
    modo_linha: Optional[str] = None
    em_cabecalho_inicial = True
    marcador_lista_pendente: Optional[str] = None
    ultima_linha_corpo: Optional[Dict[str, Any]] = None

    for pno in range(1, len(doc) + 1):
        linhas = _extrair_linhas_pagina(doc[pno - 1], pno)
        for linha in linhas:
            txt = linha["texto"]
            size = linha["size"]
            is_bold = linha["is_bold"]
            is_italic = linha["is_italic"]
            bbox_tuple = (pno, linha["x0"], linha["y0"], linha["x1"], linha["y1"])

            is_h2_inicio = size >= 13.0 and is_bold and bool(PADRAO_SECAO_H2.match(txt))
            match_q = PADRAO_PERGUNTA_KB.match(txt)
            is_q_inicio = bool(match_q) and size >= 10.0 and is_bold

            if em_cabecalho_inicial:
                if is_h2_inicio or is_q_inicio:
                    em_cabecalho_inicial = False
                else:
                    continue

            if is_h2_inicio:
                if bloco_atual is not None:
                    blocos_qa.append(_finalizar_bloco_qa(bloco_atual))
                    bloco_atual = None
                secao_atual = txt
                modo_linha = "secao_h2"
                marcador_lista_pendente = None
                ultima_linha_corpo = None
                continue

            if modo_linha == "secao_h2" and size >= 13.0 and is_bold and not is_q_inicio:
                secao_atual = f"{secao_atual} {txt}".strip()
                continue

            if is_q_inicio and match_q is not None:
                if bloco_atual is not None:
                    blocos_qa.append(_finalizar_bloco_qa(bloco_atual))
                bloco_atual = {
                    "id_pergunta": match_q.group(1).strip(),
                    "pergunta": match_q.group(2).strip(),
                    "secao": secao_atual,
                    "margem_base": linha["x0"],
                    "tambem_perguntam_partes": [],
                    "corpo_itens": [],
                    "fonte_partes": [],
                    "page_start": pno,
                    "page_end": pno,
                    "bboxes": [bbox_tuple],
                }
                modo_linha = "titulo_pergunta"
                marcador_lista_pendente = None
                ultima_linha_corpo = None
                continue

            if bloco_atual is None:
                continue

            bloco_atual["page_end"] = pno
            bloco_atual["bboxes"].append(bbox_tuple)

            if size < 9.5 and PADRAO_TAMBEM_PERGUNTAM.match(txt):
                modo_linha = "tambem_perguntam"
                conteudo_tp = PADRAO_TAMBEM_PERGUNTAM.sub("", txt).strip()
                if conteudo_tp:
                    bloco_atual["tambem_perguntam_partes"].append(conteudo_tp)
                continue

            if size < 9.5 and is_bold and PADRAO_FONTE_ITEM.match(txt):
                modo_linha = "fonte"
                conteudo_fonte = PADRAO_FONTE_ITEM.sub("", txt).strip()
                if conteudo_fonte:
                    bloco_atual["fonte_partes"].append(conteudo_fonte)
                continue

            if modo_linha == "titulo_pergunta" and size >= 10.2 and is_bold:
                bloco_atual["pergunta"] = _juntar_linha_texto(bloco_atual["pergunta"], txt)
                continue

            if modo_linha == "tambem_perguntam" and size < 9.5 and is_italic:
                bloco_atual["tambem_perguntam_partes"].append(txt)
                continue

            if modo_linha == "fonte" and size < 9.5 and is_bold:
                bloco_atual["fonte_partes"].append(txt)
                continue

            # Linha pertence ao corpo da resposta
            modo_linha = "corpo"
            if size <= 9.8 and linha["x0"] < 82.0 and (txt == "•" or re.match(r"^\d{1,2}$", txt)):
                marcador_lista_pendente = "•" if txt == "•" else f"{txt}."
                continue

            corpo_itens: List[str] = bloco_atual["corpo_itens"]
            if marcador_lista_pendente is not None:
                corpo_itens.append(f"{marcador_lista_pendente} {txt}")
                marcador_lista_pendente = None
            elif txt.startswith("•") or PADRAO_ITEM_NUMERADO.match(txt):
                corpo_itens.append(re.sub(r"\s+", " ", txt))
            elif not corpo_itens:
                corpo_itens.append(txt)
            else:
                item_anterior = corpo_itens[-1]
                eh_item_lista = item_anterior.startswith("•") or bool(PADRAO_ITEM_NUMERADO.match(item_anterior))
                margem_base = bloco_atual["margem_base"]
                voltou_margem_base = linha["x0"] <= margem_base + 2.0
                salto_vertical_grande = (
                    ultima_linha_corpo is not None
                    and ultima_linha_corpo["pagina"] == pno
                    and (linha["y0"] - ultima_linha_corpo["y0"]) >= 16.5
                    and voltou_margem_base
                )

                if (eh_item_lista and voltou_margem_base) or salto_vertical_grande:
                    corpo_itens.append(txt)
                else:
                    corpo_itens[-1] = _juntar_linha_texto(item_anterior, txt)

            ultima_linha_corpo = linha

    if bloco_atual is not None:
        blocos_qa.append(_finalizar_bloco_qa(bloco_atual))

    if not blocos_qa:
        return []

    chunks: List[ChunkDoc] = []
    for bloco_qa in blocos_qa:
        chunks.extend(
            _materializar_chunks_qa(
                bloco_qa=bloco_qa,
                titulo_documento=titulo_documento,
                codigo_documento=codigo_documento,
                max_chars=max_chars,
                overlap=overlap,
            )
        )
    return chunks


def _chunking_fallback_pdf(
    doc: Any,
    titulo_documento: str,
    pagina_inicial: int,
    max_chars: int,
    overlap: int,
) -> List[ChunkDoc]:
    """Modo de compatibilidade para PDFs genéricos fora do layout KB, aplicando max_chars e overlap."""
    chunks: List[ChunkDoc] = []
    for idx_pag in range(len(doc)):
        pno = idx_pag + pagina_inicial
        linhas = _extrair_linhas_pagina(doc[idx_pag], pno)
        texto_pag = sanitizar_texto("\n".join(l["texto"] for l in linhas))
        if not texto_pag:
            continue

        bboxes_pagina = [(pno, l["x0"], l["y0"], l["x1"], l["y1"]) for l in linhas]
        bboxes_cons, coordenadas = _consolidar_coordenadas_por_pagina(bboxes_pagina)

        sub_chunks = chunking_estrutural(
            documento=texto_pag,
            titulo_documento=titulo_documento,
            pagina=pno,
            max_chars=max_chars,
            overlap=overlap,
        )
        for ch in sub_chunks:
            ch.metadados["bboxes"] = list(bboxes_cons)
            ch.metadados["coordenadas"] = list(coordenadas)
            chunks.append(ch)
    return chunks


def _chunking_fallback_texto(
    texto: str,
    titulo_documento: str,
    pagina: int,
    max_chars: int,
    overlap: int,
) -> List[ChunkDoc]:
    """Modo de compatibilidade para strings/Markdown com limite de caracteres e overlap."""
    texto_limpo = sanitizar_texto(texto)
    if not texto_limpo:
        return []

    blocos = [b.strip() for b in re.split(r"\n\s*\n", texto_limpo) if b.strip()]
    if not blocos:
        blocos = [texto_limpo]

    chunks: List[ChunkDoc] = []
    secao_atual = "Geral"
    hierarquia_atual: List[str] = []

    for bloco in blocos:
        match_md = re.match(r"^(#{1,6})\s+(.+)$", bloco, re.MULTILINE)
        match_titulo = re.match(
            r"^(\d+(?:\.\d+)*\.?\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ\s]{3,60})",
            bloco,
        )

        if match_md:
            nivel = len(match_md.group(1))
            titulo_h = match_md.group(2).strip()
            hierarquia_atual = hierarquia_atual[: nivel - 1] + [titulo_h]
            secao_atual = titulo_h
        elif match_titulo:
            titulo_num = match_titulo.group(1).strip()
            hierarquia_atual = [titulo_num]
            secao_atual = titulo_num

        breadcrumb = _montar_breadcrumb(
            titulo_documento, hierarquia_atual, pagina, secao_fallback=secao_atual
        )
        secoes_map = {f"h{i + 1}": h for i, h in enumerate(hierarquia_atual)}

        segmentos = dividir_texto_com_overlap(bloco, max_chars=max_chars, overlap=overlap)
        total_partes = len(segmentos)

        for idx_parte, seg in enumerate(segmentos, start=1):
            metadados_dinamicos = {
                "secao": secao_atual,
                "hierarquia": list(hierarquia_atual),
                "secoes_map": secoes_map,
                "bboxes": [],
                "coordenadas": [],
                "page_start": pagina,
                "page_end": pagina,
                "types": ["text"],
                "breadcrumb": breadcrumb,
                "parte_chunk": idx_parte,
                "total_partes": total_partes,
            }
            chunks.append(
                ChunkDoc(
                    conteudo=seg,
                    secao=secao_atual,
                    pagina=pagina,
                    breadcrumb=breadcrumb,
                    metadados=metadados_dinamicos,
                )
            )

    return chunks


def chunking_estrutural(
    documento: Optional[Union[str, Any]] = None,
    titulo_documento: str = "",
    pagina: int = 1,
    max_chars: int = MAX_CHARS_PADRAO,
    overlap: int = OVERLAP_PADRAO,
    texto: Optional[str] = None,
) -> List[ChunkDoc]:
    """Particiona documentos PDF no layout KB SinergIA ou texto livre com max_chars e overlap."""
    alvo = documento if documento is not None else texto
    if alvo is None:
        return []

    is_pdf_path = isinstance(alvo, str) and (
        alvo.lower().endswith(".pdf") or os.path.exists(alvo)
    )
    is_pdf_doc = pymupdf is not None and hasattr(alvo, "page_count")

    if pymupdf is not None and (is_pdf_path or is_pdf_doc):
        with _abrir_documento_pdf(alvo) as doc:
            cabecalho = extrair_metadados_cabecalho_pdf(doc)
            titulo_efetivo = titulo_documento.strip() or (cabecalho.titulo if cabecalho else "")
            codigo_doc = cabecalho.codigo if cabecalho else None

            chunks_kb = _extrair_chunks_layout_kb_pdf(
                doc=doc,
                titulo_documento=titulo_efetivo,
                codigo_documento=codigo_doc,
                max_chars=max_chars,
                overlap=overlap,
            )
            if chunks_kb:
                return chunks_kb

            return _chunking_fallback_pdf(
                doc=doc,
                titulo_documento=titulo_efetivo,
                pagina_inicial=pagina,
                max_chars=max_chars,
                overlap=overlap,
            )

    return _chunking_fallback_texto(
        texto=str(alvo),
        titulo_documento=titulo_documento,
        pagina=pagina,
        max_chars=max_chars,
        overlap=overlap,
    )


def chunking_documento_pdf(
    caminho_pdf: str,
    titulo_documento: str = "",
    max_chars: int = MAX_CHARS_PADRAO,
    overlap: int = OVERLAP_PADRAO,
) -> List[ChunkDoc]:
    """Carrega o PDF e executa o particionamento no layout KB com limite de caracteres e overlap."""
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {caminho_pdf}")

    todos_chunks = chunking_estrutural(
        documento=caminho_pdf,
        titulo_documento=titulo_documento,
        max_chars=max_chars,
        overlap=overlap,
    )

    for idx, chunk in enumerate(todos_chunks, start=1):
        chunk.ordem = idx

    return todos_chunks
