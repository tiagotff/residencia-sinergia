"""Pipeline de sanitização e limpeza textual para documentos RAG."""

import re
import unicodedata

# Padrões para detecção de paginação e rodapés/cabeçalhos
PADRAO_PAGINACAO = re.compile(
    r"^\s*(?:p[áa]gina|p[áa]g\.?)\s+\d+(?:\s*(?:de|/)\s*\d+)?\s*$|"
    r"^\s*--\s*\d+\s*--\s*$|"
    r"^\s*-\s*\d+\s*-\s*$|"
    r"^\s*\[\s*\d+\s*\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Padrão para des-hifenização de quebras de linha (ex: "confor-\nme" -> "conforme")
PADRAO_HIFEN_QUEBRA = re.compile(
    r"([a-zA-ZÀ-ÿ]+)-\s*[\r\n]+\s*([a-zA-ZÀ-ÿ]+)",
    re.UNICODE,
)

# Padrão para caracteres de controle indesejados (form feed, null bytes, backspaces)
PADRAO_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Tags de marcação HTML injetadas por extratores externos (ex: PyMuPDF4LLM/to_chunks)
PADRAO_TAGS_MARK = re.compile(r"</?mark>", re.IGNORECASE)

# Cabeçalhos ministeriais/institucionais grudados em títulos via sublinhado Markdown
PADRAO_CABECALHO_MINISTERIAL = re.compile(
    r"_+(?:Minist[eé]rio|Secretaria|Departamento|Coordena[çc][ãa]o)[^\n]*",
    re.IGNORECASE,
)

# Tabela de conversão de ligaturas tipográficas comuns em PDFs (preservando º e ª)
TABELA_LIGATURAS = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "ft",
        "\ufb06": "st",
    }
)


def normalizar_ligaturas(texto: str) -> str:
    """Converte ligaturas tipográficas Unicode (ﬁ, ﬂ, etc.) preservando ordinais (º, ª)."""
    if not texto:
        return ""
    return unicodedata.normalize("NFC", texto.translate(TABELA_LIGATURAS))


def limpar_artefatos_marcacao_pdf(texto: str) -> str:
    """Remove tags HTML de marcação (<mark>) e cabeçalhos institucionais grudados."""
    if not texto:
        return ""
    texto = PADRAO_TAGS_MARK.sub("", texto)
    texto = PADRAO_CABECALHO_MINISTERIAL.sub("", texto)
    return texto


def remover_cabecalhos_rodapes(texto: str) -> str:
    """Estágio 1: Remove marcações de paginação, rodapés e cabeçalhos repetitivos."""
    linhas = []
    for linha in texto.splitlines():
        linha_strip = linha.strip()
        # Ignorar linhas puramente de paginação
        if PADRAO_PAGINACAO.match(linha_strip):
            continue
        linhas.append(linha)
    return "\n".join(linhas)


def deshifenizar(texto: str) -> str:
    """Estágio 2: Des-hifeniza palavras quebradas no final de linhas."""
    return PADRAO_HIFEN_QUEBRA.sub(r"\1\2", texto)


def normalizar_espacos_e_controles(texto: str) -> str:
    """Estágio 3: Remove caracteres de controle, desfaz ligaturas e normaliza espaços."""
    # Remover caracteres de controle nulos e form feed
    texto = PADRAO_CONTROLE.sub(" ", texto)

    # Desfazer ligaturas tipográficas (ﬁ -> fi, ﬂ -> fl) mantendo ordinais (13º, 1ª) e NFC
    texto = normalizar_ligaturas(texto)

    # Normalizar quebras de linha Windows para Unix
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")

    # Colapsar múltiplos espaços horizontais em um único
    texto = re.sub(r"[ \t]+", " ", texto)

    # Colapsar 3 ou mais quebras de linha consecutivas em 2
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    # Limpar espaços nas extremidades das linhas
    linhas = [l.strip() for l in texto.split("\n")]
    return "\n".join(linhas).strip()


def preservar_marcadores_juridicos(texto: str) -> str:
    """Estágio 4: Normaliza e preserva a integridade de marcadores sintáticos e jurídicos."""
    # Garantir formato canônico para Artigos: "Art. X"
    texto = re.compile(r"\bArtigo\s+(\d+)", re.IGNORECASE).sub(r"Art. \1", texto)
    texto = re.compile(r"\bArt\.\s*(\d+)").sub(r"Art. \1", texto)

    # Garantir formato canônico para Parágrafos: "§ X"
    texto = re.compile(r"§\s*(\d+)").sub(r"§ \1", texto)

    # Garantir formato canônico para Itens normativos: "Item X"
    texto = re.compile(r"\bItem\s*(\d+)", re.IGNORECASE).sub(r"Item \1", texto)

    return texto


def sanitizar_texto(texto: str) -> str:
    """Executa o pipeline completo de limpeza textual para documentos RAG."""
    if not texto:
        return ""

    t0 = limpar_artefatos_marcacao_pdf(texto)
    t1 = remover_cabecalhos_rodapes(t0)
    t2 = deshifenizar(t1)
    t3 = normalizar_espacos_e_controles(t2)
    t4 = preservar_marcadores_juridicos(t3)
    return t4.strip()
