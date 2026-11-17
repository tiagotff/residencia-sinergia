"""Subsistema RAG (Retrieval-Augmented Generation) do SinergIA.

Fornece ingestão atômica com versionamento, sanitização textual, chunking estrutural
fiel ao layout das Bases de Conhecimento (com limite de caracteres e overlap),
embeddings densos Colibri PT-BR (768d), busca híbrida com RRF (k=60) e ferramentas
para os especialistas do CrewAI.
"""

from sinergia.rag.chunking import (
    MAX_CHARS_PADRAO,
    OVERLAP_PADRAO,
    ChunkDoc,
    chunking_documento_pdf,
    chunking_estrutural,
    dividir_texto_com_overlap,
    extrair_metadados_cabecalho_pdf,
)
from sinergia.rag.embeddings import (
    GeradorEmbeddings,
    get_gerador_embeddings,
)
from sinergia.rag.ingestao import (
    ingerir_diretorio_mocks,
    ingerir_documento,
    mapear_mock_para_metadados,
)
from sinergia.rag.interfaces import (
    ContextoRAG,
    MetadadosCabecalhoKB,
    MetadadosTrecho,
    TrechoRecuperado,
)
from sinergia.rag.limpeza import (
    deshifenizar,
    limpar_artefatos_marcacao_pdf,
    normalizar_espacos_e_controles,
    normalizar_ligaturas,
    preservar_marcadores_juridicos,
    remover_cabecalhos_rodapes,
    sanitizar_texto,
)
from sinergia.rag.retrievers import (
    BuscadorHibrido,
    calcular_rrf,
)
from sinergia.rag.cli import modo_interativo
from sinergia.rag.tools import (
    consultar_base_conhecimento,
    ferramenta_busca_direitos,
    ferramenta_busca_epis,
    ferramenta_busca_pausas,
    ferramenta_busca_procedimentos,
    ferramenta_busca_saude_ocupacional,
)

__all__ = [
    # Interfaces
    "MetadadosCabecalhoKB",
    "MetadadosTrecho",
    "TrechoRecuperado",
    "ContextoRAG",
    # Embeddings
    "GeradorEmbeddings",
    "get_gerador_embeddings",
    # Limpeza
    "normalizar_ligaturas",
    "limpar_artefatos_marcacao_pdf",
    "remover_cabecalhos_rodapes",
    "deshifenizar",
    "normalizar_espacos_e_controles",
    "preservar_marcadores_juridicos",
    "sanitizar_texto",
    # Chunking
    "MAX_CHARS_PADRAO",
    "OVERLAP_PADRAO",
    "ChunkDoc",
    "dividir_texto_com_overlap",
    "extrair_metadados_cabecalho_pdf",
    "chunking_estrutural",
    "chunking_documento_pdf",
    # Recuperação
    "calcular_rrf",
    "BuscadorHibrido",
    # Ingestão
    "mapear_mock_para_metadados",
    "ingerir_documento",
    "ingerir_diretorio_mocks",
    # Ferramentas CrewAI e CLI
    "consultar_base_conhecimento",
    "ferramenta_busca_pausas",
    "ferramenta_busca_direitos",
    "ferramenta_busca_procedimentos",
    "ferramenta_busca_saude_ocupacional",
    "ferramenta_busca_epis",
    "modo_interativo",
]
