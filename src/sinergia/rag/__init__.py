"""Subsistema RAG (Retrieval-Augmented Generation) do SinergIA.

Fornece ingestão atômica com versionamento, limpeza em 4 estágios, chunking híbrido,
embeddings densos Colibri PT-BR (768d), busca híbrida com RRF (k=60) e ferramentas
para os especialistas do CrewAI.
"""

from sinergia.rag.chunking import (
    ChunkDoc,
    chunking_documento_pdf,
    chunking_estrutural,
    chunking_normativo,
    extrair_texto_pdf,
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
    MetadadosTrecho,
    TrechoRecuperado,
)
from sinergia.rag.limpeza import (
    deshifenizar,
    normalizar_espacos_e_controles,
    preservar_marcadores_juridicos,
    remover_cabecalhos_rodapes,
    sanitizar_texto,
)
from sinergia.rag.retrievers import (
    BuscadorHibrido,
    calcular_rrf,
)
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
    "MetadadosTrecho",
    "TrechoRecuperado",
    "ContextoRAG",
    # Embeddings
    "GeradorEmbeddings",
    "get_gerador_embeddings",
    # Limpeza
    "remover_cabecalhos_rodapes",
    "deshifenizar",
    "normalizar_espacos_e_controles",
    "preservar_marcadores_juridicos",
    "sanitizar_texto",
    # Chunking
    "ChunkDoc",
    "extrair_texto_pdf",
    "chunking_normativo",
    "chunking_estrutural",
    "chunking_documento_pdf",
    # Recuperação
    "calcular_rrf",
    "BuscadorHibrido",
    # Ingestão
    "mapear_mock_para_metadados",
    "ingerir_documento",
    "ingerir_diretorio_mocks",
    # Ferramentas CrewAI
    "consultar_base_conhecimento",
    "ferramenta_busca_pausas",
    "ferramenta_busca_direitos",
    "ferramenta_busca_procedimentos",
    "ferramenta_busca_saude_ocupacional",
    "ferramenta_busca_epis",
]

