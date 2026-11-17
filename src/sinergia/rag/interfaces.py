"""Contratos e modelos fortemente tipados em Pydantic para o subsistema RAG."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MetadadosCabecalhoKB(BaseModel):
    """Metadados estruturados extraídos da tabela de governança (página 1) dos PDFs da Base de Conhecimento."""

    titulo: str = Field(default="", description="Título oficial da base de conhecimento")
    codigo: Optional[str] = Field(
        default=None, description="Código identificador do documento (ex: KB-RH-ERG-001)"
    )
    versao: str = Field(default="1.0", description="Versão do documento")
    vigencia: Optional[str] = Field(default=None, description="Data de início de vigência")
    tipo_fonte: Optional[str] = Field(
        default=None, description="Tipo de fonte (ex: INTERNO, OFICIAL)"
    )
    fonte: str = Field(default="", description="Fontes normativas e áreas responsáveis")
    base_rag: Optional[str] = Field(
        default=None, description="Identificação da Base RAG e funcionalidades associadas"
    )
    responsavel: Optional[str] = Field(
        default=None, description="Equipe responsável pela manutenção da base"
    )
    aprovacao: Optional[str] = Field(
        default=None, description="Status e responsável pela aprovação"
    )
    revisao: Optional[str] = Field(
        default=None, description="Datas de revisão atual e próxima"
    )
    convencoes: Optional[str] = Field(
        default=None, description="Convenções de estrutura e precedência do documento"
    )


class MetadadosTrecho(BaseModel):
    """Metadados de procedência, versionamento e localização de um trecho documental."""

    id_documento: Optional[int] = Field(default=None, description="ID do documento no banco")
    titulo: str = Field(..., description="Título oficial do documento ou norma")
    fonte: str = Field(..., description="Origem, ato normativo ou caminho do arquivo original")
    versao: str = Field(..., description="Identificador de versão do documento")
    secao: Optional[str] = Field(default=None, description="Seção, Artigo, Item ou Capítulo")
    pagina: Optional[int] = Field(default=None, description="Número da página no documento original")
    id_base: Optional[int] = Field(default=None, description="ID da base de conhecimento (1 a 5)")
    ordem: Optional[int] = Field(default=None, description="Ordem sequencial do trecho no documento")
    breadcrumb: Optional[str] = Field(
        default=None,
        description="Trilha de navegação hierárquica (ex: Documento > Capítulo > Artigo)",
    )
    vigente: bool = Field(default=True, description="Indicador de vigência ativa do documento")
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadados estruturados flexíveis em JSONB (id_pergunta, tambem_perguntam, fonte_item, coordenadas, etc.)",
    )


class TrechoRecuperado(BaseModel):
    """Representação de um trecho de texto recuperado pelo subsistema RAG."""

    id_trecho: Optional[int] = Field(default=None, description="ID do trecho no banco de dados")
    conteudo: str = Field(..., description="Conteúdo textual do trecho")
    score: float = Field(default=0.0, description="Pontuação de relevância (RRF ou similaridade)")
    metadados: MetadadosTrecho = Field(..., description="Metadados associados ao trecho")


class ContextoRAG(BaseModel):
    """Conjunto de trechos e metadados contextuais recuperados para enriquecer o prompt."""

    query: Optional[str] = Field(default=None, description="Consulta original realizada")
    id_base: Optional[int] = Field(default=None, description="Base de conhecimento consultada")
    trechos: List[TrechoRecuperado] = Field(
        default_factory=list, description="Lista ordenada dos trechos recuperados"
    )
    total_encontrado: int = Field(default=0, description="Total de trechos recuperados")

    def formatar_para_prompt(self) -> str:
        """Formata os trechos em blocos estruturados com citação explícita para agentes CrewAI."""
        if not self.trechos:
            return "Nenhum trecho ou documento encontrado para a consulta."

        blocos = []
        for i, trecho in enumerate(self.trechos, start=1):
            meta = trecho.metadados
            linhas_meta = [f"--- [FONTE {i}]: {meta.titulo} (Versão: {meta.versao}) ---"]
            if meta.breadcrumb:
                linhas_meta.append(f"- Trilha: {meta.breadcrumb}")
            id_pergunta = meta.extra.get("id_pergunta") if meta.extra else None
            if id_pergunta:
                linhas_meta.append(f"- ID Pergunta: {id_pergunta}")
            secao_str = meta.secao if meta.secao else "Não informada"
            linhas_meta.append(f"- Seção/Artigo: {secao_str}")
            pagina_str = f"Página: {meta.pagina}" if meta.pagina is not None else "Página: N/A"
            linhas_meta.append(f"- {pagina_str}")
            linhas_meta.append(f"- Fonte: {meta.fonte}")
            fonte_item = meta.extra.get("fonte_item") if meta.extra else None
            if fonte_item:
                linhas_meta.append(f"- Fonte do Trecho: {fonte_item}")
            linhas_meta.append(f"- Score de Similaridade: {trecho.score:.4f}")
            linhas_meta.append("Conteúdo:")
            linhas_meta.append(trecho.conteudo.strip())

            blocos.append("\n".join(linhas_meta))

        return "\n\n".join(blocos)


class DetalheScoreTrecho(BaseModel):
    """Detalhamento dos scores e posições pré e pós-RRF para inspeção técnica de um trecho."""

    id_trecho: Optional[int] = Field(default=None, description="ID do trecho no banco de dados")
    conteudo: str = Field(..., description="Conteúdo textual do trecho")
    metadados: MetadadosTrecho = Field(..., description="Metadados associados ao trecho")
    rrf_score: float = Field(default=0.0, description="Pontuação final após fusão RRF")
    dense_score: float = Field(default=0.0, description="Similaridade de cosseno no ranking denso")
    dense_rank: Optional[int] = Field(default=None, description="Posição no ranking denso (1-based)")
    lex_score: float = Field(default=0.0, description="Pontuação no ranking léxico (ts_rank ou overlap)")
    lex_rank: Optional[int] = Field(default=None, description="Posição no ranking léxico (1-based)")


class EstatisticasEmbedding(BaseModel):
    """Estatísticas e amostra do vetor de embedding da consulta (768 dimensões)."""

    modelo: str = Field(
        default="tardellirs/colibri-embed-ptbr", description="Nome do modelo de embedding"
    )
    dimensoes: int = Field(default=768, description="Quantidade de dimensões do vetor")
    norma_l2: float = Field(default=1.0, description="Norma L2 do vetor da consulta")
    min_val: float = Field(default=0.0, description="Valor mínimo do vetor")
    max_val: float = Field(default=0.0, description="Valor máximo do vetor")
    mean_val: float = Field(default=0.0, description="Valor médio do vetor")
    amostra: List[float] = Field(
        default_factory=list, description="Primeiras dimensões do vetor (ex: 12 floats)"
    )
    vetor: List[float] = Field(
        default_factory=list, description="Vetor completo de 768 dimensões"
    )


class ResultadoBuscaInstrumentada(BaseModel):
    """Retorno instrumentado da busca híbrida contendo contexto final e telemetria detalhada."""

    contexto: ContextoRAG = Field(..., description="ContextoRAG padrão retornado pela busca")
    embed_ms: float = Field(default=0.0, description="Tempo de geração do embedding em ms")
    db_rrf_ms: float = Field(default=0.0, description="Tempo de busca e fusão RRF em ms")
    estatisticas_embedding: EstatisticasEmbedding = Field(
        ..., description="Estatísticas do embedding 768d da consulta"
    )
    detalhes_trechos: List[DetalheScoreTrecho] = Field(
        default_factory=list, description="Scores e ranks individuais por trecho"
    )


