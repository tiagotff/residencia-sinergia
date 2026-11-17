"""Contratos e modelos fortemente tipados em Pydantic para o subsistema RAG."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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
    # TODO: Analisar quais outros campos estruturados são importantes como metadados de primeira classe:
    # Ex: tipo_norma (Portaria, CLT, NR), orgao_emissor, data_publicacao, vigencia_inicio, vigencia_fim.
    # FIXME: Reduzir redundância do campo 'extra' JSONB - evitar armazenar dados repetidos
    # como 'secoes_map' vs 'hierarquia' e bboxes duplicadas.
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadados adicionais flexíveis persistidos em JSONB (hierarquia, bboxes, coordenadas, etc.)",
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
            secao_str = meta.secao if meta.secao else "Não informada"
            linhas_meta.append(f"- Seção/Artigo: {secao_str}")
            pagina_str = f"Página: {meta.pagina}" if meta.pagina is not None else "Página: N/A"
            linhas_meta.append(f"- {pagina_str}")
            linhas_meta.append(f"- Fonte: {meta.fonte}")
            linhas_meta.append(f"- Score de Similaridade: {trecho.score:.4f}")
            linhas_meta.append("Conteúdo:")
            linhas_meta.append(trecho.conteudo.strip())

            blocos.append("\n".join(linhas_meta))

        return "\n\n".join(blocos)

