"""Modelos de telemetria por turno e agregadores da Visão Global para o demo Streamlit."""


from pydantic import BaseModel, Field

from sinergia.rag.interfaces import DetalheScoreTrecho, EstatisticasEmbedding
from sinergia.rag.pitch.ollama_service import (
    MAPA_BASES_FERRAMENTAS,
    DecisaoRoteamento,
    MetricasOllama,
)


class TemposTurno(BaseModel):
    """Tempos desmembrados de execução de um turno completo (Roteamento + RAG + LLM)."""

    total_ms: float = Field(default=0.0, description="Tempo total ponta a ponta em ms")
    roteamento_ms: float = Field(default=0.0, description="Tempo de roteamento da base em ms")
    embed_ms: float = Field(default=0.0, description="Tempo de geração de embedding 768d em ms")
    db_rrf_ms: float = Field(default=0.0, description="Tempo de busca e fusão RRF em ms")
    llm_ms: float = Field(default=0.0, description="Tempo de inferência da resposta na LLM em ms")

    @property
    def rag_ms(self) -> float:
        """Latência combinada da etapa RAG (embedding + busca/RRF)."""
        return self.embed_ms + self.db_rrf_ms


class TelemetriaTurno(BaseModel):
    """Registro completo de telemetria e Raio-X Nerd para uma resposta da assistente."""

    id_turno: int = Field(..., description="Identificador único da mensagem da assistente")
    pergunta: str = Field(..., description="Pergunta original do usuário")
    resposta: str = Field(..., description="Resposta gerada pela LLM")
    reasoning: str = Field(default="", description="Raciocínio emitido pelo modelo durante a geração")
    citacoes: list[str] = Field(default_factory=list, description="Pílulas de citação")
    roteamento: DecisaoRoteamento = Field(..., description="Decisão autônoma de ferramenta/base")
    tempos: TemposTurno = Field(..., description="Métricas de latência por etapa")
    estatisticas_embedding: EstatisticasEmbedding = Field(
        ..., description="Estatísticas do vetor de consulta 768d"
    )
    detalhes_trechos: list[DetalheScoreTrecho] = Field(
        default_factory=list, description="Trechos recuperados com scores denso, léxico e RRF"
    )
    metricas_ollama: MetricasOllama = Field(..., description="Métricas nativas de tokens/s do Ollama")
    prompt_enviado: str = Field(default="", description="Prompt completo enviado ao Ollama")


class DistribuicaoBase(BaseModel):
    """Contagem e percentual de acionamento para uma base de conhecimento (F1..F5)."""

    id_base: int = Field(..., ge=1, le=5)
    codigo: str = Field(...)
    nome: str = Field(...)
    contagem: int = Field(default=0, ge=0)
    percentual: float = Field(default=0.0, ge=0.0, le=100.0)


class MetricasVisaoGlobal(BaseModel):
    """Métricas agregadas da sessão exibidas na Visão Global do painel direito."""

    total_consultas: int = Field(default=0)
    tempo_medio_total_ms: float = Field(default=0.0)
    tempo_medio_total_s: float = Field(default=0.0)
    latencia_media_rag_ms: float = Field(default=0.0)
    latencia_media_llm_ms: float = Field(default=0.0)
    latencia_media_llm_s: float = Field(default=0.0)
    velocidade_media_tok_s: float = Field(default=0.0)
    total_tokens_gerados: int = Field(default=0)
    total_prompt_tokens: int = Field(default=0)
    distribuicao_bases: list[DistribuicaoBase] = Field(default_factory=list)


def calcular_visao_global(turnos: list[TelemetriaTurno]) -> MetricasVisaoGlobal:
    """Agrega métricas médias de latência, tokens e distribuição F1..F5 da sessão."""
    total = len(turnos)
    contagens_por_base = {id_base: 0 for id_base in MAPA_BASES_FERRAMENTAS}

    for turno in turnos:
        id_b = turno.roteamento.id_base
        if id_b in contagens_por_base:
            contagens_por_base[id_b] += 1

    distribuicao = [
        DistribuicaoBase(
            id_base=id_base,
            codigo=info["codigo"],
            nome=info["nome"],
            contagem=contagens_por_base[id_base],
            percentual=round((contagens_por_base[id_base] / total) * 100.0, 1) if total else 0.0,
        )
        for id_base, info in sorted(MAPA_BASES_FERRAMENTAS.items())
    ]

    if total == 0:
        return MetricasVisaoGlobal(distribuicao_bases=distribuicao)

    soma_total_ms = sum(t.tempos.total_ms for t in turnos)
    soma_rag_ms = sum(t.tempos.rag_ms for t in turnos)
    soma_llm_ms = sum(t.tempos.llm_ms for t in turnos)

    velocidades_validas = [
        t.metricas_ollama.tokens_por_segundo
        for t in turnos
        if t.metricas_ollama.tokens_por_segundo > 0
    ]
    vel_media = (
        sum(velocidades_validas) / len(velocidades_validas) if velocidades_validas else 0.0
    )

    total_tokens_gerados = sum(t.metricas_ollama.eval_count for t in turnos)
    total_prompt_tokens = sum(t.metricas_ollama.prompt_eval_count for t in turnos)

    tempo_medio_total_ms = soma_total_ms / total
    latencia_media_llm_ms = soma_llm_ms / total

    return MetricasVisaoGlobal(
        total_consultas=total,
        tempo_medio_total_ms=round(tempo_medio_total_ms, 2),
        tempo_medio_total_s=round(tempo_medio_total_ms / 1000.0, 2),
        latencia_media_rag_ms=round(soma_rag_ms / total, 1),
        latencia_media_llm_ms=round(latencia_media_llm_ms, 2),
        latencia_media_llm_s=round(latencia_media_llm_ms / 1000.0, 2),
        velocidade_media_tok_s=round(vel_media, 1),
        total_tokens_gerados=total_tokens_gerados,
        total_prompt_tokens=total_prompt_tokens,
        distribuicao_bases=distribuicao,
    )
