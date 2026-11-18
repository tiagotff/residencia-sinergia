"""Contratos Pydantic e JSON Schema do Agente 2 (A2 — Bem-Estar).

Implementa o protocolo `sinergia/1` e os payloads das 4 ações oficiais do A2
conforme definido na `Documentação - SinergIA v2.2.docx` (Seção 5.1 e Anexos A,
C.3 e D.3) e nas tabelas de `db/schema/001_schema_inicial.sql`:

- `a2.responder`
- `a2.gerar_lembretes`
- `a2.recomendar_beneficios`
- `a2.orientar_epi`
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROTOCOLO_SINERGIA = "sinergia/1"

AcaoA2 = Literal[
    "a2.responder",
    "a2.gerar_lembretes",
    "a2.recomendar_beneficios",
    "a2.orientar_epi",
]

StatusEnvelopeResultado = Literal["OK", "PRECISA_HUMANO", "ERRO"]
MotivoSolicitacaoEPI = Literal["AVARIA", "PERDA", "NAO_RECEBIDO", "TAMANHO_INADEQUADO"]
StatusSolicitacaoEPI = Literal["ABERTA", "EM_ATENDIMENTO", "ATENDIDA", "CANCELADA"]

MAPA_ID_BASE_PARA_SLUG: dict[int, str] = {
    1: "pausas",
    2: "direitos_beneficios",
    3: "procedimentos",
    4: "saude_ocupacional",
    5: "epi",
}

MAPA_INTENCAO_PARA_ID_BASE: dict[str, int] = {
    "F1": 1,
    "F2": 2,
    "F3": 3,
    "F4": 4,
    "F5": 5,
    "F6": 5,
    "PAUSAS": 1,
    "DIREITOS_BENEFICIOS": 2,
    "PROCEDIMENTOS": 3,
    "SAUDE_OCUPACIONAL": 4,
    "EPI": 5,
}


# ============================================================================
# 1. Payloads das 4 Ações do Catálogo A2 (Anexos C.3 e D.3)
# ============================================================================


class PerfilColaboradorResponder(BaseModel):
    """Perfil resumido do colaborador enviado pelo A0 em `a2.responder`."""

    model_config = ConfigDict(extra="allow")

    cargo: str | None = Field(default=None, description="Cargo do colaborador.")
    setor: str | None = Field(default=None, description="Setor organizacional.")
    beneficios: list[str] = Field(
        default_factory=list,
        description="Lista de benefícios ativos do colaborador.",
    )

    @field_validator("beneficios", mode="before")
    @classmethod
    def _normalizar_beneficios(cls, valor: Any) -> list[str]:
        if valor is None:
            return []
        if isinstance(valor, str):
            item = valor.strip()
            return [item] if item else []
        return [str(v).strip() for v in valor if str(v).strip()]


class PayloadResponderTarefa(BaseModel):
    """Variáveis enviadas pelo A0 em `a2.responder` (Anexo C.3)."""

    model_config = ConfigDict(extra="allow")

    pergunta: str = Field(..., min_length=1, description="Pergunta ou relato do colaborador.")
    intencao: str | None = Field(
        default=None,
        description="Intenção opcional classificada pelo A0 (F1..F5 ou slug da base).",
    )
    perfil: PerfilColaboradorResponder | None = Field(
        default=None,
        description="Perfil do colaborador (cargo, setor, beneficios).",
    )
    historico_resumo: str | None = Field(
        default=None,
        description="Resumo da conversa mantido pelo A0.",
    )
    historico: list[dict[str, str]] | None = Field(
        default=None,
        description="Turnos recentes da sessão (usado em modo conversacional/demo).",
    )
    modelo: str | None = Field(
        default=None,
        description="Modelo Groq preferido para a execução da CrewAI.",
    )

    @model_validator(mode="before")
    @classmethod
    def _aceitar_alias_texto(cls, data: Any) -> Any:
        if isinstance(data, dict) and "pergunta" not in data and "texto" in data:
            copia = dict(data)
            copia["pergunta"] = copia["texto"]
            return copia
        return data

    @field_validator("pergunta")
    @classmethod
    def _validar_pergunta_nao_vazia(cls, valor: str) -> str:
        limpo = valor.strip()
        if not limpo:
            raise ValueError("O campo 'pergunta' não pode ser vazio.")
        return limpo


class PayloadResponderResultado(BaseModel):
    """Variáveis devolvidas pelo A2 em `a2.responder` (Anexos C.3 e D.3)."""

    model_config = ConfigDict(extra="allow")

    base: str = Field(..., description="Slug da base consultada (ex: 'direitos_beneficios', 'epi').")
    trechos_usados: int = Field(
        default=0,
        ge=0,
        description="Quantidade de trechos RAG utilizados para fundamentar a resposta.",
    )
    id_solicitacao_epi: int | None = Field(
        default=None,
        description="ID da solicitação aberta em `apoio.solicitacao_epi` quando aplicável (F5).",
    )


class JanelaLembretes(BaseModel):
    """Janela temporal enviada pelo agendador para `a2.gerar_lembretes`."""

    model_config = ConfigDict(extra="allow")

    de: datetime | str
    ate: datetime | str


class PayloadGerarLembretesTarefa(BaseModel):
    """Variáveis enviadas pelo A0 em `a2.gerar_lembretes` (Anexo C.3)."""

    model_config = ConfigDict(extra="allow")

    janela: JanelaLembretes


class PayloadGerarLembretesResultado(BaseModel):
    """Variáveis devolvidas pelo A2 em `a2.gerar_lembretes` (Anexo D.3)."""

    model_config = ConfigDict(extra="allow")

    janela: JanelaLembretes
    gerados: int = Field(..., ge=0)
    reagendados: int = Field(default=0, ge=0)


class RecomendacaoBeneficioItem(BaseModel):
    """Item individual de recomendação de benefício pela rede neural (Anexo D.3)."""

    model_config = ConfigDict(extra="allow")

    id_beneficio: int
    probabilidade: float = Field(..., ge=0.0, le=1.0)


class ExemploRecomendacaoColaborador(BaseModel):
    """Exemplo ilustrativo de recomendações por colaborador no payload de saída."""

    model_config = ConfigDict(extra="allow")

    id_colaborador: int
    recomendacoes: list[RecomendacaoBeneficioItem] = Field(default_factory=list)


class PayloadRecomendarBeneficiosTarefa(BaseModel):
    """Variáveis enviadas pelo A0 em `a2.recomendar_beneficios` (Anexo C.3)."""

    model_config = ConfigDict(extra="allow")

    competencia: str = Field(..., min_length=4, description="Competência mensal (ex: '2026-11').")


class PayloadRecomendarBeneficiosResultado(BaseModel):
    """Variáveis devolvidas pelo A2 em `a2.recomendar_beneficios` (Anexo D.3)."""

    model_config = ConfigDict(extra="allow")

    competencia: str
    modelo: str
    total: int = Field(..., ge=0)
    exemplo: ExemploRecomendacaoColaborador | None = None


class PayloadOrientarEpiTarefa(BaseModel):
    """Variáveis enviadas pelo A0 em `a2.orientar_epi` a partir do alerta A3.4 (Anexo C.3)."""

    model_config = ConfigDict(extra="allow")

    id_alerta: int
    id_colaborador: int
    id_epi: int
    situacao: str = Field(..., min_length=1)


class PayloadOrientarEpiResultado(BaseModel):
    """Payload opcional retornado em `a2.orientar_epi` (Anexo D.3)."""

    model_config = ConfigDict(extra="allow")

    id_alerta: int | None = None
    id_colaborador: int | None = None
    id_epi: int | None = None
    situacao: str | None = None


# ============================================================================
# 2. Blocos do Envelope `sinergia/1` (Anexo A e Anexo D.3)
# ============================================================================


class CorrelacaoEnvelope(BaseModel):
    """Liga a troca à conversa e à tarefa do orquestrador (Anexo A)."""

    model_config = ConfigDict(extra="allow")

    id_caso: int | None = None
    thread_id: str | None = None
    id_tarefa: str


class ContextoTarefa(BaseModel):
    """Contexto de origem e autenticação preenchido pelo A0 (Anexo A)."""

    model_config = ConfigDict(extra="allow")

    id_colaborador: int | None = None
    perfil: str = Field(
        default="colaborador",
        description="Perfil autenticado no A0 (colaborador, rh, gestor, tst, auditor, sistema).",
    )
    canal: str = Field(
        default="chat",
        description="Canal de origem (chat, e-mail, mensagem).",
    )
    data_hora: datetime | str


class DestinatarioSaida(BaseModel):
    """Destinatário de uma mensagem/lembrete/alerta (Anexo A e D.3)."""

    model_config = ConfigDict(extra="allow")

    perfil: str
    id_colaborador: int | None = None


class FonteCitada(BaseModel):
    """Fonte documental citada na saída do A2 (Anexos A e D.3)."""

    model_config = ConfigDict(extra="allow")

    base: str
    titulo: str
    versao: str


class SaidaMensagem(BaseModel):
    """Conteúdo produzido pelo A2 para o A0 entregar ao colaborador (Anexo A e D.3)."""

    model_config = ConfigDict(extra="allow")

    tipo: Literal["mensagem", "lembrete", "alerta"] = "mensagem"
    para: DestinatarioSaida | None = None
    nivel_detalhe: Literal["individual", "pseudonimizado", "agregado"] | None = None
    canal: str | None = None
    texto: str
    fontes: list[FonteCitada] = Field(default_factory=list)
    agendar_para: datetime | str | None = None


class EventoEmitido(BaseModel):
    """Fato emitido pelo A2 para o A0 orquestrar notificações/fluxos (Anexo A e D.3)."""

    model_config = ConfigDict(extra="allow")

    tipo: str = Field(
        ...,
        description="Código do evento (ex: 'solicitacao_epi.aberta', 'recomendacao.gerada').",
    )
    dados: dict[str, Any] = Field(default_factory=dict)


class DecisaoAgente(BaseModel):
    """Bloco de auditoria para registro em `orquestracao.execucao_agente` pelo A0 (Anexo A)."""

    model_config = ConfigDict(extra="allow")

    motivo: str
    usou_llm: bool
    confianca: float | None = Field(default=None, ge=0.0, le=1.0)
    modelo: str | None = None


class ErroAgente(BaseModel):
    """Bloco de diagnóstico quando `status == 'ERRO'` (Anexo A)."""

    model_config = ConfigDict(extra="allow")

    codigo: str
    mensagem: str
    reprocessavel: bool = False


# ============================================================================
# 3. Envelopes Completos `TAREFA` (A0 -> A2) e `RESULTADO` (A2 -> A0)
# ============================================================================


class TarefaA2(BaseModel):
    """Envelope completo de TAREFA recebido pelo A2 no protocolo `sinergia/1`."""

    model_config = ConfigDict(extra="allow")

    protocolo: Literal["sinergia/1"] = PROTOCOLO_SINERGIA
    id_mensagem: str = Field(
        ..., description="Identificador único da mensagem (idempotência)."
    )
    tipo: Literal["TAREFA"] = "TAREFA"
    acao: AcaoA2
    origem: Literal["A0"] = "A0"
    destino: Literal["A2"] = "A2"
    correlacao: CorrelacaoEnvelope
    contexto: ContextoTarefa
    payload: dict[str, Any] = Field(default_factory=dict)

    def validar_payload_acao(
        self,
    ) -> (
        PayloadResponderTarefa
        | PayloadGerarLembretesTarefa
        | PayloadRecomendarBeneficiosTarefa
        | PayloadOrientarEpiTarefa
    ):
        """Converte e valida `self.payload` para o modelo Pydantic específico da `acao`."""
        mapa = {
            "a2.responder": PayloadResponderTarefa,
            "a2.gerar_lembretes": PayloadGerarLembretesTarefa,
            "a2.recomendar_beneficios": PayloadRecomendarBeneficiosTarefa,
            "a2.orientar_epi": PayloadOrientarEpiTarefa,
        }
        modelo_cls = mapa[self.acao]
        return modelo_cls.model_validate(self.payload)


class ResultadoA2(BaseModel):
    """Envelope completo de RESULTADO devolvido pelo A2 ao A0 no protocolo `sinergia/1`."""

    model_config = ConfigDict(extra="allow")

    protocolo: Literal["sinergia/1"] = PROTOCOLO_SINERGIA
    id_mensagem: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Identificador único da mensagem de resultado.",
    )
    tipo: Literal["RESULTADO"] = "RESULTADO"
    acao: AcaoA2
    origem: Literal["A2"] = "A2"
    destino: Literal["A0"] = "A0"
    em_resposta_a: str = Field(..., description="id_mensagem da TAREFA original.")
    correlacao: CorrelacaoEnvelope
    status: StatusEnvelopeResultado
    estado_proposto: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    saidas: list[SaidaMensagem] = Field(default_factory=list)
    eventos: list[EventoEmitido] = Field(default_factory=list)
    decisao: DecisaoAgente | None = None
    erro: ErroAgente | None = None

    @model_validator(mode="after")
    def _validar_erro_obrigatorio(self) -> ResultadoA2:
        if self.status == "ERRO" and self.erro is None:
            raise ValueError(
                "Quando status == 'ERRO', o bloco 'erro' deve ser preenchido."
            )
        return self

    def validar_payload_acao(
        self,
    ) -> (
        PayloadResponderResultado
        | PayloadGerarLembretesResultado
        | PayloadRecomendarBeneficiosResultado
        | PayloadOrientarEpiResultado
    ):
        """Converte e valida `self.payload` do resultado conforme `self.acao`."""
        if self.acao == "a2.responder":
            return PayloadResponderResultado.model_validate(self.payload)
        if self.acao == "a2.gerar_lembretes":
            return PayloadGerarLembretesResultado.model_validate(self.payload)
        if self.acao == "a2.recomendar_beneficios":
            return PayloadRecomendarBeneficiosResultado.model_validate(self.payload)
        if self.acao == "a2.orientar_epi":
            return PayloadOrientarEpiResultado.model_validate(self.payload)
        raise ValueError(f"Ação desconhecida no ResultadoA2: {self.acao}")

    @classmethod
    def a_partir_da_tarefa(
        cls,
        tarefa: TarefaA2,
        *,
        status: StatusEnvelopeResultado,
        estado_proposto: str | None = None,
        payload: BaseModel | dict[str, Any] | None = None,
        saidas: list[SaidaMensagem | dict[str, Any]] | None = None,
        eventos: list[EventoEmitido | dict[str, Any]] | None = None,
        decisao: DecisaoAgente | dict[str, Any] | None = None,
        erro: ErroAgente | dict[str, Any] | None = None,
        id_mensagem: str | None = None,
    ) -> ResultadoA2:
        """Constrói um envelope `ResultadoA2` copiando `acao`, `correlacao` e `em_resposta_a`."""
        payload_dict = (
            payload.model_dump(mode="json", exclude_none=True)
            if isinstance(payload, BaseModel)
            else (payload or {})
        )
        return cls(
            id_mensagem=id_mensagem or str(uuid4()),
            acao=tarefa.acao,
            em_resposta_a=tarefa.id_mensagem,
            correlacao=tarefa.correlacao,
            status=status,
            estado_proposto=estado_proposto,
            payload=payload_dict,
            saidas=[
                s if isinstance(s, SaidaMensagem) else SaidaMensagem.model_validate(s)
                for s in (saidas or [])
            ],
            eventos=[
                e if isinstance(e, EventoEmitido) else EventoEmitido.model_validate(e)
                for e in (eventos or [])
            ],
            decisao=(
                decisao
                if isinstance(decisao, DecisaoAgente) or decisao is None
                else DecisaoAgente.model_validate(decisao)
            ),
            erro=(
                erro
                if isinstance(erro, ErroAgente) or erro is None
                else ErroAgente.model_validate(erro)
            ),
        )


def exportar_json_schemas_a2() -> dict[str, dict[str, Any]]:
    """Exporta os JSON Schemas oficiais do Agente 2 conforme Anexos A, C.3 e D.3."""
    return {
        "TarefaA2": TarefaA2.model_json_schema(),
        "ResultadoA2": ResultadoA2.model_json_schema(),
        "a2.responder.tarefa": PayloadResponderTarefa.model_json_schema(),
        "a2.responder.resultado": PayloadResponderResultado.model_json_schema(),
        "a2.gerar_lembretes.tarefa": PayloadGerarLembretesTarefa.model_json_schema(),
        "a2.gerar_lembretes.resultado": PayloadGerarLembretesResultado.model_json_schema(),
        "a2.recomendar_beneficios.tarefa": PayloadRecomendarBeneficiosTarefa.model_json_schema(),
        "a2.recomendar_beneficios.resultado": PayloadRecomendarBeneficiosResultado.model_json_schema(),
        "a2.orientar_epi.tarefa": PayloadOrientarEpiTarefa.model_json_schema(),
        "a2.orientar_epi.resultado": PayloadOrientarEpiResultado.model_json_schema(),
    }
