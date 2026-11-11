"""Contratos Pydantic do Agente 1 (A1 — Recepção e Processamento).

Implementa o protocolo `sinergia/1` e os payloads das 4 ações do A1 conforme
definido em `docs/Documentação - SinergIA v2.1.docx` (Seção 5.3 e Anexos A, B,
C.2, D.1, D.2, E.2 e F):

- `a1.conversar`
- `a1.processar_documento`
- `a1.confirmar`
- `a1.registrar_baixa_fisica`
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROTOCOLO_SINERGIA = "sinergia/1"

AcaoA1 = Literal[
    "a1.conversar",
    "a1.processar_documento",
    "a1.confirmar",
    "a1.registrar_baixa_fisica",
]

StatusEnvelopeResultado = Literal["OK", "PRECISA_HUMANO", "ERRO"]

EstadoCasoAfastamento = Literal[
    "NOVO",
    "RECEBIDO",
    "VALIDADO",
    "SEM_PENDENCIAS",
    "DIVERGENCIA",
    "ESCALADO",
    "EM_ACOMPANHAMENTO",
    "ENCERRADO",
]

StatusAtestadoDB = Literal["PENDENTE", "CONFIRMADO", "REJEITADO"]
OrigemDadosAtestado = Literal["IA", "MANUAL"]
LegibilidadeDocumento = Literal["ok", "ilegivel"]


# ============================================================================
# 1. Modelos de Domínio Internos (OCR, Extração LLM e Divergências)
# ============================================================================


class ResultadoOCR(BaseModel):
    """Saída padronizada do módulo `ocr.py` para consumo pelo `extrator_llm.py`."""

    model_config = ConfigDict(extra="forbid")

    arquivo_uri: str = Field(
        ..., description="Caminho/URI do arquivo no armazenamento privado."
    )
    mime: str | None = Field(
        default=None, description="Tipo MIME do arquivo (ex: image/png, application/pdf)."
    )
    sha256: str | None = Field(
        default=None, description="Hash SHA-256 calculado do arquivo."
    )
    texto_extraido: str = Field(
        default="", description="Texto bruto extraído via OCR ou camada de texto do PDF."
    )
    confianca_media: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confiança média do OCR entre 0.0 e 1.0.",
    )
    legibilidade: LegibilidadeDocumento = Field(
        default="ok",
        description="Indica se o arquivo pôde ser lido com nitidez suficiente.",
    )


class PeriodoAfastamento(BaseModel):
    """Período de afastamento extraído ou confirmado (Anexo D.2)."""

    model_config = ConfigDict(extra="allow")

    inicio: date | None = Field(default=None, description="Data de início do afastamento.")
    fim: date | None = Field(default=None, description="Data de término do afastamento.")


class ProfissionalExtraido(BaseModel):
    """Dados do profissional de saúde extraídos do atestado (Anexo D.2)."""

    model_config = ConfigDict(extra="allow")

    conselho: str | None = Field(
        default=None, description="Sigla do conselho profissional (ex: CRM, CRO, CRP)."
    )
    numero: str | None = Field(
        default=None, description="Número de registro no conselho."
    )
    uf: str | None = Field(
        default=None, description="Sigla da UF do registro profissional (2 letras)."
    )
    nome: str | None = Field(
        default=None, description="Nome do profissional de saúde (quando legível)."
    )


class AtestadoExtraido(BaseModel):
    """Estrutura oficial de `atestado_extraido` (Anexo C.2 e D.2).

    Utilizada também como schema de saída estruturada (`.with_structured_output`)
    do LLM no módulo `extrator_llm.py`.
    """

    model_config = ConfigDict(extra="allow")

    legibilidade: LegibilidadeDocumento = Field(
        default="ok",
        description="'ok' se o documento é um atestado legível; 'ilegivel' caso contrário.",
    )
    nome_no_documento: str | None = Field(
        default=None, description="Nome do paciente conforme impresso/escrito no atestado."
    )
    periodo: PeriodoAfastamento = Field(
        default_factory=PeriodoAfastamento,
        description="Datas de início e fim do afastamento.",
    )
    data_emissao: date | None = Field(
        default=None, description="Data de emissão do atestado."
    )
    profissional: ProfissionalExtraido = Field(
        default_factory=ProfissionalExtraido,
        description="Conselho, número e UF do profissional emissor.",
    )
    cid: str | None = Field(
        default=None,
        description="Código CID-10 (opcional; respeita a política de CID do Anexo G).",
    )
    municipio: str | None = Field(
        default=None, description="Município de emissão do atestado."
    )


class DivergenciaCampo(BaseModel):
    """Divergência detectada entre o atestado lido e o `registro_inicial` (Anexo D.2 e E.2)."""

    model_config = ConfigDict(extra="forbid")

    campo: str = Field(
        ..., description="Nome do campo divergente (ex: 'dias', 'inicio', 'nome')."
    )
    no_documento: Any = Field(
        ..., description="Valor lido no documento do atestado."
    )
    informado: Any = Field(
        ..., description="Valor informado anteriormente no registro inicial."
    )


# ============================================================================
# 2. Payloads Específicos das 4 Ações do Catálogo A1 (Anexo C.2 e D.2)
# ============================================================================


# --- Ação 1: a1.conversar ---
class PayloadConversarTarefa(BaseModel):
    """Variáveis enviadas pelo A0 em `a1.conversar` (Anexo C.2)."""

    model_config = ConfigDict(extra="allow")

    texto: str = Field(..., description="Mensagem enviada pelo colaborador no chat.")
    historico_resumo: str | None = Field(
        default=None, description="Resumo da conversa mantido pelo A0."
    )


class PayloadConversarResultado(BaseModel):
    """Variáveis resolvidas pelo A1 em `a1.conversar` (Anexo C.2 e D.2)."""

    model_config = ConfigDict(extra="allow")

    faltando: list[str] = Field(
        default_factory=list,
        description="Lista de itens que faltam para processar o atestado (ex: ['arquivo']).",
    )


# --- Ação 2: a1.processar_documento ---
class PayloadProcessarDocumentoTarefa(BaseModel):
    """Variáveis enviadas pelo A0 em `a1.processar_documento` (Anexo C.2)."""

    model_config = ConfigDict(extra="allow")

    arquivo_uri: str = Field(
        ..., description="Referência ao arquivo no armazenamento privado."
    )
    mime: str | None = Field(
        default=None, description="MIME type do arquivo enviado."
    )
    sha256: str | None = Field(
        default=None, description="Hash SHA-256 do arquivo enviado."
    )


class PayloadProcessarDocumentoResultado(BaseModel):
    """Variáveis resolvidas pelo A1 em `a1.processar_documento` (Anexo C.2 e D.2)."""

    model_config = ConfigDict(extra="allow")

    atestado_extraido: AtestadoExtraido
    divergencias: list[DivergenciaCampo] = Field(default_factory=list)


# --- Ação 3: a1.confirmar ---
class PayloadConfirmarTarefa(BaseModel):
    """Variáveis enviadas pelo A0 em `a1.confirmar` (Anexo C.2 e D.1)."""

    model_config = ConfigDict(extra="allow")

    atestado_confirmado: dict[str, Any] | AtestadoExtraido = Field(
        ..., description="Campos confirmados ou corrigidos pelo colaborador."
    )
    campos_corrigidos: list[str] = Field(
        default_factory=list,
        description="Lista de nomes dos campos alterados manualmente pelo colaborador.",
    )


class PayloadConfirmarValidoResultado(BaseModel):
    """Payload de `a1.confirmar` quando os dados são válidos (Anexo D.2)."""

    model_config = ConfigDict(extra="allow")

    id_atestado: int
    status_atestado: Literal["CONFIRMADO"] = "CONFIRMADO"
    origem: OrigemDadosAtestado
    prazo_entrega_fisica: date


class PayloadConfirmarInvalidoResultado(BaseModel):
    """Payload de `a1.confirmar` quando há erro de validação nas datas/regras (Anexo D.2)."""

    model_config = ConfigDict(extra="allow")

    motivo: str


# --- Ação 4: a1.registrar_baixa_fisica ---
class PayloadBaixaFisicaTarefa(BaseModel):
    """Variáveis enviadas pelo A0 em `a1.registrar_baixa_fisica` (Anexo C.2 e D.2)."""

    model_config = ConfigDict(extra="allow")

    id_atestado: int
    recebido_fisico_por: str | int = Field(
        ..., description="Identificador ou login do usuário de RH que recebeu o original."
    )
    recebido_fisico_em: datetime | str = Field(
        ..., description="Data/hora ISO 8601 do recebimento do atestado físico."
    )


class PayloadBaixaFisicaResultado(BaseModel):
    """Payload devolvido pelo A1 em `a1.registrar_baixa_fisica` (Anexo D.2)."""

    model_config = ConfigDict(extra="allow")

    id_atestado: int
    recebido_fisico_por: str | int
    recebido_fisico_em: datetime | str


# ============================================================================
# 3. Blocos do Envelope `sinergia/1` (Anexo A e Anexo B)
# ============================================================================


class CorrelacaoEnvelope(BaseModel):
    """Liga a troca ao caso de afastamento e à conversa (Anexo A)."""

    model_config = ConfigDict(extra="allow")

    id_caso: int | None = None
    thread_id: str | None = None
    id_tarefa: str


class ContextoTarefa(BaseModel):
    """Quem pediu e por onde, preenchido pelo A0 a partir do login (Anexo A)."""

    model_config = ConfigDict(extra="allow")

    id_colaborador: int
    perfil: str = Field(
        default="colaborador",
        description="Perfil autenticado no A0 (colaborador, rh, gestor, tst, auditor).",
    )
    canal: str = Field(default="chat", description="Canal de origem (chat, e-mail, mensagem).")
    data_hora: datetime | str


class RegistroInicialCaso(BaseModel):
    """Dados preenchidos pelo A0 ao abrir o caso de afastamento (Anexo B e D.1)."""

    model_config = ConfigDict(extra="allow")

    id_caso: int
    thread_id: str
    aberto_em: datetime | str
    canal: str = "chat"
    id_colaborador: int
    nome: str
    matricula: str
    id_area: int
    id_setor: int
    id_cargo: int
    jornada_inicio: str
    jornada_fim: str
    municipio: str
    uf: str
    tipo_solicitacao: str = "atestado_medico"
    inicio_informado: date | None = None
    dias_informados: int | None = None
    tem_documento: bool = False


class DestinatarioSaida(BaseModel):
    """Destinatário de uma mensagem/lembrete/alerta (Anexo A e D.2)."""

    model_config = ConfigDict(extra="allow")

    perfil: str
    id_colaborador: int | None = None


class FonteCitada(BaseModel):
    """Fonte documental citada na saída (Anexo A)."""

    model_config = ConfigDict(extra="allow")

    base: str
    titulo: str
    versao: str


class SaidaMensagem(BaseModel):
    """Conteúdo produzido pelo agente para o A0 entregar a pessoas (Anexo A e D.2)."""

    model_config = ConfigDict(extra="allow")

    tipo: Literal["mensagem", "lembrete", "alerta"] = "mensagem"
    para: DestinatarioSaida | None = None
    nivel_detalhe: Literal["individual", "pseudonimizado", "agregado"] | None = None
    texto: str
    fontes: list[FonteCitada] = Field(default_factory=list)
    agendar_para: datetime | str | None = None


class EventoEmitido(BaseModel):
    """Fato emitido pelo agente para o A0 decidir o próximo passo (Anexo A e D.2)."""

    model_config = ConfigDict(extra="allow")

    tipo: str = Field(
        ...,
        description=(
            "Código do evento (ex: 'atestado.extraido', 'atestado.registrado', "
            "'atestado.invalido', 'atestado.baixa_registrada')."
        ),
    )
    dados: dict[str, Any] = Field(default_factory=dict)


class DecisaoAgente(BaseModel):
    """Bloco de auditoria para registro em `orquestracao.execucao_agente` pelo A0 (Anexo A)."""

    model_config = ConfigDict(extra="allow")

    motivo: str
    usou_llm: bool
    confianca: float | None = None
    modelo: str | None = None


class ErroAgente(BaseModel):
    """Bloco de diagnóstico quando `status == 'ERRO'` (Anexo A)."""

    model_config = ConfigDict(extra="allow")

    codigo: str
    mensagem: str
    reprocessavel: bool = False


# ============================================================================
# 4. Envelopes Completos `TAREFA` (A0 -> A1) e `RESULTADO` (A1 -> A0)
# ============================================================================


class TarefaA1(BaseModel):
    """Envelope completo de TAREFA recebido pelo A1 no protocolo `sinergia/1`."""

    model_config = ConfigDict(extra="allow")

    protocolo: Literal["sinergia/1"] = PROTOCOLO_SINERGIA
    id_mensagem: str = Field(..., description="Identificador único da mensagem (idempotência).")
    tipo: Literal["TAREFA"] = "TAREFA"
    acao: AcaoA1
    origem: Literal["A0"] = "A0"
    destino: Literal["A1"] = "A1"
    correlacao: CorrelacaoEnvelope
    contexto: ContextoTarefa
    registro_inicial: RegistroInicialCaso | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    def validar_payload_acao(
        self,
    ) -> (
        PayloadConversarTarefa
        | PayloadProcessarDocumentoTarefa
        | PayloadConfirmarTarefa
        | PayloadBaixaFisicaTarefa
    ):
        """Converte e valida `self.payload` para o modelo Pydantic específico da `acao`."""
        mapa = {
            "a1.conversar": PayloadConversarTarefa,
            "a1.processar_documento": PayloadProcessarDocumentoTarefa,
            "a1.confirmar": PayloadConfirmarTarefa,
            "a1.registrar_baixa_fisica": PayloadBaixaFisicaTarefa,
        }
        modelo_cls = mapa[self.acao]
        return modelo_cls.model_validate(self.payload)


class ResultadoA1(BaseModel):
    """Envelope completo de RESULTADO devolvido pelo A1 ao A0 no protocolo `sinergia/1`."""

    model_config = ConfigDict(extra="allow")

    protocolo: Literal["sinergia/1"] = PROTOCOLO_SINERGIA
    id_mensagem: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Identificador único da mensagem de resultado.",
    )
    tipo: Literal["RESULTADO"] = "RESULTADO"
    acao: AcaoA1
    origem: Literal["A1"] = "A1"
    destino: Literal["A0"] = "A0"
    em_resposta_a: str = Field(..., description="id_mensagem da TAREFA original.")
    correlacao: CorrelacaoEnvelope
    status: StatusEnvelopeResultado
    estado_proposto: EstadoCasoAfastamento | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    saidas: list[SaidaMensagem] = Field(default_factory=list)
    eventos: list[EventoEmitido] = Field(default_factory=list)
    decisao: DecisaoAgente | None = None
    erro: ErroAgente | None = None

    @model_validator(mode="after")
    def _validar_erro_obrigatorio(self) -> ResultadoA1:
        if self.status == "ERRO" and self.erro is None:
            raise ValueError("Quando status == 'ERRO', o bloco 'erro' deve ser preenchido.")
        return self

    @classmethod
    def a_partir_da_tarefa(
        cls,
        tarefa: TarefaA1,
        *,
        status: StatusEnvelopeResultado,
        estado_proposto: EstadoCasoAfastamento | None = None,
        payload: BaseModel | dict[str, Any] | None = None,
        saidas: list[SaidaMensagem | dict[str, Any]] | None = None,
        eventos: list[EventoEmitido | dict[str, Any]] | None = None,
        decisao: DecisaoAgente | dict[str, Any] | None = None,
        erro: ErroAgente | dict[str, Any] | None = None,
        id_mensagem: str | None = None,
    ) -> ResultadoA1:
        """Constrói um envelope `ResultadoA1` copiando `acao`, `correlacao` e `em_resposta_a`."""
        payload_dict = (
            payload.model_dump(mode="json")
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
