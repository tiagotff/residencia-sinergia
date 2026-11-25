"""Contratos Pydantic e JSON Schema do Agente 1 (A1 — Recepção e Processamento).

Implementa o protocolo `sinergia/1` e os payloads das 4 ações do A1 conforme
definido em `docs/Documentação - SinergIA v2.1.docx` (Seção 5.3 e Anexos A, B,
C.2, D.1, D.2, E.2, F e G) e as restrições de `db/schema/001_schema_inicial.sql`:

- `a1.conversar`
- `a1.processar_documento`
- `a1.confirmar`
- `a1.registrar_baixa_fisica`
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

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
TipoDocumentoAfastamento = Literal[
    "atestado_medico", "declaracao_comparecimento", "outro"
]
StatusRevisaoCampo = Literal[
    "lido_ok", "divergente", "baixa_confianca", "nao_identificado"
]
StatusValidacaoCid = Literal[
    "CID Validada / Correta",
    "CID Não Encontrada / Inválida",
]
OrigemPreenchimentoFormulario = Literal["A1_RECEPCAO", "MANUAL"]


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
        default=None,
        description="Tipo MIME do arquivo (ex: image/png, application/pdf).",
    )
    sha256: str | None = Field(
        default=None, description="Hash SHA-256 calculado do arquivo."
    )
    texto_extraido: str = Field(
        default="",
        description="Texto bruto extraído via OCR ou camada de texto do PDF.",
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


class CampoMetadado(BaseModel):
    """Metadados de confiança, origem e descarte por baixa confiança por campo (Guia A1 v0.7)."""

    model_config = ConfigDict(extra="forbid")

    confianca: float | None = Field(default=None, ge=0.0, le=1.0)
    origem: OrigemDadosAtestado = "IA"
    vazio_por_baixa_confianca: bool = False


class PeriodoAfastamento(BaseModel):
    """Período de afastamento extraído ou confirmado (Anexo D.2)."""

    model_config = ConfigDict(extra="allow")

    inicio: date | None = Field(
        default=None, description="Data de início do afastamento."
    )
    fim: date | None = Field(
        default=None, description="Data de término do afastamento."
    )
    dias: int | None = Field(
        default=None,
        description="Quantidade inclusiva de dias (fim - inicio + 1).",
    )

    @model_validator(mode="after")
    def _preencher_dias(self) -> PeriodoAfastamento:
        if (
            self.dias is None
            and self.inicio is not None
            and self.fim is not None
            and self.fim >= self.inicio
        ):
            self.dias = (self.fim - self.inicio).days + 1
        return self

    @property
    def dias_calculados(self) -> int | None:
        """Calcula a quantidade inclusiva de dias do período (`(fim - inicio).days + 1`)."""
        if self.inicio is not None and self.fim is not None:
            return (self.fim - self.inicio).days + 1
        return None


class ProfissionalExtraido(BaseModel):
    """Dados do profissional de saúde extraídos do atestado (Anexo D.2 e 001_schema_inicial.sql)."""

    model_config = ConfigDict(extra="allow")

    conselho: str | None = Field(
        default=None,
        description="Sigla do conselho profissional (ex: CRM, CRO, CRP).",
    )
    numero: str | None = Field(
        default=None, description="Número de registro no conselho."
    )
    uf: str | None = Field(
        default=None,
        description="Sigla da UF do registro profissional (2 letras, ex: SP).",
    )
    nome: str | None = Field(
        default=None, description="Nome do profissional de saúde (quando legível)."
    )

    @field_validator("conselho", mode="before")
    @classmethod
    def _normalizar_conselho(cls, v: Any) -> str | None:
        if v is None:
            return None
        texto = str(v).strip().upper()
        return texto or None

    @field_validator("numero", mode="before")
    @classmethod
    def _normalizar_numero(cls, v: Any) -> str | None:
        if v is None:
            return None
        texto = str(v).strip()
        return texto or None

    @field_validator("uf", mode="before")
    @classmethod
    def _normalizar_uf(cls, v: Any) -> str | None:
        if v is None:
            return None
        texto = str(v).strip().upper()
        return texto or None


class AtestadoExtraido(BaseModel):
    """Estrutura oficial de `atestado_extraido` (Anexo C.2, D.2 e E.2).

    Utilizada também como schema de saída estruturada (`.with_structured_output`)
    do LLM no módulo `extrator_llm.py`.
    """

    model_config = ConfigDict(extra="allow")

    tipo_documento: TipoDocumentoAfastamento = Field(
        default="atestado_medico",
        description="Tipo do documento (atestado_medico, declaracao_comparecimento, outro).",
    )
    legibilidade: LegibilidadeDocumento = Field(
        default="ok",
        description="'ok' se o documento é um atestado legível; 'ilegivel' caso contrário.",
    )
    nome_no_documento: str | None = Field(
        default=None,
        description="Nome do paciente conforme impresso/escrito no atestado.",
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
    campos: dict[str, CampoMetadado] = Field(
        default_factory=dict,
        description="Metadados de confiança, origem e descarte por baixa confiança por campo (Guia A1 v0.7).",
    )
    confianca_campos: dict[str, float] | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="Confiança por campo (0.0 a 1.0) para aplicar o limiar de 0,8 do Anexo E.2.",
    )

    @field_validator("cid", mode="before")
    @classmethod
    def _normalizar_cid(cls, v: Any) -> str | None:
        if v is None:
            return None
        texto = str(v).strip().upper()
        return texto or None

    def aplicar_politica_e_limiar(
        self,
        *,
        extrair_cid: bool = True,
        limiar_confianca: float = 0.8,
    ) -> AtestadoExtraido:
        """Aplica a Política de CID (Anexo G) e o limiar de confiança por campo (Anexo E.2).

        Campos cuja confiança em `confianca_campos` seja inferior a `limiar_confianca`
        são esvaziados (`None`) e registrados em `campos` com `vazio_por_baixa_confianca=True`
        para que o A1 jamais adivinhe valores duvidosos.
        """
        copia = self.model_copy(deep=True)
        if not extrair_cid:
            copia.cid = None
            copia.campos.pop("cid", None)

        if copia.confianca_campos:
            c = copia.confianca_campos
            campos_meta: dict[str, CampoMetadado] = dict(copia.campos)

            def _checar_e_zerar(chave_canonica: str, alias: str | None = None) -> bool:
                score = c.get(chave_canonica)
                if score is None and alias is not None:
                    score = c.get(alias)
                if score is None or score >= limiar_confianca:
                    return False
                campos_meta[chave_canonica] = CampoMetadado(
                    confianca=round(max(0.0, min(1.0, float(score))), 4),
                    origem="IA",
                    vazio_por_baixa_confianca=True,
                )
                return True

            if _checar_e_zerar("nome_no_documento"):
                copia.nome_no_documento = None
            if _checar_e_zerar("periodo.inicio", "inicio"):
                copia.periodo.inicio = None
            if _checar_e_zerar("periodo.fim", "fim"):
                copia.periodo.fim = None
            if _checar_e_zerar("data_emissao"):
                copia.data_emissao = None
            if _checar_e_zerar("profissional.conselho", "conselho"):
                copia.profissional.conselho = None
            if _checar_e_zerar("profissional.numero", "numero"):
                copia.profissional.numero = None
            if _checar_e_zerar("profissional.uf", "uf"):
                copia.profissional.uf = None
            if _checar_e_zerar("profissional.nome"):
                copia.profissional.nome = None
            if extrair_cid and copia.cid is not None and _checar_e_zerar("cid"):
                copia.cid = None
            if _checar_e_zerar("municipio"):
                copia.municipio = None

            copia.campos = campos_meta

        if (
            copia.periodo.inicio is not None
            and copia.periodo.fim is not None
            and copia.periodo.fim >= copia.periodo.inicio
        ):
            copia.periodo.dias = (copia.periodo.fim - copia.periodo.inicio).days + 1
        else:
            copia.periodo.dias = None

        return copia


class ItemRevisaoCampo(BaseModel):
    """Campo individual exibido no espelho de revisão e validação do colaborador."""

    model_config = ConfigDict(extra="forbid")

    campo: str = Field(..., description="Chave canônica do campo (ex.: 'periodo.inicio').")
    rotulo: str = Field(..., description="Rótulo legível em português para o colaborador.")
    valor_lido: Any = Field(
        default=None, description="Valor canônico extraído do documento (ou None)."
    )
    valor_exibicao: str = Field(
        ..., description="Representação amigável em PT-BR para visualização rápida."
    )
    valor_informado_antes: Any = Field(
        default=None,
        description="Valor informado no registro_inicial quando houver divergência.",
    )
    confianca: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confiança estimada da leitura pela IA (0.0 a 1.0).",
    )
    status: StatusRevisaoCampo = Field(
        ..., description="Situação da leitura para destaque visual ao colaborador."
    )
    editavel: bool = Field(
        default=True, description="Indica se o colaborador pode alterar este campo."
    )
    obrigatorio: bool = Field(
        default=False, description="Indica se o campo é obrigatório para confirmar."
    )


class RetornoValidacaoCidFormulario(BaseModel):
    """Resultado da decisão pós-validação de CID (4A vs 4B) e auditoria do formulário."""

    model_config = ConfigDict(extra="forbid")

    codigo_cid: str
    status_validacao: StatusValidacaoCid
    cid_valida: bool
    nome_doenca: str | None = None
    descricao_oficial: str | None = None
    bloqueia_aprovacao_automatica: bool = False
    requer_revisao_humana: bool = False
    mensagem_tratamento: str | None = None
    origem_preenchimento: OrigemPreenchimentoFormulario = "A1_RECEPCAO"
    validado_em: datetime | str
    versao_tabela_cid: str


class QuadroRevisaoAtestado(BaseModel):
    """Espelho completo dos dados lidos pelo LLM para validação/correção humana."""

    model_config = ConfigDict(extra="forbid")

    titulo: str = Field(..., description="Cabeçalho curto da solicitação de revisão.")
    instrucao_colaborador: str = Field(
        ...,
        description="Orientação objetiva convidando o colaborador a confirmar ou editar.",
    )
    legibilidade: LegibilidadeDocumento
    resumo_markdown: str = Field(
        ..., description="Resumo legível em Markdown com todos os dados lidos no documento."
    )
    itens: list[ItemRevisaoCampo] = Field(
        default_factory=list,
        description="Lista ordenada de campos para renderização visual no frontend.",
    )
    permite_confirmar_direto: bool = True
    permite_editar: bool = True
    requer_novo_arquivo: bool = False
    validacao_cid: RetornoValidacaoCidFormulario | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="Status 4A/4B e auditoria de validação do CID no formulário devolvido.",
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
    revisao_colaborador: QuadroRevisaoAtestado | None = None


# --- Ação 3: a1.confirmar ---
class PayloadConfirmarTarefa(BaseModel):
    """Variáveis enviadas pelo A0 em `a1.confirmar` (Anexo C.2 e D.1)."""

    model_config = ConfigDict(extra="allow")

    atestado_confirmado: AtestadoExtraido = Field(
        ...,
        description=(
            "Campos confirmados ou corrigidos pelo colaborador (aceita tanto "
            "objeto parcial com `periodo`/`data_emissao` como no Anexo D.1 quanto "
            "todos os campos de `AtestadoExtraido`)."
        ),
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
    prazos_legais: list[str] = Field(
        default_factory=list,
        description="Marcos legais aplicáveis: 'inss_16_dias', 'exame_retorno' (Guia A1 v0.7).",
    )
    validacao_cid: RetornoValidacaoCidFormulario | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="Status 4A e auditoria de validação do CID quando aplicável.",
    )


class PayloadConfirmarInvalidoResultado(BaseModel):
    """Payload de `a1.confirmar` quando há erro de validação nas datas/regras (Anexo D.2)."""

    model_config = ConfigDict(extra="allow")

    motivo: str
    validacao_cid: RetornoValidacaoCidFormulario | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="Status 4B e auditoria de bloqueio quando o CID é inválido.",
    )


# --- Ação 4: a1.registrar_baixa_fisica ---
class PayloadBaixaFisicaTarefa(BaseModel):
    """Variáveis enviadas pelo A0 em `a1.registrar_baixa_fisica` (Anexo C.2 e D.2)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id_atestado: int
    recebido_fisico_por: str | int = Field(
        ...,
        validation_alias=AliasChoices("recebido_fisico_por", "recebido_por"),
        description="Identificador ou login do usuário de RH que recebeu o original.",
    )
    recebido_fisico_em: datetime | str = Field(
        ...,
        validation_alias=AliasChoices("recebido_fisico_em", "data_hora"),
        description="Data/hora ISO 8601 do recebimento do atestado físico.",
    )


class PayloadBaixaFisicaResultado(BaseModel):
    """Payload devolvido pelo A1 em `a1.registrar_baixa_fisica` (Anexo D.2)."""

    model_config = ConfigDict(extra="allow")

    id_atestado: int
    recebido_fisico_por: str | int
    recebido_fisico_em: datetime | str
    recebido_por: str | int | None = None
    data_hora: datetime | str | None = None

    @model_validator(mode="after")
    def _preencher_aliases_guia(self) -> PayloadBaixaFisicaResultado:
        if self.recebido_por is None:
            self.recebido_por = self.recebido_fisico_por
        if self.data_hora is None:
            self.data_hora = self.recebido_fisico_em
        return self


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
    id_empresa: int | None = Field(
        default=None, description="Identificador da empresa (Guia A1 v0.7)."
    )
    perfil: str = Field(
        default="colaborador",
        description="Perfil autenticado no A0 (colaborador, rh, gestor, tst, auditor).",
    )
    canal: str = Field(
        default="chat", description="Canal de origem (chat, e-mail, mensagem)."
    )
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
    sensivel: bool = Field(
        default=False,
        description="Indica se a saída contém dado pessoal sensível (LGPD Art. 11 / Guia A1 v0.7).",
    )


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
    confianca: float | None = Field(default=None, ge=0.0, le=1.0)
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
    id_mensagem: str = Field(
        ..., description="Identificador único da mensagem (idempotência)."
    )
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
            raise ValueError(
                "Quando status == 'ERRO', o bloco 'erro' deve ser preenchido."
            )
        return self

    def validar_payload_acao(
        self,
    ) -> (
        PayloadConversarResultado
        | PayloadProcessarDocumentoResultado
        | PayloadConfirmarValidoResultado
        | PayloadConfirmarInvalidoResultado
        | PayloadBaixaFisicaResultado
    ):
        """Converte e valida `self.payload` do resultado conforme `self.acao`."""
        if self.acao == "a1.conversar":
            return PayloadConversarResultado.model_validate(self.payload)
        if self.acao == "a1.processar_documento":
            return PayloadProcessarDocumentoResultado.model_validate(self.payload)
        if self.acao == "a1.confirmar":
            if self.estado_proposto == "VALIDADO" or "id_atestado" in self.payload:
                return PayloadConfirmarValidoResultado.model_validate(self.payload)
            return PayloadConfirmarInvalidoResultado.model_validate(self.payload)
        if self.acao == "a1.registrar_baixa_fisica":
            return PayloadBaixaFisicaResultado.model_validate(self.payload)
        raise ValueError(f"Ação desconhecida no ResultadoA1: {self.acao}")

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
            payload.model_dump(mode="json", exclude_none=False)
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


def exportar_json_schemas_a1() -> dict[str, dict[str, Any]]:
    """Exporta os JSON Schemas oficiais do Agente 1 conforme Anexo A e Anexo F."""
    return {
        "TarefaA1": TarefaA1.model_json_schema(),
        "ResultadoA1": ResultadoA1.model_json_schema(),
        "AtestadoExtraido": AtestadoExtraido.model_json_schema(),
        "QuadroRevisaoAtestado": QuadroRevisaoAtestado.model_json_schema(),
        "RetornoValidacaoCidFormulario": RetornoValidacaoCidFormulario.model_json_schema(),
        "a1.conversar.tarefa": PayloadConversarTarefa.model_json_schema(),
        "a1.conversar.resultado": PayloadConversarResultado.model_json_schema(),
        "a1.processar_documento.tarefa": PayloadProcessarDocumentoTarefa.model_json_schema(),
        "a1.processar_documento.resultado": PayloadProcessarDocumentoResultado.model_json_schema(),
        "a1.confirmar.tarefa": PayloadConfirmarTarefa.model_json_schema(),
        "a1.confirmar.resultado.valido": PayloadConfirmarValidoResultado.model_json_schema(),
        "a1.confirmar.resultado.invalido": PayloadConfirmarInvalidoResultado.model_json_schema(),
        "a1.registrar_baixa_fisica.tarefa": PayloadBaixaFisicaTarefa.model_json_schema(),
        "a1.registrar_baixa_fisica.resultado": PayloadBaixaFisicaResultado.model_json_schema(),
    }
