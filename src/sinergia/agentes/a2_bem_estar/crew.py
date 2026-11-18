"""Orquestração CrewAI + Groq do Agente 2 (A2 — Bem-Estar).

Implementa a equipe multidisciplinar da Seção 5.1 da `Documentação - SinergIA v2.2.docx`:
- **Gerente / Roteador** (`gerente_roteador`)
- **5 Especialistas (`Agent` do CrewAI)** conectados às ferramentas RAG de `src/sinergia/rag/tools.py`:
  1. `especialista_pausas` (`F1` — `ferramenta_busca_pausas`, NR-17)
  2. `especialista_direitos_beneficios` (`F2` — `ferramenta_busca_direitos`, CLT/Benefícios)
  3. `especialista_procedimentos` (`F3` — `ferramenta_busca_procedimentos`, POPs/Treinamentos)
  4. `especialista_saude_ocupacional` (`F4` — `ferramenta_busca_saude_ocupacional`, PCMSO/NR-7)
  5. `especialista_epi` (`F5` — `ferramenta_busca_epis`, NR-6/CA + reposição de EPI)
- **Redator Final** (`redator_final`) alinhado aos 5 critérios do Judge.

Captura também a telemetria instrumentada (`ResultadoBuscaInstrumentada`) para alimentar
o painel Raio-X Nerd do `chat_demo` sem poluir o envelope público `sinergia/1`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import unicodedata
from collections.abc import Callable
from typing import Any

# Desativa prompts interativos de telemetria/tracing do CrewAI em execução headless/testes
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import httpx
from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel, Field

from sinergia.agentes.a2_bem_estar.config import A2Config, obter_config_a2
from sinergia.agentes.a2_bem_estar.prompts import (
    PROMPT_GERENTE_ROTEADOR_A2,
    PROMPT_REDATOR_FINAL_A2,
    PROMPTS_ESPECIALISTAS_A2,
    construir_task_especialista,
    construir_task_redator_final,
    formatar_perfil_para_prompt,
)
from sinergia.agentes.a2_bem_estar.repositorio import (
    RepositorioA2Memoria,
    RepositorioA2Protocol,
)
from sinergia.agentes.a2_bem_estar.schemas import (
    MAPA_ID_BASE_PARA_SLUG,
    MAPA_INTENCAO_PARA_ID_BASE,
    FonteCitada,
    MotivoSolicitacaoEPI,
    PerfilColaboradorResponder,
)
from sinergia.rag.interfaces import (
    ContextoRAG,
    DetalheScoreTrecho,
    EstatisticasEmbedding,
    ResultadoBuscaInstrumentada,
    TrechoRecuperado,
)
from sinergia.rag.tools import (
    _get_buscador,
    ferramenta_busca_direitos,
    ferramenta_busca_epis,
    ferramenta_busca_pausas,
    ferramenta_busca_procedimentos,
    ferramenta_busca_saude_ocupacional,
)

logger = logging.getLogger(__name__)

MAPA_BASES_A2: dict[int, dict[str, Any]] = {
    1: {
        "id_base": 1,
        "codigo": "F1",
        "slug": "pausas",
        "agente_id": "especialista_pausas",
        "nome": "Pausas e Ergonomia (NR-17)",
        "tool_name": "ferramenta_busca_pausas",
        "tool": ferramenta_busca_pausas,
        "descricao": (
            "Consulta normas de ergonomia (NR-17), pausas para descanso, "
            "limites de digitação contínua, postura e ritmo de trabalho."
        ),
    },
    2: {
        "id_base": 2,
        "codigo": "F2",
        "slug": "direitos_beneficios",
        "agente_id": "especialista_direitos_beneficios",
        "nome": "Direitos e Benefícios (CLT)",
        "tool_name": "ferramenta_busca_direitos",
        "tool": ferramenta_busca_direitos,
        "descricao": (
            "Consulta a CLT, direitos trabalhistas, benefícios, férias, "
            "licenças e faltas justificadas (ex: casamento, nascimento, falecimento)."
        ),
    },
    3: {
        "id_base": 3,
        "codigo": "F3",
        "slug": "procedimentos",
        "agente_id": "especialista_procedimentos",
        "nome": "Procedimentos e Treinamentos (POPs)",
        "tool_name": "ferramenta_busca_procedimentos",
        "tool": ferramenta_busca_procedimentos,
        "descricao": (
            "Consulta Procedimentos Operacionais Padrão (POPs), onboarding, "
            "fluxos internos e cronogramas de treinamentos e capacitação."
        ),
    },
    4: {
        "id_base": 4,
        "codigo": "F4",
        "slug": "saude_ocupacional",
        "agente_id": "especialista_saude_ocupacional",
        "nome": "Saúde Ocupacional (PCMSO)",
        "tool_name": "ferramenta_busca_saude_ocupacional",
        "tool": ferramenta_busca_saude_ocupacional,
        "descricao": (
            "Consulta diretrizes do PCMSO (NR-7), ASO, exames admissionais, "
            "periódicos, retorno ao trabalho e programas de saúde ocupacional."
        ),
    },
    5: {
        "id_base": 5,
        "codigo": "F5",
        "slug": "epi",
        "agente_id": "especialista_epi",
        "nome": "EPIs e Fichas de CA (NR-6)",
        "tool_name": "ferramenta_busca_epis",
        "tool": ferramenta_busca_epis,
        "descricao": (
            "Consulta fichas técnicas de EPIs, códigos de Certificado de Aprovação (CA), "
            "NR-6, luvas, capacetes, óculos, protetores e prazos de substituição."
        ),
    },
}

_FRASES_CONVERSACIONAIS_EXATAS: set[str] = {
    "oi",
    "ola",
    "bom dia",
    "boa tarde",
    "boa noite",
    "e ai",
    "opa",
    "ola tudo bem",
    "oi tudo bem",
    "bom dia tudo bem",
    "boa tarde tudo bem",
    "boa noite tudo bem",
    "tudo bem",
    "tudo bom",
    "como vai",
    "como voce esta",
    "obrigado",
    "obrigada",
    "muito obrigado",
    "muito obrigada",
    "obrigado pela ajuda",
    "obrigada pela ajuda",
    "valeu",
    "agradecido",
    "ok",
    "ok obrigado",
    "beleza",
    "entendi",
    "perfeito",
    "tchau",
    "ate logo",
    "ate mais",
    "quem e voce",
    "qual seu nome",
    "qual o seu nome",
    "o que voce faz",
    "como voce funciona",
    "como pode me ajudar",
    "como voce pode me ajudar",
    "me ajude",
    "ajuda",
    "preciso de ajuda",
}

_PREFIXOS_SAUDACAO: tuple[str, ...] = (
    "bom dia",
    "boa tarde",
    "boa noite",
    "ola tudo bem",
    "oi tudo bem",
    "tudo bem",
    "tudo bom",
    "ola",
    "oi",
    "opa",
    "e ai",
)

_MARCADORES_INTENCAO_PERGUNTA: set[str] = {
    "qual",
    "quais",
    "como",
    "quando",
    "quanto",
    "quantos",
    "quantas",
    "onde",
    "quem",
    "porque",
    "pode",
    "posso",
    "podemos",
    "tenho",
    "tem",
    "preciso",
    "precisa",
    "devo",
    "deve",
    "funciona",
    "funcionam",
    "regra",
    "regras",
    "direito",
    "direitos",
    "pausa",
    "pausas",
    "limite",
    "limites",
    "validade",
    "troca",
    "substituicao",
    "dias",
    "tempo",
    "fazer",
    "explica",
    "explique",
    "sobre",
    "fale",
    "quero",
    "saber",
    "duvida",
    "consultar",
    "ver",
    "mostrar",
    "mostre",
    "listar",
    "liste",
    "resumo",
}

_TERMOS_EPI: tuple[str, ...] = (
    "epi",
    "epis",
    "luva",
    "luvas",
    "protetor auricular",
    "auricular",
    "abafador",
    "oculos",
    "viseira",
    "capacete",
    "bota",
    "botina",
    "calcado",
    "mascara",
    "avental",
    "equipamento de protecao",
)

_PADROES_MOTIVO_EPI: tuple[tuple[MotivoSolicitacaoEPI, tuple[str, ...]], ...] = (
    (
        "AVARIA",
        (
            "rasg",
            "avaria",
            "danific",
            "quebr",
            "furad",
            "estrag",
            "rachad",
            "desgast",
            "defeit",
            "trinc",
        ),
    ),
    (
        "PERDA",
        ("perdi", "perdeu", "perdido", "perdida", "sumiu", "extravi"),
    ),
    (
        "NAO_RECEBIDO",
        (
            "nao receb",
            "nunca receb",
            "sem epi",
            "sem luva",
            "sem capacete",
            "sem oculos",
            "sem protetor",
            "nao foi entregue",
            "faltando",
            "nao entreg",
        ),
    ),
    (
        "TAMANHO_INADEQUADO",
        (
            "tamanho",
            "apertad",
            "folgad",
            "largo",
            "larga",
            "inadequad",
            "nao serve",
            "numero errado",
        ),
    ),
)


def _normalizar_texto(texto: str) -> str:
    sem_acentos = "".join(
        c
        for c in unicodedata.normalize("NFKD", texto or "")
        if not unicodedata.combining(c)
    )
    limpo = re.sub(r"[^a-zA-Z0-9\s]", " ", sem_acentos.lower())
    return re.sub(r"\s+", " ", limpo).strip()


def verificar_guardrail_conversacional(pergunta: str) -> bool:
    """Detecta saudações, agradecimentos e termos soltos sem intenção de busca RAG."""
    norm = _normalizar_texto(pergunta)
    if not norm or norm in _FRASES_CONVERSACIONAIS_EXATAS:
        return True

    restante = norm
    mudou = True
    while mudou and restante:
        mudou = False
        for pref in _PREFIXOS_SAUDACAO:
            if restante == pref:
                restante = ""
                mudou = True
                break
            if restante.startswith(pref + " "):
                restante = restante[len(pref) :].strip()
                mudou = True
                break

    if not restante or restante in _FRASES_CONVERSACIONAIS_EXATAS:
        return True

    tokens_restante = restante.split()
    return (
        len(tokens_restante) == 1
        and "?" not in (pergunta or "")
        and tokens_restante[0] not in _MARCADORES_INTENCAO_PERGUNTA
    )


def detectar_relato_problema_epi(
    pergunta: str,
) -> tuple[bool, MotivoSolicitacaoEPI | None]:
    """Identifica relatos de EPI avariado, perdido, não recebido ou inadequado (`F5`)."""
    norm = _normalizar_texto(pergunta)
    if not any(termo in norm for termo in _TERMOS_EPI):
        return False, None

    for motivo, padroes in _PADROES_MOTIVO_EPI:
        if any(p in norm for p in padroes):
            return True, motivo

    return False, None


class ChamadaBaseCrewA2(BaseModel):
    """Chamada de ferramenta RAG planejada pelo Gerente/Roteador da Crew A2."""

    id_base: int = Field(..., ge=1, le=5)
    codigo_base: str
    nome_base: str
    tool_name: str
    query_busca: str
    peso: float = Field(default=1.0, ge=0.0, le=1.0)
    limite_trechos: int = Field(default=4, ge=1)


class DecisaoRoteamentoCrewA2(BaseModel):
    """Decisão de roteamento gerada pelo Gerente da Crew A2."""

    id_base: int | None = Field(default=None, ge=1, le=5)
    codigo_base: str
    nome_base: str
    tool_name: str
    query_busca: str
    modo_roteamento: str
    roteamento_ms: float = 0.0
    chamadas: list[ChamadaBaseCrewA2] = Field(default_factory=list)
    usar_rag: bool = True

    @property
    def codigos_combinados(self) -> str:
        if not self.usar_rag:
            return self.codigo_base or "CHAT"
        if self.chamadas:
            return "+".join(c.codigo_base for c in self.chamadas)
        return self.codigo_base

    @property
    def multi_base(self) -> bool:
        return len(self.chamadas) > 1


class MetricasLLMA2(BaseModel):
    """Métricas de execução da LLM no Groq / CrewAI."""

    modelo: str
    prompt_eval_count: int = 0
    eval_count: int = 0
    load_duration_ms: float = 0.0
    eval_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    tokens_por_segundo: float = 0.0


class TemposExecucaoA2(BaseModel):
    """Tempos desmembrados da execução do Agente 2 em milissegundos."""

    total_ms: float = 0.0
    roteamento_ms: float = 0.0
    embed_ms: float = 0.0
    db_rrf_ms: float = 0.0
    llm_ms: float = 0.0


class TelemetriaExecucaoA2(BaseModel):
    """Coletor rico de telemetria do Agente 2 para consumo do `chat_demo`."""

    roteamento: DecisaoRoteamentoCrewA2
    resultados_rag: list[ResultadoBuscaInstrumentada] = Field(default_factory=list)
    detalhes_trechos: list[DetalheScoreTrecho] = Field(default_factory=list)
    estatisticas_embedding: EstatisticasEmbedding = Field(
        default_factory=lambda: EstatisticasEmbedding(dimensoes=0, norma_l2=0.0)
    )
    tempos: TemposExecucaoA2 = Field(default_factory=TemposExecucaoA2)
    metricas_llm: MetricasLLMA2
    citacoes: list[str] = Field(default_factory=list)
    fontes: list[FonteCitada] = Field(default_factory=list)
    reasoning: str = ""
    prompt_enviado: str = ""
    agentes_acionados: list[str] = Field(default_factory=list)
    id_solicitacao_epi: int | None = None
    motivo_solicitacao_epi: MotivoSolicitacaoEPI | None = None


class ResultadoExecucaoCrewA2(BaseModel):
    """Resultado completo devolvido por `ExecutorCrewA2.executar_responder`."""

    resposta: str
    base_slug: str
    trechos_usados: int
    fontes: list[FonteCitada] = Field(default_factory=list)
    id_solicitacao_epi: int | None = None
    motivo_solicitacao_epi: MotivoSolicitacaoEPI | None = None
    telemetria: TelemetriaExecucaoA2


class _ThinkStreamParser:
    """Separa tokens `<think>...</think>` de conteúdo visível em streaming."""

    _OPEN_TAG = "<think>"
    _CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False
        self._reasoning_parts: list[str] = []
        self._content_parts: list[str] = []

    def feed_reasoning(self, text: str) -> bool:
        if not text:
            return False
        self._reasoning_parts.append(text)
        return True

    def feed_content(self, text: str) -> tuple[bool, bool]:
        if not text:
            return False, False
        self._buffer += text
        mudou_r = False
        mudou_c = False
        while self._buffer:
            lower_buf = self._buffer.lower()
            if not self._in_think:
                idx = lower_buf.find(self._OPEN_TAG)
                if idx != -1:
                    if idx > 0:
                        self._content_parts.append(self._buffer[:idx])
                        mudou_c = True
                    self._buffer = self._buffer[idx + len(self._OPEN_TAG) :]
                    self._in_think = True
                    continue
                keep = len(self._OPEN_TAG) - 1
                if len(self._buffer) > keep:
                    self._content_parts.append(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                    mudou_c = True
                break
            idx = lower_buf.find(self._CLOSE_TAG)
            if idx != -1:
                if idx > 0:
                    self._reasoning_parts.append(self._buffer[:idx])
                    mudou_r = True
                self._buffer = self._buffer[idx + len(self._CLOSE_TAG) :]
                self._in_think = False
                continue
            keep = len(self._CLOSE_TAG) - 1
            if len(self._buffer) > keep:
                self._reasoning_parts.append(self._buffer[:-keep])
                self._buffer = self._buffer[-keep:]
                mudou_r = True
            break
        return mudou_r, mudou_c

    def flush(self) -> None:
        if not self._buffer:
            return
        if self._in_think:
            self._reasoning_parts.append(self._buffer)
        else:
            self._content_parts.append(self._buffer)
        self._buffer = ""

    @property
    def reasoning_text(self) -> str:
        return "".join(self._reasoning_parts).strip()

    @property
    def content_text(self) -> str:
        return "".join(self._content_parts).strip()


def _extrair_citacoes_e_fontes(
    trechos: list[TrechoRecuperado],
) -> tuple[list[str], list[FonteCitada]]:
    """Extrai pílulas de citação para UI e objetos `FonteCitada` para `sinergia/1`."""
    citacoes: list[str] = []
    fontes: list[FonteCitada] = []
    vistos_cit: set[str] = set()
    vistos_fontes: set[tuple[str, str, str]] = set()

    for trecho in trechos:
        meta = trecho.metadados
        secao = meta.secao or "Geral"
        pagina = f"Pág. {meta.pagina}" if meta.pagina is not None else "Pág. N/A"
        label = f"{meta.titulo} (v{meta.versao}) · {secao} · {pagina}"
        if label not in vistos_cit:
            vistos_cit.add(label)
            citacoes.append(label)

        base_slug = MAPA_ID_BASE_PARA_SLUG.get(meta.id_base or 1, "pausas")
        chave_fonte = (base_slug, meta.titulo, meta.versao)
        if chave_fonte not in vistos_fontes:
            vistos_fontes.add(chave_fonte)
            fontes.append(
                FonteCitada(
                    base=base_slug,
                    titulo=meta.titulo,
                    versao=meta.versao,
                )
            )

    return citacoes, fontes


def _calcular_alocacao_trechos(
    pesos_brutos: list[float], orcamento_multi: int = 6, limite_unico: int = 4
) -> tuple[list[float], list[int]]:
    n = len(pesos_brutos)
    if n == 0:
        return [], []
    if n == 1:
        return [1.0], [limite_unico]

    clamped = [max(0.05, float(w)) for w in pesos_brutos]
    soma = sum(clamped)
    pesos_norm = [round(w / soma, 4) for w in clamped]
    diff = round(1.0 - sum(pesos_norm), 4)
    if diff != 0.0:
        pesos_norm[0] = round(max(0.0, min(1.0, pesos_norm[0] + diff)), 4)

    orcamento = max(n, orcamento_multi)
    alocados = [1] * n
    restante = orcamento - n
    if restante > 0:
        quotas = [p * restante for p in pesos_norm]
        inteiras = [math.floor(q) for q in quotas]
        for i in range(n):
            alocados[i] += inteiras[i]
        sobra = restante - sum(inteiras)
        if sobra > 0:
            ordenados = sorted(
                range(n),
                key=lambda i: (quotas[i] - inteiras[i], pesos_norm[i], -i),
                reverse=True,
            )
            for idx in ordenados[:sobra]:
                alocados[idx] += 1
    return pesos_norm, alocados


def _converter_decisao_externa(decisao_raw: Any) -> DecisaoRoteamentoCrewA2:
    """Converte `DecisaoRoteamento` externa (duck-typed) para `DecisaoRoteamentoCrewA2`."""
    if isinstance(decisao_raw, DecisaoRoteamentoCrewA2):
        return decisao_raw
    chamadas = [
        ChamadaBaseCrewA2(
            id_base=c.id_base,
            codigo_base=c.codigo_base,
            nome_base=c.nome_base,
            tool_name=c.tool_name,
            query_busca=c.query_busca,
            peso=c.peso,
            limite_trechos=c.limite_trechos,
        )
        for c in (getattr(decisao_raw, "chamadas", None) or [])
    ]
    return DecisaoRoteamentoCrewA2(
        id_base=getattr(decisao_raw, "id_base", None),
        codigo_base=getattr(decisao_raw, "codigo_base", "CHAT"),
        nome_base=getattr(decisao_raw, "nome_base", ""),
        tool_name=getattr(decisao_raw, "tool_name", "nenhuma"),
        query_busca=getattr(decisao_raw, "query_busca", ""),
        modo_roteamento=getattr(decisao_raw, "modo_roteamento", "tool_calling"),
        roteamento_ms=float(getattr(decisao_raw, "roteamento_ms", 0.0)),
        chamadas=chamadas,
        usar_rag=bool(getattr(decisao_raw, "usar_rag", True)),
    )


def _formatar_mensagens_para_texto(messages: str | list[dict[str, Any]]) -> str:
    if isinstance(messages, str):
        return messages
    partes: list[str] = []
    for m in messages:
        if isinstance(m, dict):
            role = str(m.get("role", "user")).upper()
            content = str(m.get("content", ""))
            partes.append(f"[{role}]\n{content}")
    return "\n\n".join(partes)


def _normalizar_historico(
    historico: list[dict[str, str]] | None,
    historico_resumo: str | None = None,
    *,
    max_mensagens: int = 4,
) -> list[dict[str, str]]:
    mensagens: list[dict[str, str]] = []
    if historico:
        for item in historico:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = (item.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                mensagens.append({"role": role, "content": content[:400]})
    elif historico_resumo and historico_resumo.strip():
        mensagens.append({"role": "user", "content": historico_resumo.strip()[:400]})
    return mensagens[-max_mensagens:]


class GroqCrewLLM(BaseLLM):
    """Adaptador `BaseLLM` da CrewAI conectado à Groq Cloud ou a serviços/mocks injetados."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        timeout_s: float = 45.0,
        servico_llm: Any | None = None,
        custom_llm: Any | None = None,
        pergunta: str = "",
        contexto_rag: ContextoRAG | None = None,
        usar_rag: bool = True,
        historico: list[dict[str, str]] | None = None,
        perfil_texto: str = "",
        info_solicitacao_epi: str = "",
        ao_receber_reasoning: Callable[[str], None] | None = None,
        ao_receber_conteudo: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(model=model)
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._servico_llm = servico_llm
        self._custom_llm = custom_llm
        self._pergunta = pergunta
        self._contexto_rag = contexto_rag or ContextoRAG()
        self._usar_rag = usar_rag
        self._historico = historico or []
        self._perfil_texto = perfil_texto
        self._info_solicitacao_epi = info_solicitacao_epi
        self._ao_receber_reasoning = ao_receber_reasoning
        self._ao_receber_conteudo = ao_receber_conteudo

        self.ultima_resposta: str = ""
        self.ultimo_reasoning: str = ""
        self.ultimo_prompt_enviado: str = ""
        self.ultima_llm_ms: float = 0.0
        self.ultimas_metricas: MetricasLLMA2 = MetricasLLMA2(modelo=model)

    def _gerar_fallback_offline(self, prompt_texto: str) -> str:
        if not self._usar_rag:
            return (
                "Olá! Sou o assistente de Bem-Estar do SinergIA. Posso ajudar você com "
                "dúvidas sobre Pausas e Ergonomia (F1/NR-17), Direitos e Benefícios (F2/CLT), "
                "Procedimentos e Treinamentos (F3/POPs), Saúde Ocupacional (F4/PCMSO) e EPIs (F5/NR-6)."
            )

        partes: list[str] = []
        if self._info_solicitacao_epi:
            partes.append(
                "Não faça a atividade de risco sem o EPI adequado e não use o equipamento danificado (NR-6). "
                f"Abri a solicitação de reposição {self._info_solicitacao_epi} e o TST e seu gestor serão avisados."
            )

        if self._contexto_rag.trechos:
            trecho = self._contexto_rag.trechos[0]
            meta = trecho.metadados
            trecho_curto = trecho.conteudo.strip()
            if len(trecho_curto) > 280:
                trecho_curto = trecho_curto[:280].rstrip() + "..."
            partes.append(
                f"Conforme {meta.titulo} (v{meta.versao}, {meta.secao or 'Geral'}): {trecho_curto}"
            )
        elif not partes:
            partes.append(
                "Chave `GROQ_API_KEY` não configurada para geração em nuvem e não há trechos locais correspondentes."
            )

        self.ultimo_prompt_enviado = prompt_texto
        return "\n\n".join(partes)

    def _chamar_groq_http(self, messages: str | list[dict[str, Any]]) -> str:
        inicio = time.perf_counter()
        if isinstance(messages, str):
            payload_msgs = [{"role": "user", "content": messages}]
        else:
            payload_msgs = [
                {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
                for m in messages
                if isinstance(m, dict)
            ]

        prompt_texto = _formatar_mensagens_para_texto(payload_msgs)
        if not self._api_key:
            self.ultima_llm_ms = round((time.perf_counter() - inicio) * 1000.0, 2)
            self.ultimo_prompt_enviado = prompt_texto
            return self._gerar_fallback_offline(prompt_texto)

        parser = _ThinkStreamParser()
        usage: dict[str, Any] = {}
        try:
            payload = {
                "model": self.model,
                "temperature": 0.1,
                "stream": True,
                "messages": payload_msgs,
            }
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            for tentativa in range(3):
                parser = _ThinkStreamParser()
                usage = {}
                try:
                    with httpx.stream(
                        "POST",
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self._timeout_s,
                    ) as resp:
                        if getattr(resp, "status_code", 200) >= 400:
                            try:
                                resp.read()
                            except (httpx.HTTPError, httpx.StreamError):
                                pass
                        resp.raise_for_status()
                        for linha in resp.iter_lines():
                            if not linha or not linha.startswith("data:"):
                                continue
                            data_str = linha[5:].strip()
                            if data_str == "[DONE]":
                                break
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices") or []
                            if choices:
                                delta = choices[0].get("delta") or {}
                                r_delta = delta.get("reasoning") or delta.get("reasoning_content") or ""
                                c_delta = delta.get("content") or ""
                                mr = parser.feed_reasoning(r_delta)
                                mr2, mc = parser.feed_content(c_delta)
                                if (mr or mr2) and self._ao_receber_reasoning and parser.reasoning_text:
                                    self._ao_receber_reasoning(parser.reasoning_text)
                                if mc and self._ao_receber_conteudo and parser.content_text:
                                    self._ao_receber_conteudo(parser.content_text)
                            if isinstance(chunk.get("usage"), dict):
                                usage = chunk["usage"]
                            elif isinstance(chunk.get("x_groq"), dict) and isinstance(
                                chunk["x_groq"].get("usage"), dict
                            ):
                                usage = chunk["x_groq"]["usage"]
                    break
                except httpx.HTTPStatusError as erro_http:
                    if getattr(erro_http.response, "status_code", None) == 429 and tentativa < 2:
                        time.sleep(min(2.0, 0.8 * (tentativa + 1)))
                        continue
                    raise

            parser.flush()
            if self._ao_receber_reasoning and parser.reasoning_text:
                self._ao_receber_reasoning(parser.reasoning_text)
            if self._ao_receber_conteudo and parser.content_text:
                self._ao_receber_conteudo(parser.content_text)

            llm_ms = round((time.perf_counter() - inicio) * 1000.0, 2)
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            eval_tokens = int(usage.get("completion_tokens") or 0)
            queue_s = float(usage.get("queue_time") or 0.0)
            completion_s = float(usage.get("completion_time") or 0.0)
            total_s = float(usage.get("total_time") or 0.0)
            tok_s = (eval_tokens / completion_s) if completion_s > 0 and eval_tokens > 0 else 0.0

            self.ultima_llm_ms = llm_ms
            self.ultimo_reasoning = parser.reasoning_text
            self.ultimo_prompt_enviado = prompt_texto
            self.ultimas_metricas = MetricasLLMA2(
                modelo=self.model,
                prompt_eval_count=prompt_tokens,
                eval_count=eval_tokens,
                load_duration_ms=round(queue_s * 1000.0, 2),
                eval_duration_ms=round(completion_s * 1000.0, 2),
                total_duration_ms=round(total_s * 1000.0, 2),
                tokens_por_segundo=round(tok_s, 1),
            )
            return parser.content_text or self._gerar_fallback_offline(prompt_texto)
        except (httpx.HTTPError, httpx.StreamError, ValueError, KeyError, TypeError) as erro:
            logger.debug("Fallback controlado na GroqCrewLLM (%s): %s", self.model, erro)
            self.ultima_llm_ms = round((time.perf_counter() - inicio) * 1000.0, 2)
            self.ultimo_prompt_enviado = prompt_texto
            return self._gerar_fallback_offline(prompt_texto)

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Executa a chamada LLM para a Task ativa da CrewAI."""
        del tools, callbacks, available_functions, kwargs

        if self._custom_llm is not None:
            t0 = time.perf_counter()
            prompt_txt = _formatar_mensagens_para_texto(messages)
            if hasattr(self._custom_llm, "call"):
                out = self._custom_llm.call(messages)
            elif hasattr(self._custom_llm, "invoke"):
                out = self._custom_llm.invoke(prompt_txt)
            elif callable(self._custom_llm):
                out = self._custom_llm(prompt_txt)
            else:
                out = str(self._custom_llm)
            texto = out.content if hasattr(out, "content") else str(out)
            self.ultima_llm_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            self.ultimo_prompt_enviado = prompt_txt
            self.ultima_resposta = str(texto).strip()
            self.ultimas_metricas = MetricasLLMA2(modelo=self.model)
            return self.ultima_resposta

        if self._servico_llm is not None:
            t0 = time.perf_counter()
            try:
                resp_servico = self._servico_llm.gerar_resposta_fundamentada(
                    pergunta=self._pergunta,
                    contexto=self._contexto_rag,
                    modelo=self.model,
                    ao_receber_reasoning=self._ao_receber_reasoning,
                    ao_receber_conteudo=self._ao_receber_conteudo,
                    usar_rag=self._usar_rag,
                    historico=self._historico,
                )
            except TypeError:
                resp_servico = self._servico_llm.gerar_resposta_fundamentada(
                    pergunta=self._pergunta,
                    contexto=self._contexto_rag,
                    modelo=self.model,
                    ao_receber_reasoning=self._ao_receber_reasoning,
                    ao_receber_conteudo=self._ao_receber_conteudo,
                )

            llm_ms = float(
                getattr(resp_servico, "llm_ms", (time.perf_counter() - t0) * 1000.0)
            )
            metricas_raw = getattr(resp_servico, "metricas", None)
            if metricas_raw is not None:
                self.ultimas_metricas = MetricasLLMA2(
                    modelo=getattr(metricas_raw, "modelo", self.model),
                    prompt_eval_count=int(getattr(metricas_raw, "prompt_eval_count", 0)),
                    eval_count=int(getattr(metricas_raw, "eval_count", 0)),
                    load_duration_ms=float(getattr(metricas_raw, "load_duration_ms", 0.0)),
                    eval_duration_ms=float(getattr(metricas_raw, "eval_duration_ms", 0.0)),
                    total_duration_ms=float(getattr(metricas_raw, "total_duration_ms", 0.0)),
                    tokens_por_segundo=float(getattr(metricas_raw, "tokens_por_segundo", 0.0)),
                )
            self.ultima_llm_ms = round(llm_ms, 2)
            self.ultimo_reasoning = str(getattr(resp_servico, "reasoning", "") or "")
            self.ultimo_prompt_enviado = str(
                getattr(resp_servico, "prompt_enviado", "")
                or _formatar_mensagens_para_texto(messages)
            )
            self.ultima_resposta = str(getattr(resp_servico, "resposta", "") or "").strip()
            return self.ultima_resposta

        self.ultima_resposta = self._chamar_groq_http(messages)
        return self.ultima_resposta


class ExecutorCrewA2:
    """Orquestra a CrewAI do Agente 2 (Gerente, 5 Especialistas F1..F5 e Redator Final)."""

    def __init__(
        self,
        *,
        config: A2Config | None = None,
        repositorio: RepositorioA2Protocol | None = None,
        buscador: Any | None = None,
        servico_llm: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        self.config = config or obter_config_a2()
        self.repositorio: RepositorioA2Protocol = repositorio or RepositorioA2Memoria()
        self.buscador = buscador
        self.servico_llm = servico_llm
        self.llm = llm

        # Constrói os 7 agentes oficiais da Crew do A2 (Seção 5.1)
        llm_placeholder = (
            llm
            if isinstance(llm, BaseLLM)
            else GroqCrewLLM(
                model=self.config.llm_model,
                api_key=self.config.groq_api_key,
                timeout_s=self.config.timeout_s,
                custom_llm=llm,
            )
        )
        self.gerente_roteador = Agent(
            role=PROMPT_GERENTE_ROTEADOR_A2["role"],
            goal=PROMPT_GERENTE_ROTEADOR_A2["goal"],
            backstory=PROMPT_GERENTE_ROTEADOR_A2["backstory"],
            llm=llm_placeholder,
            allow_delegation=False,
            verbose=False,
        )
        self.especialistas: dict[int, Agent] = {}
        for id_base, spec in PROMPTS_ESPECIALISTAS_A2.items():
            info_base = MAPA_BASES_A2[id_base]
            self.especialistas[id_base] = Agent(
                role=spec["role"],
                goal=spec["goal"],
                backstory=spec["backstory"],
                tools=[info_base["tool"]],
                llm=llm_placeholder,
                allow_delegation=False,
                verbose=False,
            )
        self.redator_final = Agent(
            role=PROMPT_REDATOR_FINAL_A2["role"],
            goal=PROMPT_REDATOR_FINAL_A2["goal"],
            backstory=PROMPT_REDATOR_FINAL_A2["backstory"],
            llm=llm_placeholder,
            allow_delegation=False,
            verbose=False,
        )

    def _obter_buscador_efetivo(self) -> Any:
        return self.buscador if self.buscador is not None else _get_buscador()

    def _decidir_heuristica(
        self,
        pergunta: str,
        inicio: float,
        historico: list[dict[str, str]] | None = None,
    ) -> DecisaoRoteamentoCrewA2:
        texto = _normalizar_texto(pergunta)
        regras = [
            (5, ("epi", "ca ", "nr 6", "luva", "capacete", "oculos", "protetor", "substituicao", "rasg")),
            (4, ("pcmso", "aso", "exame", "saude", "medico", "nr 7", "audiometria", "vacina")),
            (3, ("pop", "procedimento", "treinamento", "capacitacao", "onboarding", "cronograma")),
            (2, ("clt", "casamento", "falta", "beneficio", "ferias", "licenca", "473", "direito", "13")),
            (1, ("nr 17", "pausa", "digitacao", "ergonomia", "descanso", "postura", "toques")),
        ]
        encontradas: list[tuple[int, str, float]] = []
        for id_base, termos in regras:
            matches = sum(1 for termo in termos if termo in texto)
            if matches > 0:
                encontradas.append((id_base, pergunta, float(matches)))

        if not encontradas and historico:
            for msg in reversed(_normalizar_historico(historico)):
                if msg["role"] != "user":
                    continue
                texto_ant = _normalizar_texto(msg["content"])
                query_comb = f"{msg['content']} {pergunta}".strip()
                for id_base, termos in regras:
                    matches = sum(1 for termo in termos if termo in texto_ant)
                    if matches > 0:
                        encontradas.append((id_base, query_comb, float(matches)))
                if encontradas:
                    break

        if not encontradas:
            encontradas = [(1, pergunta, 1.0)]

        ordenados = sorted(encontradas, key=lambda x: -x[2])[:5]
        pesos_norm, limites = _calcular_alocacao_trechos([x[2] for x in ordenados])
        chamadas = []
        for (id_b, q, _), peso_n, lim in zip(ordenados, pesos_norm, limites):
            info = MAPA_BASES_A2[id_b]
            chamadas.append(
                ChamadaBaseCrewA2(
                    id_base=id_b,
                    codigo_base=info["codigo"],
                    nome_base=info["nome"],
                    tool_name=info["tool_name"],
                    query_busca=q,
                    peso=peso_n,
                    limite_trechos=lim,
                )
            )
        principal = chamadas[0]
        return DecisaoRoteamentoCrewA2(
            id_base=principal.id_base,
            codigo_base=principal.codigo_base,
            nome_base=principal.nome_base,
            tool_name=principal.tool_name,
            query_busca=principal.query_busca,
            modo_roteamento="heuristic_fallback",
            roteamento_ms=round((time.perf_counter() - inicio) * 1000.0, 2),
            chamadas=chamadas,
            usar_rag=True,
        )

    def _rotear_via_groq_http(
        self,
        *,
        pergunta: str,
        modelo_ativo: str,
        inicio: float,
        historico: list[dict[str, str]] | None = None,
    ) -> DecisaoRoteamentoCrewA2 | None:
        """Executa roteamento autônomo via tool calling na API da Groq quando `servico_llm` não foi injetado."""
        if not self.config.groq_api_key:
            return None
        tools = [
            {
                "type": "function",
                "function": {
                    "name": info["tool_name"],
                    "description": info["descricao"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Termos-chave da consulta."},
                            "peso": {"type": "number", "description": "Peso de relevância (0.1 a 1.0)."},
                        },
                        "required": ["query"],
                    },
                },
            }
            for info in MAPA_BASES_A2.values()
        ]
        mensagens = [
            {
                "role": "system",
                "content": (
                    f"{PROMPT_GERENTE_ROTEADOR_A2['goal']}\n"
                    "Selecione a(s) ferramenta(s) de busca F1..F5 estritamente relevantes para a pergunta."
                ),
            },
            *_normalizar_historico(historico, max_mensagens=2),
            {"role": "user", "content": pergunta},
        ]
        headers = {
            "Authorization": f"Bearer {self.config.groq_api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": modelo_ativo,
                    "temperature": 0.0,
                    "messages": mensagens,
                    "tools": tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                },
                timeout=self.config.timeout_s,
            )
            resp.raise_for_status()
            msg = (resp.json().get("choices") or [{}])[0].get("message", {})
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return None

            tool_to_id = {info["tool_name"]: b_id for b_id, info in MAPA_BASES_A2.items()}
            itens: list[tuple[int, str, float]] = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                t_name = fn.get("name", "")
                if t_name not in tool_to_id:
                    continue
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                q = str(args.get("query") or pergunta)
                try:
                    p = float(args.get("peso", 1.0))
                except (TypeError, ValueError):
                    p = 1.0
                itens.append((tool_to_id[t_name], q, p))

            if not itens:
                return None
            itens.sort(key=lambda x: -x[2])
            pesos_norm, limites = _calcular_alocacao_trechos([x[2] for x in itens])
            chamadas = [
                ChamadaBaseCrewA2(
                    id_base=id_b,
                    codigo_base=MAPA_BASES_A2[id_b]["codigo"],
                    nome_base=MAPA_BASES_A2[id_b]["nome"],
                    tool_name=MAPA_BASES_A2[id_b]["tool_name"],
                    query_busca=q,
                    peso=peso_n,
                    limite_trechos=lim,
                )
                for (id_b, q, _), peso_n, lim in zip(itens, pesos_norm, limites)
            ]
            principal = chamadas[0]
            return DecisaoRoteamentoCrewA2(
                id_base=principal.id_base,
                codigo_base=principal.codigo_base,
                nome_base=principal.nome_base,
                tool_name=principal.tool_name,
                query_busca=principal.query_busca,
                modo_roteamento="tool_calling",
                roteamento_ms=round((time.perf_counter() - inicio) * 1000.0, 2),
                chamadas=chamadas,
                usar_rag=True,
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as erro:
            logger.debug("Roteamento Groq indisponível (%s): %s", modelo_ativo, erro)
            return None

    def _decidir_roteamento(
        self,
        *,
        pergunta: str,
        intencao: str | None,
        modelo_ativo: str,
        historico: list[dict[str, str]] | None,
        eh_relato_epi: bool,
        servico_llm: Any | None,
    ) -> DecisaoRoteamentoCrewA2:
        inicio = time.perf_counter()

        if verificar_guardrail_conversacional(pergunta):
            return DecisaoRoteamentoCrewA2(
                id_base=None,
                codigo_base="CHAT",
                nome_base="Chat Direto (Sem RAG)",
                tool_name="nenhuma",
                query_busca="",
                modo_roteamento="guardrail_conversacional",
                roteamento_ms=round((time.perf_counter() - inicio) * 1000.0, 2),
                chamadas=[],
                usar_rag=False,
            )

        if eh_relato_epi:
            info = MAPA_BASES_A2[5]
            return DecisaoRoteamentoCrewA2(
                id_base=5,
                codigo_base=info["codigo"],
                nome_base=info["nome"],
                tool_name=info["tool_name"],
                query_busca=f"reposicao EPI NR-6 {pergunta}",
                modo_roteamento="epi_nr6_roteiro",
                roteamento_ms=round((time.perf_counter() - inicio) * 1000.0, 2),
                chamadas=[
                    ChamadaBaseCrewA2(
                        id_base=5,
                        codigo_base=info["codigo"],
                        nome_base=info["nome"],
                        tool_name=info["tool_name"],
                        query_busca=f"reposicao EPI NR-6 {pergunta}",
                        peso=1.0,
                        limite_trechos=4,
                    )
                ],
                usar_rag=True,
            )

        if servico_llm is not None:
            try:
                decisao_ext = servico_llm.decidir_base_autonomamente(
                    pergunta, modelo=modelo_ativo, historico=historico
                )
            except TypeError:
                decisao_ext = servico_llm.decidir_base_autonomamente(
                    pergunta, modelo=modelo_ativo
                )
            return _converter_decisao_externa(decisao_ext)

        if intencao:
            chave = intencao.strip().upper()
            id_base = MAPA_INTENCAO_PARA_ID_BASE.get(chave)
            if id_base is not None:
                info = MAPA_BASES_A2[id_base]
                return DecisaoRoteamentoCrewA2(
                    id_base=id_base,
                    codigo_base=info["codigo"],
                    nome_base=info["nome"],
                    tool_name=info["tool_name"],
                    query_busca=pergunta,
                    modo_roteamento="intencao_a0",
                    roteamento_ms=round((time.perf_counter() - inicio) * 1000.0, 2),
                    chamadas=[
                        ChamadaBaseCrewA2(
                            id_base=id_base,
                            codigo_base=info["codigo"],
                            nome_base=info["nome"],
                            tool_name=info["tool_name"],
                            query_busca=pergunta,
                            peso=1.0,
                            limite_trechos=4,
                        )
                    ],
                    usar_rag=True,
                )

        decisao_groq = self._rotear_via_groq_http(
            pergunta=pergunta,
            modelo_ativo=modelo_ativo,
            inicio=inicio,
            historico=historico,
        )
        if decisao_groq is not None:
            return decisao_groq

        return self._decidir_heuristica(pergunta, inicio, historico=historico)

    @staticmethod
    def _executar_busca_instrumentada(
        buscador: Any,
        chamada: ChamadaBaseCrewA2,
    ) -> ResultadoBuscaInstrumentada:
        if hasattr(buscador, "buscar_com_telemetria"):
            return buscador.buscar_com_telemetria(
                query=chamada.query_busca,
                id_base=chamada.id_base,
                limite=chamada.limite_trechos,
            )
        contexto = buscador.buscar(
            query=chamada.query_busca,
            id_base=chamada.id_base,
            limite=chamada.limite_trechos,
        )
        return ResultadoBuscaInstrumentada(
            contexto=contexto,
            embed_ms=0.0,
            db_rrf_ms=0.0,
            estatisticas_embedding=EstatisticasEmbedding(dimensoes=0, norma_l2=0.0),
            detalhes_trechos=[],
        )

    def executar_responder(
        self,
        *,
        pergunta: str,
        id_colaborador: int | None = None,
        intencao: str | None = None,
        perfil: PerfilColaboradorResponder | dict[str, Any] | None = None,
        historico_resumo: str | None = None,
        historico: list[dict[str, str]] | None = None,
        modelo: str | None = None,
        buscador: Any | None = None,
        servico_llm: Any | None = None,
        llm: Any | None = None,
        ao_avancar_etapa: Callable[[str], None] | None = None,
        ao_receber_reasoning: Callable[[str], None] | None = None,
        ao_receber_conteudo: Callable[[str], None] | None = None,
    ) -> ResultadoExecucaoCrewA2:
        """Executa o fluxo completo `a2.responder` via CrewAI + Groq + RAG instrumentado."""
        servico_ativo = servico_llm if servico_llm is not None else self.servico_llm
        llm_ativo = llm if llm is not None else self.llm
        modelo_ativo = (modelo or self.config.llm_model).strip()
        historico_norm = _normalizar_historico(historico, historico_resumo)

        # Completa perfil do repositório se não veio completo do A0
        perfil_efetivo: PerfilColaboradorResponder | dict[str, Any] | None = perfil
        if perfil_efetivo is None and id_colaborador is not None:
            perfil_db = self.repositorio.obter_perfil_colaborador(id_colaborador)
            if perfil_db is not None:
                perfil_efetivo = PerfilColaboradorResponder(
                    cargo=perfil_db.cargo,
                    setor=perfil_db.setor,
                    beneficios=perfil_db.beneficios,
                )
        perfil_texto = formatar_perfil_para_prompt(perfil_efetivo)

        eh_relato_epi, motivo_epi = detectar_relato_problema_epi(pergunta)

        if ao_avancar_etapa:
            ao_avancar_etapa("Roteando pergunta autonomamente entre as bases F1..F5 (Crew A2)...")

        decisao = self._decidir_roteamento(
            pergunta=pergunta,
            intencao=intencao,
            modelo_ativo=modelo_ativo,
            historico=historico_norm,
            eh_relato_epi=eh_relato_epi,
            servico_llm=servico_ativo,
        )

        trechos_mesclados: list[TrechoRecuperado] = []
        detalhes_mesclados: list[DetalheScoreTrecho] = []
        resultados_rag: list[ResultadoBuscaInstrumentada] = []
        embed_ms_total = 0.0
        db_rrf_ms_total = 0.0
        estatisticas_primarias: EstatisticasEmbedding | None = None

        agentes_acionados: list[str] = ["gerente_roteador"]

        if decisao.usar_rag and decisao.chamadas:
            buscador_efetivo = buscador if buscador is not None else self._obter_buscador_efetivo()
            for chamada in decisao.chamadas:
                if ao_avancar_etapa:
                    if len(decisao.chamadas) == 1:
                        ao_avancar_etapa(
                            f"Acionando especialista **{chamada.codigo_base}** ({chamada.nome_base})..."
                        )
                    else:
                        ao_avancar_etapa(
                            f"Buscando {chamada.limite_trechos} trecho(s) na base **{chamada.codigo_base}** "
                            f"({chamada.nome_base} · peso {chamada.peso * 100:.0f}%)..."
                        )
                res_rag = self._executar_busca_instrumentada(buscador_efetivo, chamada)
                resultados_rag.append(res_rag)
                trechos_mesclados.extend(res_rag.contexto.trechos)
                detalhes_mesclados.extend(res_rag.detalhes_trechos)
                embed_ms_total += float(res_rag.embed_ms)
                db_rrf_ms_total += float(res_rag.db_rrf_ms)
                if estatisticas_primarias is None:
                    estatisticas_primarias = res_rag.estatisticas_embedding
                esp_id = MAPA_BASES_A2[chamada.id_base]["agente_id"]
                if esp_id not in agentes_acionados:
                    agentes_acionados.append(esp_id)

        agentes_acionados.append("redator_final")

        # Se for relato de problema com EPI (F5 / NR-6), abre solicitação em apoio.solicitacao_epi
        id_solicitacao_epi: int | None = None
        info_solicitacao_epi = ""
        if eh_relato_epi and motivo_epi is not None and decisao.usar_rag:
            colab_id = id_colaborador if id_colaborador is not None and id_colaborador > 0 else 42
            id_epi = self.repositorio.resolver_id_epi_por_texto(
                pergunta, id_colaborador=colab_id
            )
            sol = self.repositorio.abrir_solicitacao_epi(
                id_colaborador=colab_id,
                id_epi=id_epi,
                motivo=motivo_epi,
                descricao=pergunta,
            )
            id_solicitacao_epi = sol.id_solicitacao
            info_solicitacao_epi = f"nº {sol.id_solicitacao} (status={sol.status}, motivo={sol.motivo})"

        if decisao.usar_rag:
            contexto_final = ContextoRAG(
                query=decisao.query_busca,
                id_base=decisao.id_base,
                trechos=trechos_mesclados,
                total_encontrado=len(trechos_mesclados),
            )
            estatisticas_finais = estatisticas_primarias or EstatisticasEmbedding(
                dimensoes=0, norma_l2=0.0
            )
            citacoes, fontes = _extrair_citacoes_e_fontes(trechos_mesclados)
            if eh_relato_epi and not fontes:
                fontes = [
                    FonteCitada(
                        base="epi",
                        titulo="Procedimento de reposição de EPI",
                        versao="1.2",
                    )
                ]
            if ao_avancar_etapa:
                ao_avancar_etapa(f"Redator Final sintetizando resposta via CrewAI (`{modelo_ativo}`)...")
        else:
            contexto_final = ContextoRAG(
                query=None, id_base=None, trechos=[], total_encontrado=0
            )
            estatisticas_finais = EstatisticasEmbedding(dimensoes=0, norma_l2=0.0)
            citacoes, fontes = [], []
            if ao_avancar_etapa:
                ao_avancar_etapa(
                    f"Redator Final respondendo ao colaborador via CrewAI (`{modelo_ativo}`, sem RAG)..."
                )

        # Prepara LLM da Crew para este turno
        if isinstance(llm_ativo, BaseLLM):
            crew_llm: BaseLLM = llm_ativo
        else:
            crew_llm = GroqCrewLLM(
                model=modelo_ativo,
                api_key=self.config.groq_api_key,
                timeout_s=self.config.timeout_s,
                servico_llm=servico_ativo,
                custom_llm=llm_ativo,
                pergunta=pergunta,
                contexto_rag=contexto_final,
                usar_rag=decisao.usar_rag,
                historico=historico_norm,
                perfil_texto=perfil_texto,
                info_solicitacao_epi=info_solicitacao_epi,
                ao_receber_reasoning=ao_receber_reasoning,
                ao_receber_conteudo=ao_receber_conteudo,
            )

        contexto_txt = contexto_final.formatar_para_prompt() if decisao.usar_rag else ""
        historico_txt = "\n".join(
            f"{'Colaborador' if m['role'] == 'user' else 'SinergIA'}: {m['content']}"
            for m in historico_norm
        )

        # Monta contexto dos especialistas selecionados para orientar o Redator Final na Crew
        notas_especialistas: list[str] = []
        if decisao.usar_rag and decisao.chamadas:
            for chamada in decisao.chamadas:
                notas_especialistas.append(
                    construir_task_especialista(
                        chamada.id_base,
                        pergunta=pergunta,
                        contexto_rag=contexto_txt,
                        perfil_texto=perfil_texto,
                        info_solicitacao_epi=info_solicitacao_epi,
                    )
                )

        desc_redator = construir_task_redator_final(
            pergunta=pergunta,
            perfil_texto=perfil_texto,
            historico_texto=historico_txt,
            usar_rag=decisao.usar_rag,
            info_solicitacao_epi=info_solicitacao_epi,
            contexto_rag=contexto_txt,
        )
        if notas_especialistas:
            desc_redator = (
                f"{desc_redator}\n\n"
                f"Diretrizes dos especialistas acionados ({', '.join(agentes_acionados[1:-1])}):\n"
                + "\n---\n".join(notas_especialistas[:1])
            )

        redator_turno = Agent(
            role=PROMPT_REDATOR_FINAL_A2["role"],
            goal=PROMPT_REDATOR_FINAL_A2["goal"],
            backstory=PROMPT_REDATOR_FINAL_A2["backstory"],
            llm=crew_llm,
            allow_delegation=False,
            verbose=False,
        )
        task_redator = Task(
            description=desc_redator,
            expected_output="Resposta clara, acolhedora e fundamentada em português do Brasil.",
            agent=redator_turno,
        )
        crew = Crew(
            agents=[redator_turno],
            tasks=[task_redator],
            process=Process.sequential,
            verbose=False,
            tracing=False,
        )

        t0_crew = time.perf_counter()
        crew_output = crew.kickoff()
        crew_ms = round((time.perf_counter() - t0_crew) * 1000.0, 2)

        resposta_texto = str(getattr(crew_output, "raw", "") or "").strip()
        if isinstance(crew_llm, GroqCrewLLM):
            if crew_llm.ultima_resposta and not resposta_texto:
                resposta_texto = crew_llm.ultima_resposta
            llm_ms = crew_llm.ultima_llm_ms or crew_ms
            reasoning = crew_llm.ultimo_reasoning
            prompt_enviado = crew_llm.ultimo_prompt_enviado or desc_redator
            metricas_llm = crew_llm.ultimas_metricas
        else:
            llm_ms = crew_ms
            reasoning = ""
            prompt_enviado = desc_redator
            metricas_llm = MetricasLLMA2(modelo=modelo_ativo)

        # Garante roteiro obrigatório NR-6 na resposta quando houve abertura de EPI
        if id_solicitacao_epi is not None:
            texto_lower = _normalizar_texto(resposta_texto)
            if str(id_solicitacao_epi) not in resposta_texto or "nao " not in texto_lower:
                complemento_nr6 = (
                    "Não execute a atividade de risco sem o EPI adequado e não utilize o "
                    f"equipamento danificado (NR-6). Abri o pedido de reposição nº {id_solicitacao_epi} "
                    "e o TST e seu gestor serão avisados."
                )
                resposta_texto = (
                    f"{complemento_nr6}\n\n{resposta_texto}"
                    if resposta_texto
                    else complemento_nr6
                )

        base_slug = (
            MAPA_ID_BASE_PARA_SLUG.get(decisao.id_base, "pausas")
            if decisao.usar_rag and decisao.id_base is not None
            else "chat"
        )
        total_ms = round(
            decisao.roteamento_ms + embed_ms_total + db_rrf_ms_total + llm_ms, 2
        )

        telemetria = TelemetriaExecucaoA2(
            roteamento=decisao,
            resultados_rag=resultados_rag,
            detalhes_trechos=detalhes_mesclados,
            estatisticas_embedding=estatisticas_finais,
            tempos=TemposExecucaoA2(
                total_ms=total_ms,
                roteamento_ms=decisao.roteamento_ms,
                embed_ms=round(embed_ms_total, 2),
                db_rrf_ms=round(db_rrf_ms_total, 2),
                llm_ms=round(llm_ms, 2),
            ),
            metricas_llm=metricas_llm,
            citacoes=citacoes,
            fontes=fontes,
            reasoning=reasoning,
            prompt_enviado=prompt_enviado,
            agentes_acionados=agentes_acionados,
            id_solicitacao_epi=id_solicitacao_epi,
            motivo_solicitacao_epi=motivo_epi,
        )

        return ResultadoExecucaoCrewA2(
            resposta=resposta_texto,
            base_slug=base_slug,
            trechos_usados=len(trechos_mesclados),
            fontes=fontes,
            id_solicitacao_epi=id_solicitacao_epi,
            motivo_solicitacao_epi=motivo_epi,
            telemetria=telemetria,
        )
