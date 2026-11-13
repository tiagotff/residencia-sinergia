"""Serviço de integração com Ollama local para roteamento autônomo (F1..F5) e geração RAG."""

import json
import logging
import math
import os
import re
import time
from collections.abc import Callable
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from sinergia.rag.interfaces import ContextoRAG

load_dotenv(override=True)

logger = logging.getLogger(__name__)

PRESETS_MODELOS_LOCAL: list[str] = [
    "llama3.1:8b",
    "gemma3:12b",
    "qwen3:8b",
    "deepseek-r1:14b",
    "mimo-v2.6:9b-thinking",
]

PRESETS_MODELOS_GROQ: list[str] = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-safeguard-20b",
    "allam-2-7b",
    "llama-3.1-8b-instant",
]

_TERMOS_NAO_CHAT_GROQ = ("whisper", "orpheus", "prompt-guard", "embed")

MAPA_BASES_FERRAMENTAS: dict[int, dict[str, Any]] = {
    1: {
        "id_base": 1,
        "codigo": "F1",
        "nome": "Pausas e Ergonomia (NR-17)",
        "tool_name": "ferramenta_busca_pausas",
        "descricao": (
            "Consulta normas de ergonomia (NR-17), pausas para descanso, "
            "limites de digitação contínua, postura e ritmo de trabalho."
        ),
    },
    2: {
        "id_base": 2,
        "codigo": "F2",
        "nome": "Direitos e Benefícios (CLT)",
        "tool_name": "ferramenta_busca_direitos",
        "descricao": (
            "Consulta a CLT, direitos trabalhistas, benefícios, férias, "
            "licenças e faltas justificadas (ex: casamento, nascimento, falecimento)."
        ),
    },
    3: {
        "id_base": 3,
        "codigo": "F3",
        "nome": "Procedimentos e Treinamentos (POPs)",
        "tool_name": "ferramenta_busca_procedimentos",
        "descricao": (
            "Consulta Procedimentos Operacionais Padrão (POPs), onboarding, "
            "fluxos internos e cronogramas de treinamentos e capacitação."
        ),
    },
    4: {
        "id_base": 4,
        "codigo": "F4",
        "nome": "Saúde Ocupacional (PCMSO)",
        "tool_name": "ferramenta_busca_saude_ocupacional",
        "descricao": (
            "Consulta diretrizes do PCMSO (NR-7), ASO, exames admissionais, "
            "periódicos, retorno ao trabalho e programas de saúde ocupacional."
        ),
    },
    5: {
        "id_base": 5,
        "codigo": "F5",
        "nome": "EPIs e Fichas de CA (NR-6)",
        "tool_name": "ferramenta_busca_epis",
        "descricao": (
            "Consulta fichas técnicas de EPIs, códigos de Certificado de Aprovação (CA), "
            "NR-6, luvas, capacetes, óculos, protetores e prazos de substituição."
        ),
    },
}

TOOL_NAME_TO_BASE_ID: dict[str, int] = {
    info["tool_name"]: id_base for id_base, info in MAPA_BASES_FERRAMENTAS.items()
}


class ChamadaFerramentaRAG(BaseModel):
    """Representa uma chamada individual de ferramenta RAG ponderada para uma base (F1..F5)."""

    id_base: int = Field(..., ge=1, le=5, description="ID da base escolhida (1 a 5)")
    codigo_base: str = Field(..., description="Código da base (F1 a F5)")
    nome_base: str = Field(..., description="Nome do domínio da base")
    tool_name: str = Field(..., description="Nome da ferramenta selecionada")
    query_busca: str = Field(..., description="Query refinada para busca no RAG")
    peso: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Peso normalizado de relevância (0..1)"
    )
    limite_trechos: int = Field(
        default=4, ge=1, description="Quantidade de trechos alocados pelo peso"
    )


def calcular_alocacao_trechos(
    pesos_brutos: list[float], orcamento_multi: int = 6, limite_unico: int = 4
) -> tuple[list[float], list[int]]:
    """Normaliza pesos de relevância e distribui o orçamento de trechos proporcionalmente."""
    n = len(pesos_brutos)
    if n == 0:
        return [], []
    if n == 1:
        return [1.0], [limite_unico]

    clamped = [max(0.05, float(w)) for w in pesos_brutos]
    soma = sum(clamped)
    pesos_norm = [round(w / soma, 4) for w in clamped]
    diferenca = round(1.0 - sum(pesos_norm), 4)
    if diferenca != 0.0:
        pesos_norm[0] = round(max(0.0, min(1.0, pesos_norm[0] + diferenca)), 4)

    orcamento_efetivo = max(n, orcamento_multi)
    alocados = [1] * n
    restante = orcamento_efetivo - n
    if restante > 0:
        quotas = [p * restante for p in pesos_norm]
        partes_inteiras = [math.floor(q) for q in quotas]
        for i in range(n):
            alocados[i] += partes_inteiras[i]
        sobra = restante - sum(partes_inteiras)
        if sobra > 0:
            indices_ordenados = sorted(
                range(n),
                key=lambda i: (quotas[i] - partes_inteiras[i], pesos_norm[i], -i),
                reverse=True,
            )
            for idx in indices_ordenados[:sobra]:
                alocados[idx] += 1

    return pesos_norm, alocados


class DecisaoRoteamento(BaseModel):
    """Resultado do roteamento autônomo da LLM para escolha da(s) base(s) RAG."""

    id_base: int = Field(..., ge=1, le=5, description="ID da base principal escolhida (1 a 5)")
    codigo_base: str = Field(..., description="Código da base principal (F1 a F5)")
    nome_base: str = Field(..., description="Nome do domínio da base principal")
    tool_name: str = Field(..., description="Nome da ferramenta principal selecionada")
    query_busca: str = Field(..., description="Query refinada para busca no RAG")
    modo_roteamento: str = Field(
        ..., description="Estratégia usada: tool_calling, json_fallback ou heuristic_fallback"
    )
    roteamento_ms: float = Field(default=0.0, description="Tempo gasto no roteamento em ms")
    chamadas: list[ChamadaFerramentaRAG] = Field(
        default_factory=list,
        description="Lista ordenada por peso das ferramentas acionadas no turno",
    )

    @model_validator(mode="after")
    def _garantir_chamadas_preenchidas(self) -> "DecisaoRoteamento":
        if not self.chamadas:
            self.chamadas = [
                ChamadaFerramentaRAG(
                    id_base=self.id_base,
                    codigo_base=self.codigo_base,
                    nome_base=self.nome_base,
                    tool_name=self.tool_name,
                    query_busca=self.query_busca,
                    peso=1.0,
                    limite_trechos=4,
                )
            ]
        return self

    @property
    def codigos_combinados(self) -> str:
        """Retorna os códigos das bases acionadas unidos por '+' (ex: 'F4+F5')."""
        if self.chamadas:
            return "+".join(c.codigo_base for c in self.chamadas)
        return self.codigo_base

    @property
    def multi_base(self) -> bool:
        """Indica se mais de uma base foi acionada neste turno."""
        return len(self.chamadas) > 1


class MetricasOllama(BaseModel):
    """Métricas nativas de inferência retornadas pela API /api/chat do Ollama."""

    modelo: str = Field(..., description="Nome do modelo utilizado no Ollama")
    prompt_eval_count: int = Field(default=0, description="Tokens de entrada processados")
    eval_count: int = Field(default=0, description="Tokens gerados na resposta")
    load_duration_ms: float = Field(default=0.0, description="Tempo de carregamento em ms")
    eval_duration_ms: float = Field(default=0.0, description="Tempo de geração em ms")
    total_duration_ms: float = Field(default=0.0, description="Tempo total da chamada no Ollama em ms")
    tokens_por_segundo: float = Field(default=0.0, description="Velocidade de geração (tok/s)")


class RespostaGeradaOllama(BaseModel):
    """Resposta fundamentada gerada pela LLM local junto às métricas e prompt enviado."""

    resposta: str = Field(..., description="Texto final da resposta")
    reasoning: str = Field(default="", description="Raciocínio emitido pelo modelo durante a geração")
    citacoes: list[str] = Field(default_factory=list, description="Pílulas de citação formatadas")
    metricas: MetricasOllama = Field(..., description="Métricas nativas do Ollama")
    llm_ms: float = Field(default=0.0, description="Latência total da etapa LLM em ms")
    prompt_enviado: str = Field(..., description="Prompt completo enviado para geração")


TOOL_NAME_MULTI_BASE = "ferramenta_busca_multiplas_bases"


def _construir_definicoes_tools() -> list[dict[str, Any]]:
    """Constrói o schema OpenAI/Ollama de tool calling para as 5 ferramentas RAG + multi-base."""
    tools = []
    for info in MAPA_BASES_FERRAMENTAS.values():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": info["tool_name"],
                    "description": info["descricao"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Termos-chave da consulta em português para busca híbrida.",
                            },
                            "peso": {
                                "type": "number",
                                "description": (
                                    "Peso de relevância desta base na pergunta do colaborador (0.1 a 1.0). "
                                    "Se a pergunta focar mais em um tema, atribua peso maior a ele."
                                ),
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME_MULTI_BASE,
                "description": (
                    "Obrigatória quando a pergunta abordar DOIS OU MAIS temas/bases diferentes ao mesmo tempo "
                    "(ex: saúde ocupacional + EPI, ergonomia + CLT). "
                    "Mapeamento de id_base: 1=Pausas/Ergonomia (NR-17), 2=Direitos/Benefícios (CLT), "
                    "3=Procedimentos/Treinamentos (POPs), 4=Saúde Ocupacional (PCMSO/NR-7), 5=EPIs/CA (NR-6)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "consultas": {
                            "type": "array",
                            "description": "Lista das bases relevantes (inclua apenas as estritamente relacionadas à pergunta).",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id_base": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 5,
                                        "description": "ID da base (1 a 5).",
                                    },
                                    "query": {
                                        "type": "string",
                                        "description": "Termos-chave em português específicos para essa base.",
                                    },
                                    "peso": {
                                        "type": "number",
                                        "description": "Peso de relevância (0.1 a 1.0) conforme o foco da pergunta.",
                                    },
                                },
                                "required": ["id_base", "query", "peso"],
                            },
                        }
                    },
                    "required": ["consultas"],
                },
            },
        }
    )
    return tools


def extrair_citacoes_contexto(contexto: ContextoRAG) -> list[str]:
    """Extrai pílulas concisas de citação documental a partir dos trechos recuperados."""
    citacoes: list[str] = []
    vistos = set()
    for trecho in contexto.trechos:
        meta = trecho.metadados
        secao = meta.secao or "Geral"
        pagina = f"Pág. {meta.pagina}" if meta.pagina is not None else "Pág. N/A"
        label = f"{meta.titulo} (v{meta.versao}) · {secao} · {pagina}"
        if label not in vistos:
            vistos.add(label)
            citacoes.append(label)
    return citacoes


def _remover_bloco_think(texto: str) -> str:
    """Remove blocos <think>...</think> emitidos por modelos com raciocínio."""
    limpo = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL | re.IGNORECASE)
    return limpo.strip()


class _ThinkStreamParser:
    """Separa tokens de raciocínio (campo estruturado ou tags <think>) e resposta final em streaming."""

    _OPEN_TAG = "<think>"
    _CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False
        self._reasoning_parts: list[str] = []
        self._content_parts: list[str] = []

    def feed_reasoning(self, text: str) -> bool:
        """Adiciona trecho vindo de campo estruturado de reasoning. Retorna True se houve alteração."""
        if not text:
            return False
        self._reasoning_parts.append(text)
        return True

    def feed_content(self, text: str) -> tuple[bool, bool]:
        """Processa trecho de content, separando blocos <think>...</think>. Retorna (mudou_reasoning, mudou_content)."""
        if not text:
            return False, False

        self._buffer += text
        mudou_reasoning = False
        mudou_content = False

        while self._buffer:
            lower_buf = self._buffer.lower()
            if not self._in_think:
                idx = lower_buf.find(self._OPEN_TAG)
                if idx != -1:
                    if idx > 0:
                        self._content_parts.append(self._buffer[:idx])
                        mudou_content = True
                    self._buffer = self._buffer[idx + len(self._OPEN_TAG) :]
                    self._in_think = True
                    continue

                keep = len(self._OPEN_TAG) - 1
                if len(self._buffer) > keep:
                    flush = self._buffer[:-keep]
                    self._content_parts.append(flush)
                    self._buffer = self._buffer[-keep:]
                    mudou_content = True
                break
            else:
                idx = lower_buf.find(self._CLOSE_TAG)
                if idx != -1:
                    if idx > 0:
                        self._reasoning_parts.append(self._buffer[:idx])
                        mudou_reasoning = True
                    self._buffer = self._buffer[idx + len(self._CLOSE_TAG) :]
                    self._in_think = False
                    continue

                keep = len(self._CLOSE_TAG) - 1
                if len(self._buffer) > keep:
                    flush = self._buffer[:-keep]
                    self._reasoning_parts.append(flush)
                    self._buffer = self._buffer[-keep:]
                    mudou_reasoning = True
                break

        return mudou_reasoning, mudou_content

    def flush(self) -> None:
        """Descarrega o buffer restante ao final do stream."""
        if not self._buffer:
            return
        if self._in_think:
            self._reasoning_parts.append(self._buffer)
        else:
            self._content_parts.append(self._buffer)
        self._buffer = ""

    @property
    def reasoning_text(self) -> str:
        """Texto acumulado do raciocínio até o momento."""
        return "".join(self._reasoning_parts).strip()

    @property
    def content_text(self) -> str:
        """Texto acumulado da resposta final até o momento."""
        return "".join(self._content_parts).strip()


def _mesclar_sem_duplicatas(primarios: list[str], presets: list[str]) -> list[str]:
    """Combina duas listas de nomes de modelos preservando ordem e unicidade."""
    vistos: set[str] = set()
    resultado: list[str] = []
    for item in [*primarios, *presets]:
        if item and item not in vistos:
            vistos.add(item)
            resultado.append(item)
    return resultado


class _BasePitchLLMService:
    """Comportamento compartilhado de roteamento autônomo e prompts entre provedores LLM."""

    modelo_fallback_padrao: str = "qwen2.5:3b"

    def __init__(self, modelo_padrao: str | None = None, timeout_s: float = 45.0):
        self.modelo_padrao = modelo_padrao
        self.timeout_s = timeout_s

    def listar_modelos_disponiveis(self) -> list[str]:
        raise NotImplementedError

    def selecionar_modelo_ativo(self, modelo_preferido: str | None = None) -> str:
        """Resolve o modelo ativo priorizando sempre a escolha explícita informada."""
        if modelo_preferido and modelo_preferido.strip():
            return modelo_preferido.strip()
        disponiveis = self.listar_modelos_disponiveis()
        if self.modelo_padrao and (not disponiveis or self.modelo_padrao in disponiveis):
            return self.modelo_padrao
        if disponiveis:
            return disponiveis[0]
        return self.modelo_padrao or self.modelo_fallback_padrao

    def _construir_decisao_multipla(
        self,
        itens_brutos: list[tuple[int, str, float]],
        modo: str,
        inicio: float,
    ) -> DecisaoRoteamento:
        """Consolida chamadas por base, ordena por peso e calcula a alocação proporcional."""
        consolidados_map: dict[int, tuple[list[str], float, int]] = {}
        for idx, (id_base, query, peso_bruto) in enumerate(itens_brutos):
            base_valido = id_base if id_base in MAPA_BASES_FERRAMENTAS else 1
            query_limpa = (query or "").strip() or "consulta geral"
            try:
                peso_val = max(0.05, float(peso_bruto))
            except (TypeError, ValueError):
                peso_val = 1.0

            if base_valido not in consolidados_map:
                consolidados_map[base_valido] = ([query_limpa], peso_val, idx)
            else:
                queries_existentes, peso_acum, ordem_orig = consolidados_map[base_valido]
                if query_limpa not in queries_existentes:
                    queries_existentes.append(query_limpa)
                consolidados_map[base_valido] = (
                    queries_existentes,
                    peso_acum + peso_val,
                    ordem_orig,
                )

        if not consolidados_map:
            consolidados_map[1] = (["consulta geral"], 1.0, 0)

        ordenados = sorted(
            consolidados_map.items(),
            key=lambda kv: (-kv[1][1], kv[1][2]),
        )[:5]

        pesos_norm, limites = calcular_alocacao_trechos([item[1][1] for item in ordenados])

        chamadas: list[ChamadaFerramentaRAG] = []
        for (id_b, (queries, _, _)), peso_n, limite_t in zip(ordenados, pesos_norm, limites):
            info = MAPA_BASES_FERRAMENTAS[id_b]
            chamadas.append(
                ChamadaFerramentaRAG(
                    id_base=id_b,
                    codigo_base=info["codigo"],
                    nome_base=info["nome"],
                    tool_name=info["tool_name"],
                    query_busca=" ".join(queries),
                    peso=peso_n,
                    limite_trechos=limite_t,
                )
            )

        principal = chamadas[0]
        return DecisaoRoteamento(
            id_base=principal.id_base,
            codigo_base=principal.codigo_base,
            nome_base=principal.nome_base,
            tool_name=principal.tool_name,
            query_busca=principal.query_busca,
            modo_roteamento=modo,
            roteamento_ms=round((time.perf_counter() - inicio) * 1000.0, 2),
            chamadas=chamadas,
        )

    def _construir_decisao(
        self, id_base: int, query_busca: str, modo: str, inicio: float
    ) -> DecisaoRoteamento:
        return self._construir_decisao_multipla([(id_base, query_busca, 1.0)], modo, inicio)

    def _decidir_heuristica(self, pergunta: str, inicio: float) -> DecisaoRoteamento:
        """Fallback determinístico por palavras-chave com peso proporcional ao número de matches."""
        texto = pergunta.lower()
        regras = [
            (5, ("epi", "ca ", "ca-", "nr-6", "nr 6", "luva", "capacete", "óculos", "oculos", "protetor", "substituição")),
            (4, ("pcmso", "aso", "exame", "saúde", "saude", "médico", "medico", "nr-7", "nr 7", "audiometria")),
            (3, ("pop", "procedimento", "treinamento", "capacitação", "capacitacao", "onboarding", "cronograma")),
            (2, ("clt", "casamento", "falta", "benefício", "beneficio", "férias", "ferias", "licença", "licenca", "473", "direito")),
            (1, ("nr-17", "nr 17", "pausa", "digitação", "digitacao", "ergonomia", "descanso", "postura", "toques")),
        ]
        encontradas: list[tuple[int, str, float]] = []
        for id_base, termos in regras:
            matches = sum(1 for termo in termos if termo in texto)
            if matches > 0:
                encontradas.append((id_base, pergunta, float(matches)))
        if encontradas:
            return self._construir_decisao_multipla(encontradas, "heuristic_fallback", inicio)
        return self._construir_decisao(1, pergunta, "heuristic_fallback", inicio)

    def _parsear_tool_calls(
        self, tool_calls: list[dict[str, Any]], pergunta: str, inicio: float
    ) -> DecisaoRoteamento | None:
        if not tool_calls:
            return None
        itens: list[tuple[int, str, float]] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}

            if tool_name == TOOL_NAME_MULTI_BASE:
                consultas = args.get("consultas") or []
                if isinstance(consultas, list):
                    for c in consultas:
                        if not isinstance(c, dict):
                            continue
                        id_b = self._resolver_id_base_item(c)
                        if id_b is None:
                            continue
                        q = str(c.get("query") or pergunta)
                        try:
                            p = float(c.get("peso", 1.0))
                        except (TypeError, ValueError):
                            p = 1.0
                        itens.append((id_b, q, p))
                continue

            if tool_name not in TOOL_NAME_TO_BASE_ID:
                continue
            query_busca = str(args.get("query") or pergunta)
            try:
                peso = float(args.get("peso", 1.0))
            except (TypeError, ValueError):
                peso = 1.0
            itens.append((TOOL_NAME_TO_BASE_ID[tool_name], query_busca, peso))
        if not itens:
            return None
        return self._construir_decisao_multipla(itens, "tool_calling", inicio)

    @staticmethod
    def _resolver_id_base_item(item: dict[str, Any]) -> int | None:
        """Resolve id_base a partir de inteiro, código F1..F5 ou tool_name."""
        raw_id = item.get("id_base")
        if raw_id is not None:
            if isinstance(raw_id, str) and raw_id.strip().upper().startswith("F"):
                try:
                    val = int(raw_id.strip()[1:])
                    return val if val in MAPA_BASES_FERRAMENTAS else None
                except ValueError:
                    pass
            try:
                val = int(raw_id)
                return val if val in MAPA_BASES_FERRAMENTAS else None
            except (TypeError, ValueError):
                pass
        t_name = item.get("tool_name")
        if isinstance(t_name, str) and t_name in TOOL_NAME_TO_BASE_ID:
            return TOOL_NAME_TO_BASE_ID[t_name]
        return None

    def _parsear_json_roteamento(
        self, conteudo: str, pergunta: str, inicio: float
    ) -> DecisaoRoteamento | None:
        parsed = json.loads(conteudo or "{}")
        if not isinstance(parsed, dict):
            return None

        consultas = parsed.get("consultas")
        itens: list[tuple[int, str, float]] = []
        if isinstance(consultas, list):
            for c in consultas:
                if not isinstance(c, dict):
                    continue
                try:
                    id_b = int(c.get("id_base", 0))
                except (TypeError, ValueError):
                    continue
                if id_b not in MAPA_BASES_FERRAMENTAS:
                    continue
                q = str(c.get("query") or pergunta)
                try:
                    p = float(c.get("peso", 1.0))
                except (TypeError, ValueError):
                    p = 1.0
                itens.append((id_b, q, p))
        else:
            try:
                id_base = int(parsed.get("id_base", 0))
            except (TypeError, ValueError):
                id_base = 0
            if id_base in MAPA_BASES_FERRAMENTAS:
                query_busca = str(parsed.get("query") or pergunta)
                try:
                    peso = float(parsed.get("peso", 1.0))
                except (TypeError, ValueError):
                    peso = 1.0
                itens.append((id_base, query_busca, peso))

        if not itens:
            return None
        return self._construir_decisao_multipla(itens, "json_fallback", inicio)

    def _mensagens_roteamento_tool(self, pergunta: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Você é o roteador autônomo do SinergIA. Analise a pergunta do colaborador:\n"
                    "- Se tratar de APENAS UM tema, chame a ferramenta específica (F1..F5) correspondente.\n"
                    "- Se abordar DOIS OU MAIS temas/bases diferentes (ex: saúde ocupacional e EPIs, "
                    "pausas e CLT), chame `ferramenta_busca_multiplas_bases` (ou as ferramentas em paralelo) "
                    "incluindo APENAS as bases relacionadas, com uma query específica e um peso (0.1 a 1.0) "
                    "proporcional ao foco de cada tema na pergunta."
                ),
            },
            {"role": "user", "content": pergunta},
        ]

    def _mensagens_roteamento_json(self, pergunta: str) -> list[dict[str, str]]:
        bases_desc = "\n".join(
            f"- id_base {b['id_base']} ({b['codigo']}): {b['nome']} -> {b['descricao']}"
            for b in MAPA_BASES_FERRAMENTAS.values()
        )
        return [
            {
                "role": "system",
                "content": (
                    "Escolha qual ou quais bases de conhecimento (1 a 5) respondem melhor à pergunta. "
                    "Inclua APENAS as bases diretamente relacionadas à pergunta.\n"
                    "Responda APENAS JSON válido no formato "
                    "{\"consultas\": [{\"id_base\": <int 1..5>, \"query\": \"<string>\", \"peso\": <float 0.1..1.0>}]}.\n"
                    "Atribua um peso maior à base que representa o foco principal da pergunta.\n"
                    f"Bases disponíveis:\n{bases_desc}"
                ),
            },
            {"role": "user", "content": pergunta},
        ]

    def _montar_prompts_geracao(
        self, pergunta: str, contexto: ContextoRAG
    ) -> tuple[str, str, str]:
        contexto_formatado = contexto.formatar_para_prompt()
        system_prompt = (
            "Você é o assistente SinergIA de bem-estar, ergonomia, direitos e segurança do trabalho. "
            "Responda em português de forma clara, objetiva e estritamente fundamentada nas fontes recuperadas. "
            "Cite explicitamente o título do documento, versão, seção/artigo e página presentes nas fontes. "
            "Se as fontes não trouxerem a resposta exata, informe essa limitação sem inventar dados."
        )
        user_prompt = (
            f"Pergunta do colaborador: {pergunta}\n\n"
            f"Fontes recuperadas no RAG:\n{contexto_formatado}"
        )
        prompt_completo = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
        return system_prompt, user_prompt, prompt_completo

    def _rotear_via_tool_calling(
        self, pergunta: str, modelo: str, inicio: float
    ) -> DecisaoRoteamento | None:
        raise NotImplementedError

    def _rotear_via_json_estruturado(
        self, pergunta: str, modelo: str, inicio: float
    ) -> DecisaoRoteamento | None:
        raise NotImplementedError

    def decidir_base_autonomamente(
        self, pergunta: str, modelo: str | None = None
    ) -> DecisaoRoteamento:
        """Decide autonomamente qual das 5 bases (F1..F5) consultar para a pergunta."""
        inicio = time.perf_counter()
        modelo_ativo = self.selecionar_modelo_ativo(modelo)

        try:
            decisao = self._rotear_via_tool_calling(pergunta, modelo_ativo, inicio)
            if decisao is not None:
                return decisao
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as erro:
            logger.debug("Tool calling indisponível no modelo %s: %s", modelo_ativo, erro)

        try:
            decisao = self._rotear_via_json_estruturado(pergunta, modelo_ativo, inicio)
            if decisao is not None:
                return decisao
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as erro:
            logger.debug("Fallback JSON indisponível no modelo %s: %s", modelo_ativo, erro)

        return self._decidir_heuristica(pergunta, inicio)


class OllamaPitchService(_BasePitchLLMService):
    """Cliente local do Ollama para descoberta de modelos, roteamento autônomo e geração RAG."""

    modelo_fallback_padrao = "qwen2.5:3b"

    def __init__(
        self,
        base_url: str | None = None,
        modelo_padrao: str | None = None,
        timeout_s: float = 45.0,
    ):
        super().__init__(
            modelo_padrao=modelo_padrao or os.getenv("OLLAMA_MODEL"),
            timeout_s=timeout_s,
        )
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def listar_modelos_disponiveis(self) -> list[str]:
        """Consulta GET /api/tags no Ollama e retorna modelos de chat instalados."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            resp.raise_for_status()
            dados = resp.json()
        except (httpx.HTTPError, ValueError) as erro:
            logger.debug("Falha ao listar modelos do Ollama: %s", erro)
            return []

        modelos_brutos = [
            m.get("name") or m.get("model")
            for m in dados.get("models", [])
            if m.get("name") or m.get("model")
        ]
        if not modelos_brutos:
            return []

        geradores = [
            nome
            for nome in modelos_brutos
            if "embed" not in nome.lower() and "embedding" not in nome.lower()
        ]
        return geradores if geradores else modelos_brutos

    def listar_modelos_com_presets(self) -> list[str]:
        """Retorna os modelos instalados no topo seguidos dos presets 8B–14B com raciocínio."""
        return _mesclar_sem_duplicatas(self.listar_modelos_disponiveis(), PRESETS_MODELOS_LOCAL)

    def modelo_esta_instalado(self, modelo: str) -> bool:
        """Verifica se o modelo está baixado localmente no servidor Ollama."""
        return modelo in self.listar_modelos_disponiveis()

    def _rotear_via_tool_calling(
        self, pergunta: str, modelo: str, inicio: float
    ) -> DecisaoRoteamento | None:
        payload = {
            "model": modelo,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0},
            "messages": self._mensagens_roteamento_tool(pergunta),
            "tools": _construir_definicoes_tools(),
        }
        resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        tool_calls = resp.json().get("message", {}).get("tool_calls") or []
        return self._parsear_tool_calls(tool_calls, pergunta, inicio)

    def _rotear_via_json_estruturado(
        self, pergunta: str, modelo: str, inicio: float
    ) -> DecisaoRoteamento | None:
        payload = {
            "model": modelo,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0.0},
            "messages": self._mensagens_roteamento_json(pergunta),
        }
        resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        conteudo = resp.json().get("message", {}).get("content", "{}")
        return self._parsear_json_roteamento(conteudo, pergunta, inicio)

    def gerar_resposta_fundamentada(
        self,
        pergunta: str,
        contexto: ContextoRAG,
        modelo: str | None = None,
        ao_receber_reasoning: Callable[[str], None] | None = None,
        ao_receber_conteudo: Callable[[str], None] | None = None,
    ) -> RespostaGeradaOllama:
        """Gera resposta em português fundamentada nos trechos recuperados com streaming e raciocínio."""
        modelo_ativo = self.selecionar_modelo_ativo(modelo)
        citacoes = extrair_citacoes_contexto(contexto)
        system_prompt, user_prompt, prompt_completo = self._montar_prompts_geracao(pergunta, contexto)

        inicio = time.perf_counter()
        parser = _ThinkStreamParser()
        dados_finais: dict[str, Any] = {}
        try:
            payload = {
                "model": modelo_ativo,
                "stream": True,
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_s,
            ) as resp:
                resp.raise_for_status()
                for linha in resp.iter_lines():
                    if not linha:
                        continue
                    chunk = json.loads(linha)
                    msg = chunk.get("message") or {}
                    mudou_r = parser.feed_reasoning(msg.get("thinking") or "")
                    mudou_r2, mudou_c = parser.feed_content(msg.get("content") or "")
                    if (mudou_r or mudou_r2) and ao_receber_reasoning and parser.reasoning_text:
                        ao_receber_reasoning(parser.reasoning_text)
                    if mudou_c and ao_receber_conteudo and parser.content_text:
                        ao_receber_conteudo(parser.content_text)
                    if chunk.get("done"):
                        dados_finais = chunk

            parser.flush()
            if ao_receber_reasoning and parser.reasoning_text:
                ao_receber_reasoning(parser.reasoning_text)
            if ao_receber_conteudo and parser.content_text:
                ao_receber_conteudo(parser.content_text)

            llm_ms = (time.perf_counter() - inicio) * 1000.0

            resposta_texto = parser.content_text
            prompt_tokens = int(dados_finais.get("prompt_eval_count") or 0)
            eval_tokens = int(dados_finais.get("eval_count") or 0)
            load_ms = float(dados_finais.get("load_duration") or 0) / 1_000_000.0
            eval_ms = float(dados_finais.get("eval_duration") or 0) / 1_000_000.0
            total_ms = float(dados_finais.get("total_duration") or 0) / 1_000_000.0

            tok_per_sec = (
                (eval_tokens / (eval_ms / 1000.0))
                if eval_ms > 0 and eval_tokens > 0
                else 0.0
            )

            metricas = MetricasOllama(
                modelo=modelo_ativo,
                prompt_eval_count=prompt_tokens,
                eval_count=eval_tokens,
                load_duration_ms=round(load_ms, 2),
                eval_duration_ms=round(eval_ms, 2),
                total_duration_ms=round(total_ms, 2),
                tokens_por_segundo=round(tok_per_sec, 1),
            )
            return RespostaGeradaOllama(
                resposta=resposta_texto or "Não foi possível gerar resposta com o modelo selecionado.",
                reasoning=parser.reasoning_text,
                citacoes=citacoes,
                metricas=metricas,
                llm_ms=round(llm_ms, 2),
                prompt_enviado=prompt_completo,
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as erro:
            llm_ms = (time.perf_counter() - inicio) * 1000.0
            resumo_trechos = (
                f"Base consultada retornou {len(contexto.trechos)} trecho(s) relevante(s): "
                + (contexto.trechos[0].conteudo[:260] + "..." if contexto.trechos else "nenhum trecho.")
            )
            fallback_msg = (
                f"Não foi possível conectar ao servidor Ollama local (`{self.base_url}`) para gerar "
                f"a síntese com o modelo `{modelo_ativo}` ({erro}).\n\n"
                f"Para habilitar a geração local, execute `ollama serve` e certifique-se de ter um modelo "
                f"baixado com `ollama pull {modelo_ativo}`.\n\n"
                f"**Resumo direto do RAG:** {resumo_trechos}"
            )
            return RespostaGeradaOllama(
                resposta=fallback_msg,
                citacoes=citacoes,
                metricas=MetricasOllama(modelo=modelo_ativo),
                llm_ms=round(llm_ms, 2),
                prompt_enviado=prompt_completo,
            )


class GroqPitchService(_BasePitchLLMService):
    """Cliente Cloud (Groq) compatível com OpenAI API para roteamento autônomo e geração RAG."""

    modelo_fallback_padrao = "qwen/qwen3.8-27b"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
        modelo_padrao: str | None = None,
        timeout_s: float = 45.0,
    ):
        super().__init__(
            modelo_padrao=modelo_padrao or os.getenv("LLM_MODEL"),
            timeout_s=timeout_s,
        )
        self.api_key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def listar_modelos_disponiveis(self) -> list[str]:
        """Consulta GET /models na API da Groq filtrando modelos não textuais/guard."""
        if not self.api_key:
            return []
        try:
            resp = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=5.0)
            resp.raise_for_status()
            dados = resp.json()
        except (httpx.HTTPError, ValueError) as erro:
            logger.debug("Falha ao listar modelos da Groq: %s", erro)
            return []

        ids_validos = [
            m.get("id")
            for m in dados.get("data", [])
            if m.get("id")
            and not any(termo in m["id"].lower() for termo in _TERMOS_NAO_CHAT_GROQ)
        ]

        # Priorizar modelos de raciocínio/principais antes dos auxiliares
        prioridade = {
            "qwen/qwen3.8-27b": 0,
            "openai/gpt-oss-20b": 1,
            "openai/gpt-oss-120b": 2,
            "openai/gpt-oss-safeguard-20b": 3,
            "allam-2-7b": 4,
        }
        ids_validos.sort(key=lambda mid: (prioridade.get(mid, 50), mid))
        return ids_validos

    def listar_modelos_com_presets(self) -> list[str]:
        """Retorna os modelos ativos da Groq no topo seguidos dos presets de referência."""
        return _mesclar_sem_duplicatas(self.listar_modelos_disponiveis(), PRESETS_MODELOS_GROQ)

    def modelo_esta_ativo(self, modelo: str) -> bool:
        """Verifica se o modelo consta na lista de modelos ativos da chave Groq."""
        disponiveis = self.listar_modelos_disponiveis()
        return not disponiveis or modelo in disponiveis

    def _rotear_via_tool_calling(
        self, pergunta: str, modelo: str, inicio: float
    ) -> DecisaoRoteamento | None:
        if not self.api_key:
            return None
        payload = {
            "model": modelo,
            "temperature": 0.0,
            "messages": self._mensagens_roteamento_tool(pergunta),
            "tools": _construir_definicoes_tools(),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        msg = (resp.json().get("choices") or [{}])[0].get("message", {})
        tool_calls = msg.get("tool_calls") or []
        return self._parsear_tool_calls(tool_calls, pergunta, inicio)

    def _rotear_via_json_estruturado(
        self, pergunta: str, modelo: str, inicio: float
    ) -> DecisaoRoteamento | None:
        if not self.api_key:
            return None
        payload = {
            "model": modelo,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": self._mensagens_roteamento_json(pergunta),
        }
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        conteudo = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "{}")
        return self._parsear_json_roteamento(conteudo, pergunta, inicio)

    def gerar_resposta_fundamentada(
        self,
        pergunta: str,
        contexto: ContextoRAG,
        modelo: str | None = None,
        ao_receber_reasoning: Callable[[str], None] | None = None,
        ao_receber_conteudo: Callable[[str], None] | None = None,
    ) -> RespostaGeradaOllama:
        """Gera resposta fundamentada via Groq Cloud com streaming, raciocínio e métricas de usage."""
        modelo_ativo = self.selecionar_modelo_ativo(modelo)
        citacoes = extrair_citacoes_contexto(contexto)
        system_prompt, user_prompt, prompt_completo = self._montar_prompts_geracao(pergunta, contexto)

        inicio = time.perf_counter()
        if not self.api_key:
            llm_ms = (time.perf_counter() - inicio) * 1000.0
            return RespostaGeradaOllama(
                resposta="Chave `GROQ_API_KEY` não configurada no `.env` ou ambiente.",
                citacoes=citacoes,
                metricas=MetricasOllama(modelo=modelo_ativo),
                llm_ms=round(llm_ms, 2),
                prompt_enviado=prompt_completo,
            )

        parser = _ThinkStreamParser()
        usage: dict[str, Any] = {}
        try:
            payload = {
                "model": modelo_ativo,
                "temperature": 0.1,
                "stream": True,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_s,
            ) as resp:
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
                        reasoning_delta = (
                            delta.get("reasoning") or delta.get("reasoning_content") or ""
                        )
                        content_delta = delta.get("content") or ""
                        mudou_r = parser.feed_reasoning(reasoning_delta)
                        mudou_r2, mudou_c = parser.feed_content(content_delta)
                        if (mudou_r or mudou_r2) and ao_receber_reasoning and parser.reasoning_text:
                            ao_receber_reasoning(parser.reasoning_text)
                        if mudou_c and ao_receber_conteudo and parser.content_text:
                            ao_receber_conteudo(parser.content_text)

                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    elif isinstance(chunk.get("x_groq"), dict) and isinstance(
                        chunk["x_groq"].get("usage"), dict
                    ):
                        usage = chunk["x_groq"]["usage"]

            parser.flush()
            if ao_receber_reasoning and parser.reasoning_text:
                ao_receber_reasoning(parser.reasoning_text)
            if ao_receber_conteudo and parser.content_text:
                ao_receber_conteudo(parser.content_text)

            llm_ms = (time.perf_counter() - inicio) * 1000.0

            resposta_texto = parser.content_text
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            eval_tokens = int(usage.get("completion_tokens") or 0)
            queue_s = float(usage.get("queue_time") or 0.0)
            completion_s = float(usage.get("completion_time") or 0.0)
            total_s = float(usage.get("total_time") or 0.0)

            tok_per_sec = (
                (eval_tokens / completion_s)
                if completion_s > 0 and eval_tokens > 0
                else 0.0
            )

            metricas = MetricasOllama(
                modelo=modelo_ativo,
                prompt_eval_count=prompt_tokens,
                eval_count=eval_tokens,
                load_duration_ms=round(queue_s * 1000.0, 2),
                eval_duration_ms=round(completion_s * 1000.0, 2),
                total_duration_ms=round(total_s * 1000.0, 2),
                tokens_por_segundo=round(tok_per_sec, 1),
            )
            return RespostaGeradaOllama(
                resposta=resposta_texto or "Não foi possível gerar resposta com o modelo Groq selecionado.",
                reasoning=parser.reasoning_text,
                citacoes=citacoes,
                metricas=metricas,
                llm_ms=round(llm_ms, 2),
                prompt_enviado=prompt_completo,
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as erro:
            llm_ms = (time.perf_counter() - inicio) * 1000.0
            detalhe_erro = str(erro)
            if isinstance(erro, httpx.HTTPStatusError):
                try:
                    erro.response.read()
                    msg_api = erro.response.json().get("error", {}).get("message")
                    if msg_api:
                        detalhe_erro = f"{detalhe_erro} — {msg_api}"
                except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as parse_err:
                    logger.debug("Sem detalhes JSON na resposta Groq: %s", parse_err)
            resumo_trechos = (
                f"Base consultada retornou {len(contexto.trechos)} trecho(s) relevante(s): "
                + (contexto.trechos[0].conteudo[:260] + "..." if contexto.trechos else "nenhum trecho.")
            )
            fallback_msg = (
                f"Falha ao consultar a API Groq Cloud para o modelo `{modelo_ativo}` ({detalhe_erro}).\n\n"
                f"Verifique se o modelo está ativo/habilitado na sua conta Groq ou selecione um modelo ativo "
                f"(`qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`).\n\n"
                f"**Resumo direto do RAG:** {resumo_trechos}"
            )
            return RespostaGeradaOllama(
                resposta=fallback_msg,
                citacoes=citacoes,
                metricas=MetricasOllama(modelo=modelo_ativo),
                llm_ms=round(llm_ms, 2),
                prompt_enviado=prompt_completo,
            )

