"""API HTTP isolada (FastAPI) do Agente 1 (`POST /executar`, Guia A1 v0.7)."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Response

from sinergia.agentes.a1_recepcao.config import obter_config_a1
from sinergia.agentes.a1_recepcao.extrator_llm import MockLLM
from sinergia.agentes.a1_recepcao.repositorio import (
    RepositorioA1Memoria,
    RepositorioA1Postgres,
    RepositorioA1Protocol,
)
from sinergia.agentes.a1_recepcao.schemas import (
    PROTOCOLO_SINERGIA,
    ResultadoA1,
    TarefaA1,
)
from sinergia.agentes.a1_recepcao.servico import AgenteA1, ServicoA1

_MODOS_REPO_VALIDOS = {"", "memoria", "postgres"}
_MODOS_LLM_VALIDOS = {"", "mock", "groq"}


def _ler_modos_ambiente() -> tuple[str, str]:
    modo_repo = os.getenv("A1_MODO_REPO", "").strip().lower()
    if modo_repo not in _MODOS_REPO_VALIDOS:
        raise ValueError(
            f"A1_MODO_REPO inválido: {modo_repo!r}. Valores aceitos: 'memoria' ou 'postgres'."
        )

    modo_llm = os.getenv("A1_MODO_LLM", "").strip().lower()
    if modo_llm not in _MODOS_LLM_VALIDOS:
        raise ValueError(
            f"A1_MODO_LLM inválido: {modo_llm!r}. Valores aceitos: 'mock' ou 'groq'."
        )
    return modo_repo, modo_llm


def criar_app_a1(agente: ServicoA1 | None = None) -> FastAPI:
    """Cria a aplicação FastAPI isolada do Agente 1 no protocolo `sinergia/1`."""
    app_a1 = FastAPI(
        title="SinergIA — Agente A1 (Recepção e Processamento)",
        version="0.7.0",
        description="Endpoint isolado POST /executar do protocolo sinergia/1.",
    )
    instancia: dict[str, ServicoA1 | None] = {"agente": agente}

    def _obter_agente() -> ServicoA1:
        if instancia["agente"] is None:
            cfg = obter_config_a1()
            modo_repo, modo_llm = _ler_modos_ambiente()

            repositorio: RepositorioA1Protocol | None = None
            if modo_repo == "memoria":
                repositorio = RepositorioA1Memoria()
            elif modo_repo == "postgres":
                repositorio = RepositorioA1Postgres(cfg.database_url)

            llm: Any | None = MockLLM() if modo_llm == "mock" else None
            instancia["agente"] = AgenteA1(
                repositorio=repositorio,
                llm=llm,
                config=cfg,
            )
        return instancia["agente"]

    @app_a1.get("/health")
    def health() -> dict[str, str]:
        modo_repo, modo_llm = _ler_modos_ambiente()
        cfg = obter_config_a1()
        who_local_ativo = cfg.who_icd_enabled and bool(cfg.who_icd_seed_url.strip())
        who_externa_ativo = (
            cfg.who_icd_enabled
            and bool(cfg.who_icd_client_id.strip())
            and bool(cfg.who_icd_client_secret.strip())
        )
        if who_local_ativo and who_externa_ativo:
            who_fallback = "container_local+externa"
        elif who_local_ativo:
            who_fallback = "container_local"
        elif who_externa_ativo:
            who_fallback = "externa"
        else:
            who_fallback = "somente_banco"
        return {
            "status": "ok",
            "agente": "A1",
            "protocolo": PROTOCOLO_SINERGIA,
            "modo_repositorio": modo_repo
            or ("postgres" if cfg.database_url.strip() else "memoria"),
            "modo_llm": modo_llm or ("groq" if cfg.groq_api_key.strip() else "nao_configurado"),
            "who_icd_api": "ativa" if (who_local_ativo or who_externa_ativo) else "somente_banco",
            "who_icd_fallback": who_fallback,
        }

    @app_a1.post("/executar", response_model=ResultadoA1)
    def executar_endpoint(tarefa: TarefaA1, response: Response) -> dict[str, Any]:
        agente_ativo = _obter_agente()
        resultado = agente_ativo.executar(tarefa)
        diag_cid = getattr(agente_ativo, "ultimo_diagnostico_cid", None)
        if isinstance(diag_cid, dict):
            if diag_cid.get("origem"):
                response.headers["X-Sinergia-Cid-Origem"] = str(diag_cid["origem"])
            if diag_cid.get("codigo_cid"):
                response.headers["X-Sinergia-Cid-Codigo"] = str(diag_cid["codigo_cid"])
            nome_doenca = diag_cid.get("nome_doenca") or diag_cid.get("descricao")
            if nome_doenca:
                response.headers["X-Sinergia-Cid-Nome-Doenca"] = quote(
                    str(nome_doenca), safe=""
                )
            if diag_cid.get("descricao"):
                response.headers["X-Sinergia-Cid-Descricao"] = quote(
                    str(diag_cid["descricao"]), safe=""
                )
            if diag_cid.get("capitulo"):
                response.headers["X-Sinergia-Cid-Capitulo"] = quote(
                    str(diag_cid["capitulo"]), safe=""
                )
            if diag_cid.get("versao_cid"):
                response.headers["X-Sinergia-Cid-Versao"] = str(diag_cid["versao_cid"])
            if diag_cid.get("valido") is not None:
                response.headers["X-Sinergia-Cid-Valido"] = str(
                    bool(diag_cid["valido"])
                ).lower()
            if diag_cid.get("detalhe"):
                response.headers["X-Sinergia-Cid-Detalhe"] = quote(
                    str(diag_cid["detalhe"])[:200], safe=""
                )
        return resultado

    return app_a1


app = criar_app_a1()
