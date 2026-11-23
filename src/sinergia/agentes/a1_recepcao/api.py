"""API HTTP isolada (FastAPI) do Agente 1 (`POST /executar`, Guia A1 v0.7)."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI

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
        return {
            "status": "ok",
            "agente": "A1",
            "protocolo": PROTOCOLO_SINERGIA,
            "modo_repositorio": modo_repo
            or ("postgres" if cfg.database_url.strip() else "memoria"),
            "modo_llm": modo_llm or ("groq" if cfg.groq_api_key.strip() else "nao_configurado"),
        }

    @app_a1.post("/executar", response_model=ResultadoA1)
    def executar_endpoint(tarefa: TarefaA1) -> dict[str, Any]:
        return _obter_agente().executar(tarefa)

    return app_a1


app = criar_app_a1()
