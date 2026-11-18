"""Configurações centralizadas do Agente 2 (A2 — Bem-Estar).

Carrega variáveis do arquivo `.env` (ou do ambiente) e centraliza os parâmetros
de execução do Agente 2 com CrewAI e Groq (`GROQ_API_KEY`, modelo, banco e timeout).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class A2Config:
    """Parâmetros de execução do Agente 2 (A2 — Bem-Estar)."""

    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-20b"
    database_url: str = "postgresql://sinergia:sinergia@localhost:5432/sinergia"
    timeout_s: float = 45.0

    @classmethod
    def from_env(cls) -> A2Config:
        """Instancia `A2Config` a partir das variáveis de ambiente atuais."""
        modelo_env = (
            os.getenv("SINERGIA_A2_LLM_MODEL")
            or os.getenv("LLM_MODEL")
            or "openai/gpt-oss-20b"
        ).strip()
        timeout_raw = os.getenv("SINERGIA_A2_TIMEOUT_S", "45.0").strip()
        try:
            timeout_val = float(timeout_raw)
        except ValueError:
            timeout_val = 45.0

        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            llm_model=modelo_env or "openai/gpt-oss-20b",
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql://sinergia:sinergia@localhost:5432/sinergia",
            ).strip(),
            timeout_s=timeout_val,
        )


def obter_config_a2() -> A2Config:
    """Retorna a configuração ativa do Agente 2."""
    return A2Config.from_env()
