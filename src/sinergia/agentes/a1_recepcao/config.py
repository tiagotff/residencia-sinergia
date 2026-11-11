"""Configurações centralizadas do Agente 1 (A1 — Recepção e Processamento).

Carrega variáveis do arquivo `.env` (ou do ambiente) e centraliza os parâmetros
de decisão da especificação v2.1 (Seção 5.3, Anexo E.2 e Anexo G).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _str_to_bool(valor: str | None, padrao: bool = True) -> bool:
    if valor is None:
        return padrao
    return valor.strip().lower() in {"1", "true", "t", "yes", "y", "sim", "s"}


@dataclass(frozen=True)
class A1Config:
    """Parâmetros de execução do Agente 1."""

    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    database_url: str = "postgresql://sinergia:sinergia@localhost:5432/sinergia"
    politica_extrair_cid: bool = True
    limiar_confianca_campo: float = 0.8
    prazo_entrega_fisica_dias_uteis: int = 2
    atestados_storage_dir: Path = Path(".storage/atestados")

    @classmethod
    def from_env(cls) -> A1Config:
        """Instancia `A1Config` a partir das variáveis de ambiente atuais."""
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql://sinergia:sinergia@localhost:5432/sinergia",
            ),
            politica_extrair_cid=_str_to_bool(
                os.getenv("POLITICA_EXTRAIR_CID"), padrao=True
            ),
            limiar_confianca_campo=float(
                os.getenv("LIMIAR_CONFIANCA_CAMPO", "0.8")
            ),
            prazo_entrega_fisica_dias_uteis=int(
                os.getenv("PRAZO_ENTREGA_FISICA_DIAS_UTEIS", "2")
            ),
            atestados_storage_dir=Path(
                os.getenv("ATESTADOS_STORAGE_DIR", ".storage/atestados")
            ),
        )


def obter_config_a1() -> A1Config:
    """Retorna a configuração ativa do Agente 1."""
    return A1Config.from_env()
