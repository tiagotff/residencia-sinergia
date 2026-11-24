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
    llm_timeout_segundos: float = 30.0
    database_url: str = "postgresql://sinergia:sinergia@localhost:5432/sinergia"
    politica_extrair_cid: bool = True
    politica_cid_modo: str = "extrair_restrito"
    limiar_confianca_campo: float = 0.8
    prazo_entrega_fisica_dias_uteis: int = 2
    atestados_storage_dir: Path = Path(".storage/atestados")
    max_arquivo_bytes: int = 10 * 1024 * 1024
    max_paginas_pdf: int = 5
    who_icd_enabled: bool = True
    who_icd_base_url: str = "https://id.who.int"
    who_icd_token_url: str = "https://icdaccessmanagement.who.int/connect/token"
    who_icd_client_id: str = ""
    who_icd_client_secret: str = ""
    who_icd_release_11: str = "2026-01"
    who_icd_linearization: str = "mms"
    who_icd_release_10: str = "2019"
    who_icd_language: str = "pt"
    who_icd_timeout_segundos: float = 10.0
    who_icd_seed_url: str = "http://localhost:8088"

    def __post_init__(self) -> None:
        if self.llm_timeout_segundos <= 0:
            raise ValueError("llm_timeout_segundos deve ser maior que zero.")
        if self.max_arquivo_bytes <= 0:
            raise ValueError("max_arquivo_bytes deve ser maior que zero.")
        if self.max_paginas_pdf <= 0:
            raise ValueError("max_paginas_pdf deve ser maior que zero.")
        if self.who_icd_timeout_segundos <= 0:
            raise ValueError("who_icd_timeout_segundos deve ser maior que zero.")

    @classmethod
    def from_env(cls) -> A1Config:
        """Instancia `A1Config` a partir das variáveis de ambiente atuais."""
        politica_cid_env = (os.getenv("POLITICA_CID") or "").strip().lower()
        if politica_cid_env in {"nao_extrair", "não_extrair"}:
            politica_extrair_cid = False
            politica_cid_modo = "nao_extrair"
        elif politica_cid_env in {"extrair_restrito", "extrair"}:
            politica_extrair_cid = True
            politica_cid_modo = politica_cid_env
        elif politica_cid_env:
            raise ValueError(
                f"POLITICA_CID inválida: {politica_cid_env!r}. "
                "Valores aceitos: 'nao_extrair', 'extrair_restrito' ou 'extrair'."
            )
        else:
            politica_extrair_cid = _str_to_bool(
                os.getenv("POLITICA_EXTRAIR_CID"), padrao=True
            )
            politica_cid_modo = (
                "extrair_restrito" if politica_extrair_cid else "nao_extrair"
            )

        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
            llm_timeout_segundos=float(os.getenv("LLM_TIMEOUT_SEGUNDOS", "30")),
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql://sinergia:sinergia@localhost:5432/sinergia",
            ),
            politica_extrair_cid=politica_extrair_cid,
            politica_cid_modo=politica_cid_modo,
            limiar_confianca_campo=float(
                os.getenv("LIMIAR_CONFIANCA_CAMPO", "0.8")
            ),
            prazo_entrega_fisica_dias_uteis=int(
                os.getenv("PRAZO_ENTREGA_FISICA_DIAS_UTEIS", "2")
            ),
            atestados_storage_dir=Path(
                os.getenv("ATESTADOS_STORAGE_DIR", ".storage/atestados")
            ),
            max_arquivo_bytes=int(
                os.getenv("MAX_ARQUIVO_BYTES", str(10 * 1024 * 1024))
            ),
            max_paginas_pdf=int(os.getenv("MAX_PAGINAS_PDF", "5")),
            who_icd_enabled=_str_to_bool(
                os.getenv("WHO_ICD_ENABLED"), padrao=True
            ),
            who_icd_base_url=os.getenv(
                "WHO_ICD_BASE_URL", "https://id.who.int"
            ).strip().rstrip("/"),
            who_icd_token_url=os.getenv(
                "WHO_ICD_TOKEN_URL",
                "https://icdaccessmanagement.who.int/connect/token",
            ).strip(),
            who_icd_client_id=os.getenv("WHO_ICD_CLIENT_ID", "").strip(),
            who_icd_client_secret=os.getenv("WHO_ICD_CLIENT_SECRET", "").strip(),
            who_icd_release_11=os.getenv("WHO_ICD_RELEASE_11", "2026-01").strip(),
            who_icd_linearization=os.getenv("WHO_ICD_LINEARIZATION", "mms").strip(),
            who_icd_release_10=os.getenv("WHO_ICD_RELEASE_10", "2019").strip(),
            who_icd_language=os.getenv("WHO_ICD_LANGUAGE", "pt").strip(),
            who_icd_timeout_segundos=float(
                os.getenv("WHO_ICD_TIMEOUT_SEGUNDOS", "10")
            ),
            who_icd_seed_url=os.getenv(
                "WHO_ICD_SEED_URL", "http://localhost:8088"
            ).strip().rstrip("/"),
        )


def obter_config_a1() -> A1Config:
    """Retorna a configuração ativa do Agente 1."""
    return A1Config.from_env()

