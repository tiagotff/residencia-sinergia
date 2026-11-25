"""Diagnóstico pré-voo do ambiente de integração A0 <-> A1 (sem dependência de UI)."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from sinergia.agentes.a1_recepcao.avaliacao_metricas import (
    validar_integridade_catalogo_sintetico,
)
from sinergia.agentes.a1_recepcao.cliente_cid_who import ClienteCidWHO
from sinergia.agentes.a1_recepcao.config import A1Config, obter_config_a1
from sinergia.agentes.a1_recepcao.gerador_sintetico import garantir_atestados_sinteticos
from sinergia.agentes.a1_recepcao.repositorio import RepositorioA1Postgres

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]

StatusDiagnostico = Literal["ok", "aviso", "erro"]

_TABELAS_GUIA_V07: tuple[str, ...] = (
    "atestado",
    "atestado_arquivo",
    "atestado_campo",
    "baixa_fisica",
    "cid_restrito",
)


class ItemDiagnostico(BaseModel):
    """Resultado padronizado de uma checagem pré-voo do painel."""

    nome: str
    status: StatusDiagnostico
    resumo: str
    detalhes: dict[str, Any] = Field(default_factory=dict)


def mascarar_segredo(valor: str | None) -> str:
    """Mascara valores sensíveis exibindo apenas presença e tamanho."""
    if not valor or not valor.strip():
        return "vazio"
    return f"preenchido ({len(valor.strip())} chars)"


def sanitizar_url_conexao(url: str) -> str:
    """Remove usuário e senha de URLs de conexão ou segredos de mensagens de erro."""
    sanitizado = re.sub(r"://[^/\s:@]+:[^/\s@]+@", "://***:***@", url)
    sanitizado = re.sub(r"gsk_[A-Za-z0-9_-]+", "gsk_***", sanitizado)
    sanitizado = re.sub(
        r"(?:Bearer|Basic)\s+\S+", "Bearer ***", sanitizado, flags=re.IGNORECASE
    )
    sanitizado = re.sub(
        r"(client_secret|WHO_ICD_CLIENT_SECRET)\s*[:=]\s*\S+",
        r"\1=***",
        sanitizado,
        flags=re.IGNORECASE,
    )
    return sanitizado[:260]


def verificar_variaveis_env(config: A1Config | None = None) -> ItemDiagnostico:
    """Valida presença das variáveis de ambiente do A1 e do A0 sem expor segredos."""
    cfg = config or obter_config_a1()

    chaves_a1 = {
        "GROQ_API_KEY": mascarar_segredo(cfg.groq_api_key),
        "LLM_MODEL": cfg.llm_model or "vazio",
        "LLM_TIMEOUT_SEGUNDOS": str(cfg.llm_timeout_segundos),
        "DATABASE_URL": sanitizar_url_conexao(cfg.database_url)
        if cfg.database_url.strip()
        else "vazio",
        "POLITICA_EXTRAIR_CID": str(cfg.politica_extrair_cid).lower(),
        "POLITICA_CID": cfg.politica_cid_modo,
        "LIMIAR_CONFIANCA_CAMPO": str(cfg.limiar_confianca_campo),
        "PRAZO_ENTREGA_FISICA_DIAS_UTEIS": str(cfg.prazo_entrega_fisica_dias_uteis),
        "ATESTADOS_STORAGE_DIR": str(cfg.atestados_storage_dir),
        "MAX_ARQUIVO_BYTES": str(cfg.max_arquivo_bytes),
        "MAX_PAGINAS_PDF": str(cfg.max_paginas_pdf),
        "WHO_ICD_ENABLED": str(cfg.who_icd_enabled).lower(),
        "WHO_ICD_BASE_URL": cfg.who_icd_base_url or "vazio",
        "WHO_ICD_CLIENT_ID": mascarar_segredo(cfg.who_icd_client_id),
        "WHO_ICD_CLIENT_SECRET": mascarar_segredo(cfg.who_icd_client_secret),
        "WHO_ICD_RELEASE_11": cfg.who_icd_release_11,
        "WHO_ICD_LANGUAGE": cfg.who_icd_language,
        "WHO_ICD_SEED_URL": cfg.who_icd_seed_url or "vazio",
    }

    chaves_a0 = {
        "JWT_SECRET": mascarar_segredo(os.getenv("JWT_SECRET")),
        "SMTP_HOST": os.getenv("SMTP_HOST", "").strip() or "vazio",
        "SMTP_PORT": os.getenv("SMTP_PORT", "").strip() or "vazio",
        "SMTP_USER": mascarar_segredo(os.getenv("SMTP_USER")),
        "SMTP_PASSWORD": mascarar_segredo(os.getenv("SMTP_PASSWORD")),
    }

    pendencias: list[str] = []
    if not cfg.groq_api_key.strip():
        pendencias.append("GROQ_API_KEY vazia (usando apenas MockLLM)")
    if not os.getenv("JWT_SECRET", "").strip():
        pendencias.append("JWT_SECRET vazio (necessário para login no A0)")
    if not os.getenv("SMTP_HOST", "").strip():
        pendencias.append("SMTP_HOST vazio (opcional no chat; exigido para e-mail no A0)")

    status: StatusDiagnostico = "ok" if not pendencias else "aviso"
    resumo = (
        "Variáveis principais do A1 e A0 configuradas."
        if not pendencias
        else "; ".join(pendencias)
    )
    return ItemDiagnostico(
        nome="Variáveis de Ambiente (.env)",
        status=status,
        resumo=resumo,
        detalhes={"a1": chaves_a1, "a0": chaves_a0},
    )


def verificar_postgresql(database_url: str, timeout_s: int = 2) -> ItemDiagnostico:
    """Verifica conectividade com o PostgreSQL 16, migração 003 e seed canônico."""
    url_limpa = database_url.strip()
    if not url_limpa:
        return ItemDiagnostico(
            nome="PostgreSQL 16 + Migração 003",
            status="aviso",
            resumo="DATABASE_URL vazia (modo memória ativo).",
        )
    if psycopg is None:
        return ItemDiagnostico(
            nome="PostgreSQL 16 + Migração 003",
            status="erro",
            resumo="Driver psycopg não instalado.",
        )

    inicio = time.perf_counter()
    try:
        with psycopg.connect(url_limpa, connect_timeout=timeout_s) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'afastamento'
                  AND table_name = ANY(%s)
                """,
                (list(_TABELAS_GUIA_V07),),
            )
            tabelas_encontradas = {str(row[0]) for row in cur.fetchall()}

            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'afastamento'
                  AND table_name = 'atestado'
                  AND column_name = 'id_caso'
                """
            )
            tem_id_caso = cur.fetchone() is not None

            cur.execute(
                "SELECT count(*) FROM cadastro.colaborador WHERE id_colaborador = 42"
            )
            tem_carla = bool((cur.fetchone() or [0])[0] >= 1)

            cur.execute(
                "SELECT count(*) FROM cadastro.usuario WHERE LOWER(login) = 'rh.ana'"
            )
            tem_rh = bool((cur.fetchone() or [0])[0] >= 1)
    except (psycopg.Error, OSError, ValueError) as exc:
        latencia_ms = round((time.perf_counter() - inicio) * 1000, 1)
        return ItemDiagnostico(
            nome="PostgreSQL 16 + Migração 003",
            status="erro",
            resumo=f"Banco inacessível ({sanitizar_url_conexao(str(exc))}).",
            detalhes={"latencia_ms": latencia_ms, "url": sanitizar_url_conexao(url_limpa)},
        )

    latencia_ms = round((time.perf_counter() - inicio) * 1000, 1)
    faltantes = [t for t in _TABELAS_GUIA_V07 if t not in tabelas_encontradas]
    pendencias: list[str] = []
    if faltantes:
        pendencias.append(f"tabelas ausentes em afastamento: {', '.join(faltantes)}")
    if not tem_id_caso:
        pendencias.append("coluna id_caso ausente em afastamento.atestado (aplicar 003)")
    if not tem_carla or not tem_rh:
        pendencias.append("seed canônico ausente (aplicar db/seeds/001_seed_a1.sql)")

    status: StatusDiagnostico = "ok" if not pendencias else "aviso"
    resumo = (
        f"Conectado em {latencia_ms} ms; 5 tabelas v0.7, id_caso e seed OK."
        if not pendencias
        else f"Conectado em {latencia_ms} ms, mas com pendências: {'; '.join(pendencias)}"
    )
    return ItemDiagnostico(
        nome="PostgreSQL 16 + Migração 003",
        status=status,
        resumo=resumo,
        detalhes={
            "latencia_ms": latencia_ms,
            "tabelas_v07_presentes": sorted(tabelas_encontradas),
            "tabelas_v07_faltantes": faltantes,
            "tem_coluna_id_caso": tem_id_caso,
            "seed_carla_presente": tem_carla,
            "seed_rh_ana_presente": tem_rh,
        },
    )


def verificar_endpoint_http(
    base_url: str,
    rota: str = "/health",
    *,
    timeout_s: float = 3.0,
    nome: str = "Endpoint HTTP",
    http_client: Any | None = None,
) -> ItemDiagnostico:
    """Executa GET na rota de healthcheck de um serviço HTTP (A1 ou A0)."""
    url_base = base_url.strip().rstrip("/")
    if not url_base:
        return ItemDiagnostico(
            nome=nome,
            status="aviso",
            resumo="URL não informada.",
        )

    destino = f"{url_base}{rota}"
    inicio = time.perf_counter()
    try:
        resp = (
            http_client.get(destino, timeout=timeout_s)
            if http_client is not None
            else httpx.get(destino, timeout=timeout_s)
        )
    except (httpx.HTTPError, ValueError, TypeError, OSError) as exc:
        latencia_ms = round((time.perf_counter() - inicio) * 1000, 1)
        return ItemDiagnostico(
            nome=nome,
            status="aviso",
            resumo=f"Serviço offline ou inacessível em {destino} ({sanitizar_url_conexao(str(exc))}).",
            detalhes={"url": destino, "latencia_ms": latencia_ms},
        )

    latencia_ms = round((time.perf_counter() - inicio) * 1000, 1)
    if resp.status_code != 200:
        return ItemDiagnostico(
            nome=nome,
            status="erro",
            resumo=f"{destino} respondeu HTTP {resp.status_code}.",
            detalhes={"url": destino, "status_code": resp.status_code, "latencia_ms": latencia_ms},
        )

    try:
        dados = resp.json()
    except ValueError:
        dados = {"texto": resp.text[:200]}

    return ItemDiagnostico(
        nome=nome,
        status="ok",
        resumo=f"HTTP 200 OK em {latencia_ms} ms ({destino}).",
        detalhes={"url": destino, "latencia_ms": latencia_ms, "resposta": dados},
    )


def verificar_groq_ping(
    config: A1Config | None = None,
    *,
    timeout_s: float = 10.0,
    http_client: Any | None = None,
) -> ItemDiagnostico:
    """Valida conectividade e autenticação na API da Groq consultando /openai/v1/models."""
    cfg = config or obter_config_a1()
    chave = cfg.groq_api_key.strip()
    if not chave:
        return ItemDiagnostico(
            nome="Groq Cloud API",
            status="aviso",
            resumo="GROQ_API_KEY vazia (teste em nuvem indisponível; MockLLM ativo).",
        )

    inicio = time.perf_counter()
    try:
        headers = {"Authorization": f"Bearer {chave}"}
        url = "https://api.groq.com/openai/v1/models"
        resp = (
            http_client.get(url, headers=headers, timeout=timeout_s)
            if http_client is not None
            else httpx.get(url, headers=headers, timeout=timeout_s)
        )
    except (httpx.HTTPError, ValueError, TypeError, OSError) as exc:
        latencia_ms = round((time.perf_counter() - inicio) * 1000, 1)
        return ItemDiagnostico(
            nome="Groq Cloud API",
            status="erro",
            resumo=f"Falha de rede ao consultar Groq ({sanitizar_url_conexao(str(exc))}).",
            detalhes={"latencia_ms": latencia_ms},
        )

    latencia_ms = round((time.perf_counter() - inicio) * 1000, 1)
    if resp.status_code != 200:
        return ItemDiagnostico(
            nome="Groq Cloud API",
            status="erro",
            resumo=f"Groq respondeu HTTP {resp.status_code} em {latencia_ms} ms.",
            detalhes={"status_code": resp.status_code, "latencia_ms": latencia_ms},
        )

    try:
        modelos = [str(m.get("id", "")) for m in resp.json().get("data", [])]
    except (ValueError, TypeError, AttributeError):
        modelos = []

    modelo_disponivel = cfg.llm_model in modelos if modelos else True
    status: StatusDiagnostico = "ok" if modelo_disponivel else "aviso"
    resumo = (
        f"Autenticado em {latencia_ms} ms; modelo '{cfg.llm_model}' disponível."
        if modelo_disponivel
        else f"Autenticado em {latencia_ms} ms, mas '{cfg.llm_model}' não apareceu em /models."
    )
    return ItemDiagnostico(
        nome="Groq Cloud API",
        status=status,
        resumo=resumo,
        detalhes={
            "latencia_ms": latencia_ms,
            "modelo_configurado": cfg.llm_model,
            "modelo_disponivel": modelo_disponivel,
            "total_modelos": len(modelos),
        },
    )


def verificar_catalogo_sintetico(storage_dir: Path | None = None) -> ItemDiagnostico:
    """Confere a integridade SHA-256 dos 24 arquivos sintéticos e do caso canônico 105."""
    pasta = storage_dir or obter_config_a1().atestados_storage_dir
    try:
        amostras = garantir_atestados_sinteticos(pasta)
        integridade = validar_integridade_catalogo_sintetico(amostras, pasta)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        return ItemDiagnostico(
            nome="Catálogo Sintético (SHA-256)",
            status="erro",
            resumo=f"Falha ao validar catálogo em {pasta}: {sanitizar_url_conexao(str(exc))}",
        )

    ok = bool(integridade.get("integro"))
    validos = integridade.get("arquivos_validos", 0)
    total = integridade.get("total_arquivos_verificados", 0)
    return ItemDiagnostico(
        nome="Catálogo Sintético (SHA-256)",
        status="ok" if ok else "erro",
        resumo=(
            f"{validos}/{total} arquivos íntegros; SHA-256 canônico do caso 105 preservado."
            if ok
            else f"Divergências no catálogo ({validos}/{total} válidos)."
        ),
        detalhes=integridade,
    )


def verificar_who_icd_api(
    config: A1Config | None = None,
    *,
    testar_rede: bool = False,
    codigo_teste: str = "ME84.2",
    cliente: ClienteCidWHO | None = None,
) -> ItemDiagnostico:
    """Valida configuração (e opcionalmente conectividade real) da API externa WHO ICD."""
    cfg = config or obter_config_a1()
    if not cfg.who_icd_enabled:
        return ItemDiagnostico(
            nome="API Externa WHO ICD (CID-11/CID-10)",
            status="aviso",
            resumo="WHO_ICD_ENABLED=false (validação restrita à tabela local afastamento.cid).",
            detalhes={"enabled": False, "base_url": cfg.who_icd_base_url},
        )

    cli = cliente or ClienteCidWHO(config=cfg)
    if not cli.credenciais_externas_configuradas and not cli.local_base_url:
        return ItemDiagnostico(
            nome="API Externa WHO ICD (CID-11/CID-10)",
            status="aviso",
            resumo=(
                "Container local não configurado e WHO_ICD_CLIENT_ID / WHO_ICD_CLIENT_SECRET vazios "
                "(usando apenas afastamento.cid)."
            ),
            detalhes={
                "enabled": True,
                "credenciais_configuradas": False,
                "base_url": cfg.who_icd_base_url,
                "local_base_url": cli.local_base_url,
            },
        )

    if not testar_rede:
        if cli.credenciais_externas_configuradas:
            resumo = (
                f"Fallback 1: {cli.local_base_url or 'desativado'} · "
                f"Fallback 2 (OAuth2): {cfg.who_icd_base_url} "
                f"(release {cfg.who_icd_release_11}/{cfg.who_icd_language})."
            )
            status_item: Literal["ok", "aviso", "erro"] = "ok"
        else:
            resumo = (
                f"Fallback 1: {cli.local_base_url} · "
                "Fallback 2 externo sem credenciais OAuth2."
            )
            status_item = "aviso"
        return ItemDiagnostico(
            nome="API Externa WHO ICD (CID-11/CID-10)",
            status=status_item,
            resumo=resumo,
            detalhes={
                "enabled": True,
                "credenciais_configuradas": cli.credenciais_externas_configuradas,
                "local_base_url": cli.local_base_url,
                "base_url": cfg.who_icd_base_url,
                "release_11": cfg.who_icd_release_11,
                "language": cfg.who_icd_language,
            },
        )

    inicio = time.perf_counter()
    res = cli.consultar_cid(codigo_teste)
    latencia_ms = round((time.perf_counter() - inicio) * 1000, 1)
    if res.status == "ENCONTRADO" and res.dados is not None:
        nome_doenca = res.dados.nome_doenca or res.dados.descricao
        persistido_banco = False
        if cfg.database_url.strip() and psycopg is not None:
            erros_db: tuple[type[BaseException], ...] = (
                psycopg.Error,
                OSError,
                ValueError,
                RuntimeError,
            )
            try:
                RepositorioA1Postgres(cfg.database_url).salvar_cid(
                    res.dados.codigo_cid,
                    res.dados.descricao,
                    res.dados.capitulo,
                    nome_doenca=nome_doenca,
                )
                persistido_banco = True
            except erros_db:
                persistido_banco = False
        sufixo_db = " e salvo em afastamento.cid" if persistido_banco else ""
        return ItemDiagnostico(
            nome="API Externa WHO ICD (CID-11/CID-10)",
            status="ok",
            resumo=(
                f"Conectado em {latencia_ms} ms; teste {res.dados.codigo_cid} "
                f"({res.dados.versao_cid}): {nome_doenca}{sufixo_db}."
            ),
            detalhes={
                "latencia_ms": latencia_ms,
                "codigo_teste": res.dados.codigo_cid,
                "nome_doenca": nome_doenca,
                "descricao": res.dados.descricao,
                "capitulo": res.dados.capitulo,
                "versao_cid": res.dados.versao_cid,
                "persistido_em_afastamento_cid": persistido_banco,
            },
        )
    return ItemDiagnostico(
        nome="API Externa WHO ICD (CID-11/CID-10)",
        status="erro" if res.status == "INDISPONIVEL" else "aviso",
        resumo=(
            f"Consulta de teste ({codigo_teste}) retornou {res.status} "
            f"em {latencia_ms} ms ({sanitizar_url_conexao(res.detalhe or 'sem detalhes')})."
        ),
        detalhes={"latencia_ms": latencia_ms, "status_who": res.status},
    )


def executar_diagnostico_completo(
    *,
    config: A1Config | None = None,
    url_a1_http: str = "http://localhost:8001",
    url_a0_http: str = "",
    testar_groq_rede: bool = False,
    testar_who_rede: bool = False,
) -> list[ItemDiagnostico]:
    """Executa todas as checagens pré-voo de prontidão A0 <-> A1."""
    cfg = config or obter_config_a1()
    itens = [
        verificar_variaveis_env(cfg),
        verificar_postgresql(cfg.database_url),
        verificar_who_icd_api(cfg, testar_rede=testar_who_rede),
        verificar_catalogo_sintetico(cfg.atestados_storage_dir),
    ]
    if url_a1_http.strip():
        itens.append(
            verificar_endpoint_http(
                url_a1_http, "/health", nome="FastAPI Isolada do A1 (/health)"
            )
        )
    if url_a0_http.strip():
        itens.append(
            verificar_endpoint_http(
                url_a0_http, "/health", nome="Orquestrador A0 (/health)"
            )
        )
    if testar_groq_rede:
        itens.append(verificar_groq_ping(cfg))
    return itens
