"""Cliente HTTP para a WHO ICD API (fallback externo OAuth2 + leitura do container seed).

Separação de responsabilidades:
- **Fallback em tempo de execução no Agente 1:** quando um CID não existe em
  ``afastamento.cid`` no PostgreSQL, consulta a API externa oficial da OMS
  (``https://id.who.int``) autenticada via OAuth 2.0 Client Credentials,
  suportando tanto **CID-11 (pt)** quanto **CID-10 (2019)**.
- **Seed local a partir do container Docker ``whoicd/icd-api``:** o método
  ``percorrer_arvore_icd11`` consulta o container local (ex.: ``http://localhost:8088``,
  sem OAuth2 e sem rate limit) para popular ``afastamento.cid`` em português.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import quote, urlparse, urlunparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from sinergia.agentes.a1_recepcao.config import A1Config, obter_config_a1
from sinergia.agentes.a1_recepcao.validador import (
    cid_sintaxe_valida,
    eh_cid10,
    eh_cid11,
)

_LOGGER = logging.getLogger(__name__)
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_HOSTS_LOCAIS = {"localhost", "127.0.0.1", "0.0.0.0", "icd-api", "whoicd"}


class DadosCidWHO(BaseModel):
    """Registro de CID normalizado para persistência em ``afastamento.cid``."""

    model_config = ConfigDict(extra="forbid")

    codigo_cid: str = Field(..., min_length=1, max_length=8)
    descricao: str = Field(..., min_length=1, max_length=255)
    capitulo: str = Field(..., min_length=1, max_length=10)
    versao_cid: Literal["CID-11", "CID-10"] = "CID-11"
    uri: str | None = None


class ResultadoConsultaCidWHO(BaseModel):
    """Resultado padronizado de consulta de CID na API da OMS."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ENCONTRADO", "NAO_ENCONTRADO", "INDISPONIVEL"]
    dados: DadosCidWHO | None = None
    detalhe: str | None = None


@runtime_checkable
class ClienteCidProtocol(Protocol):
    """Contrato injetável consumido pelo ``ServicoA1`` para fallback de CID."""

    def consultar_cid(self, codigo_cid: str) -> ResultadoConsultaCidWHO: ...


def _extrair_texto_idioma(campo: Any) -> str:
    """Extrai texto limpo de um campo ``LanguageSpecificText`` (JSON-LD) ou string."""
    if isinstance(campo, dict):
        valor = campo.get("@value") or campo.get("value") or ""
    elif isinstance(campo, str):
        valor = campo
    else:
        valor = ""
    limpo = _RE_HTML_TAG.sub(" ", str(valor))
    return " ".join(limpo.split()).strip()


def _truncar(texto: str, limite: int) -> str:
    texto_limpo = " ".join(texto.split()).strip()
    if len(texto_limpo) <= limite:
        return texto_limpo
    return texto_limpo[: limite - 1].rstrip() + "…"


def _reescrever_uri_para_base(uri: str, base_url: str) -> str:
    """Reescreve scheme/host de uma URI WHO (ex.: http://id.who.int/...) para ``base_url``."""
    uri_limpa = uri.strip()
    if not uri_limpa:
        return ""
    if uri_limpa.startswith("/"):
        return f"{base_url.rstrip('/')}{uri_limpa}"
    parsed_uri = urlparse(uri_limpa)
    parsed_base = urlparse(base_url.rstrip("/"))
    if not parsed_uri.scheme or not parsed_base.scheme:
        return uri_limpa
    return urlunparse(
        (
            parsed_base.scheme,
            parsed_base.netloc,
            parsed_uri.path,
            parsed_uri.params,
            parsed_uri.query,
            parsed_uri.fragment,
        )
    )


def _eh_host_local(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").strip().lower()
    return host in _HOSTS_LOCAIS or host.endswith(".local")


def _derivar_capitulo(codigo_cid: str, capitulo_hint: str | None = None) -> str:
    if capitulo_hint and capitulo_hint.strip():
        return capitulo_hint.strip()[:10]
    codigo = codigo_cid.strip().upper()
    if eh_cid11(codigo):
        return codigo[:2][:10]
    return (codigo[:1] or "N/D")[:10]


class ClienteCidWHO:
    """Cliente ``httpx`` para a WHO ICD API (externa OAuth2 e container local)."""

    def __init__(
        self,
        *,
        config: A1Config | None = None,
        base_url: str | None = None,
        token_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        release_11: str | None = None,
        linearization: str | None = None,
        release_10: str | None = None,
        language: str | None = None,
        timeout_segundos: float | None = None,
        requer_oauth: bool | None = None,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 3,
    ) -> None:
        cfg = config or obter_config_a1()
        self.base_url = (base_url if base_url is not None else cfg.who_icd_base_url).strip().rstrip("/")
        self.token_url = (token_url if token_url is not None else cfg.who_icd_token_url).strip()
        self.client_id = (client_id if client_id is not None else cfg.who_icd_client_id).strip()
        self.client_secret = (
            client_secret if client_secret is not None else cfg.who_icd_client_secret
        ).strip()
        self.release_11 = (release_11 if release_11 is not None else cfg.who_icd_release_11).strip()
        self.linearization = (
            linearization if linearization is not None else cfg.who_icd_linearization
        ).strip()
        self.release_10 = (release_10 if release_10 is not None else cfg.who_icd_release_10).strip()
        self.language = (language if language is not None else cfg.who_icd_language).strip() or "pt"
        self.timeout_segundos = float(
            timeout_segundos if timeout_segundos is not None else cfg.who_icd_timeout_segundos
        )
        self.requer_oauth = (
            requer_oauth if requer_oauth is not None else not _eh_host_local(self.base_url)
        )
        self._transport = transport
        self._max_retries = max(1, int(max_retries))
        self._token_lock = threading.RLock()
        self._access_token: str | None = None
        self._token_expira_em: float = 0.0

    @property
    def credenciais_configuradas(self) -> bool:
        """Indica se o cliente possui o necessário para autenticar (ou se dispensa OAuth2)."""
        if not self.requer_oauth:
            return True
        return bool(self.client_id and self.client_secret)

    def _criar_http_client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self.timeout_segundos,
            transport=self._transport,
            follow_redirects=True,
        )

    def _obter_token_oauth(
        self,
        client: httpx.Client,
        *,
        forcar_renovacao: bool = False,
    ) -> str | None:
        if not self.requer_oauth:
            return None
        if not (self.client_id and self.client_secret):
            raise RuntimeError(
                "Credenciais WHO_ICD_CLIENT_ID / WHO_ICD_CLIENT_SECRET não configuradas."
            )

        with self._token_lock:
            agora = time.monotonic()
            if (
                not forcar_renovacao
                and self._access_token
                and agora < self._token_expira_em
            ):
                return self._access_token

            resp = client.post(
                self.token_url,
                data={
                    "grant_type": "client_credentials",
                    "scope": "icdapi_access",
                },
                auth=(self.client_id, self.client_secret),
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            dados = resp.json()
            token = str(dados.get("access_token") or "").strip()
            if not token:
                raise RuntimeError("Resposta OAuth2 da OMS sem campo access_token.")
            expires_in = float(dados.get("expires_in") or 3600)
            margem = min(60.0, max(expires_in * 0.1, 5.0))
            self._access_token = token
            self._token_expira_em = time.monotonic() + max(expires_in - margem, 10.0)
            return token

    def _headers(self, token: str | None, *, language: str | None = None) -> dict[str, str]:
        headers = {
            "API-Version": "v2",
            "Accept": "application/json",
            "Accept-Language": language or self.language,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _get_json(
        self,
        client: httpx.Client,
        url: str,
        *,
        language: str | None = None,
    ) -> tuple[int, dict[str, Any] | None]:
        """Executa GET com renovação automática de token em 401 e backoff em 429/5xx."""
        forcar_token = False
        ultima_status = 0
        for tentativa in range(self._max_retries):
            token = self._obter_token_oauth(client, forcar_renovacao=forcar_token)
            forcar_token = False
            resp = client.get(url, headers=self._headers(token, language=language))
            ultima_status = resp.status_code

            if resp.status_code == 401 and self.requer_oauth and tentativa + 1 < self._max_retries:
                forcar_token = True
                continue

            if resp.status_code in {429, 502, 503, 504} and tentativa + 1 < self._max_retries:
                retry_after = resp.headers.get("Retry-After")
                try:
                    espera = min(float(retry_after), 5.0) if retry_after else min(0.5 * (2**tentativa), 4.0)
                except ValueError:
                    espera = min(0.5 * (2**tentativa), 4.0)
                time.sleep(espera)
                continue

            if resp.status_code == 404:
                return 404, None
            if resp.status_code == 200:
                return 200, resp.json()
            return resp.status_code, None

        return ultima_status, None

    def verificar_saude(self) -> bool:
        """Verifica se o endpoint WHO configurado (externo ou container local) está acessível."""
        if not self.credenciais_configuradas:
            return False
        url = f"{self.base_url}/icd/release/11/{self.release_11}/{self.linearization}"
        try:
            with self._criar_http_client() as client:
                status, _ = self._get_json(client, url, language=self.language)
                return status == 200
        except (httpx.HTTPError, RuntimeError, ValueError):
            return False

    def _consultar_cid11(
        self, client: httpx.Client, codigo_norm: str
    ) -> ResultadoConsultaCidWHO:
        code_enc = quote(codigo_norm, safe="")
        url_info = (
            f"{self.base_url}/icd/release/11/{self.release_11}/"
            f"{self.linearization}/codeinfo/{code_enc}"
        )
        status, info = self._get_json(client, url_info, language=self.language)
        if status == 404:
            return ResultadoConsultaCidWHO(
                status="NAO_ENCONTRADO",
                detalhe=f"Código CID-11 '{codigo_norm}' não encontrado na OMS.",
            )
        if status != 200 or not isinstance(info, dict):
            return ResultadoConsultaCidWHO(
                status="INDISPONIVEL",
                detalhe=f"API WHO ICD-11 retornou HTTP {status}.",
            )

        codigo_oficial = str(info.get("code") or codigo_norm).strip().upper()
        stem_uri = str(
            info.get("stemId") or info.get("linearizationUri") or ""
        ).strip()
        if not stem_uri:
            return ResultadoConsultaCidWHO(
                status="NAO_ENCONTRADO",
                detalhe=f"Resposta codeinfo sem stemId para '{codigo_norm}'.",
            )

        url_entidade = _reescrever_uri_para_base(stem_uri, self.base_url)
        status_ent, entidade = self._get_json(client, url_entidade, language=self.language)
        descricao = ""
        capitulo_hint: str | None = None
        if status_ent == 200 and isinstance(entidade, dict):
            class_kind = str(entidade.get("classKind") or "category").strip().lower()
            if class_kind in {"chapter", "block"}:
                return ResultadoConsultaCidWHO(
                    status="NAO_ENCONTRADO",
                    detalhe=f"'{codigo_norm}' corresponde a {class_kind}, não a uma categoria CID.",
                )
            descricao = _extrair_texto_idioma(entidade.get("title"))
            codigo_ent = str(entidade.get("code") or "").strip().upper()
            if codigo_ent and len(codigo_ent) <= 8:
                codigo_oficial = codigo_ent
            capitulo_hint = str(entidade.get("chapter") or "").strip() or None

        if not descricao:
            descricao = f"CID-11 {codigo_oficial}"

        return ResultadoConsultaCidWHO(
            status="ENCONTRADO",
            dados=DadosCidWHO(
                codigo_cid=codigo_oficial[:8],
                descricao=_truncar(descricao, 255),
                capitulo=_derivar_capitulo(codigo_oficial, capitulo_hint),
                versao_cid="CID-11",
                uri=stem_uri,
            ),
            detalhe="Validado via WHO ICD-11 API",
        )

    def _consultar_cid10(
        self, client: httpx.Client, codigo_norm: str
    ) -> ResultadoConsultaCidWHO:
        code_enc = quote(codigo_norm, safe="")
        url = f"{self.base_url}/icd/release/10/{self.release_10}/{code_enc}"
        # Releases de ICD-10 na API da OMS usam inglês ('en') como idioma padrão
        status, entidade = self._get_json(client, url, language="en")
        if status == 404:
            return ResultadoConsultaCidWHO(
                status="NAO_ENCONTRADO",
                detalhe=f"Código CID-10 '{codigo_norm}' não encontrado na OMS.",
            )
        if status != 200 or not isinstance(entidade, dict):
            return ResultadoConsultaCidWHO(
                status="INDISPONIVEL",
                detalhe=f"API WHO ICD-10 retornou HTTP {status}.",
            )

        class_kind = str(entidade.get("classKind") or "").strip().lower()
        if class_kind and class_kind != "category":
            return ResultadoConsultaCidWHO(
                status="NAO_ENCONTRADO",
                detalhe=f"'{codigo_norm}' corresponde a '{class_kind}', não a uma categoria CID-10.",
            )

        codigo_oficial = str(entidade.get("code") or codigo_norm).strip().upper()
        descricao = _extrair_texto_idioma(entidade.get("title")) or f"CID-10 {codigo_oficial}"
        uri = str(entidade.get("@id") or url)
        return ResultadoConsultaCidWHO(
            status="ENCONTRADO",
            dados=DadosCidWHO(
                codigo_cid=codigo_oficial[:8],
                descricao=_truncar(descricao, 255),
                capitulo=_derivar_capitulo(codigo_oficial),
                versao_cid="CID-10",
                uri=uri,
            ),
            detalhe="Validado via WHO ICD-10 API",
        )

    def consultar_cid(self, codigo_cid: str) -> ResultadoConsultaCidWHO:
        """Consulta um código CID-11 ou CID-10 na API WHO configurada."""
        codigo_norm = (codigo_cid or "").strip().upper()
        if not codigo_norm or not cid_sintaxe_valida(codigo_norm):
            return ResultadoConsultaCidWHO(
                status="NAO_ENCONTRADO",
                detalhe=f"Sintaxe CID inválida: {codigo_cid!r}",
            )

        if not self.credenciais_configuradas:
            return ResultadoConsultaCidWHO(
                status="INDISPONIVEL",
                detalhe=(
                    "WHO_ICD_CLIENT_ID / WHO_ICD_CLIENT_SECRET não configurados para "
                    f"{self.base_url}."
                ),
            )

        try:
            with self._criar_http_client() as client:
                if eh_cid11(codigo_norm):
                    return self._consultar_cid11(client, codigo_norm)
                if eh_cid10(codigo_norm):
                    return self._consultar_cid10(client, codigo_norm)
                return ResultadoConsultaCidWHO(
                    status="NAO_ENCONTRADO",
                    detalhe=f"Formato CID não reconhecido: {codigo_norm!r}",
                )
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            _LOGGER.warning(
                "Falha ao consultar CID %s na WHO API (%s): %s",
                codigo_norm,
                self.base_url,
                type(exc).__name__,
            )
            return ResultadoConsultaCidWHO(
                status="INDISPONIVEL",
                detalhe=f"Erro ao consultar API WHO ({type(exc).__name__}).",
            )

    def percorrer_arvore_icd11(
        self,
        *,
        limite: int | None = None,
        progresso_cb: Callable[[int, int], None] | None = None,
    ) -> list[DadosCidWHO]:
        """Percorre a árvore ICD-11 (tipicamente no container Docker local ``whoicd/icd-api``).

        Extrai todas as entidades ``classKind == "category"`` com código válido (``<= 8`` chars)
        e descrição no idioma configurado (padrão ``pt``).
        """
        url_raiz = f"{self.base_url}/icd/release/11/{self.release_11}/{self.linearization}"
        encontrados: dict[str, DadosCidWHO] = {}
        visitados: set[str] = set()
        fila: deque[tuple[str, str]] = deque()

        with self._criar_http_client() as client:
            status, raiz = self._get_json(client, url_raiz, language=self.language)
            if status != 200 or not isinstance(raiz, dict):
                raise RuntimeError(
                    f"Não foi possível acessar a raiz ICD-11 em {url_raiz} (HTTP {status})."
                )

            for child_uri in raiz.get("child") or []:
                if isinstance(child_uri, str) and child_uri.strip():
                    fila.append((_reescrever_uri_para_base(child_uri, self.base_url), ""))

            processados = 0
            while fila:
                if limite is not None and len(encontrados) >= limite:
                    break
                uri_atual, capitulo_atual = fila.popleft()
                if uri_atual in visitados:
                    continue
                visitados.add(uri_atual)

                status_no, no = self._get_json(client, uri_atual, language=self.language)
                processados += 1
                if status_no != 200 or not isinstance(no, dict):
                    continue

                class_kind = str(no.get("classKind") or "").strip().lower()
                codigo_no = str(no.get("code") or "").strip().upper()
                prox_capitulo = capitulo_atual
                if class_kind == "chapter":
                    prox_capitulo = (codigo_no or str(no.get("chapter") or ""))[:10]

                if (
                    class_kind == "category"
                    and codigo_no
                    and len(codigo_no) <= 8
                    and cid_sintaxe_valida(codigo_no)
                ):
                    descricao = _extrair_texto_idioma(no.get("title")) or f"CID-11 {codigo_no}"
                    cap = _derivar_capitulo(codigo_no, prox_capitulo)
                    encontrados[codigo_no] = DadosCidWHO(
                        codigo_cid=codigo_no,
                        descricao=_truncar(descricao, 255),
                        capitulo=cap,
                        versao_cid="CID-11",
                        uri=str(no.get("@id") or uri_atual),
                    )
                    if progresso_cb is not None:
                        progresso_cb(len(encontrados), processados)
                    if limite is not None and len(encontrados) >= limite:
                        break

                for child_uri in no.get("child") or []:
                    if isinstance(child_uri, str) and child_uri.strip():
                        child_local = _reescrever_uri_para_base(child_uri, self.base_url)
                        if child_local not in visitados:
                            fila.append((child_local, prox_capitulo))

        return list(encontrados.values())
