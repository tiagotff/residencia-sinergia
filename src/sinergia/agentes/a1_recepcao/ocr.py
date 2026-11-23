"""Leitura e OCR de atestados armazenados no repositorio privado do A1."""

from __future__ import annotations

import hashlib
import mimetypes
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
from PIL import Image, ImageChops, ImageStat, UnidentifiedImageError

from sinergia.agentes.a1_recepcao.config import obter_config_a1
from sinergia.agentes.a1_recepcao.schemas import ResultadoOCR

try:
    import pymupdf
except ImportError:  # pragma: no cover - dependencia declarada no projeto
    pymupdf = None

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover - depende do ambiente OCR
    RapidOCR = None

_MIME_POR_EXTENSAO = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "application/pdf",
}


def _resolver_arquivo(arquivo_uri: str) -> Path:
    """Resolve caminho relativo ao armazenamento privado ou um URI file://."""
    if not isinstance(arquivo_uri, str) or not arquivo_uri.strip():
        raise ValueError("arquivo_uri não pode ser vazio.")
    if "\x00" in arquivo_uri:
        raise ValueError("arquivo_uri inválido: contém byte nulo.")

    uri = arquivo_uri.strip()
    if uri.startswith(("\\\\", "//")):
        raise ValueError("Caminhos UNC de rede não são permitidos por segurança.")

    caminho_nativo = Path(uri)
    if caminho_nativo.is_absolute():
        return caminho_nativo.expanduser().resolve(strict=False)

    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme.lower() != "file":
        raise ValueError("Somente arquivos locais ou URIs file:// são aceitos.")
    if parsed.scheme:
        if parsed.netloc and parsed.netloc.lower() not in {"", "localhost"}:
            raise ValueError(
                "URIs file:// com host remoto não são permitidas por segurança."
            )
        caminho = Path(url2pathname(parsed.path))
        return caminho.expanduser().resolve(strict=False)

    storage_dir = obter_config_a1().atestados_storage_dir
    ja_prefixado = bool(
        storage_dir.parts
        and caminho_nativo.parts[: len(storage_dir.parts)] == storage_dir.parts
    )
    if ja_prefixado or caminho_nativo.exists():
        caminho = caminho_nativo
    else:
        caminho = storage_dir / caminho_nativo

    resolvido = caminho.expanduser().resolve(strict=False)
    storage_resolvido = storage_dir.expanduser().resolve(strict=False)
    cwd_resolvido = Path.cwd().resolve(strict=False)
    if ".." in caminho_nativo.parts and not resolvido.is_relative_to(storage_resolvido):
        raise ValueError(
            "Caminho de arquivo fora do diretório de armazenamento permitido "
            "(path traversal bloqueado)."
        )
    if not (
        resolvido.is_relative_to(storage_resolvido)
        or (
            (ja_prefixado or caminho_nativo.exists())
            and resolvido.is_relative_to(cwd_resolvido)
        )
    ):
        raise ValueError(
            "Caminho de arquivo fora do diretório de armazenamento permitido "
            "(path traversal bloqueado)."
        )
    return resolvido


def _calcular_sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _detectar_mime(caminho: Path, mime: str | None) -> str | None:
    if mime:
        mime_normalizado = mime.strip().lower()
        if mime_normalizado in {"image/png", "image/jpeg", "application/pdf"}:
            return mime_normalizado

    sufixo_mime = _MIME_POR_EXTENSAO.get(caminho.suffix.lower())
    if sufixo_mime:
        return sufixo_mime
    return mimetypes.guess_type(caminho.name)[0]


def _imagem_esta_em_branco(imagem: Image.Image) -> bool:
    if imagem.width < 20 or imagem.height < 20:
        return True

    escala_de_cinza = imagem.convert("L")
    estatisticas = ImageStat.Stat(escala_de_cinza)
    if estatisticas.extrema[0][0] == estatisticas.extrema[0][1]:
        return True

    fundo = Image.new("L", escala_de_cinza.size, round(estatisticas.mean[0]))
    diferenca = ImageChops.difference(escala_de_cinza, fundo)
    if diferenca.getbbox() is None:
        return True

    try:
        return estatisticas.stddev[0] < 2.0
    except (TypeError, ValueError):
        return True


@lru_cache(maxsize=1)
def _obter_motor_rapidocr() -> Any:
    if RapidOCR is None:
        raise RuntimeError(
            "Dependência OCR indisponível: instale rapidocr-onnxruntime."
        )
    return RapidOCR()


def _executar_rapidocr(imagem: Image.Image) -> tuple[str, float | None]:
    motor = _obter_motor_rapidocr()
    retorno = motor(np.asarray(imagem))
    itens = None
    if hasattr(retorno, "box_res_list"):
        caixas = getattr(retorno, "box_res_list", None)
        textos_retorno = getattr(retorno, "txts", None)
        confiancas_retorno = getattr(retorno, "scores", None)
        if not caixas or not textos_retorno or not confiancas_retorno:
            return "", None
        itens = [
            (caixa, texto, confianca)
            for caixa, texto, confianca in zip(
                caixas,
                textos_retorno,
                confiancas_retorno,
                strict=True,
            )
        ]
    elif isinstance(retorno, tuple) and retorno:
        itens = retorno[0]

    if not itens:
        return "", None
    textos: list[str] = []
    confiancas: list[float] = []
    for item in itens:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        texto = str(item[1]).strip()
        try:
            confianca = float(item[2])
        except (TypeError, ValueError):
            confianca = 0.0
        if texto:
            textos.append(texto)
            confiancas.append(confianca)

    texto = "\n".join(textos).strip()
    media = sum(confiancas) / len(confiancas) if confiancas else None
    return texto, media


def _ocr_imagem(imagem: Image.Image) -> tuple[str, float | None]:
    if _imagem_esta_em_branco(imagem):
        return "", 0.0
    return _executar_rapidocr(imagem)


class _ErroLimitePaginasPDF(ValueError):
    """Indica que o PDF excede o limite de páginas configurado para atestados."""


def _assinatura_valida(caminho: Path, mime_detectado: str) -> bool:
    """Valida magic bytes básicos do arquivo antes da decodificação."""
    try:
        with caminho.open("rb") as arq:
            cabecalho = arq.read(8)
    except OSError:
        return False

    if mime_detectado == "image/png":
        return cabecalho.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_detectado == "image/jpeg":
        return cabecalho.startswith(b"\xff\xd8\xff")
    if mime_detectado == "application/pdf":
        return cabecalho.startswith(b"%PDF-")
    return False


def _processar_imagem(caminho: Path) -> tuple[str, float | None]:
    try:
        with Image.open(caminho) as imagem:
            imagem.load()
            imagem = imagem.convert("RGB")
            return _ocr_imagem(imagem)
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
    ) as exc:
        raise ValueError(f"Arquivo de imagem corrompido ou ilegível: {exc}") from exc


def _processar_pdf(caminho: Path, max_paginas: int = 5) -> tuple[str, float | None]:
    if pymupdf is None:
        raise RuntimeError("Dependência PDF indisponível: instale pymupdf.")

    try:
        documento = pymupdf.open(caminho)
    except Exception as exc:
        raise ValueError(f"Arquivo PDF corrompido ou ilegível: {exc}") from exc

    textos: list[str] = []
    confiancas: list[float] = []
    try:
        if documento.page_count == 0:
            return "", 0.0
        if documento.page_count > max_paginas:
            raise _ErroLimitePaginasPDF(
                f"PDF excede o limite máximo de {max_paginas} páginas para atestados."
            )

        for pagina in documento:
            texto_nativo = pagina.get_text("text").strip()
            if len(texto_nativo) >= 20:
                textos.append(texto_nativo)
                confiancas.append(1.0)
                continue

            pixmap = pagina.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0), alpha=False)
            imagem = Image.frombytes(
                "RGB", (pixmap.width, pixmap.height), pixmap.samples
            )
            texto, confianca = _ocr_imagem(imagem)
            if texto:
                textos.append(texto)
            if confianca is not None:
                confiancas.append(confianca)
    finally:
        documento.close()

    texto_completo = "\n".join(textos).strip()
    media = sum(confiancas) / len(confiancas) if confiancas else None
    return texto_completo, media


def processar_arquivo_atestado(
    arquivo_uri: str,
    mime: str | None = None,
    sha256_esperado: str | None = None,
) -> ResultadoOCR:
    """Calcula o hash e executa OCR em imagem PNG/JPEG ou PDF."""
    cfg = obter_config_a1()
    caminho = _resolver_arquivo(arquivo_uri)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de atestado não encontrado: {caminho}")

    mime_detectado = _detectar_mime(caminho, mime)
    if mime_detectado not in {
        "image/png",
        "image/jpeg",
        "application/pdf",
    }:
        raise ValueError(
            f"Tipo de arquivo não suportado: {mime_detectado or caminho.suffix}"
        )

    tamanho_bytes = caminho.stat().st_size
    if tamanho_bytes > cfg.max_arquivo_bytes:
        raise ValueError(
            f"Arquivo excede o limite máximo permitido de {cfg.max_arquivo_bytes} bytes."
        )

    sha256_calculado = _calcular_sha256(caminho)
    if sha256_esperado:
        esperado = sha256_esperado.strip().lower()
        if len(esperado) != 64:
            raise ValueError("sha256_esperado deve conter exatamente 64 caracteres.")
        try:
            int(esperado, 16)
        except ValueError as exc:
            raise ValueError("sha256_esperado deve ser hexadecimal.") from exc
        if sha256_calculado != esperado:
            raise ValueError("Hash SHA-256 do arquivo não corresponde ao esperado.")

    if not _assinatura_valida(caminho, mime_detectado):
        return ResultadoOCR(
            arquivo_uri=str(caminho),
            mime=mime_detectado,
            sha256=sha256_calculado,
            texto_extraido="",
            confianca_media=0.0,
            legibilidade="ilegivel",
        )

    try:
        if mime_detectado == "application/pdf":
            texto, confianca = _processar_pdf(caminho, max_paginas=cfg.max_paginas_pdf)
        else:
            texto, confianca = _processar_imagem(caminho)
    except _ErroLimitePaginasPDF as exc:
        raise ValueError(str(exc)) from exc
    except (OSError, ValueError):
        return ResultadoOCR(
            arquivo_uri=str(caminho),
            mime=mime_detectado,
            sha256=sha256_calculado,
            texto_extraido="",
            confianca_media=0.0,
            legibilidade="ilegivel",
        )

    texto_limpo = texto.strip()
    if not texto_limpo or (
        confianca is not None and confianca < 0.30
    ):
        legibilidade = "ilegivel"
    else:
        legibilidade = "ok"

    return ResultadoOCR(
        arquivo_uri=str(caminho),
        mime=mime_detectado,
        sha256=sha256_calculado,
        texto_extraido=texto_limpo,
        confianca_media=confianca,
        legibilidade=legibilidade,
    )
