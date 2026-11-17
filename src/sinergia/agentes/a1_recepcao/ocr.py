"""Leitura e OCR de atestados armazenados no repositorio privado do A1."""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlparse

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

    uri = arquivo_uri.strip()
    caminho_nativo = Path(uri)
    if caminho_nativo.is_absolute():
        caminho = caminho_nativo
    else:
        parsed = urlparse(uri)
        if parsed.scheme and parsed.scheme.lower() != "file":
            raise ValueError("Somente arquivos locais ou URIs file:// são aceitos.")
        if parsed.scheme:
            caminho = Path(unquote(parsed.path))
            if parsed.netloc and parsed.netloc not in {"", "localhost"}:
                caminho = Path(f"//{parsed.netloc}{parsed.path}")
        else:
            caminho = obter_config_a1().atestados_storage_dir / caminho_nativo

    if not caminho.is_absolute():
        caminho = caminho.expanduser().resolve(strict=False)
    else:
        caminho = caminho.expanduser().resolve(strict=False)
    return caminho


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

    fundo = Image.new("L", escala_de_cinza.size, estatisticas.mean[0])
    diferenca = ImageChops.difference(escala_de_cinza, fundo)
    if diferenca.getbbox() is None:
        return True

    try:
        return estatisticas.stddev[0] < 2.0
    except (TypeError, ValueError):
        return True


def _executar_rapidocr(imagem: Image.Image) -> tuple[str, float | None]:
    if RapidOCR is None:
        raise RuntimeError(
            "Dependência OCR indisponível: instale rapidocr-onnxruntime."
        )

    motor = RapidOCR()
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


def _processar_imagem(caminho: Path) -> tuple[str, float | None]:
    try:
        with Image.open(caminho) as imagem:
            imagem.load()
            imagem = imagem.convert("RGB")
            return _ocr_imagem(imagem)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Arquivo de imagem corrompido ou ilegível: {exc}") from exc


def _processar_pdf(caminho: Path) -> tuple[str, float | None]:
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

        for pagina in documento:
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

    try:
        if mime_detectado == "application/pdf":
            texto, confianca = _processar_pdf(caminho)
        else:
            texto, confianca = _processar_imagem(caminho)
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
