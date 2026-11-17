"""Testes da Task 1A.1 - leitura privada e OCR de atestados."""

from __future__ import annotations

import hashlib

import pymupdf
import pytest
from PIL import Image

from sinergia.agentes.a1_recepcao import ocr, processar_arquivo_atestado


def test_processa_png_calcula_hash_e_confere_sha256(tmp_path, monkeypatch) -> None:
    imagem = tmp_path / "atestado.png"
    Image.new("RGB", (300, 200), "white").save(imagem)

    monkeypatch.setattr(
        ocr,
        "_executar_rapidocr",
        lambda _imagem: ("ATESTADO MÉDICO\nCarla Souza", 0.95),
    )
    monkeypatch.setattr(
        ocr,
        "_imagem_esta_em_branco",
        lambda _imagem: False,
    )

    sha256 = hashlib.sha256(imagem.read_bytes()).hexdigest()
    resultado = processar_arquivo_atestado(
        str(imagem), mime="image/png", sha256_esperado=sha256.upper()
    )

    assert resultado.sha256 == sha256
    assert resultado.mime == "image/png"
    assert "Carla Souza" in resultado.texto_extraido
    assert resultado.confianca_media == 0.95
    assert resultado.legibilidade == "ok"


def test_imagem_em_branco_retorna_ilegivel(tmp_path) -> None:
    imagem = tmp_path / "branco.png"
    Image.new("RGB", (300, 200), "white").save(imagem)

    resultado = processar_arquivo_atestado(str(imagem), mime="image/png")

    assert resultado.texto_extraido == ""
    assert resultado.confianca_media == 0.0
    assert resultado.legibilidade == "ilegivel"


def test_imagem_corrompida_retorna_ilegivel(tmp_path) -> None:
    imagem = tmp_path / "corrompida.jpg"
    imagem.write_bytes(b"conteudo nao eh uma imagem jpg")

    resultado = processar_arquivo_atestado(str(imagem), mime="image/jpeg")

    assert resultado.sha256 == hashlib.sha256(imagem.read_bytes()).hexdigest()
    assert resultado.texto_extraido == ""
    assert resultado.legibilidade == "ilegivel"


def test_hash_divergente_deve_ser_rejeitado(tmp_path) -> None:
    imagem = tmp_path / "atestado.png"
    Image.new("RGB", (300, 200), "white").save(imagem)

    with pytest.raises(ValueError, match="não corresponde"):
        processar_arquivo_atestado(
            str(imagem),
            mime="image/png",
            sha256_esperado="0" * 64,
        )


def test_arquivo_inexistente_deve_ser_rejeitado(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        processar_arquivo_atestado(str(tmp_path / "ausente.pdf"))


def test_pdf_em_branco_retorna_ilegivel(tmp_path) -> None:
    pdf = tmp_path / "branco.pdf"
    documento = pymupdf.open()
    pagina = documento.new_page(width=300, height=200)
    pagina.draw_rect(pagina.rect, color=(1, 1, 1), fill=(1, 1, 1))
    documento.save(pdf)
    documento.close()

    resultado = processar_arquivo_atestado(str(pdf), mime="application/pdf")

    assert resultado.legibilidade == "ilegivel"
    assert resultado.texto_extraido == ""
