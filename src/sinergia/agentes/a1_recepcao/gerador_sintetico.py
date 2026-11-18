"""Gerador determinístico de atestados sintéticos (`.png` e `.pdf`) com gabarito.

Implementa a parte de massa de testes da Task 2B.1 do plano de desenvolvimento
do Agente 1 (`docs/plano_desenvolvimento_a1.md`), produzindo amostras com marca
d'água explícita ``DADO SINTETICO (IA/TESTES) - SEM VALOR LEGAL`` e manifesto
`gabarito_atestados.json` para:

- Exercitar o caso canônico `105` do Anexo D.2 (`atestado_carla_105`) sem
  alterar o SHA-256 utilizado na fixture oficial.
- Testar variedade de conselhos (`CRM`, `CRO`, `CRP`), com e sem CID,
  declaração de comparecimento, ruído leve de digitalização e documento
  borrado/ilegível.
- Avaliar a métrica da Seção 10 (>= 80% de extração sem correção manual).
"""

from __future__ import annotations

import hashlib
import json
import random
import textwrap
from datetime import date
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pydantic import BaseModel, ConfigDict, Field

from sinergia.agentes.a1_recepcao.config import obter_config_a1
from sinergia.agentes.a1_recepcao.schemas import (
    AtestadoExtraido,
    PeriodoAfastamento,
    ProfissionalExtraido,
)

SHA256_CANONICO_CARLA_105_PNG = (
    "b5e6b4363c1e2377824c6af62171176c4f0b78b88f789f4425cbf8b7620ed47f"
)


class _EspecificacaoAmostra(BaseModel):
    """Definição interna de uma amostra sintética de atestado."""

    model_config = ConfigDict(extra="forbid")

    id_amostra: str
    titulo: str
    cenario: str
    clinica: str
    cabecalho: str
    corpo: str
    linha_cid: str | None = None
    local_data: str
    profissional_nome: str
    registro_texto: str
    modo_visual: str = "limpo"
    esperado: AtestadoExtraido
    campos_corrigidos_sugeridos: list[str] = Field(default_factory=list)


class GabaritoAtestadoSintetico(BaseModel):
    """Registro do manifesto `gabarito_atestados.json` para avaliação e testes."""

    model_config = ConfigDict(extra="forbid")

    id_amostra: str
    titulo: str
    arquivo_png: str
    arquivo_pdf: str
    sha256_png: str
    sha256_pdf: str
    esperado: AtestadoExtraido
    cenario: str
    campos_corrigidos_sugeridos: list[str] = Field(default_factory=list)


_ESPECIFICACOES_ATESTADOS: tuple[_EspecificacaoAmostra, ...] = (
    _EspecificacaoAmostra(
        id_amostra="atestado_carla_105",
        titulo="Atestado médico canônico — Caso 105 (Carla Souza, 3 dias)",
        cenario="canonico_anexo_d2",
        clinica="CLINICA MEDICA CAMPINAS",
        cabecalho="ATESTADO MEDICO",
        corpo=(
            "Atesto para os devidos fins que a paciente Carla Souza necessita de "
            "3 (tres) dias de afastamento de suas atividades laborais a partir de "
            "09/11/2026 (periodo de 09/11/2026 a 11/11/2026)."
        ),
        linha_cid="CID: J06.9",
        local_data="Campinas - SP, 09 de novembro de 2026.",
        profissional_nome="Dr. Marcos Lima",
        registro_texto="CRM/SP 123456",
        esperado=AtestadoExtraido(
            legibilidade="ok",
            nome_no_documento="Carla Souza",
            periodo=PeriodoAfastamento(
                inicio=date(2026, 11, 9), fim=date(2026, 11, 11)
            ),
            data_emissao=date(2026, 11, 9),
            profissional=ProfissionalExtraido(
                conselho="CRM", numero="123456", uf="SP"
            ),
            cid="J06.9",
            municipio="Campinas",
        ),
    ),
    _EspecificacaoAmostra(
        id_amostra="atestado_lombar_cid_m545",
        titulo="Atestado médico ortopédico (5 dias, CID M54.5)",
        cenario="afastamento_lombar_com_cid",
        clinica="CLINICA ORTOPEDICA CAMPINAS",
        cabecalho="ATESTADO MEDICO",
        corpo=(
            "Atesto para os devidos fins que a paciente Carla Souza necessita de "
            "5 (cinco) dias de afastamento de suas atividades laborais, no periodo "
            "de 16/11/2026 a 20/11/2026."
        ),
        linha_cid="CID: M54.5",
        local_data="Campinas - SP, 16 de novembro de 2026.",
        profissional_nome="Dra. Helena Costa",
        registro_texto="CRM/SP 654321",
        esperado=AtestadoExtraido(
            legibilidade="ok",
            nome_no_documento="Carla Souza",
            periodo=PeriodoAfastamento(
                inicio=date(2026, 11, 16), fim=date(2026, 11, 20)
            ),
            data_emissao=date(2026, 11, 16),
            profissional=ProfissionalExtraido(
                conselho="CRM", numero="654321", uf="SP"
            ),
            cid="M54.5",
            municipio="Campinas",
        ),
    ),
    _EspecificacaoAmostra(
        id_amostra="atestado_sem_cid_crm",
        titulo="Atestado médico sem CID (sigilo médico CFM 1.658/2002)",
        cenario="sem_cid_sigilo_medico",
        clinica="CENTRO MEDICO CAMPINAS",
        cabecalho="ATESTADO MEDICO",
        corpo=(
            "Atesto para os devidos fins que a paciente Carla Souza necessita de "
            "2 (dois) dias de afastamento de suas atividades laborais, no periodo "
            "de 23/11/2026 a 24/11/2026."
        ),
        linha_cid=None,
        local_data="Campinas - SP, 23 de novembro de 2026.",
        profissional_nome="Dr. Ricardo Alves",
        registro_texto="CRM/SP 334455",
        esperado=AtestadoExtraido(
            legibilidade="ok",
            nome_no_documento="Carla Souza",
            periodo=PeriodoAfastamento(
                inicio=date(2026, 11, 23), fim=date(2026, 11, 24)
            ),
            data_emissao=date(2026, 11, 23),
            profissional=ProfissionalExtraido(
                conselho="CRM", numero="334455", uf="SP"
            ),
            cid=None,
            municipio="Campinas",
        ),
    ),
    _EspecificacaoAmostra(
        id_amostra="atestado_odontologico_cro",
        titulo="Atestado odontológico (1 dia, CRO/SP)",
        cenario="odontologico_cro",
        clinica="CLINICA ODONTOLOGICA PAULISTA",
        cabecalho="ATESTADO ODONTOLOGICO",
        corpo=(
            "Atesto para os devidos fins que a paciente Carla Souza necessita de "
            "1 (um) dia de afastamento de suas atividades laborais no periodo de "
            "25/11/2026 a 25/11/2026."
        ),
        linha_cid="CID: Z00.0",
        local_data="São Paulo - SP, 25 de novembro de 2026.",
        profissional_nome="Dra. Beatriz Prado",
        registro_texto="CRO/SP 98765",
        esperado=AtestadoExtraido(
            legibilidade="ok",
            nome_no_documento="Carla Souza",
            periodo=PeriodoAfastamento(
                inicio=date(2026, 11, 25), fim=date(2026, 11, 25)
            ),
            data_emissao=date(2026, 11, 25),
            profissional=ProfissionalExtraido(
                conselho="CRO", numero="98765", uf="SP"
            ),
            cid="Z00.0",
            municipio="São Paulo",
        ),
    ),
    _EspecificacaoAmostra(
        id_amostra="atestado_psicologico_crp",
        titulo="Atestado psicológico (7 dias, CRP/SP, CID F41.1)",
        cenario="psicologico_crp",
        clinica="INSTITUTO DE SAUDE MENTAL CAMPINAS",
        cabecalho="ATESTADO PSICOLOGICO",
        corpo=(
            "Atesto que a paciente Carla Souza, em acompanhamento psicologico, "
            "necessita de 7 (sete) dias de afastamento de suas atividades laborais, "
            "no periodo de 01/12/2026 a 07/12/2026."
        ),
        linha_cid="CID: F41.1",
        local_data="Campinas - SP, 01 de dezembro de 2026.",
        profissional_nome="Dra. Juliana Mendes",
        registro_texto="CRP/SP 54321",
        esperado=AtestadoExtraido(
            legibilidade="ok",
            nome_no_documento="Carla Souza",
            periodo=PeriodoAfastamento(
                inicio=date(2026, 12, 1), fim=date(2026, 12, 7)
            ),
            data_emissao=date(2026, 12, 1),
            profissional=ProfissionalExtraido(
                conselho="CRP", numero="54321", uf="SP"
            ),
            cid="F41.1",
            municipio="Campinas",
        ),
    ),
    _EspecificacaoAmostra(
        id_amostra="declaracao_comparecimento_1dia",
        titulo="Declaração de comparecimento (1 dia, CRM/SP)",
        cenario="declaracao_comparecimento",
        clinica="HOSPITAL MUNICIPAL DE SAO PAULO",
        cabecalho="DECLARACAO DE COMPARECIMENTO",
        corpo=(
            "Atesto que a paciente Carla Souza, esteve em atendimento medico e "
            "necessita de 1 (um) dia de dispensa no periodo de 10/12/2026 a 10/12/2026."
        ),
        linha_cid="CID: Z00.0",
        local_data="São Paulo - SP, 10 de dezembro de 2026.",
        profissional_nome="Dr. Fernando Rocha",
        registro_texto="CRM/SP 778899",
        esperado=AtestadoExtraido(
            legibilidade="ok",
            nome_no_documento="Carla Souza",
            periodo=PeriodoAfastamento(
                inicio=date(2026, 12, 10), fim=date(2026, 12, 10)
            ),
            data_emissao=date(2026, 12, 10),
            profissional=ProfissionalExtraido(
                conselho="CRM", numero="778899", uf="SP"
            ),
            cid="Z00.0",
            municipio="São Paulo",
        ),
    ),
    _EspecificacaoAmostra(
        id_amostra="atestado_ruido_leve_manual",
        titulo="Atestado com ruído leve de digitalização (revisão manual)",
        cenario="ruido_leve_revisao_manual",
        clinica="CLINICA MEDICA CAMPINAS",
        cabecalho="ATESTADO MEDICO",
        corpo=(
            "Atesto para os devidos fins que a paciente Carla Souza necessita de "
            "4 (quatro) dias de afastamento de suas atividades laborais, no periodo "
            "de 14/12/2026 a 17/12/2026."
        ),
        linha_cid=None,
        local_data="Campinas - SP, 14 de dezembro de 2026.",
        profissional_nome="Dr. Andre Nogueira",
        registro_texto="CRM/SP 112233",
        modo_visual="ruido_leve",
        esperado=AtestadoExtraido(
            legibilidade="ok",
            nome_no_documento="Carla Souza",
            periodo=PeriodoAfastamento(
                inicio=date(2026, 12, 14), fim=date(2026, 12, 17)
            ),
            data_emissao=date(2026, 12, 14),
            profissional=ProfissionalExtraido(
                conselho="CRM", numero="112233", uf="SP"
            ),
            cid=None,
            municipio="Campinas",
        ),
        campos_corrigidos_sugeridos=["periodo.fim"],
    ),
    _EspecificacaoAmostra(
        id_amostra="atestado_inss_nr7_30dias",
        titulo="Atestado médico longo (30 dias — marco INSS e exame retorno NR-07)",
        cenario="afastamento_longo_inss_nr7",
        clinica="CLINICA ORTOPEDICA CAMPINAS",
        cabecalho="ATESTADO MEDICO",
        corpo=(
            "Atesto que a paciente Carla Souza, necessita de 30 (trinta) dias "
            "de afastamento de suas atividades laborais, no periodo de "
            "01/11/2026 a 30/11/2026."
        ),
        linha_cid="CID: M54.5",
        local_data="Campinas - SP, 01 de novembro de 2026.",
        profissional_nome="Dr. Eduardo Tavares",
        registro_texto="CRM/SP 445566",
        esperado=AtestadoExtraido(
            legibilidade="ok",
            nome_no_documento="Carla Souza",
            periodo=PeriodoAfastamento(
                inicio=date(2026, 11, 1), fim=date(2026, 11, 30)
            ),
            data_emissao=date(2026, 11, 1),
            profissional=ProfissionalExtraido(
                conselho="CRM", numero="445566", uf="SP"
            ),
            cid="M54.5",
            municipio="Campinas",
        ),
    ),
    _EspecificacaoAmostra(
        id_amostra="atestado_borrado_ilegivel",
        titulo="Imagem borrada / foto tremida (ilegível)",
        cenario="ilegivel_borrado",
        clinica="CLINICA MEDICA",
        cabecalho="ATESTADO MEDICO",
        corpo="Documento com foto tremida e baixo contraste para testar rejeicao de leitura.",
        linha_cid=None,
        local_data="Campinas - SP",
        profissional_nome="Dr. Indistinto",
        registro_texto="CRM/SP 000000",
        modo_visual="borrado",
        esperado=AtestadoExtraido(legibilidade="ilegivel"),
    ),
    _EspecificacaoAmostra(
        id_amostra="atestado_escuro_subexposto",
        titulo="Foto escura / subexposta de baixo contraste (ilegível)",
        cenario="ilegivel_subexposto",
        clinica="CLINICA MEDICA",
        cabecalho="ATESTADO MEDICO",
        corpo="Foto capturada sem iluminacao adequada tornando o texto ilegivel.",
        linha_cid=None,
        local_data="Campinas - SP",
        profissional_nome="Dr. Sombra",
        registro_texto="CRM/SP 000000",
        modo_visual="subexposto",
        esperado=AtestadoExtraido(legibilidade="ilegivel"),
    ),
    _EspecificacaoAmostra(
        id_amostra="atestado_cortado_incompleto",
        titulo="Foto cortada sem datas e sem CRM (ilegível / incompleto)",
        cenario="ilegivel_cortado",
        clinica="CLINICA MEDICA CAMPINAS",
        cabecalho="ATESTADO MEDICO",
        corpo="Atesto que a paciente Carla Souza, esteve em avaliacao clinica nesta data.",
        linha_cid=None,
        local_data="",
        profissional_nome="",
        registro_texto="",
        modo_visual="cortado",
        esperado=AtestadoExtraido(legibilidade="ilegivel"),
    ),
    _EspecificacaoAmostra(
        id_amostra="atestado_manchado_rasurado",
        titulo="Documento manchado/rasurado sobre datas e CRM (ilegível)",
        cenario="ilegivel_rasurado",
        clinica="CLINICA MEDICA CAMPINAS",
        cabecalho="ATESTADO MEDICO",
        corpo="Atesto que a paciente Carla Souza,",
        linha_cid=None,
        local_data="",
        profissional_nome="",
        registro_texto="",
        modo_visual="rasurado",
        esperado=AtestadoExtraido(legibilidade="ilegivel"),
    ),
)


def _sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arq:
        for bloco in iter(lambda: arq.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _formatar_caminho_portavel(caminho: Path) -> str:
    try:
        return caminho.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(caminho)


def _carregar_fonte(tamanho: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidatos = (
        Path("C:/Windows/Fonts/verdana.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for fonte in candidatos:
        if fonte.is_file():
            return ImageFont.truetype(str(fonte), tamanho)
    return ImageFont.load_default(size=tamanho)


def _renderizar_imagem_atestado(spec: _EspecificacaoAmostra) -> Image.Image:
    if spec.modo_visual == "borrado":
        largura, altura = 960, 620
        imagem = Image.new("RGB", (largura, altura), (212, 214, 218))
        draw = ImageDraw.Draw(imagem)
        fonte_media = _carregar_fonte(22)
        draw.text((80, 120), spec.clinica, fill=(195, 198, 202), font=fonte_media)
        draw.text((80, 180), spec.cabecalho, fill=(190, 194, 198), font=fonte_media)
        draw.text((80, 240), spec.corpo, fill=(192, 196, 200), font=fonte_media)
        draw.text((80, 320), spec.registro_texto, fill=(194, 196, 200), font=fonte_media)
        imagem = imagem.filter(ImageFilter.GaussianBlur(radius=11))
        rng = random.Random(105)
        pixels = imagem.load()
        for _ in range(1800):
            px = rng.randint(0, largura - 1)
            py = rng.randint(0, altura - 1)
            tom = rng.randint(175, 235)
            pixels[px, py] = (tom, tom, tom)
        return imagem

    if spec.modo_visual == "subexposto":
        largura, altura = 960, 620
        imagem = Image.new("RGB", (largura, altura), (18, 20, 24))
        draw = ImageDraw.Draw(imagem)
        fonte_media = _carregar_fonte(22)
        draw.text((80, 120), spec.clinica, fill=(26, 28, 32), font=fonte_media)
        draw.text((80, 180), spec.cabecalho, fill=(28, 30, 34), font=fonte_media)
        draw.text((80, 240), spec.corpo, fill=(27, 29, 33), font=fonte_media)
        imagem = imagem.filter(ImageFilter.GaussianBlur(radius=2.5))
        rng = random.Random(404)
        pixels = imagem.load()
        for _ in range(2500):
            px = rng.randint(0, largura - 1)
            py = rng.randint(0, altura - 1)
            tom = rng.randint(8, 36)
            pixels[px, py] = (tom, tom, tom)
        return imagem

    if spec.modo_visual == "cortado":
        largura, altura = 1440, 430
        imagem = Image.new("RGB", (largura, altura), "white")
        draw = ImageDraw.Draw(imagem)
        fonte_clinica = _carregar_fonte(34)
        fonte_titulo = _carregar_fonte(42)
        fonte_corpo = _carregar_fonte(32)
        fonte_marca = _carregar_fonte(20)
        draw.rectangle([(36, 36), (largura - 36, altura)], outline=(44, 62, 80), width=4)
        draw.rectangle(
            [(36, 36), (largura - 36, 130)],
            fill=(238, 243, 249),
            outline=(44, 62, 80),
            width=3,
        )
        draw.text((70, 62), spec.clinica, fill=(24, 48, 72), font=fonte_clinica)
        draw.text(
            (largura - 560, 68),
            "DADO SINTETICO (IA/TESTES) - SEM VALOR LEGAL",
            fill=(130, 145, 165),
            font=fonte_marca,
        )
        draw.text((70, 174), spec.cabecalho, fill=(15, 23, 42), font=fonte_titulo)
        draw.line([(70, 236), (largura - 70, 236)], fill=(180, 192, 208), width=3)
        y_cursor = 280
        for linha in textwrap.wrap(spec.corpo, width=68):
            draw.text((70, y_cursor), linha, fill=(20, 20, 20), font=fonte_corpo)
            y_cursor += 50
        return imagem

    largura, altura = 1440, 960
    imagem = Image.new("RGB", (largura, altura), "white")
    draw = ImageDraw.Draw(imagem)

    fonte_clinica = _carregar_fonte(34)
    fonte_titulo = _carregar_fonte(42)
    fonte_corpo = _carregar_fonte(32)
    fonte_rodape = _carregar_fonte(31)
    fonte_marca = _carregar_fonte(20)

    # Moldura institucional
    draw.rectangle([(36, 36), (largura - 36, altura - 36)], outline=(44, 62, 80), width=4)
    draw.rectangle(
        [(36, 36), (largura - 36, 130)],
        fill=(238, 243, 249),
        outline=(44, 62, 80),
        width=3,
    )

    draw.text((70, 62), spec.clinica, fill=(24, 48, 72), font=fonte_clinica)
    draw.text(
        (largura - 560, 68),
        "DADO SINTETICO (IA/TESTES) - SEM VALOR LEGAL",
        fill=(130, 145, 165),
        font=fonte_marca,
    )

    draw.text((70, 174), spec.cabecalho, fill=(15, 23, 42), font=fonte_titulo)
    draw.line([(70, 236), (largura - 70, 236)], fill=(180, 192, 208), width=3)

    y_cursor = 280
    for linha in textwrap.wrap(spec.corpo, width=68):
        draw.text((70, y_cursor), linha, fill=(20, 20, 20), font=fonte_corpo)
        y_cursor += 50

    if spec.modo_visual == "rasurado":
        # Manchas/tarjas escuras obliterando completamente período, emissão e CRM
        draw.rectangle([(65, 345), (1240, 560)], fill=(26, 28, 32))
        draw.ellipse([(140, 410), (760, 650)], fill=(38, 40, 46))
        draw.rectangle([(65, 730), (660, 865)], fill=(26, 28, 32))
        draw.ellipse([(420, 700), (880, 880)], fill=(32, 34, 40))
        draw.text(
            (largura - 470, 865),
            "SINERGIA • AMBIENTE DE TESTES",
            fill=(165, 175, 188),
            font=fonte_marca,
        )
        return imagem

    if spec.linha_cid:
        y_cursor += 20
        draw.text((70, y_cursor), spec.linha_cid, fill=(20, 20, 20), font=fonte_corpo)
        y_cursor += 50

    y_cursor += 34
    draw.text((70, y_cursor), spec.local_data, fill=(35, 35, 35), font=fonte_rodape)

    draw.line([(70, 760), (600, 760)], fill=(90, 95, 105), width=2)
    draw.text((70, 778), spec.profissional_nome, fill=(20, 20, 20), font=fonte_rodape)
    draw.text((70, 822), spec.registro_texto, fill=(20, 20, 20), font=fonte_rodape)

    draw.text(
        (largura - 470, 865),
        "SINERGIA • AMBIENTE DE TESTES",
        fill=(165, 175, 188),
        font=fonte_marca,
    )

    if spec.modo_visual == "ruido_leve":
        rng = random.Random(2026)
        pixels = imagem.load()
        for _ in range(1800):
            px = rng.randint(42, largura - 42)
            py = rng.randint(42, altura - 42)
            tom = rng.randint(155, 225)
            pixels[px, py] = (tom, tom, tom)
        imagem = imagem.rotate(0.45, fillcolor="white")

    return imagem


def _salvar_pdf_da_imagem(caminho_png: Path, caminho_pdf: Path) -> None:
    with Image.open(caminho_png) as img:
        largura, altura = img.size
    doc = pymupdf.open()
    pagina = doc.new_page(width=largura, height=altura)
    pagina.insert_image(pagina.rect, filename=str(caminho_png))
    doc.save(caminho_pdf)
    doc.close()


def garantir_atestados_sinteticos(
    diretorio: Path | None = None,
    *,
    sobrescrever: bool = False,
) -> list[GabaritoAtestadoSintetico]:
    """Gera os 12 pares de atestados sintéticos (`.png` + `.pdf`) e `gabarito_atestados.json`.

    Quando `sobrescrever=False` (padrão), arquivos já existentes (em especial
    `atestado_carla_105.png` e `.pdf`, cujo SHA-256 integra a fixture oficial)
    são preservados intactos.
    """
    destino = diretorio or obter_config_a1().atestados_storage_dir
    destino.mkdir(parents=True, exist_ok=True)

    gabarito: list[GabaritoAtestadoSintetico] = []
    for spec in _ESPECIFICACOES_ATESTADOS:
        caminho_png = destino / f"{spec.id_amostra}.png"
        caminho_pdf = destino / f"{spec.id_amostra}.pdf"

        preservar_canonico = (
            spec.id_amostra == "atestado_carla_105"
            and caminho_png.is_file()
            and _sha256_arquivo(caminho_png) == SHA256_CANONICO_CARLA_105_PNG
        )

        if not preservar_canonico and (sobrescrever or not caminho_png.is_file()):
            imagem = _renderizar_imagem_atestado(spec)
            imagem.save(caminho_png, format="PNG")

        if not preservar_canonico and (sobrescrever or not caminho_pdf.is_file()):
            _salvar_pdf_da_imagem(caminho_png, caminho_pdf)

        gabarito.append(
            GabaritoAtestadoSintetico(
                id_amostra=spec.id_amostra,
                titulo=spec.titulo,
                arquivo_png=_formatar_caminho_portavel(caminho_png),
                arquivo_pdf=_formatar_caminho_portavel(caminho_pdf),
                sha256_png=_sha256_arquivo(caminho_png),
                sha256_pdf=_sha256_arquivo(caminho_pdf),
                esperado=spec.esperado,
                cenario=spec.cenario,
                campos_corrigidos_sugeridos=list(spec.campos_corrigidos_sugeridos),
            )
        )

    manifesto_path = destino / "gabarito_atestados.json"
    manifesto_dados = {
        "versao": "2.2",
        "total_amostras": len(gabarito),
        "amostras": [item.model_dump(mode="json") for item in gabarito],
    }
    manifesto_path.write_text(
        json.dumps(manifesto_dados, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return gabarito
