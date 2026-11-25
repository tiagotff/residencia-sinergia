"""Componentes visuais e CSS monocromático claro para o painel de integração A0 <-> A1."""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import Any, Literal

import streamlit as st

PALETA_PAINEL: dict[str, str] = {
    "bg_principal": "#FFFFFF",
    "bg_secundario": "#FAFAFA",
    "bg_superficie": "#F4F4F5",
    "borda": "#E4E4E7",
    "texto_principal": "#09090B",
    "texto_secundario": "#71717A",
    "primario": "#18181B",
    "ok_bg": "#ECFDF5",
    "ok_fg": "#065F46",
    "ok_borda": "#A7F3D0",
    "aviso_bg": "#FFFBEB",
    "aviso_fg": "#92400E",
    "aviso_borda": "#FDE68A",
    "erro_bg": "#FEF2F2",
    "erro_fg": "#991B1B",
    "erro_borda": "#FECACA",
    "api_bg": "#EFF6FF",
    "api_fg": "#1D4ED8",
    "api_borda": "#BFDBFE",
}

VarianteBadge = Literal["ok", "aviso", "erro", "neutro", "api"]


def obter_css_painel() -> str:
    """Retorna CSS leve mantendo a sidebar visível e reforçando a estética shadcn light."""
    p = PALETA_PAINEL
    return f"""
<style>
  :root {{
    color-scheme: light !important;
  }}

  .sinergia-painel-titulo {{
    font-size: 1.4rem;
    font-weight: 700;
    color: {p["texto_principal"]};
    letter-spacing: -0.02em;
    margin: 0;
  }}

  .sinergia-painel-sub {{
    font-size: 0.88rem;
    color: {p["texto_secundario"]};
    margin: 0.2rem 0 0.8rem 0;
  }}

  .sinergia-kpi {{
    background: {p["bg_secundario"]};
    border: 1px solid {p["borda"]};
    border-radius: 0.5rem;
    padding: 0.7rem 0.9rem;
  }}

  .sinergia-kpi-rotulo {{
    font-size: 0.75rem;
    font-weight: 500;
    color: {p["texto_secundario"]};
  }}

  .sinergia-kpi-valor {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {p["texto_principal"]};
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.15rem;
  }}

  .sinergia-kpi-sub {{
    font-size: 0.73rem;
    color: {p["texto_secundario"]};
    margin-top: 0.15rem;
  }}

  .sinergia-badge {{
    display: inline-block;
    padding: 0.18rem 0.58rem;
    border-radius: 9999px;
    font-size: 0.74rem;
    font-weight: 600;
    line-height: 1.3;
    border: 1px solid transparent;
  }}

  .sinergia-badge--ok {{
    background: {p["ok_bg"]};
    color: {p["ok_fg"]};
    border-color: {p["ok_borda"]};
  }}

  .sinergia-badge--aviso {{
    background: {p["aviso_bg"]};
    color: {p["aviso_fg"]};
    border-color: {p["aviso_borda"]};
  }}

  .sinergia-badge--erro {{
    background: {p["erro_bg"]};
    color: {p["erro_fg"]};
    border-color: {p["erro_borda"]};
  }}

  .sinergia-badge--neutro {{
    background: {p["bg_superficie"]};
    color: {p["primario"]};
    border-color: {p["borda"]};
  }}

  .sinergia-badge--api {{
    background: {p["api_bg"]};
    color: {p["api_fg"]};
    border-color: {p["api_borda"]};
  }}

  .sinergia-cid-box {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.45rem 0.65rem;
    padding: 0.55rem 0.78rem;
    border-radius: 0.5rem;
    border: 1px solid {p["borda"]};
    background: {p["bg_secundario"]};
    margin: 0.45rem 0 0.65rem 0;
    font-size: 0.82rem;
    color: {p["texto_principal"]};
  }}

  .sinergia-cid-box--ok {{
    background: {p["ok_bg"]};
    border-color: {p["ok_borda"]};
  }}

  .sinergia-cid-box--api {{
    background: {p["api_bg"]};
    border-color: {p["api_borda"]};
  }}

  .sinergia-cid-box--erro {{
    background: {p["erro_bg"]};
    border-color: {p["erro_borda"]};
  }}

  .sinergia-cid-box--aviso {{
    background: {p["aviso_bg"]};
    border-color: {p["aviso_borda"]};
  }}

  .sinergia-stepper {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin: 0.5rem 0 0.8rem 0;
  }}

  .sinergia-step {{
    padding: 0.28rem 0.7rem;
    border-radius: 0.45rem;
    font-size: 0.76rem;
    font-weight: 600;
    background: {p["bg_superficie"]};
    color: {p["texto_secundario"]};
    border: 1px solid {p["borda"]};
  }}

  .sinergia-step--ativo {{
    background: {p["primario"]};
    color: {p["bg_principal"]};
    border-color: {p["primario"]};
  }}

  .sinergia-step--hitl {{
    background: {p["aviso_bg"]};
    color: {p["aviso_fg"]};
    border-color: {p["aviso_borda"]};
  }}

  .sinergia-status-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: center;
    background: {p["bg_secundario"]};
    border: 1px solid {p["borda"]};
    border-radius: 0.55rem;
    padding: 0.45rem 0.75rem;
    margin-bottom: 0.75rem;
  }}

  .sinergia-status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.76rem;
    color: {p["texto_secundario"]};
  }}

  .sinergia-status-pill strong {{
    color: {p["texto_principal"]};
    font-weight: 600;
  }}

  .sinergia-campo-card {{
    background: {p["bg_principal"]};
    border: 1px solid {p["borda"]};
    border-radius: 0.5rem;
    padding: 0.55rem 0.75rem;
    margin-bottom: 0.45rem;
  }}

  .sinergia-campo-card--divergente {{
    background: {p["aviso_bg"]};
    border-color: {p["aviso_borda"]};
  }}

  .sinergia-campo-card--baixa_confianca {{
    background: {p["aviso_bg"]};
    border-style: dashed;
    border-color: {p["aviso_borda"]};
  }}

  .sinergia-campo-card--nao_identificado {{
    background: {p["bg_secundario"]};
  }}

  .sinergia-campo-topo {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.2rem;
  }}

  .sinergia-campo-rotulo {{
    font-size: 0.73rem;
    font-weight: 600;
    color: {p["texto_secundario"]};
    text-transform: uppercase;
    letter-spacing: 0.02em;
  }}

  .sinergia-campo-valor {{
    font-size: 0.94rem;
    font-weight: 600;
    color: {p["texto_principal"]};
    word-break: break-word;
  }}

  .sinergia-campo-antes {{
    font-size: 0.74rem;
    color: {p["aviso_fg"]};
    margin-top: 0.2rem;
  }}

  .sinergia-campo-doenca {{
    margin-top: 0.35rem;
    padding-top: 0.3rem;
    border-top: 1px dashed {p["borda"]};
    font-size: 0.78rem;
    line-height: 1.35;
    color: {p["texto_secundario"]};
  }}

  .sinergia-campo-doenca strong {{
    color: {p["texto_principal"]};
  }}

  .sinergia-diff-box {{
    background: {p["bg_secundario"]};
    border: 1px solid {p["borda"]};
    border-radius: 0.5rem;
    padding: 0.55rem 0.75rem;
    margin: 0.5rem 0;
  }}

  .sinergia-diff-row {{
    font-size: 0.8rem;
    color: {p["texto_principal"]};
    padding: 0.12rem 0;
  }}

  .sinergia-diff-antes {{
    color: {p["texto_secundario"]};
    text-decoration: line-through;
  }}

  .sinergia-diff-depois {{
    font-weight: 600;
    color: {p["ok_fg"]};
  }}
</style>
"""


def aplicar_estilo_painel() -> None:
    """Injeta o CSS monocromático do painel na página Streamlit."""
    st.markdown(obter_css_painel(), unsafe_allow_html=True)


def renderizar_badge_status(texto: str, variante: VarianteBadge = "neutro") -> str:
    """Gera HTML seguro de badge de status."""
    return (
        f'<span class="sinergia-badge sinergia-badge--{variante}">'
        f"{html.escape(texto)}</span>"
    )


def _descrever_origem_cid(diag: dict[str, Any]) -> tuple[str, VarianteBadge]:
    """Mapeia `origem` do diagnóstico CID para rótulo curto e variante visual."""
    origem = str(diag.get("origem") or "")
    if origem == "BANCO_LOCAL":
        return "🗄️ Banco Local (afastamento.cid)", "ok"
    if origem == "CONTAINER_WHO_LOCAL_CACHE_ASIDE":
        versao = str(diag.get("versao_cid") or "CID")
        return f"🐳 Container Local WHO ({versao} · cache-aside)", "api"
    if origem == "API_WHO_EXTERNA_CACHE_ASIDE":
        versao = str(diag.get("versao_cid") or "CID")
        return f"🌐 API Externa OMS ({versao} · cache-aside)", "api"
    if origem == "API_WHO_NAO_ENCONTRADO":
        return "❌ Não encontrado (Banco + API OMS)", "erro"
    if origem == "SINTAXE_INVALIDA":
        return "❌ Sintaxe CID inválida", "erro"
    if origem == "API_WHO_INDISPONIVEL":
        return "⚠️ API OMS indisponível (fallback gracioso)", "aviso"
    if origem == "NAO_ENCONTRADO_SEM_API":
        return "ℹ️ Ausente no banco (API OMS não configurada)", "neutro"
    return f"CID ({origem or 'desconhecido'})", "neutro"


def renderizar_badge_origem_cid(diag: dict[str, Any] | None) -> str:
    """Renderiza badge compacto indicando se o CID veio do Banco Local ou da API Externa OMS."""
    if not diag:
        return ""
    rotulo, variante = _descrever_origem_cid(diag)
    codigo = str(diag.get("codigo_cid") or "").strip()
    texto = f"{rotulo} · {codigo}" if codigo else rotulo
    return renderizar_badge_status(texto, variante)


def renderizar_card_origem_cid(diag: dict[str, Any] | None) -> str:
    """Renderiza faixa informativa destacando origem do CID (Banco Local vs API Externa OMS)."""
    if not diag:
        return ""
    rotulo, variante = _descrever_origem_cid(diag)
    badge = renderizar_badge_status(rotulo, variante)
    codigo = html.escape(str(diag.get("codigo_cid") or "—"))
    nome_doenca = str(diag.get("nome_doenca") or diag.get("descricao") or "").strip()
    capitulo = str(diag.get("capitulo") or "").strip()
    detalhe = str(diag.get("detalhe") or "").strip()
    origem = str(diag.get("origem") or "")
    status_validacao = str(diag.get("status_validacao_cid") or "").strip()
    bloqueia_aprov = bool(diag.get("bloqueia_aprovacao_automatica"))
    origem_preench = str(diag.get("origem_preenchimento") or "").strip()
    validado_em = str(diag.get("validado_em") or "").strip()
    versao_tabela = str(diag.get("versao_tabela_cid") or "").strip()

    partes = [badge, f"<strong>CID {codigo}</strong>"]
    if status_validacao:
        var_status: VarianteBadge = "ok" if diag.get("valido") is True else "erro"
        partes.append(renderizar_badge_status(status_validacao, var_status))
    if nome_doenca:
        partes.append(
            f"<span>— <strong>Nome da doença:</strong> {html.escape(nome_doenca)}</span>"
        )
    descricao = str(diag.get("descricao") or "").strip()
    if descricao and descricao != nome_doenca:
        partes.append(
            f'<span style="color:#52525B;">· <strong>Descrição:</strong> {html.escape(descricao)}</span>'
        )
    if capitulo:
        partes.append(
            f'<span style="color:#71717A;">(Cap. {html.escape(capitulo)})</span>'
        )
    if origem in {"CONTAINER_WHO_LOCAL_CACHE_ASIDE", "API_WHO_EXTERNA_CACHE_ASIDE"}:
        partes.append(
            '<span style="color:#1D4ED8;font-weight:600;">· Salvo em afastamento.cid</span>'
        )
    if bloqueia_aprov:
        partes.append(
            '<span style="color:#991B1B;font-weight:600;">· Aprovação automática bloqueada (revisão humana)</span>'
        )
    if not nome_doenca and detalhe:
        partes.append(f'<span style="color:#71717A;">({html.escape(detalhe)})</span>')

    auditoria_itens: list[str] = []
    if origem_preench:
        rotulo_origem = "Manual" if origem_preench == "MANUAL" else "A1 Recepção"
        auditoria_itens.append(f"Origem: {html.escape(rotulo_origem)}")
    if validado_em:
        auditoria_itens.append(f"Validado em: {html.escape(validado_em)}")
    if versao_tabela:
        auditoria_itens.append(f"Tabela CID: {html.escape(versao_tabela)}")
    if auditoria_itens:
        partes.append(
            '<div style="width:100%;font-size:0.74rem;color:#52525B;margin-top:0.15rem;">'
            f'<strong>Auditoria:</strong> {" · ".join(auditoria_itens)}'
            "</div>"
        )

    classe_box = f"sinergia-cid-box sinergia-cid-box--{variante}"
    return f'<div class="{classe_box}">{" ".join(partes)}</div>'


def renderizar_card_kpi(rotulo: str, valor: str, subtitulo: str = "") -> str:
    """Gera HTML seguro de cartão de métrica/KPI."""
    sub_html = (
        f'<div class="sinergia-kpi-sub">{html.escape(subtitulo)}</div>'
        if subtitulo
        else ""
    )
    return (
        '<div class="sinergia-kpi">'
        f'<div class="sinergia-kpi-rotulo">{html.escape(rotulo)}</div>'
        f'<div class="sinergia-kpi-valor">{html.escape(valor)}</div>'
        f"{sub_html}"
        "</div>"
    )


def renderizar_stepper_estados(
    estado_atual: str,
    aguardando_hitl: bool,
    baixa_registrada: bool = False,
) -> str:
    """Renderiza o stepper visual do ciclo de vida A0 <-> A1."""
    etapas = [
        ("NOVO", "1. NOVO"),
        ("RECEBIDO", "2. RECEBIDO (HITL)" if aguardando_hitl else "2. RECEBIDO"),
        ("VALIDADO", "3. VALIDADO"),
        ("BAIXA", "4. BAIXA FÍSICA"),
    ]
    itens_html: list[str] = []
    for chave, rotulo in etapas:
        classe = "sinergia-step"
        if chave == "BAIXA" and baixa_registrada:
            classe += " sinergia-step--ativo"
        elif chave == estado_atual:
            classe += " sinergia-step--hitl" if (chave == "RECEBIDO" and aguardando_hitl) else " sinergia-step--ativo"
        itens_html.append(f'<span class="{classe}">{html.escape(rotulo)}</span>')
    return f'<div class="sinergia-stepper">{"".join(itens_html)}</div>'


def renderizar_barra_status_topo(
    itens: Sequence[tuple[str, str, VarianteBadge]],
) -> str:
    """Renderiza pílulas de status ao vivo no cabeçalho do painel."""
    partes: list[str] = []
    for rotulo, valor, variante in itens:
        badge = renderizar_badge_status(valor, variante)
        partes.append(
            f'<div class="sinergia-status-pill"><strong>{html.escape(rotulo)}:</strong> {badge}</div>'
        )
    return f'<div class="sinergia-status-bar">{"".join(partes)}</div>'


def renderizar_detalhes_doenca_cid(diag_cid: dict[str, Any] | None) -> str:
    """Renderiza nome e descrição da doença associada ao CID para exibição no card."""
    if not diag_cid:
        return ""
    nome_doenca = str(diag_cid.get("nome_doenca") or "").strip()
    descricao = str(diag_cid.get("descricao") or "").strip()
    if not nome_doenca and descricao:
        nome_doenca = descricao
    if not nome_doenca and not descricao:
        return ""

    linhas: list[str] = []
    if nome_doenca:
        linhas.append(
            f"<div><strong>Nome da doença:</strong> {html.escape(nome_doenca)}</div>"
        )
    if descricao and descricao != nome_doenca:
        linhas.append(
            f"<div><strong>Descrição:</strong> {html.escape(descricao)}</div>"
        )
    return f'<div class="sinergia-campo-doenca">{"".join(linhas)}</div>'


def renderizar_card_campo_revisao(
    item: dict[str, Any],
    diag_cid: dict[str, Any] | None = None,
) -> str:
    """Renderiza um cartão visual do Espelho de Leitura da IA (`ItemRevisaoCampo`)."""
    status = str(item.get("status", "lido_ok"))
    rotulo = str(item.get("rotulo") or item.get("campo") or "")
    valor_exib = str(item.get("valor_exibicao") or "Não identificado")
    confianca = item.get("confianca")
    antes = item.get("valor_informado_antes")

    classe_card = "sinergia-campo-card"
    if status in {"divergente", "baixa_confianca", "nao_identificado"}:
        classe_card += f" sinergia-campo-card--{status}"

    if status == "divergente":
        badge_html = renderizar_badge_status("⚠ Divergência", "aviso")
    elif status == "baixa_confianca":
        badge_html = renderizar_badge_status("? Baixa confiança", "aviso")
    elif status == "nao_identificado":
        badge_html = renderizar_badge_status("— Não identificado", "neutro")
    else:
        pct = (
            f" ({round(float(confianca) * 100)}%)"
            if isinstance(confianca, (int, float)) and float(confianca) > 0
            else ""
        )
        badge_html = renderizar_badge_status(f"✓ Lido pela IA{pct}", "ok")

    doenca_html = ""
    if item.get("campo") == "cid":
        diag_item = diag_cid if diag_cid is not None else item.get("diagnostico_cid")
        if isinstance(diag_item, dict):
            doenca_html = renderizar_detalhes_doenca_cid(diag_item)

    antes_html = ""
    if status == "divergente" and antes is not None:
        antes_html = (
            f'<div class="sinergia-campo-antes">'
            f"Informado antes: <code>{html.escape(str(antes))}</code>"
            "</div>"
        )

    return (
        f'<div class="{classe_card}">'
        '<div class="sinergia-campo-topo">'
        f'<span class="sinergia-campo-rotulo">{html.escape(rotulo)}</span>'
        f"{badge_html}"
        "</div>"
        f'<div class="sinergia-campo-valor">{html.escape(valor_exib)}</div>'
        f"{doenca_html}"
        f"{antes_html}"
        "</div>"
    )


def renderizar_alerta_diff(diffs: Sequence[dict[str, str]]) -> str:
    """Renderiza o resumo visual Antes (IA) -> Depois (Manual) na edição guiada."""
    if not diffs:
        return ""
    linhas: list[str] = []
    for diff in diffs:
        rotulo = html.escape(diff.get("rotulo", diff.get("campo", "")))
        antes = html.escape(diff.get("antes_ia", ""))
        depois = html.escape(diff.get("depois_manual", ""))
        linhas.append(
            f'<div class="sinergia-diff-row"><strong>{rotulo}:</strong> '
            f'<span class="sinergia-diff-antes">{antes}</span> → '
            f'<span class="sinergia-diff-depois">{depois}</span></div>'
        )
    return f'<div class="sinergia-diff-box">{"".join(linhas)}</div>'
