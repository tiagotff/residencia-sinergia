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
}

VarianteBadge = Literal["ok", "aviso", "erro", "neutro"]


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


def renderizar_card_campo_revisao(item: dict[str, Any]) -> str:
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
